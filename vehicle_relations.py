"""
Suy luận QUAN HỆ giữa các xe (không phải thuộc tính của 1 xe đơn lẻ) từ
quỹ đạo (trajectory) đã lưu trong mỗi Event -- trả lời trực tiếp các câu
query như "xe nào chạy trước xe nào", "xe nào vượt xe nào", "có va chạm
không" MÀ KHÔNG CẦN embedding đoán mò, vì đây là suy luận HÌNH HỌC trên toạ
độ tâm bbox theo thời gian đã trích xuất chính xác từ tracking.

QUAN TRỌNG -- đã cân nhắc kỹ vì đây là phần dễ sai (theo đúng lo ngại ban
đầu khi thảo luận chiến thuật): camera giao thông tĩnh, xe che khuất nhau
liên tục là chuyện BÌNH THƯỜNG (không phải va chạm), và "vượt" cũng cần
phân biệt với "đổi làn nhẹ" hay "chạy song song rồi tự tách ra". Vì vậy mọi
rule ở đây đều:
  1. Chỉ áp dụng cho 2 track có THỜI GIAN HOẠT ĐỘNG CHỒNG LẤN đáng kể (tránh
     so sánh 2 xe không hề xuất hiện cùng lúc).
  2. Gắn kèm "confidence" + "note" nhắc rõ đây là GỢI Ý ứng viên cần xem lại
     bằng mắt (qua representative_frame_names của track liên quan), KHÔNG
     phải khẳng định chắc chắn -- đặc biệt với va chạm, nơi false positive
     có hậu quả nghiêm trọng nếu người dùng tin tưởng tuyệt đối.
"""
from __future__ import annotations

from typing import List, Optional


def _traj_bounds(event) -> tuple:
    ts = [p["t"] for p in event.trajectory]
    return (min(ts), max(ts)) if ts else (event.t_start, event.t_end)


def _overlap_window(e1, e2) -> Optional[tuple]:
    s1, t1 = _traj_bounds(e1)
    s2, t2 = _traj_bounds(e2)
    lo, hi = max(s1, s2), min(t1, t2)
    return (lo, hi) if hi > lo else None


def _interp_position(trajectory: List[dict], t: float) -> Optional[dict]:
    """Nội suy tuyến tính vị trí tâm bbox tại thời điểm t từ 2 điểm quỹ đạo
    gần nhất. Trả None nếu t nằm ngoài khoảng trajectory."""
    if not trajectory:
        return None
    if t <= trajectory[0]["t"]:
        return trajectory[0]
    if t >= trajectory[-1]["t"]:
        return trajectory[-1]
    for i in range(len(trajectory) - 1):
        a, b = trajectory[i], trajectory[i + 1]
        if a["t"] <= t <= b["t"]:
            if b["t"] == a["t"]:
                return a
            frac = (t - a["t"]) / (b["t"] - a["t"])
            return {
                "cx": a["cx"] + frac * (b["cx"] - a["cx"]),
                "cy": a["cy"] + frac * (b["cy"] - a["cy"]),
                "w": a["w"] + frac * (b["w"] - a["w"]),
                "h": a["h"] + frac * (b["h"] - a["h"]),
            }
    return trajectory[-1]


def _direction_of_travel(trajectory: List[dict]) -> Optional[tuple]:
    """Vector hướng di chuyển thô (dx, dy) từ điểm đầu tới điểm cuối quỹ
    đạo -- dùng để xác định trục "trước/sau" theo đúng hướng xe đang chạy
    (không cứng nhắc theo trục x hay y của khung hình, vì xe có thể chạy
    chéo trong góc camera nghiêng)."""
    if len(trajectory) < 2:
        return None
    p0, p1 = trajectory[0], trajectory[-1]
    dx, dy = p1["cx"] - p0["cx"], p1["cy"] - p0["cy"]
    mag = (dx ** 2 + dy ** 2) ** 0.5
    if mag < 5:  # xe gần như đứng yên -- không đủ tin cậy để xác định hướng
        return None
    return (dx / mag, dy / mag)


_VEHICLE_CLASSES = ("car", "motorcycle", "bus", "truck", "bicycle")


def _is_likely_same_physical_object(e1, e2, dist_thresh_px: float = 40.0, size_ratio_thresh: float = 0.15) -> bool:
    """
    ĐÃ PHÁT HIỆN KHI TEST THẬT: ByteTrack đôi khi "đánh mất" 1 track đang
    gần như đứng yên (vd. xe tải đỗ/dừng lâu) rồi gán NHẦM track_id MỚI cho
    CHÍNH đối tượng đó (do bị che khuất thoáng qua hoặc detect bị bỏ sót vài
    frame). Kết quả: 2 "track" khác nhau nhưng thực chất là 1 xe DUY NHẤT
    đứng yên tại chỗ, có thời gian hoạt động gần kề/chồng lấn nhau và vị trí
    + kích thước bbox GẦN NHƯ GIỐNG HỆT NHAU.

    Nếu không lọc trường hợp này, detect_collision_candidates() sẽ báo
    "va chạm" giữa 1 xe với CHÍNH NÓ (đã xác nhận bằng cách so trajectory
    thật: track 3 và track 133 trên N001-V001.mov có toạ độ tâm bbox lệch
    nhau <5px suốt thời gian chồng lấn -- rõ ràng là cùng 1 xe tải đứng yên,
    không phải 2 xe khác nhau).

    So sánh vị trí trung bình trong khoảng thời gian CHỒNG LẤN của 2 track:
    nếu khoảng cách tâm bbox nhỏ hơn dist_thresh_px VÀ kích thước lệch nhau
    dưới size_ratio_thresh (tỉ lệ) trong SUỐT khoảng chồng lấn, coi là cùng
    1 vật thể -- loại khỏi mọi phân tích quan hệ (không phải "va chạm",
    không phải "vượt", không tính là 2 xe khác nhau).
    """
    window = _overlap_window(e1, e2)
    if window is None:
        return False
    t_lo, t_hi = window
    n_checks = max(3, int((t_hi - t_lo) * 2))
    for k in range(n_checks + 1):
        t = t_lo + (t_hi - t_lo) * k / n_checks
        p1, p2 = _interp_position(e1.trajectory, t), _interp_position(e2.trajectory, t)
        if not p1 or not p2:
            continue
        dist = ((p1["cx"] - p2["cx"]) ** 2 + (p1["cy"] - p2["cy"]) ** 2) ** 0.5
        avg_size = (p1["w"] + p1["h"] + p2["w"] + p2["h"]) / 4 or 1.0
        if dist > dist_thresh_px:
            return False
        size_diff = abs(p1["w"] - p2["w"]) + abs(p1["h"] - p2["h"])
        if size_diff / (avg_size * 2) > size_ratio_thresh:
            return False
    return True


def detect_order_and_overtake(
    events: List,
    min_overlap_sec: float = 1.0,
    overtake_min_swap_px: float = 25.0,
) -> List[dict]:
    """
    Với mỗi cặp track phương tiện có thời gian hoạt động chồng lấn đủ lâu
    VÀ đang di chuyển theo hướng tương tự nhau (khả năng cùng làn/hướng
    đường), so sánh vị trí dọc theo trục hướng di chuyển tại đầu và cuối
    khoảng chồng lấn:
      - Nếu thứ tự trước/sau KHÔNG đổi suốt khoảng chồng lấn -> quan hệ
        "in_front_of" / "behind" ổn định.
      - Nếu thứ tự ĐẢO NGƯỢC (xe A ban đầu sau, kết thúc lại trước xe B,
        lệch đủ xa overtake_min_swap_px) -> "overtake" (A vượt B).

    Đây là suy luận GẦN ĐÚNG dựa trên hình học 2D chiếu từ góc camera (không
    phải toạ độ thế giới thực), nên với xe ở xa/góc nghiêng có thể kém chính
    xác hơn xe ở gần -- luôn trả kèm representative_frame_names để xem lại.
    """
    vehicle_events = [e for e in events if e.cls_name in _VEHICLE_CLASSES and len(e.trajectory) >= 2]
    relations = []

    for i in range(len(vehicle_events)):
        for j in range(i + 1, len(vehicle_events)):
            e1, e2 = vehicle_events[i], vehicle_events[j]
            window = _overlap_window(e1, e2)
            if window is None or window[1] - window[0] < min_overlap_sec:
                continue
            if _is_likely_same_physical_object(e1, e2):
                continue  # cùng 1 xe bị tracker đổi track_id giữa chừng -- xem ghi chú hàm trên

            dir1, dir2 = _direction_of_travel(e1.trajectory), _direction_of_travel(e2.trajectory)
            if dir1 is None or dir2 is None:
                continue
            # cùng hướng nếu góc giữa 2 vector hướng < ~60 độ (dot product > 0.5)
            same_direction = (dir1[0] * dir2[0] + dir1[1] * dir2[1]) > 0.5
            if not same_direction:
                continue

            t_lo, t_hi = window
            p1_start, p2_start = _interp_position(e1.trajectory, t_lo), _interp_position(e2.trajectory, t_lo)
            p1_end, p2_end = _interp_position(e1.trajectory, t_hi), _interp_position(e2.trajectory, t_hi)
            if not all([p1_start, p2_start, p1_end, p2_end]):
                continue

            # chiếu độ lệch vị trí lên trục hướng di chuyển trung bình để
            # biết ai "ở trước" (proj lớn hơn = đã đi xa hơn theo hướng di chuyển)
            avg_dir = ((dir1[0] + dir2[0]) / 2, (dir1[1] + dir2[1]) / 2)

            def proj(p):
                return p["cx"] * avg_dir[0] + p["cy"] * avg_dir[1]

            diff_start = proj(p1_start) - proj(p2_start)
            diff_end = proj(p1_end) - proj(p2_end)

            e1_ahead_start = diff_start > 0
            e1_ahead_end = diff_end > 0

            if e1_ahead_start == e1_ahead_end:
                # thứ tự ổn định suốt khoảng chồng lấn -- quan hệ trước/sau, không phải vượt
                ahead, behind = (e1, e2) if e1_ahead_end else (e2, e1)
                relations.append({
                    "relation": "in_front_of",
                    "front_track_id": ahead.track_id, "front_cls": ahead.cls_name,
                    "behind_track_id": behind.track_id, "behind_cls": behind.cls_name,
                    "t_start": round(t_lo, 2), "t_end": round(t_hi, 2),
                    "frame_names_front": ahead.representative_frame_names,
                    "frame_names_behind": behind.representative_frame_names,
                })
            elif abs(diff_end - diff_start) >= overtake_min_swap_px:
                # thứ tự đảo ngược VÀ đủ lớn để không phải nhiễu đo đạc -- ứng viên "vượt"
                overtaker, overtaken = (e1, e2) if e1_ahead_end else (e2, e1)
                relations.append({
                    "relation": "overtake",
                    "overtaker_track_id": overtaker.track_id, "overtaker_cls": overtaker.cls_name,
                    "overtaken_track_id": overtaken.track_id, "overtaken_cls": overtaken.cls_name,
                    "t_start": round(t_lo, 2), "t_end": round(t_hi, 2),
                    "frame_names_overtaker": overtaker.representative_frame_names,
                    "frame_names_overtaken": overtaken.representative_frame_names,
                    "note": "Suy luận từ đảo thứ tự vị trí theo hướng di chuyển -- gợi ý ứng viên, "
                            "NÊN xem lại frame/video thật để xác nhận (có thể là đổi làn thường, "
                            "không nhất thiết là vượt ẩu).",
                })

    return relations


def detect_collision_candidates(
    events: List,
    proximity_px: float = 30.0,
    min_speed_drop_ratio: float = 0.5,
) -> List[dict]:
    """
    Ứng viên VA CHẠM: 2 track phương tiện có bbox GẦN CHẠM NHAU (khoảng
    cách tâm bbox < proximity_px, đã chuẩn hoá thô theo kích thước bbox)
    TẠI CÙNG THỜI ĐIỂM, VÀ ít nhất 1 trong 2 xe có tốc độ di chuyển GIẢM
    ĐỘT NGỘT ngay sau thời điểm đó (dấu hiệu phanh gấp/dừng do va chạm,
    phân biệt với việc chỉ đơn giản đi ngang qua nhau).

    CẢNH BÁO QUAN TRỌNG: đây là rule THÔ, tỉ lệ báo nhầm CAO trên camera
    toàn cảnh giao lộ (xe che khuất nhau khi rẽ, dừng đèn đỏ sát nhau, hoặc
    góc camera khiến 2 xe ở xa nhau trong thực tế trông như chạm nhau trong
    ảnh 2D, đều có thể trùng điều kiện này dù KHÔNG hề va chạm). Vì vậy:
      - Luôn trả về kèm cả 2 representative_frame_names để bắt buộc xem lại
        bằng mắt trước khi kết luận.
      - Không dùng để tự động cảnh báo/hành động, chỉ dùng làm ứng viên gợi
        ý trong kết quả search khi người dùng hỏi về va chạm.
    """
    vehicle_events = [e for e in events if e.cls_name in _VEHICLE_CLASSES and len(e.trajectory) >= 2]
    candidates = []

    for i in range(len(vehicle_events)):
        for j in range(i + 1, len(vehicle_events)):
            e1, e2 = vehicle_events[i], vehicle_events[j]
            window = _overlap_window(e1, e2)
            if window is None:
                continue
            if _is_likely_same_physical_object(e1, e2):
                continue  # cùng 1 xe bị tracker đổi track_id giữa chừng -- KHÔNG phải va chạm
            t_lo, t_hi = window
            # quét vài mốc thời gian trong cửa sổ chồng lấn (đủ dày để không bỏ sót va chạm ngắn)
            n_samples = max(3, int((t_hi - t_lo) * 4))
            for k in range(n_samples + 1):
                t = t_lo + (t_hi - t_lo) * k / n_samples
                p1, p2 = _interp_position(e1.trajectory, t), _interp_position(e2.trajectory, t)
                if not p1 or not p2:
                    continue
                dist = ((p1["cx"] - p2["cx"]) ** 2 + (p1["cy"] - p2["cy"]) ** 2) ** 0.5
                # chuẩn hoá theo kích thước trung bình 2 bbox -- 2 xe to ở gần camera
                # "gần nhau" theo pixel tuyệt đối vẫn có thể còn cách xa thực tế
                avg_size = ((p1["w"] + p1["h"] + p2["w"] + p2["h"]) / 4) or 1.0
                if dist / avg_size > (proximity_px / 50.0):
                    continue

                speed_drop = _check_speed_drop(e1.trajectory, t) or _check_speed_drop(e2.trajectory, t)
                if speed_drop:
                    candidates.append({
                        "event_type": "possible_collision",
                        "track_id_1": e1.track_id, "cls_1": e1.cls_name,
                        "track_id_2": e2.track_id, "cls_2": e2.cls_name,
                        "t": round(t, 2),
                        "frame_names_1": e1.representative_frame_names,
                        "frame_names_2": e2.representative_frame_names,
                        "note": "CẢNH BÁO: chỉ là ỨNG VIÊN dựa trên rule hình học thô (2 bbox gần "
                                "nhau + tốc độ giảm đột ngột), tỉ lệ báo nhầm cao trên camera toàn "
                                "cảnh (xe che khuất nhau, dừng sát đèn đỏ...). BẮT BUỘC xem lại "
                                "video/frame thật trước khi kết luận có va chạm.",
                    })
                    break  # đủ 1 mốc khớp trong cửa sổ này là đủ để báo ứng viên, tránh trùng lặp

    return candidates


def _check_speed_drop(trajectory: List[dict], t: float, window: float = 1.0) -> bool:
    """Kiểm tra tốc độ NGAY SAU thời điểm t có giảm mạnh so với tốc độ NGAY
    TRƯỚC đó không (dấu hiệu phanh gấp/dừng đột ngột)."""
    before = [p for p in trajectory if t - window <= p["t"] < t]
    after = [p for p in trajectory if t < p["t"] <= t + window]
    if len(before) < 2 or len(after) < 2:
        return False

    def avg_speed(points):
        dist = sum(
            ((points[i + 1]["cx"] - points[i]["cx"]) ** 2 + (points[i + 1]["cy"] - points[i]["cy"]) ** 2) ** 0.5
            for i in range(len(points) - 1)
        )
        dt = points[-1]["t"] - points[0]["t"]
        return dist / dt if dt > 0 else 0.0

    speed_before, speed_after = avg_speed(before), avg_speed(after)
    if speed_before < 1e-3:
        return False
    return (speed_after / speed_before) < 0.5


def _is_moving_at(trajectory: List[dict], t: float, window: float = 1.0, min_speed_px_per_sec: float = 15.0) -> bool:
    """True nếu vật thể đang di chuyển với tốc độ đáng kể TẠI THỜI ĐIỂM t cụ
    thể (không phải trung bình cả track) -- xem giải thích chi tiết trong
    detect_side_by_side() về lý do cần kiểm tra này (tránh báo "đi cạnh" cho
    xe đang đậu/dừng chỉ vì 1 xe khác đi ngang qua gần đó).

    ĐÃ SỬA (phát hiện thực nghiệm): trajectory chỉ lưu tối đa ~20 điểm
    downsample cho CẢ track (có thể dài hàng chục giây) -- với window cố
    định 1.0s như thiết kế ban đầu, nhiều thời điểm t sẽ KHÔNG có đủ 2 điểm
    trong window dù xe THỰC SỰ đang di chuyển nhanh (khoảng cách giữa 2 điểm
    downsample liên tiếp có thể > 2s với track dài). Cách sửa: nếu không đủ
    điểm trong window ban đầu, MỞ RỘNG dần window (nhân đôi, tối đa vài lần)
    cho tới khi có đủ 2 điểm HOẶC hết giới hạn mở rộng -- vẫn ưu tiên window
    hẹp trước để giữ tính "tức thời" khi có đủ dữ liệu dày, chỉ nới ra khi
    cần thiết."""
    for w in (window, window * 2, window * 4, window * 8):
        nearby = [p for p in trajectory if t - w <= p["t"] <= t + w]
        if len(nearby) >= 2:
            nearby = sorted(nearby, key=lambda p: p["t"])
            dist = sum(
                ((nearby[i + 1]["cx"] - nearby[i]["cx"]) ** 2 + (nearby[i + 1]["cy"] - nearby[i]["cy"]) ** 2) ** 0.5
                for i in range(len(nearby) - 1)
            )
            dt = nearby[-1]["t"] - nearby[0]["t"]
            speed = dist / dt if dt > 0 else 0.0
            return speed >= min_speed_px_per_sec
    return False


def detect_side_by_side(
    events: List,
    min_overlap_sec: float = 0.6,
    max_lateral_ratio: float = 1.5,
    max_longitudinal_ratio: float = 0.8,
    min_same_direction_dot: float = 0.7,
) -> List[dict]:
    """
    Phát hiện 2 xe ĐI SONG SONG/CẠNH NHAU (khác với "in_front_of"/"overtake"
    trong detect_order_and_overtake(), vốn xét theo TRỤC DỌC hướng di chuyển)
    -- đây là phần còn thiếu để trả lời trực tiếp kiểu câu hỏi "xe tải xanh
    dương đi BÊN CẠNH ô tô đỏ" (xem thảo luận với người dùng).

    CÁCH LÀM (đã kiểm chứng bằng trajectory + frame thật trên N001-V001.mov,
    xem ví dụ track 160/161 -- 1 ô tô và 1 xe máy chở 2 người đi song song rõ
    ràng trong frame thật đã xem tại t=8.5s):
      1. Yêu cầu 2 xe di chuyển CÙNG HƯỚNG (dot product giữa 2 vector hướng >
         min_same_direction_dot -- chặt hơn detect_order_and_overtake() một
         chút vì "cạnh nhau" cần 2 xe THỰC SỰ cùng làn/hướng, không chỉ gần
         giống hướng).
      2. Dựng hệ trục cục bộ tại mỗi thời điểm: trục "dọc" (longitudinal)
         theo hướng di chuyển trung bình, trục "ngang" (lateral) VUÔNG GÓC
         với trục dọc -- đây là cách duy nhất hợp lý để định nghĩa "trái/
         phải" trên camera nhìn nghiêng (không thể dùng thẳng trục x của
         khung hình, vì hướng đường có thể chéo so với khung hình).
      3. "Cạnh nhau" tại 1 thời điểm nếu: độ lệch NGANG (lateral) đủ nhỏ so
         với kích thước xe (max_lateral_ratio) NHƯNG > 0 (có lệch ngang thật,
         không phải chồng lên nhau -- 2 xe chồng gần khít nhau đã bị lọc bởi
         _is_likely_same_physical_object) VÀ độ lệch DỌC (longitudinal) đủ
         nhỏ (max_longitudinal_ratio) -- tức 2 xe THỰC SỰ ngang hàng nhau,
         không phải 1 xe đã vượt lên trước xa.
      4. Yêu cầu trạng thái "cạnh nhau" duy trì liên tục ít nhất
         min_overlap_sec (không chỉ 1 khung hình thoáng qua khi 2 xe tình cờ
         cắt ngang nhau lúc đổi làn) -- tránh báo nhầm các trường hợp thoáng
         qua không phải "đi cùng nhau" thật.

    Trả về danh sách dict {track_id_1, cls_1, track_id_2, cls_2, side (xe 1
    ở "left"/"right" so với xe 2, theo trục ngang cục bộ), t_start, t_end,
    frame_names_1, frame_names_2, note}.

    GIỚI HẠN: cũng là suy luận hình học 2D trên góc camera nghiêng, không
    phải toạ độ thế giới thực -- 2 xe ở khoảng cách xa nhau thật (1 xe ở làn
    ngoài cùng, 1 xe ở vỉa hè bên kia đường) có thể trông "gần" nhau trên
    ảnh nếu camera ở xa. LUÔN xem lại frame thật trước khi tin.
    """
    vehicle_events = [e for e in events if e.cls_name in _VEHICLE_CLASSES and len(e.trajectory) >= 2]
    results = []

    for i in range(len(vehicle_events)):
        for j in range(i + 1, len(vehicle_events)):
            e1, e2 = vehicle_events[i], vehicle_events[j]
            window = _overlap_window(e1, e2)
            if window is None or window[1] - window[0] < min_overlap_sec:
                continue
            if _is_likely_same_physical_object(e1, e2):
                continue

            dir1, dir2 = _direction_of_travel(e1.trajectory), _direction_of_travel(e2.trajectory)
            if dir1 is None or dir2 is None:
                continue
            dot = dir1[0] * dir2[0] + dir1[1] * dir2[1]
            if dot < min_same_direction_dot:
                continue
            avg_dir = ((dir1[0] + dir2[0]) / 2, (dir1[1] + dir2[1]) / 2)
            mag = (avg_dir[0] ** 2 + avg_dir[1] ** 2) ** 0.5
            if mag < 1e-6:
                continue
            avg_dir = (avg_dir[0] / mag, avg_dir[1] / mag)
            perp_dir = (-avg_dir[1], avg_dir[0])  # xoay 90 độ -- trục "ngang" cục bộ

            t_lo, t_hi = window
            n_samples = max(3, int((t_hi - t_lo) * 4))
            side_by_side_spans = []  # danh sách (t_start, t_end, lateral_signed) liên tục
            current_span_start = None
            current_lateral_signs = []

            for k in range(n_samples + 1):
                t = t_lo + (t_hi - t_lo) * k / n_samples
                p1, p2 = _interp_position(e1.trajectory, t), _interp_position(e2.trajectory, t)
                is_side_by_side = False
                lateral_signed = 0.0
                if p1 and p2:
                    # PHẢI kiểm tra CẢ 2 xe đang thực sự DI CHUYỂN tại chính
                    # thời điểm t này -- KHÔNG chỉ dựa vào hướng trung bình
                    # toàn track (dir1/dir2 ở trên tính từ điểm đầu-cuối cả
                    # track, có thể "giả tạo" nếu xe đứng yên phần lớn thời
                    # gian rồi di chuyển 1 đoạn ngắn). Đã phát hiện thực
                    # nghiệm: track 205 (xe đậu, gần như đứng yên suốt
                    # 12s-40s) bị báo "đi cạnh" HÀNG LOẠT xe máy khác chỉ vì
                    # đi ngang qua gần vị trí nó đậu -- SAI, vì xe đậu không
                    # "đi cạnh" xe đang chạy. Yêu cầu tốc độ tức thời của CẢ
                    # 2 xe (trong cửa sổ nhỏ quanh t) đủ lớn mới tính.
                    if not (_is_moving_at(e1.trajectory, t) and _is_moving_at(e2.trajectory, t)):
                        pass
                    else:
                        dx, dy = p2["cx"] - p1["cx"], p2["cy"] - p1["cy"]
                        lateral = dx * perp_dir[0] + dy * perp_dir[1]
                        longitudinal = dx * avg_dir[0] + dy * avg_dir[1]
                        avg_size = ((p1["w"] + p1["h"] + p2["w"] + p2["h"]) / 4) or 1.0
                        if abs(longitudinal) <= avg_size * max_longitudinal_ratio and \
                           0 < abs(lateral) <= avg_size * max_lateral_ratio:
                            is_side_by_side = True
                            lateral_signed = lateral

                if is_side_by_side:
                    if current_span_start is None:
                        current_span_start = t
                    current_lateral_signs.append(lateral_signed)
                else:
                    if current_span_start is not None:
                        side_by_side_spans.append((current_span_start, t, current_lateral_signs))
                    current_span_start = None
                    current_lateral_signs = []
            if current_span_start is not None:
                side_by_side_spans.append((current_span_start, t_hi, current_lateral_signs))

            for span_start, span_end, laterals in side_by_side_spans:
                if span_end - span_start < min_overlap_sec or not laterals:
                    continue
                avg_lateral = sum(laterals) / len(laterals)
                # lateral > 0 nghĩa là e2 lệch về phía "bên phải" của hướng di
                # chuyển e1 (theo quy ước xoay 90 độ (-dy,dx) ở trên) -- tức
                # e1 ở bên TRÁI của e2, e2 ở bên PHẢI của e1.
                if avg_lateral > 0:
                    left_e, right_e = e1, e2
                else:
                    left_e, right_e = e2, e1
                results.append({
                    "event_type": "side_by_side",
                    "left_track_id": left_e.track_id, "left_cls": left_e.cls_name,
                    "right_track_id": right_e.track_id, "right_cls": right_e.cls_name,
                    "t_start": round(span_start, 2), "t_end": round(span_end, 2),
                    "frame_names_left": left_e.representative_frame_names,
                    "frame_names_right": right_e.representative_frame_names,
                    "note": "Ứng viên 'đi cạnh nhau' dựa trên suy luận hình học 2D (lệch ngang nhỏ, lệch dọc "
                            "nhỏ, cùng hướng di chuyển, duy trì liên tục) -- trái/phải xác định THEO HƯỚNG DI "
                            "CHUYỂN của xe, không phải theo khung hình camera. CẦN xem lại frame thật để xác "
                            "nhận, đặc biệt với xe ở xa/góc camera nghiêng.",
                })

    return results
