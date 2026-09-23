"""
LỌC TRÙNG LẶP TRACK (deduplicate) trước khi đưa Event vào bất kỳ module suy
luận quan hệ/vị trí nào khác -- ĐÂY LÀ BƯỚC QUAN TRỌNG mới thêm sau khi phát
hiện thực nghiệm (kiểm tra bằng frame thật trên N001-V001.mov) rằng lỗi
"track_id giả/trùng lặp cho CÙNG 1 vật thể" phổ biến hơn nhiều so với tưởng
tượng ban đầu, và ảnh hưởng trực tiếp tới ĐỘ CHÍNH XÁC VỊ TRÍ mà các module
quan hệ không gian (vehicle_relations.py, module "đi cạnh nhau" mới,
density_timeline.py, zone_events.py) đều dựa vào.

PHÁT HIỆN THỰC NGHIỆM (không suy đoán, đã xem trực tiếp frame + trajectory
thật của N001-V001.mov):
  1. Vật thể ĐỨNG YÊN LÂU (xe tải đỗ, người đứng chờ đèn đỏ trên xe máy) rất
     dễ bị ByteTrack/YOLO tạo track_id MỚI nhiều lần trong lúc track chính
     vẫn còn tồn tại -- track 3 (xe tải đứng yên 0-60s) bị "double-detect"
     thành track_id mới (133, 164, 181, 634, 679) tổng cộng 5 LẦN suốt video,
     mỗi lần vài giây, bbox gần trùng khít (~290x290px, tâm lệch <10px).
  2. Người ngồi trên xe máy dừng đèn đỏ bị YOLO nhận nhầm class "person"
     (thay vì "motorcycle", có lẽ vì tư thế ngồi che khuất phần lớn xe),
     rồi lặp lại kiểu lỗi #1: track 196 (person, đứng yên 1 chỗ) chồng track
     ID mới nhiều lần (246, 296, 403, 546) -- ĐÁNG CHÚ Ý: các track "mới" đó
     lại được gán class "motorcycle" (không phải "person") dù CÙNG 1 vị trí
     -- tức bbox chồng khít gần như tuyệt đối nhưng model đôi lúc phân loại
     đúng là xe máy đôi lúc lại phân thành người, chứng tỏ đây LÀ CÙNG 1 vật
     thể (người + xe máy dính liền) bị model "phân vân" giữa 2 nhãn.
  3. Vì thế: LỌC TRÙNG LẶP PHẢI ÁP DỤNG CHO MỌI CẶP CLASS (không chỉ 2 track
     cùng class như thiết kế ban đầu trong vehicle_relations.py), miễn là vị
     trí+kích thước bbox chồng khít trong suốt khoảng thời gian overlap.

CÁCH LÀM: coi 2 Event là "cùng 1 vật thể" nếu khoảng thời gian hoạt động có
overlap ĐỦ LỚN VÀ trong suốt overlap đó, tâm bbox luôn cách nhau rất gần
(dist_thresh_px) VÀ kích thước bbox luôn gần giống nhau (size_ratio_thresh) --
dùng lại chính xác logic đã kiểm chứng trong
vehicle_relations._is_likely_same_physical_object(), chỉ mở rộng để không
giới hạn theo class nữa.

Khi phát hiện 1 nhóm track trùng lặp (có thể >2 track, như trường hợp track 3
với 5 "bản sao"), MERGE thành 1 Event duy nhất:
  - Giữ track_id của Event có t_start SỚM NHẤT (thường cũng là track có
    n_detections nhiều nhất -- track "chính").
  - Gộp trajectory của mọi track trong nhóm, sắp theo t, downsample lại.
  - t_start/t_end lấy min/max toàn nhóm.
  - representative_frame_* giữ nguyên của track chính (đã chọn frame chất
    lượng cao nhất, không cần chọn lại).
  - cls_name giữ của track chính -- nếu các track trong nhóm có class khác
    nhau (trường hợp #2 ở trên), ưu tiên class KHÔNG PHẢI "person" nếu có
    (vì rất có thể là xe cộ bị nhận nhầm class lúc đứng yên, xem note trên
    mỗi merge để biết rõ đã xảy ra xung đột class).

GIỚI HẠN: đây vẫn là rule dựa trên khoảng cách/kích thước 2D -- 2 vật thể
KHÁC NHAU nhưng đứng cực sát nhau, cùng kích thước (vd 2 xe máy giống hệt đậu
sát nhau, không di chuyển suốt overlap) About lý thuyết có thể bị merge nhầm.
Đã đặt ngưỡng khá chặt (dist_thresh_px=40, size_ratio_thresh=0.15, ÁP DỤNG
CHO TOÀN BỘ overlap chứ không chỉ 1 điểm) để giảm rủi ro này, và luôn ghi lại
merged_from_track_ids để có thể truy vết/kiểm tra lại nếu nghi ngờ.

BỔ SUNG (yêu cầu người dùng: "nhận diện cùng 1 vật thể xuyên suốt của event
đó chứ không tách 1 vật ra thành nhiều vật khác nhau"): logic ở trên (từ đầu
file tới đây) CHỈ xử lý được vật ĐỨNG YÊN bị double-detect (2 track OVERLAP
thời gian, vị trí trùng khít). Còn 1 kiểu tách track khác hẳn -- xe ĐANG DI
CHUYỂN bị che khuất tạm thời (bởi xe khác, cột đèn, hoặc ra khỏi rồi vào lại
khung hình) khiến track cũ KẾT THÚC và track mới XUẤT HIỆN SAU ĐÓ, KHÔNG
OVERLAP thời gian với nhau -- xem _is_likely_occlusion_split() bên dưới, xử
lý bằng cách NGOẠI SUY vị trí theo vận tốc tức thời lúc mất track, so khớp
với vị trí thật lúc track mới xuất hiện. Đây là kiểu merge có RỦI RO NHẦM
CAO HƠN (dự đoán chuyển động tuyến tính, không phải vị trí trùng khít trực
tiếp) nên có thể tắt riêng qua enable_occlusion_merge=False trong
deduplicate_events() nếu nghi ngờ, mà không mất phần gộp track đứng yên.
"""
from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger("traffic_pipeline.track_dedup")


def _traj_bounds(event) -> tuple:
    ts = [p["t"] for p in event.trajectory]
    return (min(ts), max(ts)) if ts else (event.t_start, event.t_end)


def _overlap_window(e1, e2):
    s1, t1 = _traj_bounds(e1)
    s2, t2 = _traj_bounds(e2)
    lo, hi = max(s1, s2), min(t1, t2)
    return (lo, hi) if hi > lo else None


def _interp_position(trajectory: List[dict], t: float):
    if not trajectory:
        return None
    pts = trajectory
    if t <= pts[0]["t"]:
        return pts[0]
    if t >= pts[-1]["t"]:
        return pts[-1]
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        if a["t"] <= t <= b["t"]:
            frac = (t - a["t"]) / (b["t"] - a["t"]) if b["t"] > a["t"] else 0.0
            return {
                "cx": a["cx"] + frac * (b["cx"] - a["cx"]),
                "cy": a["cy"] + frac * (b["cy"] - a["cy"]),
                "w": a["w"] + frac * (b["w"] - a["w"]),
                "h": a["h"] + frac * (b["h"] - a["h"]),
            }
    return pts[-1]


def _is_same_physical_object(e1, e2, dist_thresh_px: float = 40.0, size_ratio_thresh: float = 0.15,
                              min_overlap_sec: float = 0.3) -> bool:
    """Giống hệt logic đã kiểm chứng trong vehicle_relations.py, nhưng KHÔNG
    yêu cầu e1.cls_name == e2.cls_name -- xem GHI CHÚ THỰC NGHIỆM #2 ở đầu
    file (person/motorcycle cùng 1 vật thể bị phân loại khác nhau)."""
    window = _overlap_window(e1, e2)
    if window is None or (window[1] - window[0]) < min_overlap_sec:
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


def _downsample_dict_trajectory(points: List[dict], max_points: int = 20) -> List[dict]:
    """Giống hệt event_timeline._downsample_trajectory() về Ý TƯỞNG (rải đều
    theo index, tối đa max_points điểm) nhưng nhận trực tiếp List[dict]
    {"t","cx","cy","w","h"} đã có sẵn (trajectory của Event, KHÔNG phải
    List[Detection] thô như hàm gốc trong event_timeline.py -- 2 kiểu dữ
    liệu khác nhau nên viết hàm riêng thay vì gọi nhầm hàm gốc)."""
    n = len(points)
    if n <= max_points:
        return points
    step = n / max_points
    return [points[int(i * step)] for i in range(max_points)]


def _merge_group(group: List, downsample_fn=None, occlusion: bool = False):
    """Gộp 1 nhóm Event (đã xác định là cùng 1 vật thể) thành 1 Event duy
    nhất -- giữ lại object của track có t_start sớm nhất (coi là "chính"),
    chỉnh sửa TRỰC TIẾP trên object đó (mutate) rồi trả về, để giữ nguyên mọi
    field khác (vehicle_color, representative_frame_*, ...) đã tính sẵn cho
    track chính, không cần tính lại.

    occlusion: True nếu nhóm này được ghép qua _is_likely_occlusion_split()
    (xe đang di chuyển bị che khuất tạm thời) thay vì _is_same_physical_object()
    (vật đứng yên bị double-detect) -- chỉ ảnh hưởng nội dung merge_note để
    người xem lại biết case nào đáng tin cậy hơn (đứng yên: vị trí trùng
    khít trực tiếp, đáng tin hơn; occlusion: dựa trên ngoại suy chuyển động,
    cần xem lại bằng mắt kỹ hơn -- xem GIỚI HẠN trong _is_likely_occlusion_split)."""
    group_sorted = sorted(group, key=lambda e: e.t_start)
    main = group_sorted[0]
    others = group_sorted[1:]

    all_classes = {e.cls_name for e in group}
    if len(all_classes) > 1 and main.cls_name == "person" and any(c != "person" for c in all_classes):
        # Ưu tiên class KHÔNG PHẢI person nếu có xung đột (xem GHI CHÚ #2) --
        # nhiều khả năng là xe bị nhận nhầm thành người lúc đứng yên, không
        # phải ngược lại (COCO detector hiếm khi nhầm xe thành người ở mọi
        # frame nhưng lại đúng "person" đúng dáng người ngồi rất phổ biến).
        non_person = next(e for e in group if e.cls_name != "person")
        main.cls_name = non_person.cls_name

    main.t_start = min(e.t_start for e in group)
    main.t_end = max(e.t_end for e in group)

    merged_traj = []
    for e in group:
        merged_traj.extend(e.trajectory)
    merged_traj.sort(key=lambda p: p["t"])
    if downsample_fn is not None:
        merged_traj = downsample_fn(merged_traj)
    main.trajectory = merged_traj

    main.n_detections = sum(e.n_detections for e in group)
    setattr(main, "merged_from_track_ids", [e.track_id for e in others])
    if occlusion:
        setattr(main, "merge_note",
                f"Đã gộp {len(group)} track (id={[e.track_id for e in group]}) vì NGOẠI SUY chuyển động cho "
                f"thấy rất có thể là CÙNG 1 xe đang di chuyển bị CHE KHUẤT TẠM THỜI (xe khác/cột đèn/mép khung "
                f"hình) rồi ByteTrack/YOLO gán track_id mới khi xe xuất hiện lại -- ĐỘ TIN CẬY THẤP HƠN case "
                f"đứng yên (dựa trên dự đoán vận tốc không đổi, không phải vị trí trùng khít trực tiếp), LUÔN "
                f"xem representative_frame_names để xác nhận lại bằng mắt trước khi tin.")
    else:
        setattr(main, "merge_note",
                f"Đã gộp {len(group)} track (id={[e.track_id for e in group]}) vì phát hiện chồng khít vị trí+kích "
                f"thước bbox trong toàn bộ thời gian overlap -- rất có thể là CÙNG 1 vật thể bị ByteTrack/YOLO tạo "
                f"track_id mới nhiều lần (thường xảy ra với vật đứng yên lâu). Xem representative_frame_names để "
                f"xác nhận lại bằng mắt nếu nghi ngờ.")
    return main


def _velocity_near(trajectory: List[dict], at_end: bool, window: float = 1.0) -> "tuple | None":
    """Vận tốc TỨC THỜI (px/s theo cx,cy) gần đầu (at_end=False) hoặc gần
    cuối (at_end=True) quỹ đạo -- dùng đúng ý tưởng progressively-expanding
    window đã kiểm chứng trong vehicle_relations._is_moving_at() (trajectory
    downsample ~20 điểm nên window cố định dễ không đủ 2 điểm), NHƯNG ở đây
    cần cả HƯỚNG lẫn ĐỘ LỚN vận tốc (không chỉ True/False như _is_moving_at)
    để ngoại suy vị trí, nên viết riêng thay vì tái dùng thẳng.

    Trả None nếu track quá ngắn/đứng yên (không đủ tin cậy để ngoại suy)."""
    if len(trajectory) < 2:
        return None
    ref_t = trajectory[-1]["t"] if at_end else trajectory[0]["t"]
    for w in (window, window * 2, window * 4, window * 8):
        if at_end:
            nearby = [p for p in trajectory if ref_t - w <= p["t"] <= ref_t]
        else:
            nearby = [p for p in trajectory if ref_t <= p["t"] <= ref_t + w]
        if len(nearby) >= 2:
            nearby = sorted(nearby, key=lambda p: p["t"])
            a, b = nearby[0], nearby[-1]
            dt = b["t"] - a["t"]
            if dt <= 0:
                continue
            vx = (b["cx"] - a["cx"]) / dt
            vy = (b["cy"] - a["cy"]) / dt
            return (vx, vy)
    return None


def _is_likely_occlusion_split(e1, e2, max_gap_sec: float = 2.5, max_predict_err_px: float = 90.0,
                                size_ratio_thresh: float = 0.35, min_speed_px_per_sec: float = 15.0) -> bool:
    """
    PHÁT HIỆN THỰC NGHIỆM (bổ sung sau _is_same_physical_object ở trên):
    logic cũ chỉ xử lý được vật ĐỨNG YÊN bị double-detect (2 track OVERLAP
    thời gian, vị trí trùng khít). Nhưng còn 1 kiểu tách track KHÁC hẳn: xe
    ĐANG DI CHUYỂN bị 1 xe khác (hoặc cột đèn, biển báo, mép khung hình) che
    khuất một lúc ngắn -- track cũ (e1) KẾT THÚC lúc bị che, rồi model detect
    lại "xe mới" (e2) vài frame sau đó tại vị trí cách đó 1 đoạn (đúng bằng
    quãng đường xe đi được trong lúc bị che) -- 2 track này KHÔNG OVERLAP
    thời gian (trái ngược hoàn toàn với case đứng yên ở trên), nên
    _is_same_physical_object() (yêu cầu overlap) không bao giờ bắt được.

    CÁCH XỬ LÝ: coi e1 kết thúc trước, e2 bắt đầu sau (không overlap, cách
    nhau tối đa max_gap_sec) là CÙNG 1 vật thể nếu:
      1. e1 đang di chuyển thật sự (không phải đứng yên) ngay trước khi mất --
         dùng _velocity_near(e1.trajectory, at_end=True).
      2. NGOẠI SUY vị trí e1 sẽ ở đâu tại thời điểm e2 xuất hiện (vị trí cuối
         + vận tốc * khoảng thời gian gap) -- nếu vị trí ngoại suy đó gần vị
         trí THẬT của e2 lúc bắt đầu (trong max_predict_err_px), rất có khả
         năng là CÙNG xe tiếp tục đi theo quán tính.
      3. Kích thước bbox 2 track không lệch quá nhiều (size_ratio_thresh nới
         hơn case đứng yên vì góc nhìn xe có thể đổi chút khi di chuyển qua
         chỗ bị che).
      4. Hướng đi (nếu e2 đủ dài để tính được) phải THUẬN với hướng e1 đang đi
         (không bắt buộc nếu e2 quá ngắn/mới xuất hiện chưa đủ điểm).

    GIỚI HẠN (khác với case đứng yên -- rủi ro nhầm cao hơn, cần thận trọng
    hơn khi tin kết quả):
      - Đây là dự đoán chuyển động TUYẾN TÍNH đơn giản (vận tốc không đổi) --
        xe rẽ ngoặt hoặc đổi tốc độ đột ngột trong lúc bị che sẽ làm ngoại suy
        sai, dẫn đến BỎ SÓT merge (an toàn hơn merge nhầm, nhưng vẫn là hạn
        chế cần biết).
      - 2 xe CÙNG LOẠI, CÙNG HƯỚNG, đi sát nhau (xe sau "thế chỗ" xe trước
        ngay khi xe trước bị che) về lý thuyết vẫn có thể bị ghép nhầm thành
        1 xe -- đã đặt max_gap_sec khá ngắn (2.5s) và max_predict_err_px vừa
        phải (90px, ~1 làn xe máy ở khung hình cỡ 1920x1080) để giảm rủi ro,
        nhưng KHÔNG loại trừ hoàn toàn được bằng hình học 2D thuần -- luôn
        cần xem representative_frame_names để xác nhận lại bằng mắt.
      - merge_note của các case occlusion sẽ ghi rõ "occlusion" để phân biệt
        với case đứng yên (đáng tin cậy hơn), giúp người xem lại biết case
        nào cần kiểm tra kỹ hơn.
    """
    if e1.t_end <= e2.t_start:
        early, late = e1, e2
    elif e2.t_end <= e1.t_start:
        early, late = e2, e1
    else:
        return False  # có overlap thời gian -- không phải occlusion-split, để _is_same_physical_object xử lý

    # ĐÃ PHÁT HIỆN THỰC NGHIỆM (xem frame thật): case đứng yên ở
    # _is_same_physical_object() cho phép khác class (person/motorcycle) vì
    # có BẰNG CHỨNG TRỰC TIẾP là vị trí trùng khít CÙNG THỜI ĐIỂM (overlap).
    # Ở ĐÂY (occlusion, KHÔNG overlap) chỉ có ngoại suy chuyển động làm bằng
    # chứng -- rủi ro nhầm cao hơn nhiều. Đã bắt gặp thật: track20 (người đi
    # bộ đang bước đều giữa đường, t=0.56-1.84s) bị ngoại suy trùng khớp
    # ngẫu nhiên với track140 (1 XE MÁY KHÁC hoàn toàn, đang chạy tới đúng
    # chỗ đó 1.44s sau) chỉ vì tốc độ đi bộ (~85px/s) đủ để qua ngưỡng
    # min_speed_px_per_sec và vị trí ngoại suy tình cờ khớp -- xem ảnh so
    # sánh: rõ ràng là 2 người/xe khác nhau. Vì vậy BẮT BUỘC CÙNG CLASS cho
    # occlusion-merge (khác hẳn case đứng yên), trừ ngoại lệ xử lý riêng nếu
    # cần sau này.
    if early.cls_name != late.cls_name:
        return False

    gap = late.t_start - early.t_end
    if gap <= 0 or gap > max_gap_sec:
        return False

    if not early.trajectory or not late.trajectory:
        return False

    vel = _velocity_near(early.trajectory, at_end=True)
    if vel is None:
        return False
    speed = (vel[0] ** 2 + vel[1] ** 2) ** 0.5
    if speed < min_speed_px_per_sec:
        return False  # early gần như đứng yên lúc mất track -- không phải occlusion đang di chuyển

    last_pt = early.trajectory[-1]
    predicted_cx = last_pt["cx"] + vel[0] * gap
    predicted_cy = last_pt["cy"] + vel[1] * gap

    first_pt_late = late.trajectory[0]
    err = ((predicted_cx - first_pt_late["cx"]) ** 2 + (predicted_cy - first_pt_late["cy"]) ** 2) ** 0.5
    if err > max_predict_err_px:
        return False

    avg_size = (last_pt["w"] + last_pt["h"] + first_pt_late["w"] + first_pt_late["h"]) / 4 or 1.0
    size_diff = abs(last_pt["w"] - first_pt_late["w"]) + abs(last_pt["h"] - first_pt_late["h"])
    if size_diff / (avg_size * 2) > size_ratio_thresh:
        return False

    # Nếu late đủ dài để có hướng riêng, kiểm tra thuận hướng với early --
    # bỏ qua kiểm tra này nếu late quá ngắn/mới xuất hiện (không đủ tin cậy).
    vel_late = _velocity_near(late.trajectory, at_end=False)
    if vel_late is not None:
        speed_late = (vel_late[0] ** 2 + vel_late[1] ** 2) ** 0.5
        if speed_late >= min_speed_px_per_sec:
            dot = (vel[0] * vel_late[0] + vel[1] * vel_late[1]) / (speed * speed_late)
            if dot < 0.3:  # gần như ngược hướng hoặc vuông góc -- khó là cùng xe tiếp tục đi thẳng
                return False

    return True


def deduplicate_events(events: List, dist_thresh_px: float = 40.0, size_ratio_thresh: float = 0.15,
                        enable_occlusion_merge: bool = True, occlusion_max_gap_sec: float = 2.5,
                        occlusion_max_predict_err_px: float = 90.0) -> List:
    """
    events: danh sách Event đầy đủ (mọi class, KHÔNG lọc trước theo
    _VEHICLE_CLASSES như vehicle_relations.py -- vì lỗi này xảy ra CẢ với
    class "person", xem GHI CHÚ THỰC NGHIỆM).

    Trả về danh sách Event MỚI (đã merge các nhóm trùng lặp) -- số lượng
    Event có thể GIẢM so với đầu vào. Mọi Event khác (không thuộc nhóm nào bị
    merge) giữ nguyên object cũ, không copy/không đổi.

    enable_occlusion_merge: BẬT/TẮT riêng phần gộp do OCCLUSION (xe đang di
    chuyển bị che khuất tạm thời rồi xuất hiện lại track_id mới) -- xem
    _is_likely_occlusion_split(). Tách riêng cờ này (mặc định BẬT) vì đây là
    kiểu merge có RỦI RO NHẦM CAO HƠN case đứng yên (dựa trên ngoại suy
    chuyển động, không phải vị trí trùng khít trực tiếp) -- nếu nghi ngờ kết
    quả gộp occlusion sai, có thể tắt bằng cách gọi
    deduplicate_events(events, enable_occlusion_merge=False) mà không mất
    phần gộp track đứng yên (vẫn đáng tin cậy hơn, luôn bật).

    QUAN TRỌNG: hàm này nên chạy 1 LẦN, NGAY SAU build_events_from_tracks(),
    TRƯỚC MỌI module suy luận khác (vehicle_relations, zone_events,
    density_timeline, lane_direction, module "đi cạnh nhau" mới) -- vì mọi
    module đó đều giả định 1 track_id = 1 vật thể thật, và merge càng trễ
    càng khó áp dụng nhất quán (vd density_timeline đã đếm sai trước khi kịp
    merge thì không sửa lại được nữa).
    """
    n = len(events)
    parent = list(range(n))  # union-find để gộp nhóm >2 track (vd track 3 với 5 "bản sao")

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    n_pairs_checked = 0
    n_pairs_merged = 0
    n_occlusion_merged = 0
    for i in range(n):
        for j in range(i + 1, n):
            e1, e2 = events[i], events[j]
            window = _overlap_window(e1, e2)
            if window is not None and (window[1] - window[0]) >= 0.3:
                n_pairs_checked += 1
                if _is_same_physical_object(e1, e2, dist_thresh_px, size_ratio_thresh):
                    union(i, j)
                    n_pairs_merged += 1
                    continue
            if enable_occlusion_merge and _is_likely_occlusion_split(
                    e1, e2, max_gap_sec=occlusion_max_gap_sec,
                    max_predict_err_px=occlusion_max_predict_err_px):
                union(i, j)
                n_occlusion_merged += 1

    groups = {}
    for idx in range(n):
        root = find(idx)
        groups.setdefault(root, []).append(events[idx])

    result = []
    n_groups_merged = 0
    for group in groups.values():
        if len(group) == 1:
            result.append(group[0])
        else:
            n_groups_merged += 1
            is_occlusion_group = len(group) == 2 and _overlap_window(group[0], group[1]) is None
            result.append(_merge_group(group, downsample_fn=_downsample_dict_trajectory,
                                        occlusion=is_occlusion_group))

    logger.info(f"deduplicate_events: đã kiểm tra {n_pairs_checked} cặp track có overlap thời gian "
                f"({n_pairs_merged} trùng lặp) + phát hiện {n_occlusion_merged} cặp nghi occlusion "
                f"(xe di chuyển bị che khuất tạm thời) -- gộp thành {n_groups_merged} nhóm, "
                f"còn lại {len(result)}/{n} track sau khi lọc.")
    return result
