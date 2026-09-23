"""
Tự học hướng di chuyển "chuẩn" của từng vùng trong khung hình TỪ VÀO CHÍNH
quy đạo đa số xe đã track được trong video -- KHÔNG cần nhập tay hướng
làn, KHÔNG cần model AI riêng, và không cần biết bản đồ đường thuạt thực tế.

Cách làm: chia khung hình thành lưới ô vuông (grid cell) cố định kích
thước. Với mỗi ô, gộp vector hướng di chuyển của MỌI điểm quy đạo
(từ mọi track phương tiện) đi qua ô đó, rồi tính vector trung bình (hướng
đa số -- đúng vậy vì giao thông thật luôn có đa số xe đi đúng chiều).

Sau đó, với 1 track bất kỳ, so hướng di chuyển của nó tại mỗi điểm
quy đạo với hướng đa số đã học ở ô đó -- nếu lệch quá nhiều (góc
> ~120 độ, tức đi ngược hẳn) ở đủ nhiều điểm trên quy đạo, coi là
ứng viên "đi sai làn/ngược chiều".

GIớI HẠN QUAN TRỌNG (nói rõ để không tạo ảo giác về độ chính xác):
  - Cần ĐỦ xe mẫu đi qua mỗi ô mới học được hướng đúng -- video ngắn
    hoặc ô ít xe sẽ không đủ tin cậy (có đánh dấu min_samples, ô không đủ
    mẫu sẽ KHÔNG dùng để so sánh, tránh báo sai hàng loạt).
  - Giao lộ (nơi xe rẽ) có hướng đa dạng THẬT (rẽ trái/phải/đi thẳng đều
    đúng) nên dễ bị coi nhầm là "sai chiều" nếu hướng đa số học được chỉ
    nghiêng về 1 hướng -- vì vậy CHỈ áp dụng ràng buộc cho ô có độ đồng
    thuận hướng cao (circular variance thấp) -- ô giao lộ nhiều hướng khác
    nhau sẽ tự động BỊ LOẠI, không dùng để gán cờ sai chiều.
  - ĐÃ PHÁT HIỆN THỰC TẾ (kiểm chứng bằng frame thật trên N001-V001.mov):
    circular-variance-filter một mình KHÔNG đủ để loại toàn bộ giao lộ --
    xe đi trên ĐƯỜNG CẮT NGANG (vd Le Hong Phong cắt An Duong Vuong) gần
    khu vực giao lộ có thể tạo ra 1 ô với hướng khá đồng nhất (vì tất cả xe
    rẽ/băng ngang tại đó đi cùng 1 hướng cắt ngang), nhưng đó là hướng của
    ĐƯỜNG KHÁC chứ không phải "đi ngược chiều" trên cùng 1 đường. Track 211
    và track 780 (car) bị gắn cờ sai theo cách này trong thử nghiệm thực tế
    -- cả 2 đều là xe băng ngang qua giao lộ, KHÔNG phải đi ngược chiều.
    Do đó module này áp dụng thêm 1 lớp lọc: loại các ô lưới nằm TRONG hoặc
    SÁT vùng giao lộ đã biết (dùng CROSSWALK_POLYGON làm neo tham chiếu vị
    trí giao lộ, mở rộng thêm biên độ) khỏi việc dùng để gán cờ sai chiều,
    xem intersection_polygon/intersection_margin_px bên dưới. Đây vẫn CHỈ
    là giảm thiểu, không loại bỏ hoàn toàn rủi ro -- luôn cần xem lại frame
    thật trước khi kết luận.
  - Luôn là ứng viên gợi ý, kèm representative_frame_names để xem lại.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import List, Optional


def _cell_key(cx: float, cy: float, cell_size: float) -> tuple:
    return (int(cx // cell_size), int(cy // cell_size))


def _point_near_polygon(x: float, y: float, polygon: List[tuple], margin_px: float) -> bool:
    """True nếu điểm (x,y) nằm TRONG polygon hoặc trong khoảng margin_px
    quanh bounding box của polygon -- dùng làm cách xấp xỉ đơn giản, rẻ, để
    loại các ô lưới gần khu vực giao lộ (nơi CROSSWALK_POLYGON đã đo) khỏi
    việc học/so hướng, vì tại đó nhiều dòng xe hợp lệ cắt nhau (xem giới hạn
    nêu trong docstring module). Không cần chính xác tuyệt đối theo hình học
    -- chỉ cần đủ để loại các ô lưới sát giao lộ."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (min(xs) - margin_px) <= x <= (max(xs) + margin_px) and \
           (min(ys) - margin_px) <= y <= (max(ys) + margin_px)


def _cell_center(key: tuple, cell_size: float) -> tuple:
    gx, gy = key
    return (gx * cell_size + cell_size / 2.0, gy * cell_size + cell_size / 2.0)


def _track_direction_vectors(trajectory: List[dict]) -> List[dict]:
    """Trả về danh sách vector hướng di chuyển tại từng đoạn liên tiếp của
    quỹ đạo: [{"cx", "cy", "dx", "dy"}] -- (cx, cy) là vị trí trung điểm
    đoạn, (dx, dy) là vector hướng đã chuẩn hoá (unit vector)."""
    vectors = []
    for i in range(len(trajectory) - 1):
        a, b = trajectory[i], trajectory[i + 1]
        dx, dy = b["cx"] - a["cx"], b["cy"] - a["cy"]
        mag = math.hypot(dx, dy)
        if mag < 3:  # đứng yên gần như tuyệt đối -- không mang thông tin hướng, bỏ qua
            continue
        vectors.append({
            "cx": (a["cx"] + b["cx"]) / 2, "cy": (a["cy"] + b["cy"]) / 2,
            "dx": dx / mag, "dy": dy / mag,
        })
    return vectors


class LaneDirectionModel:
    def __init__(self, cell_size: float = 80.0, min_samples: int = 8, max_circular_std: float = 0.7,
                 intersection_polygon: Optional[List[tuple]] = None, intersection_margin_px: float = 150.0):
        """
        cell_size: kích thước ô lưới (pixel) -- nhỏ hơn thì phân giải theo
        không gian tốt hơn nhưng cần nhiều mẫu hơn mới đủ tin cậy; 80px là
        điểm cân bằng hợp lý cho khung hình 1920x1080 của camera giao thông
        điển hình (khoảng 24x14 ô).
        min_samples: số vector hướng tối thiểu trong 1 ô mới coi là đủ dữ
        liệu để học hướng đa số (ô ít mẫu quá dễ học sai do nhiễu).
        max_circular_std: ngưỡng độ lệch hướng trong ô (0 = mọi vector cùng
        hướng tuyệt đối, ~1.4 = hướng ngẫu nhiên hoàn toàn) -- ô có độ lệch
        cao hơn ngưỡng này (thường là giao lộ, nơi xe rẽ nhiều hướng khác
        nhau) sẽ KHÔNG dùng để phát hiện sai chiều, xem docstring module.
        intersection_polygon: polygon xấp xỉ vị trí giao lộ (mặc định dùng
        config.CROSSWALK_POLYGON nếu có sẵn, vì giao lộ luôn ở ngay cạnh
        vạch kẻ đường đã đo) -- các ô lưới nằm trong/gần polygon này (theo
        intersection_margin_px) sẽ KHÔNG dùng để học/so hướng, vì đây là nơi
        nhiều luồng xe hợp lệ (rẽ, băng ngang từ đường khác) cắt nhau và
        riêng circular-variance-filter không đủ để loại hết, xem docstring.
        intersection_margin_px: biên độ mở rộng quanh bounding box của
        intersection_polygon (pixel) -- 150px là ước lượng thận trọng, rộng
        hơn kích thước 1 ô lưới, để trùm cả phần đường ngay sát giao lộ nơi
        xe mới rẽ vào/ra còn chưa ổn định theo hướng làn chính.
        """
        self.cell_size = cell_size
        self.min_samples = min_samples
        self.max_circular_std = max_circular_std
        if intersection_polygon is None:
            try:
                import config
                intersection_polygon = getattr(config, "CROSSWALK_POLYGON", None)
            except ImportError:
                intersection_polygon = None
        self.intersection_polygon = intersection_polygon
        self.intersection_margin_px = intersection_margin_px
        self.cell_dominant_dir: dict = {}  # {(gx,gy): (dx, dy)}
        self.cell_n_samples: dict = {}

    def _cell_is_near_intersection(self, key: tuple) -> bool:
        if not self.intersection_polygon:
            return False
        cx, cy = _cell_center(key, self.cell_size)
        return _point_near_polygon(cx, cy, self.intersection_polygon, self.intersection_margin_px)

    def fit(self, events: List) -> None:
        """Học hướng đa số cho mỗi ô lưới từ TOÀN BỘ quỹ đạo phương tiện
        trong events đã có (KHÔNG cần nhãn/tương tác gì thêm)."""
        from vehicle_relations import _VEHICLE_CLASSES

        cell_vectors = defaultdict(list)
        for e in events:
            if e.cls_name not in _VEHICLE_CLASSES:
                continue
            for v in _track_direction_vectors(e.trajectory):
                key = _cell_key(v["cx"], v["cy"], self.cell_size)
                cell_vectors[key].append((v["dx"], v["dy"]))

        for key, vecs in cell_vectors.items():
            if self._cell_is_near_intersection(key):
                continue  # ô sát giao lộ -- nhiều luồng xe hợp lệ cắt nhau, không dùng để học/so hướng (xem docstring)
            n = len(vecs)
            if n < self.min_samples:
                continue
            mean_dx = sum(v[0] for v in vecs) / n
            mean_dy = sum(v[1] for v in vecs) / n
            mag = math.hypot(mean_dx, mean_dy)
            # circular variance xấp xỉ: 1 - |vector trung bình| (mag gần 1
            # nghĩa là các vector đều cùng hướng, gần 0 nghĩa là hướng rất
            # phân tán -- công thức chuẩn cho circular statistics).
            circular_var = 1.0 - mag
            if circular_var > self.max_circular_std or mag < 1e-6:
                continue  # ô có hướng quá phân tán (giao lộ, khúc rẽ) -- bỏ qua, không dùng để so sánh
            self.cell_dominant_dir[key] = (mean_dx / mag, mean_dy / mag)
            self.cell_n_samples[key] = n

    def n_reliable_cells(self) -> int:
        return len(self.cell_dominant_dir)

    def check_track(self, trajectory: List[dict], wrong_way_angle_deg: float = 120.0,
                     min_wrong_fraction: float = 0.5, min_checked_segments: int = 4) -> Optional[dict]:
        """
        So hướng từng đoạn quỹ đạo với hướng đa số đã học ở ô tương ứng.
        Trả về dict {n_checked, n_wrong, wrong_fraction, avg_angle_deg} nếu
        CÓ ĐỦ đoạn được so sánh (nằm trong ô đã học hướng tin cậy) VÀ tỉ lệ
        đoạn "đi ngược" (góc lệch > wrong_way_angle_deg) đạt ngưỡng
        min_wrong_fraction -- ngược lại trả None (không đủ bằng chứng hoặc
        không phát hiện đi ngược).

        min_checked_segments: số đoạn quỹ đạo TỐI THIỂU phải rơi vào ô đã
        học hướng tin cậy mới coi kết quả có ý nghĩa thống kê. Phát hiện
        thực tế trên N001-V001.mov: chỉ 1-2 đoạn nằm ngay biên giữa 1 ô tin
        cậy và 1 ô bị loại (gần giao lộ) có thể ngẫu nhiên lệch hướng, tạo
        cờ giả dù cả quỹ đạo track rõ ràng cùng hướng với luồng xe xung
        quanh (track 351, track 556) -- yêu cầu tối thiểu 4 đoạn giúp loại
        các trường hợp nhiễu 1-2 điểm biên này.
        """
        vectors = _track_direction_vectors(trajectory)
        checked, wrong, angles = 0, 0, []
        for v in vectors:
            key = _cell_key(v["cx"], v["cy"], self.cell_size)
            dom = self.cell_dominant_dir.get(key)
            if dom is None:
                continue
            checked += 1
            cos_angle = max(-1.0, min(1.0, v["dx"] * dom[0] + v["dy"] * dom[1]))
            angle_deg = math.degrees(math.acos(cos_angle))
            angles.append(angle_deg)
            if angle_deg > wrong_way_angle_deg:
                wrong += 1

        if checked < min_checked_segments:
            return None
        wrong_fraction = wrong / checked
        if wrong_fraction < min_wrong_fraction:
            return None
        return {
            "n_checked": checked, "n_wrong": wrong,
            "wrong_fraction": round(wrong_fraction, 2),
            "avg_angle_deg": round(sum(angles) / len(angles), 1),
        }


def detect_wrong_way_vehicles(events: List, lane_model: Optional[LaneDirectionModel] = None) -> List[dict]:
    """
    events: TOÀN BỘ event (dùng để fit model nếu lane_model=None, và để
    kiểm tra từng track phương tiện).
    lane_model: truyền sẵn nếu đã fit từ trước (vd fit trên video dài hơn
    rồi áp dụng cho từng đoạn) -- mặc định tự fit ngay trên events này.
    """
    from vehicle_relations import _VEHICLE_CLASSES

    if lane_model is None:
        lane_model = LaneDirectionModel()
        lane_model.fit(events)

    if lane_model.n_reliable_cells() == 0:
        return []  # không đủ dữ liệu để học hướng bất kỳ ô nào -- không báo gì, tránh đoán mò

    results = []
    for e in events:
        if e.cls_name not in _VEHICLE_CLASSES or len(e.trajectory) < 2:
            continue
        check = lane_model.check_track(e.trajectory)
        if check is not None:
            results.append({
                "event_type": "possible_wrong_way",
                "track_id": e.track_id, "cls_name": e.cls_name,
                "t_start": e.t_start, "t_end": e.t_end, "location": e.location,
                **check,
                "frame_names": e.representative_frame_names,
                "note": "Ứng viên dựa trên so sánh hướng di chuyển với hướng đa số xe khác đã học "
                        "được tại cùng vị trí trong khung hình -- CẦN xem lại video/frame thật, "
                        "đặc biệt cẩn thận gần giao lộ (xe rẽ có thể bị hiểu nhầm là đi ngược).",
            })
    return results
