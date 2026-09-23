"""
Sự kiện dựa trên VÙNG KHÔNG GIAN CỐ ĐỊNH (zone) trong khung hình -- khác với
composite events trong event_timeline.py (dựa trên thời gian/đèn) và
vehicle_relations.py (dựa trên quan hệ giữa 2 track). Ở đây quan tâm tới vị
trí track SO VỚI 1 VÙNG CỐ ĐỊNH đã đo trước (vd. vạch kẻ đường crosswalk).

Đã cài:
  1. pedestrian_crossing_outside_crosswalk -- người đi bộ có quỹ đạo đi qua
     đường (băng ngang, không chỉ đứng/đi dọc vỉa hè) NHƯNG không đi trong
     vùng CROSSWALK_POLYGON -- ứng viên "băng qua đường sai vạch".
  2. vehicle_stopped_on_crosswalk -- phương tiện có track NẰM (một phần)
     TRONG vùng crosswalk VÀ gần như đứng yên đủ lâu (dấu hiệu dừng/đỗ đè
     lên vạch, không phải chỉ đi ngang qua) -- ứng viên "dừng sai vạch".

CẢNH BÁO (nhất quán với các module khác): đây là suy luận RULE-BASED trên
toạ độ 2D chiếu từ góc camera, không phải toạ độ thế giới thực -- luôn kèm
representative_frame_names để xem lại bằng mắt trước khi kết luận là vi
phạm thật. Đặc biệt "băng qua sai vạch" dễ nhầm với người đi bộ chỉ đứng
gần đó hoặc đi dọc vỉa hè mà quỹ đạo lấn nhẹ vào lòng đường.
"""
from __future__ import annotations

from typing import List, Optional

try:
    import cv2
    import numpy as np
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def _point_in_polygon(x: float, y: float, polygon: List[tuple]) -> bool:
    """Ray casting thuần Python -- không bắt buộc phải có cv2/numpy (dù
    project này luôn có sẵn), để module này độc lập, dễ test/tái sử dụng."""
    n = len(polygon)
    inside = False
    px, py = polygon[-1]
    for cx, cy in polygon:
        if ((cy > y) != (py > y)) and (x < (px - cx) * (y - cy) / (py - cy + 1e-9) + cx):
            inside = not inside
        px, py = cx, cy
    return inside


def _fraction_of_trajectory_in_zone(trajectory: List[dict], polygon: List[tuple]) -> float:
    if not trajectory:
        return 0.0
    n_in = sum(1 for p in trajectory if _point_in_polygon(p["cx"], p["cy"], polygon))
    return n_in / len(trajectory)


def _is_crossing_motion(trajectory: List[dict], polygon: List[tuple]) -> bool:
    """Ước lượng thô: người đi bộ 'băng qua đường' nếu quỹ đạo di chuyển
    ngang qua BỀ RỘNG của polygon (theo trục x xấp xỉ) một khoảng đáng kể,
    chứ không chỉ đứng yên hoặc đi dọc theo 1 cạnh (vd. đi trên vỉa hè song
    song với đường). Dùng bounding box của polygon làm tham chiếu bề rộng."""
    if len(trajectory) < 2:
        return False
    xs = [p["cx"] for p in trajectory]
    poly_xs = [p[0] for p in polygon]
    poly_width = max(poly_xs) - min(poly_xs)
    travel_x = max(xs) - min(xs)
    return travel_x > poly_width * 0.3  # đi ngang ít nhất ~30% bề rộng vạch mới coi là "băng qua"


def detect_pedestrian_crossing_outside_crosswalk(
    events: List,
    crosswalk_polygon: List[tuple],
    max_in_zone_fraction: float = 0.15,
) -> List[dict]:
    """
    events: danh sách Event (chỉ xét cls_name == "person").
    Với mỗi track người đi bộ CÓ DẤU HIỆU BĂNG QUA ĐƯỜNG (xem
    _is_crossing_motion) nhưng tỉ lệ điểm quỹ đạo nằm TRONG vùng crosswalk
    THẤP (< max_in_zone_fraction) -- ứng viên băng qua ngoài vạch kẻ đường.
    """
    results = []
    for e in events:
        if e.cls_name != "person" or len(e.trajectory) < 2:
            continue
        if not _is_crossing_motion(e.trajectory, crosswalk_polygon):
            continue
        frac_in = _fraction_of_trajectory_in_zone(e.trajectory, crosswalk_polygon)
        if frac_in < max_in_zone_fraction:
            results.append({
                "event_type": "pedestrian_crossing_outside_crosswalk",
                "track_id": e.track_id, "t_start": e.t_start, "t_end": e.t_end,
                "location": e.location,
                "fraction_in_crosswalk": round(frac_in, 2),
                "frame_names": e.representative_frame_names,
                "note": "Ứng viên dựa trên quỹ đạo băng ngang đường nhưng phần lớn nằm ngoài "
                        "vùng vạch kẻ đường đã đo -- CẦN xem lại frame/video thật để xác nhận, "
                        "có thể là người đi bộ ở gần vỉa hè bị đo lệch nhẹ do góc camera.",
            })
    return results


def detect_vehicle_stopped_on_crosswalk(
    events: List,
    crosswalk_polygon: List[tuple],
    min_in_zone_fraction: float = 0.3,
    min_stationary_sec: float = 3.0,
) -> List[dict]:
    """
    events: danh sách Event (chỉ xét class phương tiện).
    Ứng viên "dừng đè vạch": tỉ lệ điểm quỹ đạo nằm TRONG crosswalk đủ cao
    (>= min_in_zone_fraction, khác với đi ngang qua chỉ chạm 1 chút) VÀ xe
    gần như đứng yên (biến thiên vị trí nhỏ) trong khoảng thời gian đủ lâu
    (>= min_stationary_sec) khi đang ở trong vùng đó.
    """
    from vehicle_relations import _VEHICLE_CLASSES

    results = []
    for e in events:
        if e.cls_name not in _VEHICLE_CLASSES or len(e.trajectory) < 2:
            continue
        in_zone_points = [p for p in e.trajectory if _point_in_polygon(p["cx"], p["cy"], crosswalk_polygon)]
        frac_in = len(in_zone_points) / len(e.trajectory)
        if frac_in < min_in_zone_fraction or len(in_zone_points) < 2:
            continue

        t_span = in_zone_points[-1]["t"] - in_zone_points[0]["t"]
        if t_span < min_stationary_sec:
            continue

        # kiểm tra "gần như đứng yên" trong khoảng đó: độ lệch vị trí nhỏ so với kích thước xe
        xs = [p["cx"] for p in in_zone_points]
        ys = [p["cy"] for p in in_zone_points]
        avg_size = sum(p["w"] + p["h"] for p in in_zone_points) / (2 * len(in_zone_points))
        spread = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
        if avg_size > 0 and spread / avg_size > 0.5:
            continue  # di chuyển đáng kể trong lúc ở vùng -- chỉ đang đi ngang qua, không dừng

        results.append({
            "event_type": "vehicle_stopped_on_crosswalk",
            "track_id": e.track_id, "cls_name": e.cls_name,
            "t_start": round(in_zone_points[0]["t"], 2), "t_end": round(in_zone_points[-1]["t"], 2),
            "location": e.location,
            "fraction_in_crosswalk": round(frac_in, 2),
            "frame_names": e.representative_frame_names,
            "note": "Ứng viên dựa trên rule (phần lớn quỹ đạo nằm trong vạch kẻ đường + gần như "
                    "đứng yên đủ lâu) -- CẦN xem lại frame/video thật, có thể là xe đang nhích chậm "
                    "do kẹt xe chứ không cố ý dừng đè vạch.",
        })
    return results
