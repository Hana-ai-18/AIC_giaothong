"""
Thống kê MẬT ĐỘ GIAO THÔNG theo từng khoảng thời gian cố định (vd mỗi 10s)
xuyên suốt video -- khác với `traffic_jam` trong event_timeline.py (chỉ báo
NHỊ PHÂN khi vượt ngưỡng đông xe tại 1 thời điểm), cái này trả về TOÀN BỘ
đường cong mật độ theo thời gian, cho phép trả lời:
  - "Lúc nào đông xe nhất / vắng nhất trong video?"
  - So sánh mật độ giữa các khoảng thời gian khác nhau.
  - Vẽ biểu đồ mật độ theo thời gian (đưa thẳng dữ liệu ra là mảng số).

Đây là phép ĐẾM đơn giản trên Event đã có (không cần model/suy luận thêm),
nên chính xác 100% theo đúng số track đã detect được -- không có rủi ro
false positive như collision/crosswalk ở các module khác.
"""
from __future__ import annotations

from typing import List


def build_density_timeline(
    events: List,
    video_duration_sec: float,
    bucket_sec: float = 10.0,
) -> List[dict]:
    """
    Chia video thành các khoảng (bucket) liên tiếp độ dài bucket_sec, đếm số
    lượng track ĐANG HOẠT ĐỘNG (t_start < bucket_end và t_end > bucket_start)
    trong mỗi khoảng, tách riêng theo loại (vehicle vs person) để hữu ích
    hơn khi query (vd "lúc nào đông người đi bộ nhất" khác "lúc nào kẹt xe").

    Trả về danh sách dict {t_start, t_end, n_vehicles, n_persons, n_total,
    by_class: {class_name: count}} -- xuất thẳng ra JSON, hoặc dùng
    matplotlib vẽ biểu đồ mật độ theo thời gian (xem notebook Kaggle).
    """
    from vehicle_relations import _VEHICLE_CLASSES

    n_buckets = max(1, int(video_duration_sec // bucket_sec) + 1)
    timeline = []

    for i in range(n_buckets):
        b_start = i * bucket_sec
        b_end = b_start + bucket_sec
        overlapping = [e for e in events if e.t_start < b_end and e.t_end > b_start]

        by_class = {}
        for e in overlapping:
            by_class[e.cls_name] = by_class.get(e.cls_name, 0) + 1

        n_vehicles = sum(c for cls, c in by_class.items() if cls in _VEHICLE_CLASSES)
        n_persons = by_class.get("person", 0)

        timeline.append({
            "t_start": round(b_start, 1), "t_end": round(b_end, 1),
            "n_vehicles": n_vehicles, "n_persons": n_persons,
            "n_total": n_vehicles + n_persons,
            "by_class": by_class,
        })

    return timeline


def summarize_density(timeline: List[dict]) -> dict:
    """Tóm tắt nhanh: khoảng nào đông nhất/vắng nhất -- tiện cho hiển thị
    hoặc trả lời trực tiếp câu query 'lúc nào đông xe nhất' mà không cần
    quét lại toàn bộ timeline."""
    if not timeline:
        return {}
    busiest = max(timeline, key=lambda b: b["n_total"])
    quietest = min(timeline, key=lambda b: b["n_total"])
    avg_total = sum(b["n_total"] for b in timeline) / len(timeline)
    return {
        "busiest_window": busiest,
        "quietest_window": quietest,
        "avg_vehicles_per_window": round(avg_total, 2),
        "n_windows": len(timeline),
    }
