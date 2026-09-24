"""
Sinh mô tả tự nhiên (tiếng Việt) cho mỗi Event -- KHÔNG dùng LLM (tránh phụ
thuộc API ngoài + chi phí + độ trễ khi chạy hàng loạt trên Kaggle), chỉ ghép
template từ các field có sẵn trên Event. Mô tả này dùng để:
  1. Hiển thị nhanh cho người xem kết quả (dễ đọc hơn JSON thô).
  2. Làm input cho TEXT EMBEDDING (SigLIP2/Qwen3 text encoder hoặc bất kỳ
     text embedding model nào backend đang dùng) -- tạo ra 1 vector ngữ
     nghĩa cho từng event, bổ sung thêm 1 "arm" tìm kiếm ngữ nghĩa LINH
     HOẠT hơn exact-match field thuần, trong khi vẫn tận dụng được toàn bộ
     metadata có cấu trúc đã trích xuất chính xác (thời gian, vị trí, loại
     xe, đèn, biển số) -- không phải suy đoán từ pixel như embedding ảnh
     thuần, nên mô tả sinh ra được đảm bảo đúng sự thật 100% với những gì
     pipeline đã phát hiện.

Không tự bịa thông tin không có trong Event -- nếu field nào None/không rõ
thì bỏ qua phần đó trong câu, tránh tạo mô tả sai lệch khiến embedding học
nhầm.
"""
from __future__ import annotations

from typing import List, Optional

_CLASS_VI = {
    "car": "ô tô", "truck": "xe tải", "bus": "xe buýt",
    "motorcycle": "xe máy", "bicycle": "xe đạp", "person": "người đi bộ",
}

_LIGHT_VI = {"red": "đèn đỏ", "yellow": "đèn vàng", "green": "đèn xanh"}

_COLOR_VI = {
    "black": "đen", "blue": "xanh dương", "brown": "nâu", "green": "xanh lá",
    "grey": "xám", "red": "đỏ", "silver": "bạc", "white": "trắng", "yellow": "vàng",
}


def _fmt_time(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m}:{s:02d}" if m else f"{s}s"


def describe_event(e) -> str:
    """e: Event (từ event_timeline.py). Trả về 1 câu mô tả tiếng Việt."""
    # Ưu tiên loại xe CHI TIẾT (vd "xe cứu thương") nếu vehicle_type_refine.py
    # đã phát hiện được (confidence đủ cao) -- cụ thể hơn nhãn COCO gốc, giúp
    # câu mô tả/text-embedding phân biệt tốt hơn khi search. Nếu không có
    # (None -- phổ biến nhất, vì phần lớn xe chỉ là car/motorcycle bình
    # thường), dùng nhãn gốc như cũ.
    type_detail = getattr(e, "vehicle_type_detail", None)
    if type_detail and type_detail.get("vehicle_type_vi"):
        cls_vi = type_detail["vehicle_type_vi"]
    else:
        cls_vi = _CLASS_VI.get(e.cls_name, e.cls_name)
    color_prefix = ""
    if e.vehicle_color and e.vehicle_color in _COLOR_VI:
        color_prefix = f"{_COLOR_VI[e.vehicle_color]} "

    parts = [f"{cls_vi.capitalize()} màu {color_prefix}".strip() if color_prefix else cls_vi.capitalize()]
    parts[0] += f" xuất hiện từ giây {_fmt_time(e.t_start)} đến {_fmt_time(e.t_end)}"

    # Màu áo (person_pose.py + event_timeline.py's shirt_color) -- áp dụng
    # cho "person" (người đi bộ) VÀ xe 2 bánh có người ngồi khớp được vùng
    # áo. Chỉ thêm khi có giá trị thật (không suy đoán), đúng nguyên tắc
    # "không tự bịa" của module này.
    shirt_color = getattr(e, "shirt_color", None)
    if shirt_color and shirt_color in _COLOR_VI:
        if e.cls_name == "person":
            parts.append(f"mặc áo màu {_COLOR_VI[shirt_color]}")
        else:
            parts.append(f"người lái mặc áo màu {_COLOR_VI[shirt_color]}")

    position_desc = getattr(e, "position_description", None)
    if position_desc:
        parts.append(position_desc)

    if e.location:
        parts.append(f"tại {e.location}")

    if e.light_at_start and e.light_at_start != "unknown":
        light_desc = _LIGHT_VI.get(e.light_at_start, e.light_at_start)
        if e.light_at_end and e.light_at_end != e.light_at_start and e.light_at_end in _LIGHT_VI:
            parts.append(f"lúc đèn chuyển từ {light_desc} sang {_LIGHT_VI[e.light_at_end]}")
        else:
            parts.append(f"lúc {light_desc}")

    duration = e.t_end - e.t_start
    if duration >= 20:
        parts.append(f"dừng lại khá lâu ({duration:.0f} giây)")

    return ", ".join(parts) + "."


def describe_composite(c: dict) -> str:
    """c: 1 dict trong danh sách trả về từ detect_composite_events()."""
    event_type = c.get("event_type")
    loc = f" tại {c['location']}" if c.get("location") else ""

    if event_type == "vehicle_present_during_red_light":
        cls_vi = _CLASS_VI.get(c["cls_name"], c["cls_name"])
        return f"{cls_vi.capitalize()} xuất hiện lúc đèn đỏ{loc}, từ giây {_fmt_time(c['t_start'])} đến {_fmt_time(c['t_end'])}."
    if event_type == "red_light_runner":
        cls_vi = _CLASS_VI.get(c["cls_name"], c["cls_name"])
        return (f"{cls_vi.capitalize()} có khả năng vượt đèn đỏ{loc} trong khoảng giây "
                f"{_fmt_time(c['t_start'])} đến {_fmt_time(c['t_end'])} (đèn chuyển đỏ trong lúc xe đang ở giao lộ).")
    if event_type == "abnormal_long_stop":
        cls_vi = _CLASS_VI.get(c["cls_name"], c["cls_name"])
        return f"{cls_vi.capitalize()} dừng lại bất thường lâu ({c['duration_sec']}s){loc}, từ giây {_fmt_time(c['t_start'])}."
    if event_type == "traffic_jam":
        return f"Ùn tắc giao thông{loc} với khoảng {c['n_vehicles']} phương tiện, từ giây {_fmt_time(c['t_start'])} đến {_fmt_time(c['t_end'])}."
    if event_type == "pedestrian_crossing_outside_crosswalk":
        return (f"Người đi bộ có khả năng băng qua đường ngoài vạch kẻ đường{loc}, từ giây "
                f"{_fmt_time(c['t_start'])} đến {_fmt_time(c['t_end'])}.")
    if event_type == "vehicle_stopped_on_crosswalk":
        cls_vi = _CLASS_VI.get(c.get("cls_name"), c.get("cls_name", ""))
        return (f"{cls_vi.capitalize()} có khả năng dừng đè lên vạch kẻ đường dành cho người đi bộ{loc}, "
                f"từ giây {_fmt_time(c['t_start'])} đến {_fmt_time(c['t_end'])}.")
    if event_type == "possible_wrong_way":
        cls_vi = _CLASS_VI.get(c.get("cls_name"), c.get("cls_name", ""))
        return (f"{cls_vi.capitalize()} có khả năng đi sai làn/ngược chiều so với đa số phương tiện khác{loc}, "
                f"từ giây {_fmt_time(c['t_start'])} đến {_fmt_time(c['t_end'])} "
                f"(lệch hướng trung bình {c.get('avg_angle_deg', '?')}°).")
    return str(c)


def build_all_descriptions(events: List, composites: List[dict], extra_events: Optional[List[dict]] = None) -> List[dict]:
    """
    Trả về danh sách dict {video_id, track_id (None nếu composite),
    event_type, t_start, t_end, description, frame_names} -- ĐÚNG format
    phẳng dễ đẩy thẳng vào text-embedding + Elasticsearch/Milvus, giữ
    nguyên frame_names để join lại với ảnh khi hiển thị kết quả search.

    extra_events: các danh sách sự kiện dạng dict khác có cùng field cơ bản
    (event_type, t_start, t_end, frame_names, ...) -- vd kết quả từ
    zone_events.py (băng qua sai vạch, dừng đè vạch) và lane_direction.py
    (possible_wrong_way) -- gộp chung vào đây để MỌI loại sự kiện đều có mô
    tả tự nhiên dùng cho text-embedding, không chỉ riêng track/composite.
    """
    out = []
    for e in events:
        out.append({
            "video_id": e.video_id, "track_id": e.track_id,
            "event_type": "track", "cls_name": e.cls_name,
            "t_start": e.t_start, "t_end": e.t_end,
            "description": describe_event(e),
            "frame_names": e.representative_frame_names,
            "vehicle_color": e.vehicle_color,
            "vehicle_type_detail": getattr(e, "vehicle_type_detail", None),
            "position_grid_vi": getattr(e, "position_grid_vi", None),
            "position_grid_en": getattr(e, "position_grid_en", None),
            "crosswalk_fraction": getattr(e, "crosswalk_fraction", None),
            "in_crosswalk": getattr(e, "in_crosswalk", False),
            "position_description": getattr(e, "position_description", None),
        })
    for c in composites:
        out.append({
            "video_id": None, "track_id": c.get("track_id"),
            "event_type": c["event_type"], "cls_name": c.get("cls_name"),
            "t_start": c["t_start"], "t_end": c["t_end"],
            "description": describe_composite(c),
            "frame_names": [], "vehicle_color": None, "vehicle_type_detail": None,
        })
    for c in (extra_events or []):
        out.append({
            "video_id": None, "track_id": c.get("track_id"),
            "event_type": c["event_type"], "cls_name": c.get("cls_name"),
            "t_start": c["t_start"], "t_end": c["t_end"],
            "description": describe_composite(c),
            "frame_names": c.get("frame_names", []), "vehicle_color": None, "vehicle_type_detail": None,
        })
    return out
