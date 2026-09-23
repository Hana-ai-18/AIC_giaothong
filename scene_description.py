"""
Sinh MÔ TẢ CẢNH (scene description) -- ghép NHIỀU xe/người đang cùng xuất
hiện tại 1 mốc thời gian thành 1 CÂU DUY NHẤT đầy đủ ngữ cảnh, đúng như ví dụ
người dùng đưa ra: "xe tải màu xanh dương đi bên cạnh xe ô tô màu đỏ và đang
đèn đỏ, phía trước có 3 xe máy".

KHÁC VỚI event_description.py: file đó mô tả TỪNG Event riêng lẻ ("1 câu / 1
track"), và vehicle_relations.py chỉ trả cặp quan hệ RỜI RẠC (A cạnh B, A
trước B...). Module này GỘP nhiều nguồn đã có (color, type, side_by_side,
order_and_overtake, đèn giao thông, đếm nhóm xe máy phía trước) tại CÙNG 1
THỜI ĐIỂM thành 1 câu tường thuật, giống cách người thật mô tả 1 khung cảnh.

CÁCH LÀM:
  1. Chọn ra các MỐC THỜI GIAN đại diện để dựng cảnh -- không dựng cho MỌI
     giây (sẽ ra quá nhiều câu trùng lặp gần giống nhau, tốn chi phí
     embedding vô ích) mà chỉ dựng tại thời điểm bắt đầu của MỖI quan hệ
     side_by_side/order_and_overtake đã phát hiện (đây là các mốc "có gì đó
     đáng mô tả" -- 2 xe đang cạnh nhau, xe này trước xe kia) -- tái sử dụng
     kết quả đã tính từ vehicle_relations.py, không cần suy luận lại.
  2. Tại mỗi mốc, tìm mọi Event đang HOẠT ĐỘNG (t_start <= t <= t_end) gần
     vị trí của cặp xe trung tâm (chủ ngữ của câu) để đếm nhóm xe cùng loại
     gần đó theo hướng "phía trước" -- dùng lại _direction_of_travel và phép
     chiếu toạ độ đã có trong vehicle_relations.py (không phát minh lại).
  3. Ghép câu theo mẫu tiếng Việt tự nhiên: "<màu> <loại xe A> đi bên
     <trái/phải> <màu> <loại xe B>, lúc <đèn>, phía trước có <N> <loại xe C>"
     -- CHỈ điền phần nào có dữ liệu thật (không bịa, giống nguyên tắc đã
     áp dụng trong event_description.py).

GIỚI HẠN: đây là suy luận GHÉP NHIỀU RULE ĐÃ CÓ, nên kế thừa toàn bộ giới hạn
của các rule gốc (side_by_side, order_and_overtake đều là suy luận hình học
2D, CẦN xem lại frame thật). Ngoài ra: đếm "phía trước có N xe máy" là đếm
THEO Ô/VÙNG XẤP XỈ phía trước theo hướng di chuyển, KHÔNG phải đếm chính xác
tuyệt đối (2 xe máy đứng cạnh nhau xa hơn 1 chút vẫn có thể bị đếm hoặc bỏ
sót tuỳ ngưỡng khoảng cách) -- luôn kèm frame_names để xem lại.
"""
from __future__ import annotations

from typing import List, Optional

from vehicle_relations import (
    _VEHICLE_CLASSES, _direction_of_travel, _interp_position, _is_moving_at,
    detect_side_by_side, detect_order_and_overtake,
)

_CLASS_VI = {
    "car": "ô tô", "truck": "xe tải", "bus": "xe buýt",
    "motorcycle": "xe máy", "bicycle": "xe đạp", "person": "người đi bộ",
}
_LIGHT_VI = {"red": "đèn đỏ", "yellow": "đèn vàng", "green": "đèn xanh"}
_COLOR_VI = {
    "black": "đen", "blue": "xanh dương", "brown": "nâu", "green": "xanh lá",
    "grey": "xám", "red": "đỏ", "silver": "bạc", "white": "trắng", "yellow": "vàng",
}

# Bán kính (px) xung quanh trục hướng di chuyển để tính "phía trước có N xe
# máy" -- xấp xỉ 1 làn đường thực tế trên camera cỡ trung bình. Thận trọng,
# không quá rộng để tránh đếm cả xe làn khác đi vào.
FRONT_ZONE_LATERAL_PX = 120.0
FRONT_ZONE_LONGITUDINAL_MIN_PX = 20.0
FRONT_ZONE_LONGITUDINAL_MAX_PX = 300.0


def _describe_vehicle(e, fallback_cls_vi: Optional[str] = None) -> str:
    """'<màu> <loại xe>' -- vd 'xanh dương xe tải', ưu tiên vehicle_type_detail
    nếu có (xem event_description.py, cùng nguyên tắc), rồi tới màu. Nếu là
    xe 2 bánh và khớp được màu áo người lái (shirt_color, xem person_pose.py
    + event_timeline.py), thêm vào cuối -- vd 'xe máy màu đen, người lái mặc
    áo màu vàng' -- giúp phân biệt 2 xe máy CÙNG MÀU/LOẠI trong câu mô tả
    (đúng tinh thần ví dụ người dùng yêu cầu ban đầu: mô tả càng chi tiết
    càng dễ tìm kiếm lại đúng cảnh)."""
    type_detail = getattr(e, "vehicle_type_detail", None)
    if type_detail and type_detail.get("vehicle_type_vi"):
        cls_vi = type_detail["vehicle_type_vi"]
    else:
        cls_vi = fallback_cls_vi or _CLASS_VI.get(e.cls_name, e.cls_name)
    color_vi = _COLOR_VI.get(e.vehicle_color) if e.vehicle_color else None
    desc = f"{cls_vi} màu {color_vi}" if color_vi else cls_vi

    shirt_color = getattr(e, "shirt_color", None)
    if shirt_color and e.cls_name in ("motorcycle", "bicycle") and shirt_color in _COLOR_VI:
        desc += f", người lái mặc áo màu {_COLOR_VI[shirt_color]}"
    return desc


def _count_vehicles_ahead(events: List, subject, t: float, exclude_track_ids: set) -> dict:
    """Đếm số xe (theo từng loại) đang ở PHÍA TRƯỚC subject (theo hướng di
    chuyển của subject) tại thời điểm t, trong 1 vùng hình chữ nhật hẹp dọc
    theo hướng đó (xem FRONT_ZONE_*). Trả về dict {cls_name: count}."""
    subj_dir = _direction_of_travel(subject.trajectory)
    subj_pos = _interp_position(subject.trajectory, t)
    if subj_dir is None or subj_pos is None:
        return {}
    perp_dir = (-subj_dir[1], subj_dir[0])

    counts: dict = {}
    for e in events:
        if e.track_id in exclude_track_ids or e.track_id == subject.track_id:
            continue
        if e.cls_name not in _VEHICLE_CLASSES:
            continue
        if not (e.t_start <= t <= e.t_end):
            continue
        pos = _interp_position(e.trajectory, t)
        if pos is None:
            continue
        dx, dy = pos["cx"] - subj_pos["cx"], pos["cy"] - subj_pos["cy"]
        longitudinal = dx * subj_dir[0] + dy * subj_dir[1]
        lateral = dx * perp_dir[0] + dy * perp_dir[1]
        if FRONT_ZONE_LONGITUDINAL_MIN_PX <= longitudinal <= FRONT_ZONE_LONGITUDINAL_MAX_PX and \
           abs(lateral) <= FRONT_ZONE_LATERAL_PX:
            counts[e.cls_name] = counts.get(e.cls_name, 0) + 1
    return counts


def _light_phrase(light_lookup, t: float) -> Optional[str]:
    if light_lookup is None:
        return None
    light = light_lookup(t)
    if not light or light == "unknown":
        return None
    return _LIGHT_VI.get(light, light)


def build_scene_descriptions(events: List, light_lookup=None,
                              side_by_side: Optional[List[dict]] = None,
                              order_relations: Optional[List[dict]] = None) -> List[dict]:
    """
    events: toàn bộ Event (đã qua track_dedup.deduplicate_events()).
    light_lookup: hàm (t) -> "red"/"yellow"/"green"/None, giống ocr/light
    lookup trong main_pipeline.py (dùng cache theo giây đã có sẵn, KHÔNG mở
    lại video) -- có thể None nếu không có (câu mô tả sẽ bỏ qua phần đèn).
    side_by_side, order_relations: kết quả đã tính từ detect_side_by_side()/
    detect_order_and_overtake() -- truyền vào để KHÔNG tính lại (2 hàm này
    không rẻ, O(n^2) track), nếu None thì hàm tự tính.

    Trả về danh sách dict {t, subject_track_ids, description, frame_names}
    -- mỗi dict là 1 "cảnh" tại 1 mốc thời gian, dùng cho text-embedding
    giống các mô tả khác trong pipeline.
    """
    if side_by_side is None:
        side_by_side = detect_side_by_side(events)
    if order_relations is None:
        order_relations = detect_order_and_overtake(events)

    by_id = {e.track_id: e for e in events}
    scenes = []

    for sbs in side_by_side:
        left = by_id.get(sbs["left_track_id"])
        right = by_id.get(sbs["right_track_id"])
        if left is None or right is None:
            continue
        t = sbs["t_start"]  # mốc bắt đầu quan hệ -- thời điểm "đáng mô tả nhất"

        left_desc = _describe_vehicle(left)
        right_desc = _describe_vehicle(right)
        parts = [f"{left_desc.capitalize()} đi bên trái {right_desc}"]

        light_phrase = _light_phrase(light_lookup, t)
        if light_phrase:
            parts.append(f"lúc {light_phrase}")

        exclude = {left.track_id, right.track_id}
        counts = _count_vehicles_ahead(events, left, t, exclude)
        for cls_name, n in counts.items():
            cls_vi = _CLASS_VI.get(cls_name, cls_name)
            parts.append(f"phía trước có {n} {cls_vi}" if n > 1 else f"phía trước có 1 {cls_vi}")

        location = getattr(left, "location", None) or getattr(right, "location", None)
        if location:
            parts.append(f"tại {location}")

        description = ", ".join(parts) + "."
        frame_names = list(dict.fromkeys(
            (left.representative_frame_names or []) + (right.representative_frame_names or [])
        ))
        scenes.append({
            "t": round(t, 2),
            "subject_track_ids": [left.track_id, right.track_id],
            "description": description,
            "frame_names": frame_names,
        })

    for rel in order_relations:
        if rel["relation"] != "in_front_of":
            continue
        front = by_id.get(rel["front_track_id"])
        behind = by_id.get(rel["behind_track_id"])
        if front is None or behind is None:
            continue
        t = rel["t_start"]

        # ĐÃ PHÁT HIỆN THỰC NGHIỆM: detect_order_and_overtake() (module cũ,
        # không sửa ở đây để tránh ảnh hưởng ngược tới các chỗ khác đang
        # dùng nó) gán "in_front_of" cho MỌI cặp track cùng hướng có overlap
        # thời gian đủ lâu -- kể cả khi 1 hoặc cả 2 xe ĐANG ĐỨNG YÊN (vd xe
        # tải đậu bị coi là "phía trước" hàng chục xe khác chỉ vì chúng đi
        # qua gần đó cùng lúc). "X chạy sau Y" chỉ có ý nghĩa mô tả cảnh khi
        # CẢ 2 đang thực sự di chuyển -- lọc bằng _is_moving_at() giống
        # detect_side_by_side(), tránh sinh hàng loạt câu vô nghĩa kiểu "xe
        # máy chạy sau xe tải" khi xe tải đang đậu im 1 chỗ.
        if not (_is_moving_at(front.trajectory, t) and _is_moving_at(behind.trajectory, t)):
            continue

        front_desc = _describe_vehicle(front)
        behind_desc = _describe_vehicle(behind)
        parts = [f"{behind_desc.capitalize()} chạy sau {front_desc}"]

        light_phrase = _light_phrase(light_lookup, t)
        if light_phrase:
            parts.append(f"lúc {light_phrase}")

        location = getattr(front, "location", None) or getattr(behind, "location", None)
        if location:
            parts.append(f"tại {location}")

        description = ", ".join(parts) + "."
        frame_names = list(dict.fromkeys(
            (front.representative_frame_names or []) + (behind.representative_frame_names or [])
        ))
        scenes.append({
            "t": round(t, 2),
            "subject_track_ids": [front.track_id, behind.track_id],
            "description": description,
            "frame_names": frame_names,
        })

    return scenes
