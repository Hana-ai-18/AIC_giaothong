"""
Gộp toàn bộ output JSON (events, scene_descriptions, relations) của MỌI
video đã chạy trong OUT_DIR thành 1 file `traffic_index.parquet` DUY NHẤT
-- đúng schema đã thống nhất với backend PixelPals thật (xem
polars_search.py trong repo backend: mỗi dòng nạp thẳng vào RAM 1 lần bằng
polars.read_parquet, mọi truy vấn sau đó chạy trong RAM, không đụng lại
Milvus).

1 DÒNG = 1 EVENT (1 track_id đã build xong, sau dedup/merge) -- KHÔNG phải
1 dòng/detection/frame, vì mục đích là lọc/filter theo thuộc tính
(vehicle_color, shirt_color, quan hệ...), không phải suy luận theo frame.

Các quan hệ (side_by_side, order_and_overtake) được "join" ngược lại vào
đúng các track_id liên quan dưới dạng list cột `relations` (JSON string,
vì polars/parquet không có kiểu "list các dict lồng nhau" tiện lợi bằng
text) -- cách này để 1 dòng track vẫn tra được đầy đủ quan hệ nó tham gia
mà không phải join 2 bảng lúc filter.

CHỦ Ý DÙNG POLARS (không phải pandas): đúng thư viện backend thật
(polars_search.py) đã dùng cho object_index.parquet, tránh phải dạy code
2 API khác nhau cho 2 index song song trong cùng 1 backend.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List

import polars as pl

# Các suffix output phụ (không phải file event chính <video_id>.json) --
# dùng để loại chúng ra khi glob "*.json" tìm file event, và để tự suy ra
# tên file liên quan (descriptions/relations/scene_descriptions) từ mỗi
# video_id tìm được.
_AUX_SUFFIXES = ("_descriptions.json", "_scene_descriptions.json", "_relations.json", "_density.json",
                  "_scene_context.json")


def _find_event_files(out_dir: str) -> List[str]:
    return sorted(
        f for f in glob.glob(os.path.join(out_dir, "*.json"))
        if not any(f.endswith(suf) for suf in _AUX_SUFFIXES)
    )


def _load_json(path: str, default):
    if not os.path.isfile(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _relations_for_track(relations: dict, track_id: int) -> List[dict]:
    """Trích mọi quan hệ (side_by_side, order_and_overtake, va chạm...) có
    nhắc tới track_id này, gắn kèm loại quan hệ -- để lúc query biết được
    "track X đang cạnh track Y" mà không cần join riêng bảng relations."""
    out = []
    for r in relations.get("side_by_side", []):
        if track_id in (r.get("track_id_1"), r.get("track_id_2")):
            out.append({"type": "side_by_side", **r})
    for r in relations.get("order_and_overtake", []):
        if track_id in (r.get("track_id_1"), r.get("track_id_2")):
            out.append({"type": "order_and_overtake", **r})
    for r in relations.get("collision_candidates", []):
        if track_id in (r.get("track_id_1"), r.get("track_id_2")):
            out.append({"type": "collision_candidate", **r})
    return out


def build_index_for_video(out_dir: str, video_id: str) -> List[Dict]:
    """Đọc mọi file output của 1 video, trả về list dict (1 dict/track) sẵn
    sàng nạp vào DataFrame. Không raise nếu thiếu file phụ (relations/scene
    description) -- bỏ qua field liên quan, không suy đoán."""
    events = _load_json(os.path.join(out_dir, f"{video_id}.json"), [])
    relations = _load_json(os.path.join(out_dir, f"{video_id}_relations.json"), {})
    scene_descs = _load_json(os.path.join(out_dir, f"{video_id}_scene_descriptions.json"), [])
    descs = _load_json(os.path.join(out_dir, f"{video_id}_descriptions.json"), [])

    # scene_description áp cho CẢ CỤM xe (nhiều track cùng lúc) -- map
    # track_id -> list câu mô tả cảnh nó xuất hiện trong đó, để mỗi dòng
    # track có thể mang theo câu mô tả cảnh liên quan (dùng cho arm RRF
    # embedding scene_description phía backend).
    scene_by_track: Dict[int, List[str]] = {}
    for sd in scene_descs:
        # đúng tên field thật do scene_description.py::build_scene_descriptions()
        # trả về ("subject_track_ids", không phải "track_ids") -- xem
        # scene_description.py dòng ~179/223.
        for tid in sd.get("subject_track_ids", []) or []:
            scene_by_track.setdefault(tid, []).append(sd.get("description", ""))

    # description per-event (1-1 với track qua event_description.py) --
    # map track_id -> description (event_type == "track").
    desc_by_track: Dict[int, str] = {
        d["track_id"]: d["description"]
        for d in descs
        if d.get("event_type") == "track" and d.get("track_id") is not None
    }

    rows = []
    for e in events:
        tid = e["track_id"]
        rows.append({
            "video_id": video_id,
            "track_id": tid,
            "cls_name": e.get("cls_name"),
            "t_start": e.get("t_start"),
            "t_end": e.get("t_end"),
            "duration_sec": round((e.get("t_end") or 0) - (e.get("t_start") or 0), 3),
            "n_detections": e.get("n_detections"),
            "location": e.get("location"),
            "light_at_start": e.get("light_at_start"),
            "light_at_end": e.get("light_at_end"),
            "vehicle_color": e.get("vehicle_color"),
            "vehicle_color_conf": e.get("vehicle_color_conf"),
            "vehicle_color_method": e.get("vehicle_color_method"),
            "shirt_color": e.get("shirt_color"),
            "shirt_color_conf": e.get("shirt_color_conf"),
            "shirt_color_method": e.get("shirt_color_method"),
            "vehicle_type_detail_en": (e.get("vehicle_type_detail") or {}).get("vehicle_type"),
            "vehicle_type_detail_vi": (e.get("vehicle_type_detail") or {}).get("vehicle_type_vi"),
            "vehicle_type_detail_conf": (e.get("vehicle_type_detail") or {}).get("confidence"),
            "merged_from_track_ids": json.dumps(e.get("merged_from_track_ids")) if e.get("merged_from_track_ids") else None,
            "merge_note": e.get("merge_note"),
            "position_grid_vi": e.get("position_grid_vi"),
            "position_grid_en": e.get("position_grid_en"),
            "crosswalk_fraction": e.get("crosswalk_fraction"),
            "in_crosswalk": e.get("in_crosswalk"),
            "position_description": e.get("position_description"),
            "representative_frame_names": json.dumps(e.get("representative_frame_names") or []),
            # Quỹ đạo đầy đủ (đã downsample sẵn, tối đa 20 điểm/track, xem
            # event_timeline.py::_downsample_trajectory) -- LƯU THÊM vào
            # index để các converter/tool khác (vd
            # convert_traffic_to_object_index.py, ghép sang schema
            # aic26_pipeline) có bbox pixel thật của từng điểm thời gian,
            # không phải suy đoán/để trống. Trước đây field này CÓ trong
            # Event nhưng KHÔNG được xuất ra parquet, khiến các converter
            # khác không có bbox thật để dùng.
            "trajectory": json.dumps(e.get("trajectory") or []),
            "description": desc_by_track.get(tid),
            "scene_descriptions": json.dumps(scene_by_track.get(tid, [])),
            "relations": json.dumps(_relations_for_track(relations, tid)),
        })
    return rows


# Schema tường minh -- tránh polars tự suy luận kiểu SAI khi 1 số video có
# cột toàn giá trị None (vd video không phát hiện shirt_color nào cả) --
# lỗi thật đã gặp trước đây trong project này khi ghép nhiều DataFrame có
# cột toàn null (polars suy ra Null type, concat với video khác có giá trị
# thật sẽ lỗi kiểu). Khai schema cứng để luôn nhất quán giữa mọi video.
INDEX_SCHEMA = {
    "video_id": pl.Utf8, "track_id": pl.Int64, "cls_name": pl.Utf8,
    "t_start": pl.Float64, "t_end": pl.Float64, "duration_sec": pl.Float64,
    "n_detections": pl.Int64, "location": pl.Utf8,
    "light_at_start": pl.Utf8, "light_at_end": pl.Utf8,
    "vehicle_color": pl.Utf8, "vehicle_color_conf": pl.Float64, "vehicle_color_method": pl.Utf8,
    "shirt_color": pl.Utf8, "shirt_color_conf": pl.Float64, "shirt_color_method": pl.Utf8,
    "vehicle_type_detail_en": pl.Utf8, "vehicle_type_detail_vi": pl.Utf8, "vehicle_type_detail_conf": pl.Float64,
    "merged_from_track_ids": pl.Utf8, "merge_note": pl.Utf8,
    "position_grid_vi": pl.Utf8, "position_grid_en": pl.Utf8,
    "crosswalk_fraction": pl.Float64, "in_crosswalk": pl.Boolean,
    "position_description": pl.Utf8,
    "representative_frame_names": pl.Utf8, "trajectory": pl.Utf8, "description": pl.Utf8,
    "scene_descriptions": pl.Utf8, "relations": pl.Utf8,
}


def build_traffic_index(out_dir: str, parquet_out_path: str) -> pl.DataFrame:
    """Quét MỌI video có file event trong out_dir, gộp thành 1 DataFrame,
    ghi ra parquet_out_path. Trả về chính DataFrame đó để notebook có thể
    xem nhanh (df.head(), thống kê...) mà không phải đọc lại file."""
    event_files = _find_event_files(out_dir)
    all_rows: List[Dict] = []
    for f in event_files:
        video_id = os.path.basename(f)[:-5]  # bỏ ".json"
        all_rows.extend(build_index_for_video(out_dir, video_id))

    if not all_rows:
        df = pl.DataFrame(schema=INDEX_SCHEMA)
    else:
        df = pl.DataFrame(all_rows, schema=INDEX_SCHEMA)

    df.write_parquet(parquet_out_path)
    return df
