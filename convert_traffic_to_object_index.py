"""
convert_traffic_to_object_index.py
====================================
Chuyển `traffic_index.parquet` (traffic_pipeline, 1 dòng = 1 TRACK xuyên
suốt video, dùng cho video N001-V001...N040-V003) sang ĐÚNG SCHEMA
`index.parquet` mà `aic26_pipeline/src/export_parquet.py` xuất ra (1 dòng =
1 OBJECT trong 1 KEYFRAME cụ thể) -- để backend PixelPals đọc được cả 2 tập
video theo CÙNG 1 schema quen thuộc (dù xuất ra 2 file riêng, KHÔNG ghép
chung 1 file -- xem ghi chú "KHÔNG GHÉP FILE" bên dưới).

KHÔNG GHÉP FILE: module này CHỈ convert, KHÔNG tự động nối/ghép với
index.parquet của aic26_pipeline nữa (đã bỏ theo yêu cầu) -- xuất ra 1 file
riêng `traffic_as_object_index.parquet`, backend/bạn tự quyết định có ghép
2 file lại hay đọc riêng từng file theo video_id.

VÌ SAO KHÔNG THỂ GHÉP THẲNG (2 schema khác BẢN CHẤT, không chỉ khác tên cột):
  - aic26_pipeline: 1 dòng = 1 lần object xuất hiện trong 1 KEYFRAME rời rạc
    (video được cắt keyframe theo scene-detect, có thể vài chục tới vài trăm
    keyframe/video) -- object KHÔNG có khái niệm "track_id"/"t_start-t_end"
    xuyên suốt, mỗi object là 1 lần detect độc lập trên 1 ảnh tĩnh.
  - traffic_pipeline: 1 dòng = 1 TRACK (ByteTrack) xuyên suốt nhiều giây,
    trajectory là danh sách điểm downsample theo THỜI GIAN (giây thực,
    tối đa 20 điểm/track, xem event_timeline.py::_downsample_trajectory).

  Converter này COI MỖI ĐIỂM trong `trajectory` (đã có bbox pixel thật:
  cx/cy/w/h, xem build_traffic_index.py -- ĐÃ THÊM cột "trajectory" để giữ
  dữ liệu này, trước đó bị bỏ qua khi xuất parquet) là 1 "keyframe ảo" tại
  đúng giây đó -- tạo 1 dòng object cho MỖI điểm quỹ đạo, dùng CHUNG
  metadata track (màu/loại xe/track_id) cho mọi điểm của cùng track, nhưng
  TÍNH RIÊNG bbox/vị trí (horizontal/vertical/depth/cx_norm/cy_norm/
  area_ratio) cho từng điểm theo ĐÚNG công thức spatial.py -- vì xe/người
  di chuyển nên vị trí khác nhau ở mỗi điểm trong quỹ đạo, không nên dùng
  chung 1 vị trí cho toàn track.

  Đây là cách xấp xỉ hợp lý nhất có thể làm được mà KHÔNG chạy lại detect
  trên video N (đã có sẵn dữ liệu track, không cần trích xuất lại) -- LƯU Ý:
  số "keyframe" tạo ra theo cách này (tối đa 20 điểm/track) không phải là
  scene-detect thật như aic26_pipeline làm, chỉ là các điểm mẫu quỹ đạo đã
  downsample sẵn.

CÁC TRƯỜNG KHÔNG THỂ SUY RA TỪ traffic_pipeline (để None, KHÔNG suy đoán/bịa):
  - gender/age/age_min/age_max/face_conf: traffic_pipeline KHÔNG chạy face
    analysis (InsightFace) -- luôn None cho mọi dòng "person" tạo ra ở đây.
  - upper_color/lower_color: traffic_pipeline gọi field này "shirt_color"
    (chỉ có màu ÁO TRÊN, không tách áo/quần bằng SCHP như aic26_pipeline) --
    map thẳng shirt_color -> upper_color, để lower_color = None (KHÔNG bịa
    giá trị suy đoán từ upper_color).
  - clothing_mask_source: "heuristic" nếu có shirt_color (khớp đúng bản
    chất -- vehicle_color.py's HSV cho vùng crop qua pose-keypoint, không
    phải SCHP human-parsing thật như aic26_pipeline), None nếu không có.
  - faiss_idx: luôn -1 (chưa qua CLIP embed) -- query_engine.py ĐÃ xử lý an
    toàn giá trị -1 (chỉ loại khỏi bước rerank semantic CLIP, filter cấu
    trúc SQL/Polars vẫn hoạt động bình thường -- xem query_engine.py dòng
    ~187: `c.get("faiss_idx", -1) >= 0`).
  - frame_idx (số thứ tự frame NGUYÊN theo fps gốc): traffic_pipeline lưu
    theo GIÂY (timestamp), không có frame_idx gốc trong trajectory point --
    để None, KHÔNG suy ngược từ timestamp*fps vì có thể sai lệch do
    frame_stride biến thiên giữa các đoạn (streaming segment, xem session
    trước). THAY VÀO ĐÓ, dùng cột MỚI `timestamp_ms` (mili-giây, tính CHÍNH
    XÁC và TRỰC TIẾP từ timestamp thật -- round(t * 1000), không suy đoán
    gì thêm) để backend tra đúng video_id + thời điểm -> tự lấy được đúng
    frame nếu cần, KHÔNG phụ thuộc frame_idx/fps nào cả. `keyframe_id`/
    `frame_id` cũng đã theo đúng quy ước "{video_id}_{timestamp_ms}.jpg"
    (khớp _frame_name() trong event_timeline.py) nên backend có thể parse
    ngược lại video_id + timestamp_ms từ chính tên này nếu muốn.

GIỚI HẠN KHÁC:
  - scene_context (cây/nhà/cửa sổ...), composite/relations/lane-direction
    của traffic_pipeline KHÔNG xuất vào index.parquet này -- schema
    `objects` của aic26 không có chỗ cho "sự kiện" hay bối cảnh không gắn
    track_id cụ thể. Vẫn đọc được đầy đủ qua các file JSON traffic_pipeline
    xuất riêng (<video_id>_relations.json, _scene_context.json...), không
    mất dữ liệu, chỉ không nằm trong `traffic_as_object_index.parquet` này.
  - keyframe_id/frame_id dùng tên "{video_id}_{time_ms}.jpg" (đúng quy ước
    _frame_name() trong event_timeline.py) -- KHÔNG khớp tên file ảnh thật
    nào trên đĩa (traffic_pipeline không xuất ảnh keyframe riêng như
    aic26_pipeline, chỉ xử lý trực tiếp trên video) -- nếu backend cần ảnh
    thật cho các dòng này, phải tự trích frame tại đúng timestamp từ video
    gốc, KHÔNG nằm trong phạm vi converter này (chỉ chuyển đổi bảng).

Chạy (ví dụ, trong notebook Kaggle sau khi build_traffic_index() xong):
    python3 convert_traffic_to_object_index.py \
        --traffic_parquet /kaggle/working/traffic_index.parquet \
        --frame_width 1920 --frame_height 1080 \
        --out /kaggle/working/traffic_as_object_index.parquet
"""
from __future__ import annotations

import argparse
import json
from typing import List, Optional

import polars as pl

# ----------------------------------------------------------------------
# spatial.py's bins (aic26_pipeline/configs/config.yaml, mặc định) -- SAO
# CHÉP ĐÚNG giá trị mặc định để không phải import cả package aic26_pipeline
# (2 codebase độc lập, không muốn tạo phụ thuộc chéo). Nếu aic26_pipeline
# đổi config này, PHẢI sửa lại đây cho khớp.
_HORIZONTAL_BINS = ["left", "center-left", "center", "center-right", "right"]
_VERTICAL_BINS = ["top", "middle", "bottom"]
_DEPTH_BINS = ["near", "mid", "far"]

# Ngưỡng area_ratio cho gần/trung/xa -- ĐÚNG số trong spatial.py's describe_bbox()
_DEPTH_NEAR_THRESHOLD = 0.15
_DEPTH_MID_THRESHOLD = 0.03

# Map tên màu traffic_pipeline (vehicle_color.py/event_timeline.py) -> tên
# màu trong bảng REFERENCE_COLORS của aic26_pipeline/color_naming.py --
# CHỈ đổi tên khi 2 bên rõ ràng CÙNG 1 màu (vd "grey"/"gray" chỉ khác chính
# tả Anh-Anh/Anh-Mỹ), KHÔNG suy diễn màu gần đúng khi tên khác hẳn.
_COLOR_NAME_MAP = {
    "grey": "gray",
    # các màu còn lại của traffic_pipeline (black, blue, brown, green, red,
    # silver, white, yellow) đã trùng khớp tên trong REFERENCE_COLORS, giữ
    # nguyên không cần map.
}


def _map_color(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    return _COLOR_NAME_MAP.get(name, name)


def _describe_point_like_spatial_py(cx: float, cy: float, w: float, h: float,
                                     frame_width: int, frame_height: int) -> dict:
    """Tính lại ĐÚNG công thức spatial.py::describe_bbox() từ 1 điểm
    trajectory {"cx","cy","w","h"} (toạ độ pixel tâm bbox + kích thước) của
    traffic_pipeline -- traffic_pipeline lưu tâm+kích thước, không lưu
    x1y1x2y2 trực tiếp, nên suy ngược lại bbox trước khi áp công thức."""
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    cx_norm = cx / frame_width
    cy_norm = cy / frame_height
    area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / float(frame_width * frame_height)

    h_idx = max(0, min(int(cx_norm * len(_HORIZONTAL_BINS)), len(_HORIZONTAL_BINS) - 1))
    v_idx = max(0, min(int(cy_norm * len(_VERTICAL_BINS)), len(_VERTICAL_BINS) - 1))

    if area_ratio > _DEPTH_NEAR_THRESHOLD:
        depth = _DEPTH_BINS[0]
    elif area_ratio > _DEPTH_MID_THRESHOLD:
        depth = _DEPTH_BINS[1]
    else:
        depth = _DEPTH_BINS[2]

    return {
        "bbox_x1": round(x1, 2), "bbox_y1": round(y1, 2),
        "bbox_x2": round(x2, 2), "bbox_y2": round(y2, 2),
        "cx_norm": round(cx_norm, 4), "cy_norm": round(cy_norm, 4),
        "area_ratio": round(area_ratio, 5),
        "horizontal": _HORIZONTAL_BINS[h_idx], "vertical": _VERTICAL_BINS[v_idx],
        "depth": depth,
    }


# Schema đích PHẢI khớp CHÍNH XÁC cột + kiểu dữ liệu mà
# aic26_pipeline/src/export_parquet.py xuất ra -- để polars.concat() nối
# được 2 DataFrame không lỗi kiểu.
OBJECT_INDEX_SCHEMA = {
    "object_id": pl.Utf8, "keyframe_id": pl.Utf8, "video_id": pl.Utf8,
    "frame_idx": pl.Int64, "timestamp_sec": pl.Float64,
    "label": pl.Utf8, "source": pl.Utf8, "conf": pl.Float64,
    "bbox_x1": pl.Float64, "bbox_y1": pl.Float64, "bbox_x2": pl.Float64, "bbox_y2": pl.Float64,
    "horizontal": pl.Utf8, "vertical": pl.Utf8, "depth": pl.Utf8,
    "cx_norm": pl.Float64, "cy_norm": pl.Float64, "area_ratio": pl.Float64,
    "color_dominant": pl.Utf8, "color_dominant_ratio": pl.Float64,
    "color_secondary": pl.Utf8, "color_secondary_ratio": pl.Float64,
    "gender": pl.Utf8, "age": pl.Int64, "age_min": pl.Int64, "age_max": pl.Int64, "face_conf": pl.Float64,
    "upper_color": pl.Utf8, "upper_color_ratio": pl.Float64,
    "lower_color": pl.Utf8, "lower_color_ratio": pl.Float64,
    "clothing_mask_source": pl.Utf8,
    "faiss_idx": pl.Int64,
    "frame_id": pl.Utf8,
    "upper_color_source": pl.Utf8, "lower_color_source": pl.Utf8,
    # CỘT MỚI (không có trong aic26_pipeline gốc, chỉ traffic_pipeline có) --
    # timestamp mili-giây CHÍNH XÁC, thay thế vai trò frame_idx khi không có
    # frame_idx thật -- xem GHI CHÚ trong docstring converter này. Nếu ghép
    # chung với DataFrame gốc aic26 sau này, cột này sẽ là None ở các dòng
    # aic26 (không có timestamp_ms) -- polars xử lý None bình thường.
    "timestamp_ms": pl.Int64,
}


def convert(traffic_parquet_path: str, frame_width: int, frame_height: int) -> pl.DataFrame:
    df_tracks = pl.read_parquet(traffic_parquet_path)
    rows: List[dict] = []

    for track in df_tracks.iter_rows(named=True):
        video_id = track["video_id"]
        track_id = track["track_id"]
        cls_name = track["cls_name"]

        trajectory = json.loads(track.get("trajectory") or "[]")
        if not trajectory:
            # Track cũ chưa qua build_traffic_index.py bản mới (chưa có cột
            # "trajectory") -- KHÔNG bịa vị trí, chỉ xuất 1 dòng object với
            # bbox/vị trí để None (rơi về nhánh dự phòng bên dưới).
            trajectory = [None]

        color_dominant = _map_color(track.get("vehicle_color"))
        color_dominant_ratio = track.get("vehicle_color_conf")
        upper_color = _map_color(track.get("shirt_color"))
        upper_color_ratio = track.get("shirt_color_conf")
        clothing_mask_source = "heuristic" if upper_color is not None else None

        label = cls_name
        vt_detail_en = track.get("vehicle_type_detail_en")
        if vt_detail_en:
            label = vt_detail_en  # loại xe chi tiết hơn (vd "van") nếu có, giống cách aic26 dùng "label" mở
        type_conf = track.get("vehicle_type_detail_conf")

        for i, point in enumerate(trajectory):
            if point is not None:
                spatial = _describe_point_like_spatial_py(
                    point["cx"], point["cy"], point["w"], point["h"], frame_width, frame_height,
                )
                timestamp_sec = point.get("t", track.get("t_start"))
                timestamp_ms = int(round(timestamp_sec * 1000))
                keyframe_id = f"{video_id}_{timestamp_ms}.jpg"
            else:
                spatial = {k: None for k in (
                    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                    "cx_norm", "cy_norm", "area_ratio", "horizontal", "vertical", "depth",
                )}
                timestamp_sec = track.get("t_start")
                timestamp_ms = int(round(timestamp_sec * 1000)) if timestamp_sec is not None else None
                keyframe_id = f"{video_id}_track{track_id}"

            object_id = f"{keyframe_id}_trk{track_id}_o{i}"
            rows.append({
                "object_id": object_id,
                "keyframe_id": keyframe_id,
                "video_id": video_id,
                "frame_idx": None,  # traffic_pipeline lưu theo giây (timestamp), không có frame_idx gốc đáng tin cậy để suy ngược -- dùng timestamp_ms bên dưới thay thế
                "timestamp_sec": timestamp_sec,
                "timestamp_ms": timestamp_ms,
                "label": label,
                "source": "closed_vocab",
                "conf": type_conf if type_conf is not None else track.get("avg_conf"),
                **spatial,
                "color_dominant": color_dominant, "color_dominant_ratio": color_dominant_ratio,
                "color_secondary": None, "color_secondary_ratio": None,
                "gender": None, "age": None, "age_min": None, "age_max": None, "face_conf": None,
                "upper_color": upper_color, "upper_color_ratio": upper_color_ratio,
                "lower_color": None, "lower_color_ratio": None,
                "clothing_mask_source": clothing_mask_source,
                "faiss_idx": -1,
                "frame_id": keyframe_id,
                "upper_color_source": clothing_mask_source, "lower_color_source": None,
            })

    if not rows:
        return pl.DataFrame(schema=OBJECT_INDEX_SCHEMA)
    return pl.DataFrame(rows, schema=OBJECT_INDEX_SCHEMA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traffic_parquet", required=True, help="Đường dẫn traffic_index.parquet (traffic_pipeline)")
    ap.add_argument("--out", required=True, help="Đường dẫn ghi file parquet ĐÃ CONVERT -- schema aic26")
    ap.add_argument("--frame_width", type=int, default=1920)
    ap.add_argument("--frame_height", type=int, default=1080)
    args = ap.parse_args()

    df_converted = convert(args.traffic_parquet, args.frame_width, args.frame_height)
    df_converted.write_parquet(args.out)
    print(f"Đã convert {df_converted.height} dòng object từ track trong {args.traffic_parquet} -> {args.out}")


if __name__ == "__main__":
    main()
