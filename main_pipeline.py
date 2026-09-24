"""
Pipeline chính: chạy detect+track -> OCR overlay theo giây (cache) -> đèn
giao thông theo giây (cache) -> build event-timeline -> xuất JSON.

HỖ TRỢ 2 CHẾ ĐỘ:
  1) --video_path <file>      : chạy 1 video đơn (demo nhanh)
  2) --video_dir <thư mục>    : chạy HÀNG LOẠT mọi video trong thư mục (dùng
                                 trên Kaggle, xem notebook đi kèm) -- mỗi
                                 video ra 1 file events/<video_id>.json,
                                 video_id = tên file không đuôi.

Chạy: python3 main_pipeline.py --video_dir /kaggle/input/traffic-videos --out_dir ./events
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import glob
import json
import logging
import math
import os
from typing import Dict, List, Optional

import cv2

from detect_track import run_detection_tracking
from ocr_overlay import ocr_overlay, OverlayInfo
from traffic_light import detect_light_color
from event_timeline import build_events_from_tracks, detect_composite_events, Event
from event_description import build_all_descriptions
from vehicle_relations import detect_order_and_overtake, detect_collision_candidates, detect_side_by_side
from zone_events import detect_pedestrian_crossing_outside_crosswalk, detect_vehicle_stopped_on_crosswalk
from density_timeline import build_density_timeline, summarize_density
from lane_direction import LaneDirectionModel, detect_wrong_way_vehicles
from vehicle_type_refine import VehicleTypeRefiner, refine_vehicle_types
from scene_context_classes import detect_scene_context
from track_dedup import deduplicate_events
from scene_description import build_scene_descriptions
import config
import mem_guard

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("traffic_pipeline.main")

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


class _ReopeningFrameReader:
    """Đọc lại frame theo frame_idx bất kỳ (không tuần tự -- _select_representative()
    trong event_timeline.py chọn frame theo ĐỘ NÉT/KÍCH THƯỚC bbox, không theo
    thứ tự thời gian, nên gọi frame_reader() nhảy lung tung qua lại nhiều vị
    trí khác nhau trong video, có thể hàng nghìn lần với video nhiều track).

    NGUYÊN NHÂN RAM TRÀN DẦN ĐÃ XÁC ĐỊNH: seek ngẫu nhiên (cap.set(CAP_PROP_POS_FRAMES))
    lặp đi lặp lại rất nhiều lần trên CÙNG 1 cv2.VideoCapture, với video .mov/H.264
    long-GOP, khiến buffer giải mã nội bộ của FFmpeg backend (native C, KHÔNG
    phải object Python nên gc.collect()/del không giải phóng được) tích lũy
    dần theo số lần seek -- càng nhiều track/càng về sau video càng nặng, đúng
    hiện tượng quan sát được (đoạn 5, 6, 7... của cùng 1 video mới bắt đầu tràn).

    FIX (2 LỚP, để KHÔNG phụ thuộc 1 con số cố định):
    1) LƯỚI AN TOÀN CỐ ĐỊNH: cứ REOPEN_EVERY lần đọc thì đóng/mở lại, bất kể
       RAM đang cao hay thấp -- phòng trường hợp không đo được RAM (thiếu
       psutil) thì vẫn có 1 cơ chế dọn tối thiểu.
    2) TỰ ĐỘNG THEO RAM THẬT (mem_guard.py, ưu tiên hơn, kiểm tra mỗi lần
       đọc): đo RSS tiến trình / tổng RAM hệ thống qua psutil -- vượt 70%
       thì đóng/mở lại NGAY dù chưa tới mốc REOPEN_EVERY. Cách này thích
       nghi được với video ngắn/dài khác nhau và với RAM khác nhau giữa các
       session Kaggle (29GB bản mới, 13GB bản cũ), thay vì đoán 1 con số
       đếm cứng có thể sai với video/máy khác.

    Đóng/mở lại VideoCapture là cách DUY NHẤT chắc chắn giải phóng buffer
    decoder native mà OpenCV/FFmpeg giữ bên trong -- không có API nào khác
    để "flush" nó khi vẫn dùng chung 1 VideoCapture."""

    REOPEN_EVERY = 300  # lưới an toàn cố định, dùng khi KHÔNG đo được RAM (thiếu psutil)

    def __init__(self, video_path: str, stride: int):
        self.video_path = video_path
        self.stride = stride
        self._cap = cv2.VideoCapture(video_path)
        self._n_reads = 0

    def _reopen(self):
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(self.video_path)

    def read(self, frame_idx: int):
        freed = mem_guard.free_memory_if_needed(self._reopen, context="_ReopeningFrameReader")
        if not freed and self._n_reads > 0 and self._n_reads % self.REOPEN_EVERY == 0:
            self._reopen()
            gc.collect()
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx * self.stride)
        ok, frame = self._cap.read()
        self._n_reads += 1
        return frame if ok else None

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def build_second_level_cache(video_path: str, duration_sec: float):
    """OCR + đèn giao thông chỉ cần chạy 1 LẦN MỖI GIÂY (đồng hồ overlay chỉ
    đổi theo giây, đèn không đổi màu quá nhanh) -- tiết kiệm rất nhiều so
    với chạy trên mọi frame. Trả về 2 dict {giây_nguyên: giá_trị}.

    LƯU Ý QUAN TRỌNG: vùng crop OCR/đèn trong config.py được ĐO CHO ĐÚNG 1
    CAMERA (N001-V001.mov, 1920x1080, overlay+đèn ở vị trí cố định của
    camera đó). Nếu video khác có overlay/đèn ở vị trí khác, các con số
    trong config.py PHẢI được đo lại -- xem hướng dẫn ở đầu notebook."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    ocr_cache: Dict[int, OverlayInfo] = {}
    light_cache: Dict[int, str] = {}

    n_seconds = int(math.ceil(duration_sec))
    logger.info(f"Xây cache OCR + đèn giao thông cho {n_seconds} giây (1 frame/giây)...")

    # Seek tuần tự (mỗi giây 1 lần, luôn tiến về phía trước) ít nghiêm trọng
    # hơn seek ngẫu nhiên (xem _ReopeningFrameReader ở trên), nhưng với video
    # rất dài (hàng nghìn giây) vẫn tích lũy buffer decoder native đáng kể
    # theo cùng cơ chế. Dùng CÙNG 2 lớp bảo vệ: lưới an toàn cố định
    # (REOPEN_EVERY_SEC) + tự động theo RAM thật đo qua mem_guard.py (ưu
    # tiên hơn, thích nghi theo video dài/ngắn và RAM thật của session).
    REOPEN_EVERY_SEC = 300
    _cap_holder = {"cap": cap}

    def _reopen_ocr_cap():
        _cap_holder["cap"].release()
        _cap_holder["cap"] = cv2.VideoCapture(video_path)

    for sec in range(n_seconds):
        freed = mem_guard.free_memory_if_needed(_reopen_ocr_cap, context="build_second_level_cache")
        if not freed and sec > 0 and sec % REOPEN_EVERY_SEC == 0:
            _reopen_ocr_cap()
            gc.collect()
        cap = _cap_holder["cap"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
        ok, frame = cap.read()
        if not ok:
            break
        ocr_cache[sec] = ocr_overlay(frame)
        light_cache[sec] = detect_light_color(frame)
        if sec % 60 == 0:
            logger.info(f"  đã OCR/đèn tới giây {sec}/{n_seconds}")

    _cap_holder["cap"].release()
    return ocr_cache, light_cache


def make_lookups(ocr_cache: Dict[int, OverlayInfo], light_cache: Dict[int, str]):
    def ocr_lookup(t: float) -> Optional[OverlayInfo]:
        return ocr_cache.get(int(t))

    def light_lookup(t: float) -> Optional[str]:
        return light_cache.get(int(t))

    return ocr_lookup, light_lookup


def event_to_dict(e: Event) -> dict:
    return dataclasses.asdict(e)


def demo_queries(events, ocr_cache, duration_sec=None):
    print("\n" + "=" * 70)
    print("DEMO QUERY 1: lọc theo khoảng thời gian (giây 0-60) + loại xe")
    print("=" * 70)
    for e in events:
        if e.t_start <= 60 and e.cls_name in ("car", "truck", "bus"):
            print(f"  track={e.track_id:4d} class={e.cls_name:10s} color={e.vehicle_color} "
                  f"[{e.t_start:6.1f}s - {e.t_end:6.1f}s] loc={e.location}")

    print("\n" + "=" * 70)
    print("DEMO QUERY 2: composite events (đèn đỏ, vượt đèn, dừng lâu, kẹt xe)")
    print("=" * 70)
    composites = detect_composite_events(events)
    by_type = {}
    for c in composites:
        by_type.setdefault(c["event_type"], []).append(c)
    for etype, items in by_type.items():
        print(f"  [{etype}] tổng {len(items)}:")
        for c in items[:5]:
            loc = c.get("location") or "?"
            print(f"    [{c['t_start']:6.1f}s - {c['t_end']:6.1f}s] tại {loc}")

    if ocr_cache:
        print("\n" + "=" * 70)
        print("DEMO QUERY 3: OCR overlay đọc được ở vài mốc giây")
        print("=" * 70)
        for sec in sorted(ocr_cache.keys())[:5]:
            info = ocr_cache[sec]
            print(f"  giây {sec}: location={info.location!r} time={info.time_str!r}")

    print("\n" + "=" * 70)
    print("DEMO QUERY 4: mô tả tự nhiên sinh tự động (dùng để text-embedding)")
    print("=" * 70)
    from event_description import describe_event
    for e in events[:5]:
        print(f"  {describe_event(e)}")

    print("\n" + "=" * 70)
    print("DEMO QUERY 5: quan hệ giữa các xe (trước/sau, vượt, va chạm)")
    print("=" * 70)
    relations = detect_order_and_overtake(events)
    overtakes = [r for r in relations if r["relation"] == "overtake"]
    print(f"  Tổng {len(relations)} quan hệ trước/sau, trong đó {len(overtakes)} là 'vượt' (overtake):")
    for r in overtakes[:10]:
        print(f"    track {r['overtaker_track_id']} ({r['overtaker_cls']}) vượt "
              f"track {r['overtaken_track_id']} ({r['overtaken_cls']}) lúc "
              f"{r['t_start']:.1f}s-{r['t_end']:.1f}s")
    collisions = detect_collision_candidates(events)
    print(f"  Ứng viên va chạm (CẦN xem lại bằng mắt): {len(collisions)}")
    for c in collisions[:5]:
        print(f"    track {c['track_id_1']} ({c['cls_1']}) & track {c['track_id_2']} ({c['cls_2']}) lúc {c['t']:.1f}s")

    from vehicle_relations import detect_side_by_side
    side_by_side = detect_side_by_side(events)
    print(f"  Xe đi cạnh nhau (side-by-side): {len(side_by_side)}")
    for s in side_by_side[:5]:
        print(f"    trái=track {s['left_track_id']} ({s['left_cls']}) / phải=track {s['right_track_id']} "
              f"({s['right_cls']}) lúc {s['t_start']:.1f}s-{s['t_end']:.1f}s")

    from scene_description import build_scene_descriptions
    scenes = build_scene_descriptions(events, light_lookup=None, side_by_side=side_by_side, order_relations=relations)
    print(f"  Mô tả cảnh gộp nhiều xe (scene description): {len(scenes)}")
    for s in scenes[:5]:
        print(f"    [{s['t']:.1f}s] {s['description']}")

    print("\n" + "=" * 70)
    print("DEMO QUERY 6: sự kiện theo vùng (băng qua sai vạch, dừng đè vạch)")
    print("=" * 70)
    ped_out = detect_pedestrian_crossing_outside_crosswalk(events, config.CROSSWALK_POLYGON)
    veh_stop = detect_vehicle_stopped_on_crosswalk(events, config.CROSSWALK_POLYGON)
    print(f"  Người đi bộ băng qua ngoài vạch: {len(ped_out)}")
    for p in ped_out[:5]:
        print(f"    track {p['track_id']} lúc {p['t_start']:.1f}s-{p['t_end']:.1f}s (trong vạch: {p['fraction_in_crosswalk']*100:.0f}%)")
    print(f"  Xe dừng đè vạch: {len(veh_stop)}")
    for v in veh_stop[:5]:
        print(f"    track {v['track_id']} ({v['cls_name']}) lúc {v['t_start']:.1f}s-{v['t_end']:.1f}s")

    print("\n" + "=" * 70)
    print("DEMO QUERY 7: xe đi sai làn/ngược chiều (tự học hướng từ đa số xe)")
    print("=" * 70)
    lane_model = LaneDirectionModel()
    lane_model.fit(events)
    wrong_way = detect_wrong_way_vehicles(events, lane_model=lane_model)
    print(f"  Đã học được hướng đa số tin cậy cho {lane_model.n_reliable_cells()} ô lưới.")
    print(f"  Ứng viên đi sai làn/ngược chiều (CẦN xem lại bằng mắt): {len(wrong_way)}")
    for w in wrong_way[:5]:
        print(f"    track {w['track_id']} ({w['cls_name']}) lúc {w['t_start']:.1f}s-{w['t_end']:.1f}s "
              f"lệch {w['avg_angle_deg']}° trên {w['n_checked']} đoạn quỹ đạo")

    if duration_sec:
        print("\n" + "=" * 70)
        print("DEMO QUERY 8: mật độ giao thông theo thời gian (bucket 10s)")
        print("=" * 70)
        timeline = build_density_timeline(events, duration_sec, bucket_sec=10.0)
        summary = summarize_density(timeline)
        if summary:
            b = summary["busiest_window"]
            q = summary["quietest_window"]
            print(f"  Đông nhất: {b['t_start']:.0f}s-{b['t_end']:.0f}s ({b['n_total']} đối tượng, "
                  f"{b['n_vehicles']} xe + {b['n_persons']} người)")
            print(f"  Vắng nhất: {q['t_start']:.0f}s-{q['t_end']:.0f}s ({q['n_total']} đối tượng)")
            print(f"  Trung bình: {summary['avg_vehicles_per_window']:.1f} đối tượng/khoảng 10s")

    print("\n" + "=" * 70)
    print("DEMO QUERY 9: màu áo người đi bộ / người ngồi xe máy (person_pose.py)")
    print("=" * 70)
    n_shirt = sum(1 for e in events if getattr(e, "shirt_color", None))
    n_person_or_2wheeler = sum(1 for e in events if e.cls_name in ("person", "motorcycle", "bicycle"))
    print(f"  Xác định được màu áo cho {n_shirt}/{n_person_or_2wheeler} track người/xe 2 bánh "
          f"(số còn lại: không đủ keypoint tin cậy, hoặc confidence màu dưới ngưỡng -- xem person_pose.py).")
    for e in events:
        if getattr(e, "shirt_color", None):
            who = "người đi bộ" if e.cls_name == "person" else f"người lái {e.cls_name}"
            print(f"    track {e.track_id} ({who}): áo màu {e.shirt_color} (conf={e.shirt_color_conf})")


_COLOR_CLASSIFIER = None  # cache toàn cục -- load model màu xe 1 lần cho mọi video
_TYPE_REFINER = None      # cache toàn cục -- load model YOLOE (loại xe chi tiết) 1 lần cho mọi video
_POSE_ESTIMATOR = None    # cache toàn cục -- load model YOLO-pose (vùng áo) 1 lần cho mọi video


def _get_color_classifier():
    global _COLOR_CLASSIFIER
    if not config.ENABLE_VEHICLE_COLOR:
        return None
    if _COLOR_CLASSIFIER is None:
        from vehicle_color import VehicleColorClassifier
        logger.info("Đang load model màu xe (tự động dùng fallback HSV nếu model lỗi/thiếu)...")
        _COLOR_CLASSIFIER = VehicleColorClassifier(model_path=config.VEHICLE_COLOR_MODEL_PATH)
    return _COLOR_CLASSIFIER


def _get_pose_estimator():
    global _POSE_ESTIMATOR
    if not getattr(config, "ENABLE_SHIRT_COLOR", True):
        return None
    if _POSE_ESTIMATOR is None:
        from person_pose import PersonPoseEstimator
        logger.info("Đang load model YOLO-pose (xác định vùng áo cho người đi bộ/người ngồi xe máy)...")
        _POSE_ESTIMATOR = PersonPoseEstimator(model_name=getattr(config, "POSE_MODEL_NAME", "yolov8s-pose.pt"))
    return _POSE_ESTIMATOR


def _get_type_refiner(force: bool = False):
    """force=True: load refiner (dùng chung YOLOE model) BẤT KỂ
    ENABLE_VEHICLE_TYPE_DETAIL đang bật/tắt -- cần cho scene_context_classes.py
    (ENABLE_SCENE_CONTEXT_CLASSES độc lập với ENABLE_VEHICLE_TYPE_DETAIL, xem
    config.py) vì 2 tính năng này dùng CHUNG 1 model YOLOE để tránh tải model
    2 lần, nhưng có thể bật/tắt riêng từng cái."""
    global _TYPE_REFINER
    if not force and not getattr(config, "ENABLE_VEHICLE_TYPE_DETAIL", True):
        return None
    if _TYPE_REFINER is None:
        logger.info("Đang load model YOLOE (dùng chung cho loại xe chi tiết + class bối cảnh phụ)...")
        _TYPE_REFINER = VehicleTypeRefiner()
    return _TYPE_REFINER


def process_one_video(video_path: str, out_dir: str, max_frames: Optional[int], stride: int,
                       show_demo: bool = False) -> str:
    """Chạy toàn bộ pipeline cho 1 video, ghi ra out_dir/<video_id>.json.
    Trả về đường dẫn file JSON đã ghi."""
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{video_id}.json")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_sec = total_frames / fps if fps > 0 else 0
    cap.release()
    logger.info(f"[{video_id}] {duration_sec:.1f}s, {fps:.1f}fps -- bắt đầu xử lý...")

    if max_frames is not None:
        duration_for_cache = min(duration_sec, max_frames * stride / fps + 5)
    else:
        duration_for_cache = duration_sec

    logger.info(f"[{video_id}] BƯỚC 1: Detection + Tracking (YOLOv8 + ByteTrack)")
    detections = run_detection_tracking(video_path=video_path, max_frames=max_frames, frame_stride=stride)

    logger.info(f"[{video_id}] BƯỚC 2: OCR overlay + đèn giao thông (cache theo giây)")
    ocr_cache, light_cache = build_second_level_cache(video_path, duration_for_cache)
    ocr_lookup, light_lookup = make_lookups(ocr_cache, light_cache)

    logger.info(f"[{video_id}] BƯỚC 3: Build event-timeline từ track")
    color_classifier = _get_color_classifier()
    pose_estimator = _get_pose_estimator()
    # frame_reader: đọc lại đúng frame gốc theo frame_idx (frame_idx đã tính
    # theo stride khi detect+track, xem detect_track.py -- timestamp =
    # frame_idx * stride / fps, nên frame thật trong video tương ứng nằm ở
    # vị trí frame_idx * stride). _select_representative() trong
    # event_timeline.py chọn frame theo ĐỘ NÉT/KÍCH THƯỚC bbox (KHÔNG theo
    # thứ tự thời gian), nên frame_reader() bị gọi seek NGẪU NHIÊN qua lại
    # rất nhiều vị trí khác nhau trong video -- với video dài/nhiều track,
    # có thể tới hàng nghìn lần seek trên CÙNG 1 VideoCapture.
    #
    # ĐÃ XÁC ĐỊNH QUA THỰC NGHIỆM: đây là nguyên nhân RAM tăng dần không
    # giới hạn (buffer giải mã nội bộ của FFmpeg/libavcodec bên trong
    # VideoCapture tích lũy theo số lần seek ngẫu nhiên qua nhiều GOP khác
    # nhau -- bộ nhớ NATIVE C, gc.collect()/del KHÔNG giải phóng được, chỉ
    # có cách đóng hẳn rồi mở lại VideoCapture mới giải phóng). Dùng
    # _ReopeningFrameReader thay vì 1 VideoCapture mở suốt để tự động đóng
    # /mở lại định kỳ, chặn tích lũy này -- xem docstring class ở trên.
    _frame_reader_obj = _ReopeningFrameReader(video_path, stride)
    _frame_reader = _frame_reader_obj.read

    events = build_events_from_tracks(
        detections, video_id=video_id, ocr_lookup=ocr_lookup, light_lookup=light_lookup,
        frame_reader=_frame_reader, color_classifier=color_classifier, pose_estimator=pose_estimator,
    )

    # Giải phóng ngay danh sách Detection thô (1 dòng/đối tượng/frame -- với
    # video dài có thể hàng trăm nghìn phần tử) VÀ ocr_cache/light_cache (1
    # entry/giây -- với video dài hàng nghìn giây cũng tích lũy đáng kể)
    # NGAY SAU KHI build_events_from_tracks() đã trích hết thông tin cần
    # thiết vào events -- không cần giữ tới cuối hàm nữa. gc.collect() ép
    # dọn ngay thay vì chờ garbage collector Python tự chạy theo chu kỳ
    # (mặc định có thể trễ, giữ RAM cao không cần thiết qua các bước sau).
    del detections, ocr_cache, light_cache
    gc.collect()

    logger.info(f"[{video_id}] BƯỚC 3-dedup: Lọc track trùng lặp (cùng 1 vật thể bị track nhầm nhiều lần -- "
                f"xem track_dedup.py, phổ biến với vật đứng yên lâu)")
    n_before_dedup = len(events)
    events = deduplicate_events(events)
    if len(events) != n_before_dedup:
        logger.info(f"[{video_id}] Đã gộp {n_before_dedup - len(events)} track trùng lặp "
                    f"({n_before_dedup} -> {len(events)}).")

    logger.info(f"[{video_id}] BƯỚC 3a: Nhận diện loại xe chi tiết (YOLOE open-vocabulary, "
                f"chỉ chạy 1 lần/track trên frame đại diện tốt nhất, không chạy toàn video)")
    type_refiner = _get_type_refiner()
    if type_refiner is not None:
        n_types = refine_vehicle_types(events, type_refiner, _frame_reader)
        logger.info(f"[{video_id}] Đã phát hiện loại xe chi tiết cho {n_types}/{len(events)} track.")

    logger.info(f"[{video_id}] BƯỚC 3a-2: Mở rộng phạm vi class bối cảnh (cây/nhà/cửa sổ/bánh xe/..., "
                f"đối chiếu mẫu BTC N001-V001.zip) -- chỉ lấy mẫu vài frame, KHÔNG gắn vào track cụ thể")
    scene_context = {}
    if getattr(config, "ENABLE_SCENE_CONTEXT_CLASSES", True):
        context_refiner = type_refiner if type_refiner is not None else _get_type_refiner(force=True)
        n_context_sample_frames = getattr(config, "SCENE_CONTEXT_N_SAMPLE_FRAMES", 5)
        scene_context = detect_scene_context(
            context_refiner, _frame_reader, n_total_frames=int(total_frames),
            n_samples=n_context_sample_frames,
        )
        logger.info(f"[{video_id}] Đã phát hiện {len(scene_context)} loại bối cảnh phụ "
                    f"(vd cây/nhà/cửa sổ) trong {n_context_sample_frames} frame mẫu.")

    _frame_reader_obj.close()
    gc.collect()

    logger.info(f"[{video_id}] BƯỚC 3b: Composite event + quan hệ + vùng (crosswalk) + mật độ + mô tả")
    composites = detect_composite_events(events)
    order_relations = detect_order_and_overtake(events)
    side_by_side_relations = detect_side_by_side(events)
    collision_candidates = detect_collision_candidates(events)
    ped_outside_crosswalk = detect_pedestrian_crossing_outside_crosswalk(events, config.CROSSWALK_POLYGON)
    vehicle_on_crosswalk = detect_vehicle_stopped_on_crosswalk(events, config.CROSSWALK_POLYGON)
    density_timeline = build_density_timeline(events, duration_sec, bucket_sec=10.0)
    density_summary = summarize_density(density_timeline)

    # Hướng làn "chuẩn" TỰ HỌC từ chính quỹ đạo đa số xe trong video này (xem
    # lane_direction.py) -- không cần nhập tay, không cần model AI riêng.
    # lane_model.n_reliable_cells() == 0 nghĩa là video quá ngắn/ít xe để học
    # -- lúc đó detect_wrong_way_vehicles() tự trả về [] (không đoán mò).
    lane_model = LaneDirectionModel()
    lane_model.fit(events)
    wrong_way_candidates = detect_wrong_way_vehicles(events, lane_model=lane_model)
    logger.info(f"[{video_id}] Học hướng làn: {lane_model.n_reliable_cells()} ô lưới tin cậy, "
                f"{len(wrong_way_candidates)} ứng viên đi sai làn/ngược chiều.")

    descriptions = build_all_descriptions(
        events, composites,
        extra_events=ped_outside_crosswalk + vehicle_on_crosswalk + wrong_way_candidates,
    )

    logger.info(f"[{video_id}] BƯỚC 3c: Mô tả cảnh gộp nhiều xe (scene description) -- "
                f"'xe A cạnh xe B, đèn X, phía trước có N xe máy'")
    scene_descriptions = build_scene_descriptions(
        events, light_lookup=light_lookup,
        side_by_side=side_by_side_relations, order_relations=order_relations,
    )
    logger.info(f"[{video_id}] Đã sinh {len(scene_descriptions)} mô tả cảnh.")

    os.makedirs(out_dir, exist_ok=True)
    desc_path = os.path.join(out_dir, f"{video_id}_descriptions.json")
    with open(desc_path, "w", encoding="utf-8") as f:
        json.dump(descriptions, f, ensure_ascii=False, indent=2)

    scene_desc_path = os.path.join(out_dir, f"{video_id}_scene_descriptions.json")
    with open(scene_desc_path, "w", encoding="utf-8") as f:
        json.dump(scene_descriptions, f, ensure_ascii=False, indent=2)

    # File RIÊNG, KHÔNG gộp vào events chính -- xem docstring scene_context_classes.py
    # về lý do tách biệt (bối cảnh tĩnh không gắn với track_id nào cụ thể).
    scene_context_path = os.path.join(out_dir, f"{video_id}_scene_context.json")
    with open(scene_context_path, "w", encoding="utf-8") as f:
        json.dump(scene_context, f, ensure_ascii=False, indent=2)

    relations_path = os.path.join(out_dir, f"{video_id}_relations.json")
    with open(relations_path, "w", encoding="utf-8") as f:
        json.dump({
            "order_and_overtake": order_relations,
            "side_by_side": side_by_side_relations,
            "collision_candidates": collision_candidates,
            "pedestrian_crossing_outside_crosswalk": ped_outside_crosswalk,
            "vehicle_stopped_on_crosswalk": vehicle_on_crosswalk,
            "wrong_way_candidates": wrong_way_candidates,
            "lane_direction_model_info": {
                "n_reliable_cells": lane_model.n_reliable_cells(),
                "cell_size": lane_model.cell_size,
                "note": "Số ô lưới học được hướng đa số tin cậy -- 0 nghĩa là video quá ngắn/ít "
                        "xe để tự học, wrong_way_candidates khi đó luôn rỗng (không đoán mò).",
            },
        }, f, ensure_ascii=False, indent=2)

    density_path = os.path.join(out_dir, f"{video_id}_density.json")
    with open(density_path, "w", encoding="utf-8") as f:
        json.dump({"timeline": density_timeline, "summary": density_summary}, f, ensure_ascii=False, indent=2)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([event_to_dict(e) for e in events], f, ensure_ascii=False, indent=2)
    logger.info(f"[{video_id}] Đã ghi {len(events)} event, {len(descriptions)} mô tả, "
                f"{len(scene_descriptions)} mô tả cảnh gộp, "
                f"{len(order_relations)} quan hệ trước/sau-vượt, {len(side_by_side_relations)} quan hệ cạnh nhau, "
                f"{len(collision_candidates)} ứng viên va chạm, "
                f"{len(ped_outside_crosswalk)} người băng sai vạch, {len(vehicle_on_crosswalk)} xe dừng đè vạch, "
                f"{len(wrong_way_candidates)} ứng viên đi sai làn/ngược chiều.")

    if show_demo:
        demo_queries(events, ocr_cache, duration_sec=duration_for_cache)

    return out_path


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video_path", type=str, help="Chạy 1 video đơn")
    group.add_argument("--video_dir", type=str, help="Chạy hàng loạt mọi video trong thư mục này")
    parser.add_argument("--out_dir", type=str, default="./events",
                         help="Thư mục ghi kết quả (mỗi video 1 file <video_id>.json)")
    parser.add_argument("--max_frames", type=int, default=None,
                         help="Giới hạn số frame detect+track MỖI video để demo nhanh (None = chạy hết)")
    parser.add_argument("--stride", type=int, default=2,
                         help="Bỏ bớt frame khi detect+track (2 = xử lý 1/2 số frame)")
    parser.add_argument("--demo", action="store_true", help="In demo query sau khi xử lý xong")
    args = parser.parse_args()

    if args.video_path:
        video_paths = [args.video_path]
    else:
        video_paths = sorted(
            p for p in glob.glob(os.path.join(args.video_dir, "*"))
            if p.lower().endswith(VIDEO_EXTENSIONS)
        )
        if not video_paths:
            raise SystemExit(f"Không tìm thấy video nào trong {args.video_dir} (đuôi hợp lệ: {VIDEO_EXTENSIONS})")
        logger.info(f"Tìm thấy {len(video_paths)} video trong {args.video_dir}.")

    results = []
    for i, vp in enumerate(video_paths):
        logger.info(f"=== Video {i+1}/{len(video_paths)}: {vp} ===")
        try:
            out_path = process_one_video(vp, args.out_dir, args.max_frames, args.stride, show_demo=args.demo)
            results.append({"video": vp, "status": "ok", "out": out_path})
        except Exception as e:
            logger.error(f"LỖI xử lý {vp}: {type(e).__name__}: {e}")
            results.append({"video": vp, "status": "error", "error": str(e)})

    n_ok = sum(1 for r in results if r["status"] == "ok")
    logger.info(f"=== HOÀN TẤT: {n_ok}/{len(results)} video xử lý thành công. Kết quả tại {args.out_dir}/ ===")
    if any(r["status"] == "error" for r in results):
        logger.warning("Video lỗi:")
        for r in results:
            if r["status"] == "error":
                logger.warning(f"  {r['video']}: {r['error']}")


if __name__ == "__main__":
    main()
