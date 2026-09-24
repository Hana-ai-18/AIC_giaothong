"""
Detection + tracking bằng YOLOv8 + ByteTrack (built-in trong ultralytics --
model.track() tự động dùng ByteTrack, gắn track_id ổn định xuyên suốt
nhiều frame cho mỗi đối tượng).

Chạy trực tiếp trên video, KHÔNG cần decode thủ công -- ultralytics tự đọc
video qua OpenCV nội bộ, xử lý từng frame, trả generator kết quả.

DÙNG `yolov8s.pt` (small) THAY VÌ `yolov8n.pt` (nano, mặc định cũ) -- ĐÃ ĐỔI
để tăng độ chính xác detect, đối chiếu yêu cầu "mạnh hơn BTC": nano là bản
NHẸ/NHANH NHẤT trong họ YOLOv8 nhưng đánh đổi bằng recall/precision thấp
nhất (dễ bỏ sót object nhỏ/xa, dễ nhầm class khi vật thể bị che khuất một
phần) -- theo benchmark công bố của Ultralytics trên COCO val, `yolov8s` có
mAP cao hơn `yolov8n` rõ rệt (~44.9 so với ~37.3 mAP50-95) trong khi vẫn
chạy realtime được trên GPU T4 của Kaggle (chậm hơn nano nhưng không đáng
kể so với tổng thời gian pipeline, vốn đã bottleneck ở OCR/pose/color chứ
không phải riêng detect). Nếu cần chính xác hơn nữa và chấp nhận chậm hơn,
đổi thành "yolov8m.pt" (medium, mAP ~50.2) -- không khuyến nghị "yolov8l/x"
trên CPU/Kaggle free-tier vì quá chậm cho việc chạy hàng loạt nhiều video.

`conf_threshold` HẠ TỪ 0.35 XUỐNG 0.25 -- ngưỡng 0.35 cũ bỏ sót một số
object có confidence trung bình (đặc biệt xe/người ở xa, một phần bị che
khuất), nhưng hạ ngưỡng đơn thuần dễ tăng false positive nếu KHÔNG có lọc
hậu kỳ đi kèm. Đã có sẵn 2 lớp lọc hậu kỳ trong pipeline giảm rủi ro này:
(1) MIN_EVENT_DURATION_SEC (config.py) loại track quá ngắn (dấu hiệu nhiễu
detect chớp nhoáng thường có confidence thấp gần ngưỡng), (2) track_dedup.py
gộp/loại track trùng lặp. Nếu sau khi chạy thật thấy false positive tăng rõ
rệt (nhiều track rất ngắn, avg_conf sát 0.25), nên tăng lại về 0.3 thay vì
0.35 (điểm cân bằng giữa 2 lần thử).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from config import RELEVANT_CLASSES, VIDEO_PATH
import mem_guard

logger = logging.getLogger("traffic_pipeline.detect_track")


# slots=True: giảm ~40-50% bộ nhớ mỗi Detection object so với dataclass
# thường (bỏ __dict__ per-instance, dùng slot cố định) -- QUAN TRỌNG vì đây
# là nguồn RAM lớn nhất trong cả pipeline khi video dài: `detections` tích
# lũy TOÀN BỘ danh sách này trong RAM tới khi build_events_from_tracks() xử
# lý xong (không thể stream theo track vì ByteTrack cần thấy hết trước khi
# biết track nào dài/ngắn). Với video vài chục nghìn frame x vài chục vật
# thể/frame, có thể là hàng trăm nghìn tới cả triệu Detection -- slots giúp
# giảm đáng kể RAM đỉnh mà không đổi bất kỳ logic/API nào ở nơi khác.
@dataclass(slots=True)
class Detection:
    frame_idx: int
    timestamp: float
    track_id: int
    cls_id: int
    cls_name: str
    bbox: tuple  # (x1, y1, x2, y2)
    conf: float


def run_detection_tracking(
    video_path: str = VIDEO_PATH,
    model_name: str = "yolov8s.pt",
    conf_threshold: float = 0.25,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    device: str = "cpu",
) -> List[Detection]:
    """
    Chạy YOLOv8 + ByteTrack trên video, trả về danh sách Detection phẳng
    (1 dòng = 1 đối tượng ở 1 frame). frame_stride > 1 để bỏ bớt frame nếu
    cần chạy nhanh hơn trên CPU (không ảnh hưởng track_id vì ByteTrack vẫn
    theo dõi được qua các frame cách quãng, miễn stride không quá lớn).

    max_frames: giới hạn số frame xử lý (để demo nhanh trên video dài) --
    None nghĩa là chạy hết video.
    """
    model = YOLO(model_name)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    logger.info(f"Bắt đầu detect+track trên {video_path} (fps gốc={fps:.2f}, stride={frame_stride})")

    detections: List[Detection] = []
    frame_idx = -1
    n_processed = 0

    # persist=True: giữ track_id xuyên suốt qua các lần gọi (bắt buộc để
    # ByteTrack hoạt động đúng theo thời gian, không phải detect độc lập
    # từng frame).
    results_gen = model.track(
        source=video_path, conf=conf_threshold, classes=list(RELEVANT_CLASSES.keys()),
        persist=True, stream=True, verbose=False, device=device, tracker="bytetrack.yaml",
        vid_stride=frame_stride,
    )

    for result in results_gen:
        frame_idx += 1
        timestamp = frame_idx * frame_stride / fps

        if result.boxes is None or result.boxes.id is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        for box, tid, cid, conf in zip(boxes, track_ids, cls_ids, confs):
            detections.append(Detection(
                frame_idx=frame_idx, timestamp=round(timestamp, 3),
                track_id=int(tid), cls_id=int(cid),
                cls_name=RELEVANT_CLASSES.get(int(cid), str(cid)),
                bbox=tuple(float(v) for v in box), conf=float(conf),
            ))

        n_processed += 1
        if n_processed % 100 == 0:
            logger.info(f"  đã xử lý {n_processed} frame ({timestamp:.1f}s), {len(detections)} detection tích lũy")
            # `detections` PHẢI giữ hết trong RAM tới khi hàm này return
            # (ByteTrack cần thấy toàn bộ trước khi build_events_from_tracks()
            # phân track) -- không có cách "dọn giữa chừng" ở đây như
            # _ReopeningFrameReader, nên chỉ CẢNH BÁO SỚM nếu RAM đã cao,
            # để người dùng biết cần giảm STRIDE/video ngắn hơn cho lần
            # chạy tiếp theo, thay vì chỉ phát hiện khi đã tràn/bị OOM-kill.
            mem_guard.check_and_warn_if_critical(context="run_detection_tracking")
        if max_frames is not None and n_processed >= max_frames:
            logger.info(f"Đạt max_frames={max_frames}, dừng sớm để demo nhanh.")
            break

    logger.info(f"Xong: {n_processed} frame xử lý, {len(detections)} detection, "
                f"{len(set(d.track_id for d in detections))} track riêng biệt.")
    return detections
