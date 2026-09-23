"""
Detection + tracking bằng YOLOv8 + ByteTrack (built-in trong ultralytics --
model.track() tự động dùng ByteTrack, gắn track_id ổn định xuyên suốt
nhiều frame cho mỗi đối tượng).

Chạy trực tiếp trên video, KHÔNG cần decode thủ công -- ultralytics tự đọc
video qua OpenCV nội bộ, xử lý từng frame, trả generator kết quả.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from config import RELEVANT_CLASSES, VIDEO_PATH

logger = logging.getLogger("traffic_pipeline.detect_track")


@dataclass
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
    model_name: str = "yolov8n.pt",
    conf_threshold: float = 0.35,
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
        if max_frames is not None and n_processed >= max_frames:
            logger.info(f"Đạt max_frames={max_frames}, dừng sớm để demo nhanh.")
            break

    logger.info(f"Xong: {n_processed} frame xử lý, {len(detections)} detection, "
                f"{len(set(d.track_id for d in detections))} track riêng biệt.")
    return detections
