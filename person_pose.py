"""
XÁC ĐỊNH VÙNG ÁO (thân trên) của người đi bộ VÀ người ngồi trên xe máy, dùng
model YOLO-POSE (17 keypoint chuẩn COCO) thay vì đoán tỉ lệ cố định trên bbox
người -- để nhận diện MÀU ÁO chính xác hơn (đưa vùng áo, không phải cả bbox
người lẫn nền/quần/mũ bảo hiểm, vào vehicle_color.py's HSV/K-means classifier).

TẠI SAO KHÔNG DÙNG HEURISTIC TỈ LỆ CỐ ĐỊNH (vd "15%-45% chiều cao bbox"):
người dùng yêu cầu "chỉ đánh phần áo để xem áo màu gì" -- với góc camera
giao thông nhìn từ trên cao/chéo, người đi bộ đứng thẳng và người ngồi xe máy
cúi người có tỉ lệ đầu/thân/chân trong bbox 2D RẤT KHÁC NHAU (người ngồi xe
máy bbox thường chỉ gồm nửa trên cơ thể + xe, người đi bộ bbox gồm đủ đầu tới
chân) -- 1 tỉ lệ cố định sẽ đúng cho trường hợp này nhưng cắt trúng nền/mũ
bảo hiểm cho trường hợp khác. Dùng pose estimation lấy toạ độ VAI + HÔNG thật
sự của từng người, không phụ thuộc tỉ lệ bbox.

ĐÃ KIỂM CHỨNG THỰC NGHIỆM (chạy trên frame thật của N001-V001.mov) trước khi
chọn model:
  - `yolov8n-pose.pt` (bản nano, nhẹ nhất): conf phát hiện người RẤT THẤP
    (0.06-0.14) trên người ở xa/nhỏ trong camera giao thông góc cao -- KHÔNG
    ĐỦ TIN CẬY, sẽ bỏ sót phần lớn người trong khung hình nếu dùng ngưỡng
    conf hợp lý (vd 0.3+).
  - `yolov8s-pose.pt` (bản small): conf phát hiện người 0.6-0.7 trên CÙNG
    frame đó, keypoint vai/hông đạt conf 0.95-0.98 -- ĐỦ TIN CẬY. Đây là
    model được chọn dùng (tải qua ultralytics/GitHub releases, KHÔNG qua
    HuggingFace nên không bị chặn trong môi trường dev bị hạn chế mạng).
  - Đã verify bằng mắt: bbox áo tính từ vai+hông của model này khớp đúng
    vùng áo thật trên nhiều người khác nhau (đứng, ngồi xe máy, các màu áo
    khác nhau) -- xem ảnh so sánh trong quá trình phát triển.

CÁCH TÍNH VÙNG ÁO: lấy 4 keypoint {vai trái, vai phải, hông trái, hông phải}
(index COCO 5,6,11,12) có conf đủ cao (>= MIN_KEYPOINT_CONF), dựng bbox bao
quanh các điểm này, rồi nới thêm biên nhỏ (PAD_RATIO) vì áo thường rộng hơn 1
chút so với đường nối vai-hông. CẦN ÍT NHẤT 2 keypoint đủ tin cậy để tính (vd
người quay lưng chỉ thấy 1 bên vai/hông, hoặc bị che khuất 1 phần) -- nếu
không đủ, trả None (KHÔNG đoán mò bằng bbox người, vì đó lại quay về đúng vấn
đề đang muốn tránh).

GIỚI HẠN (đã quan sát thấy khi test):
  - Người đứng giữa đám đông xe cộ, bị che khuất nhiều bởi xe/người khác,
    keypoint vai/hông có thể có conf thấp -> bbox áo bị bỏ qua (None) cho
    người đó, dù bản thân người đó vẫn được track như 1 Event bình thường
    (chỉ thiếu field shirt_color, không ảnh hưởng track/vị trí).
  - Model được train trên dữ liệu ảnh người ở góc nhìn thông thường (ngang
    tầm mắt), không chuyên biệt cho góc camera giám sát trên cao -- độ chính
    xác keypoint có thể giảm ở các tư thế bất thường (vd người cúi rạp người
    lái xe máy tốc độ cao).
  - Đây là model + bước xử lý THÊM vào pipeline (không dùng chung batch với
    YOLO detect+track vehicle/person chính) -- làm tăng thời gian xử lý mỗi
    frame có người, cần cân nhắc nếu chạy video dài/số lượng lớn.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger("traffic_pipeline.person_pose")

# Index keypoint chuẩn COCO-17 (thứ tự cố định do model YOLO-pose quy định).
_KP_LSHOULDER, _KP_RSHOULDER, _KP_LHIP, _KP_RHIP = 5, 6, 11, 12

MIN_KEYPOINT_CONF = 0.3   # dưới ngưỡng này coi như keypoint không đáng tin
MIN_KEYPOINTS_NEEDED = 2  # cần ít nhất 2/4 điểm (vai/hông) đủ tin cậy để dựng bbox áo
PAD_RATIO = 0.12          # nới thêm biên quanh bbox vai-hông (áo thường rộng hơn khung xương 1 chút)


@dataclass
class ShirtBox:
    bbox: tuple       # (x1, y1, x2, y2) trong toạ độ frame gốc
    n_keypoints: int  # số keypoint (trong 4 điểm vai/hông) đủ tin cậy đã dùng để tính -- 2, 3 hoặc 4
    person_bbox: tuple  # bbox người gốc (từ YOLO-pose), để đối chiếu/debug


class PersonPoseEstimator:
    """Bọc model YOLO-pose (mặc định yolov8s-pose.pt) -- chạy 1 lần trên 1
    frame, trả về danh sách ShirtBox cho MỌI người phát hiện được (kể cả khi
    không đủ keypoint để tính bbox áo, entry đó có bbox=None để caller biết
    "có người nhưng không xác định được vùng áo", KHÔNG lặng lẽ bỏ qua)."""

    def __init__(self, model_name: str = "yolov8s-pose.pt", conf_threshold: float = 0.25):
        self.conf_threshold = conf_threshold
        try:
            from ultralytics import YOLO
            self._model = YOLO(model_name)
            logger.info(f"Đã tải model pose '{model_name}' để xác định vùng áo.")
        except Exception as exc:
            logger.warning(f"Không tải được model pose '{model_name}' ({exc}) -- "
                            f"TẮT tính năng nhận diện màu áo (sẽ trả None cho mọi frame).")
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def estimate(self, frame_bgr: np.ndarray) -> List[Optional[ShirtBox]]:
        """Chạy pose estimation trên 1 frame, trả về list ShirtBox (hoặc None
        cho người không đủ keypoint tin cậy). Trả [] nếu model không khả dụng
        hoặc không phát hiện người nào."""
        if self._model is None:
            return []
        results = self._model.predict(frame_bgr, conf=self.conf_threshold, verbose=False)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
            return []

        h, w = frame_bgr.shape[:2]
        out: List[Optional[ShirtBox]] = []
        for i in range(len(r.boxes)):
            person_bbox = tuple(float(v) for v in r.boxes.xyxy[i].tolist())
            kpts_xy = r.keypoints.xy[i]
            kpts_conf = r.keypoints.conf[i] if r.keypoints.conf is not None else None
            if kpts_conf is None:
                out.append(None)
                continue

            idxs = (_KP_LSHOULDER, _KP_RSHOULDER, _KP_LHIP, _KP_RHIP)
            pts, confs = [], []
            for idx in idxs:
                c = float(kpts_conf[idx].item())
                if c >= MIN_KEYPOINT_CONF:
                    x, y = kpts_xy[idx].tolist()
                    pts.append((x, y))
                    confs.append(c)

            if len(pts) < MIN_KEYPOINTS_NEEDED:
                out.append(None)
                continue

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            pad_x = (x2 - x1) * PAD_RATIO or 5.0
            pad_y = (y2 - y1) * PAD_RATIO or 5.0
            x1, x2 = max(0, x1 - pad_x), min(w, x2 + pad_x)
            y1, y2 = max(0, y1 - pad_y), min(h, y2 + pad_y)
            if x2 <= x1 or y2 <= y1:
                out.append(None)
                continue

            out.append(ShirtBox(bbox=(x1, y1, x2, y2), n_keypoints=len(pts), person_bbox=person_bbox))
        return out


def match_shirt_box_to_bbox(shirt_boxes: List[Optional[ShirtBox]], target_bbox: tuple,
                             iou_thresh: float = 0.3) -> Optional[ShirtBox]:
    """Khớp 1 bbox người/xe máy (từ track chính của pipeline, class "person"
    hoặc người ngồi trên "motorcycle") với ShirtBox gần nhất theo IoU giữa
    person_bbox (từ pose model) và target_bbox (từ detect+track chính) --
    2 model detect ĐỘC LẬP nên bbox không trùng khít tuyệt đối, cần khớp
    bằng độ chồng lấn thay vì so toạ độ chính xác."""
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    best, best_iou = None, iou_thresh
    for sb in shirt_boxes:
        if sb is None:
            continue
        score = iou(sb.person_bbox, target_bbox)
        if score > best_iou:
            best, best_iou = sb, score
    return best
