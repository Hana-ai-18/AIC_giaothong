"""
Nhận diện biển số xe Việt Nam (ALPR) -- detect vùng biển số + OCR ký tự.

QUAN TRỌNG (đã test thật trên video N001-V001.mov, xem ghi chú cuối file):
biển số trong camera giám sát giao thông toàn cảnh (wide-angle, không zoom)
thường chỉ chiếm 25-60px chiều rộng trong khung 1920x1080 -- QUÁ NHỎ để OCR
đọc chính xác ký tự trong đa số trường hợp. Model này chỉ đọc được tương đối
tốt với xe ở RẤT GẦN camera (trong khoảng 5-10m đầu). Vì vậy module này:
  1. Luôn lưu lại bbox + ảnh crop biển số (dù OCR đọc được hay không) để
     người dùng có thể xem bằng mắt khi cần xác minh thủ công.
  2. Gắn plate_conf (độ tin cậy detect vùng biển số) + ocr_low_confidence
     (cờ cảnh báo khi text đọc được có khả năng sai cao) vào kết quả, KHÔNG
     coi text OCR là chính xác tuyệt đối.
  3. Dùng để lọc/gợi ý trong search (vd. "biển số có chứa 43F7"), không dùng
     để khẳng định chắc chắn biển số chính xác 100%.

Model: torch.hub YOLOv5 (bắt buộc dùng API cũ, KHÔNG tương thích package
`ultralytics` mới -- xem hubconf.py của yolov5 repo) với 2 model .pt:
  - LP_detector_nano_61.pt : detect vùng biển số trong ảnh gốc
  - LP_ocr_nano_62.pt      : detect + phân loại từng ký tự trong crop biển số
Cả 2 model + code helper lấy từ:
  https://github.com/trungdinh22/License-Plate-Recognition
"""
from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger("traffic_pipeline.plate_ocr")

# ĐÃ TEST: PyTorch >=2.6 đổi mặc định weights_only=True làm torch.hub.load
# yolov5 custom model bị lỗi UnpicklingError. Vá lại để weights_only=False
# (an toàn vì ta tự kiểm soát nguồn file .pt, tải từ repo LP_REPO_DIR).
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


@dataclass
class PlateResult:
    bbox: tuple  # (x1, y1, x2, y2) trong ảnh gốc
    plate_conf: float  # độ tin cậy detect VÙNG biển số (không phải độ tin cậy text)
    plate_text: Optional[str]  # None nếu không đọc được ký tự nào
    ocr_low_confidence: bool  # True nếu text đọc được nên nghi ngờ (xem lý do trong _read_plate_text)
    crop_path: Optional[str] = None  # đường dẫn ảnh crop đã lưu, để xem bằng mắt khi cần


class PlateRecognizer:
    def __init__(
        self,
        lp_repo_dir: str,
        detector_conf: float = 0.35,
        ocr_conf: float = 0.25,
        device: str = "cpu",
    ):
        """
        lp_repo_dir: đường dẫn tới thư mục clone của
            https://github.com/trungdinh22/License-Plate-Recognition
            (phải có sẵn: <lp_repo_dir>/yolov5 (clone repo ultralytics/yolov5
            v7.0) và <lp_repo_dir>/model/LP_detector_nano_61.pt +
            LP_ocr_nano_62.pt). Xem notebook Kaggle: đã tự động clone + tải
            đúng 2 repo này trước khi import module này.
        """
        self.lp_repo_dir = lp_repo_dir
        yolov5_dir = os.path.join(lp_repo_dir, "yolov5")
        if lp_repo_dir not in sys.path:
            sys.path.insert(0, lp_repo_dir)

        self.detector = torch.hub.load(
            yolov5_dir, "custom",
            path=os.path.join(lp_repo_dir, "model", "LP_detector_nano_61.pt"),
            force_reload=False, source="local", verbose=False, device=device,
        )
        self.detector.conf = detector_conf

        self.ocr_model = torch.hub.load(
            yolov5_dir, "custom",
            path=os.path.join(lp_repo_dir, "model", "LP_ocr_nano_62.pt"),
            force_reload=False, source="local", verbose=False, device=device,
        )
        self.ocr_model.conf = ocr_conf

        from function import utils_rotate  # module trong lp_repo_dir
        self._utils_rotate = utils_rotate

        logger.info("Đã load xong model ALPR (detector + OCR nano).")

    def _read_plate_text(self, crop_bgr: np.ndarray) -> tuple:
        """
        Thử đọc text trên 1 crop biển số, với NHIỀU scale + góc xoay khác
        nhau (đã test thật: scale=3 với size= truyền ĐÚNG kích thước ảnh đã
        upscale -- KHÔNG để mặc định size=640 -- mới cho kết quả ổn định
        nhất; scale quá lớn/nhỏ làm model bỏ sót hoặc nhiễu ký tự).

        Trả về (text, ocr_low_confidence). ocr_low_confidence=True khi số
        ký tự tìm được nằm ở biên ngưỡng chấp nhận (7 hoặc 10, dễ nhầm) HOẶC
        crop gốc nhỏ hơn 45px chiều rộng (biển số càng nhỏ, khả năng model
        ghép nhầm ký tự càng cao -- xem ghi chú đầu file, đã verify thực tế
        trên video giao thông thật)."""
        h0, w0 = crop_bgr.shape[:2]
        small_plate = w0 < 45

        for scale in (3, 4, 2):
            up = cv2.resize(crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            for cc in range(2):
                for ct in range(2):
                    rotated = self._utils_rotate.deskew(up, cc, ct)
                    text, n_chars = self._detect_chars(rotated)
                    if text is not None:
                        low_conf = small_plate or n_chars in (7, 10)
                        return text, low_conf
        return None, True

    def _detect_chars(self, im: np.ndarray):
        h, w = im.shape[:2]
        results = self.ocr_model(im, size=max(h, w) + 32)
        bb_list = results.pandas().xyxy[0].values.tolist()
        n = len(bb_list)
        if n < 7 or n > 10:
            return None, n

        center_list = []
        y_sum = 0.0
        for bb in bb_list:
            xc, yc = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
            y_sum += yc
            center_list.append([xc, yc, bb[-1]])

        l_point = min(center_list, key=lambda c: c[0])
        r_point = max(center_list, key=lambda c: c[0])
        lp_type = "1"
        if l_point[0] != r_point[0]:
            a = (l_point[1] - r_point[1]) / (l_point[0] - r_point[0])
            b = l_point[1] - a * l_point[0]
            for c in center_list:
                if not math.isclose(a * c[0] + b, c[1], abs_tol=3):
                    lp_type = "2"
                    break

        y_mean = y_sum / n
        if lp_type == "2":
            line1 = sorted((c for c in center_list if c[1] <= y_mean), key=lambda c: c[0])
            line2 = sorted((c for c in center_list if c[1] > y_mean), key=lambda c: c[0])
            text = "".join(str(c[2]) for c in line1) + "-" + "".join(str(c[2]) for c in line2)
        else:
            text = "".join(str(c[2]) for c in sorted(center_list, key=lambda c: c[0]))
        return text, n

    @staticmethod
    def vote_plate_texts(texts: List[str]) -> tuple:
        """
        Kết hợp nhiều kết quả OCR (từ nhiều frame khác nhau của CÙNG 1 xe)
        bằng CHAR-LEVEL MAJORITY VOTING theo từng vị trí ký tự, thay vì so
        khớp cả chuỗi (gần như không bao giờ khớp 100% giữa các frame --
        xem ghi chú thực nghiệm cuối file: 5 frame của cùng 1 xe cho ra 5
        kết quả OCR khác nhau như "48-42477", "494-4477F", "44F-4457",
        "44-44777" -- nhưng CÙNG một số ký tự ở gần đúng vị trí lặp lại
        nhiều lần, vd. ký tự '4' và '7' xuất hiện nhất quán).

        Chỉ vote trên các text có CÙNG chiều dài phổ biến nhất (loại text
        có độ dài lệch nhiều, khả năng cao là OCR bỏ sót/thừa ký tự làm
        lệch toàn bộ vị trí).

        Trả về (voted_text, agreement_ratio) -- agreement_ratio = tỉ lệ
        trung bình các vị trí có đa số phiếu đồng thuận (0.0-1.0), dùng làm
        chỉ báo độ tin cậy kết hợp: càng gần 1.0 càng nhiều frame đồng ý
        với ký tự tại vị trí đó.
        """
        texts = [t for t in texts if t]
        if not texts:
            return None, 0.0
        if len(texts) == 1:
            return texts[0], 0.0  # chỉ 1 mẫu -- không đủ để vote, giữ nguyên nhưng đánh dấu tin cậy thấp

        from collections import Counter
        len_counts = Counter(len(t) for t in texts)
        common_len, _ = len_counts.most_common(1)[0]
        candidates = [t for t in texts if len(t) == common_len]
        if len(candidates) < 2:
            return max(texts, key=len), 0.0

        voted_chars = []
        agreements = []
        for pos in range(common_len):
            col = Counter(t[pos] for t in candidates)
            best_char, best_count = col.most_common(1)[0]
            voted_chars.append(best_char)
            agreements.append(best_count / len(candidates))

        voted_text = "".join(voted_chars)
        agreement_ratio = sum(agreements) / len(agreements)
        return voted_text, agreement_ratio

    def recognize(
        self,
        frame_bgr: np.ndarray,
        crop_out_dir: Optional[str] = None,
        crop_name_prefix: str = "plate",
    ) -> List[PlateResult]:
        """Detect mọi vùng biển số trong 1 frame + thử đọc text từng vùng.
        Luôn trả kết quả cho MỌI vùng detect được (kể cả khi không đọc được
        text) -- để còn crop_path cho người dùng xem lại bằng mắt."""
        det = self.detector(frame_bgr, size=640)
        boxes = det.pandas().xyxy[0].values.tolist()

        results = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            plate_conf = float(box[4])
            x1, y1 = max(0, x1 - 3), max(0, y1 - 3)
            crop = frame_bgr[y1:y2 + 3, x1:x2 + 3]
            if crop.size == 0:
                continue

            text, low_conf = self._read_plate_text(crop)

            crop_path = None
            if crop_out_dir is not None:
                os.makedirs(crop_out_dir, exist_ok=True)
                crop_path = os.path.join(crop_out_dir, f"{crop_name_prefix}_{i}.jpg")
                cv2.imwrite(crop_path, crop)

            results.append(PlateResult(
                bbox=(x1, y1, x2, y2), plate_conf=round(plate_conf, 3),
                plate_text=text, ocr_low_confidence=low_conf, crop_path=crop_path,
            ))
        return results


"""
GHI CHÚ KẾT QUẢ TEST THẬT (2026-09-23, trên N001-V001.mov + N071-V001.mov):
- Detector (vùng biển số) hoạt động tốt: tìm đúng vị trí biển số trên xe ở
  khoảng cách gần-trung bình, dù đôi khi có false positive (vd. nhầm biển
  quảng cáo LED thành biển số -- xem plate_conf để lọc, nên ưu tiên
  plate_conf > 0.5).
- OCR ký tự: THẤT BẠI hoàn toàn (0/5-9 vùng) trên frame có xe ở xa/trung
  bình (biển số <45px chiều rộng) dù đã thử upscale nhiều mức -- đây là
  giới hạn thông tin gốc của ảnh (camera toàn cảnh giao lộ, không phải
  camera zoom biển số), KHÔNG khắc phục được bằng thuật toán.
- Với xe ở RẤT gần (biển số ~55-60px), sau khi sửa lỗi truyền sai tham số
  `size=` cho model OCR (mặc định 640 làm distort ảnh đã upscale), OCR bắt
  đầu ra kết quả có cấu trúc hợp lệ (2 dòng, 8 ký tự) nhưng NỘI DUNG ký tự
  vẫn có sai số đáng kể so với biển số thật đọc bằng mắt thường.
=> KẾT LUẬN: dùng field plate_text như một tín hiệu GỢI Ý (kèm
   ocr_low_confidence + crop_path để xem lại), KHÔNG dùng để tìm kiếm
   exact-match biển số một cách tin tưởng tuyệt đối. Nên đưa vào search như
   1 arm yếu trong RRF (trọng số thấp) hoặc chỉ dùng field crop_path để cho
   người dùng tự xem lại khi hệ thống lọc ra ứng viên bằng các tín hiệu
   khác (thời gian, vị trí, loại xe) trước.
"""
