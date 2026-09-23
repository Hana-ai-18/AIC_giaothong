"""
Nhận diện màu xe (vehicle color classification) -- 2 tầng:

  1. TẦNG CHÍNH: model TFLite nhẹ (9 lớp màu: black/blue/brown/green/grey/
     red/silver/white/yellow), tải từ HuggingFace Space
     https://huggingface.co/spaces/sujithvamshi/vehicle-color-recognition
     (model.tflite, input 224x224, chuẩn hoá /255.0).

     LƯU Ý QUAN TRỌNG: model này CHƯA được tự kiểm tra chạy thật trên frame
     video giao thông (môi trường sandbox lúc phát triển bị chặn truy cập
     huggingface.co theo chính sách bảo mật, không tải được để test). Trên
     Kaggle (có thể tải HuggingFace bình thường), notebook có 1 bước kiểm
     tra nhanh (giống bước kiểm tra crop OCR) để tự xác nhận model tải và
     chạy đúng TRƯỚC khi xử lý hàng loạt -- xem "Bước kiểm tra màu xe"
     trong notebook.

  2. TẦNG DỰ PHÒNG (tự động, không cần cấu hình): nếu model không tải được
     hoặc lỗi khi chạy (file hỏng, thiếu thư viện, sai định dạng...), TỰ
     ĐỘNG chuyển sang thuật toán màu chủ đạo bằng HSV clustering (không cần
     model, luôn chạy được, độ chính xác thấp hơn model chuyên biệt nhưng
     đủ dùng để lọc thô "xe màu gì" trong nhiều trường hợp phổ biến).

  method trả về trong VehicleColorResult cho biết ĐANG DÙNG TẦNG NÀO, để
  không tạo ảo giác về độ chính xác khi thực ra đang chạy fallback.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("traffic_pipeline.vehicle_color")

_TFLITE_CLASSES = ["black", "blue", "brown", "green", "grey", "red", "silver", "white", "yellow"]

# Dải HSV cho fallback thuật toán -- ánh xạ về cùng 9 tên màu như model, để
# 2 tầng cho ra kết quả có thể so sánh/gộp được với nhau.
_HSV_COLOR_RANGES = [
    # (tên, [(lo_h, lo_s, lo_v), (hi_h, hi_s, hi_v)], ...) -- có thể nhiều dải cho 1 màu (đỏ vòng qua 0/179)
    ("red", [((0, 70, 50), (10, 255, 255)), ((170, 70, 50), (179, 255, 255))]),
    ("yellow", [((15, 70, 80), (35, 255, 255))]),
    ("green", [((36, 40, 40), (85, 255, 255))]),
    ("blue", [((86, 40, 40), (130, 255, 255))]),
    ("brown", [((5, 50, 20), (20, 200, 150))]),
]
# Các màu trung tính (đen/trắng/xám/bạc) không phân biệt được bằng Hue, phải
# xét riêng qua kênh Saturation + Value.


def _dominant_hsv_via_kmeans(hsv_pixels: np.ndarray, k: int = 3,
                              center_dist: Optional[np.ndarray] = None) -> np.ndarray:
    """
    ĐÃ SỬA (phát hiện lỗi khi test thật): lấy trung bình (mean) toàn bộ
    pixel trong bbox cho kết quả SAI vì bbox object-detection luôn lẫn 1
    phần nền (kính xe tối màu, khoảng trống nhựa đường/vỉa hè...) -- trung
    bình bị kéo lệch về màu xám dù thân xe rõ ràng có màu khác (vd. xe màu
    vàng đồng/gold bị nhận nhầm thành "grey" vì kính+bóng tối chiếm nhiều
    pixel hơn phần sơn xe sáng màu).

    Dùng K-MEANS PHÂN CỤM màu (trên toàn bộ pixel, kể cả nền) rồi CHỌN CỤM
    LỚN NHẤT LOẠI TRỪ pixel quá tối (giả định là bóng/kính/gầm xe, không
    phải màu sơn) -- ổn định hơn nhiều so với mean vì cô lập được màu chủ
    đạo thực sự của thân xe.

    ĐÃ SỬA LẦN 2 (phát hiện thực nghiệm sâu hơn, sau khi thử sửa upstream
    bằng lọc tỉ lệ khung hình bbox nhưng KHÔNG giải quyết được): "cụm lớn
    nhất" vẫn có thể là NỀN chứ không phải thân xe, theo CẢ 2 HƯỚNG trái
    ngược nhau -- đã verify trên 2 case thật:
      - Xe con màu TRẮNG (track "145"): cụm đúng (V~227, thân xe/nóc xe)
        chỉ chiếm 37.9% pixel, còn cụm SAI (V~93, kính chắn gió + bóng tối)
        chiếm 54.9% -> "lớn nhất" chọn nhầm cụm TỐI.
      - Xe máy màu ĐEN (track202): cụm đúng (V~71, thân xe) chỉ chiếm
        27.1%, còn cụm SAI (V~178, mặt đường sáng lọt vào bbox) chiếm
        41.2% -> "lớn nhất" chọn nhầm cụm SÁNG.
    Không có ngưỡng độ sáng/kích thước cụm nào sửa được cả 2 hướng cùng
    lúc (đã thử "ưu tiên cụm sáng nhất trong các cụm đủ lớn" -- sửa được
    ca xe trắng nhưng làm HỎNG ca xe máy đen, đổi thành "bạc").

    Phát hiện thêm: dùng vị trí pixel THAY VÌ chỉ màu để chọn cụm -- object
    detection bbox có margin 15% loại bớt viền nhưng nền (nhựa đường, kính
    xe ở góc, khoảng trống) vẫn có xu hướng lọt vào GÓC/RÌA của crop nhiều
    hơn, trong khi thân xe (đối tượng chính giữa bbox) chiếm phần TRUNG TÂM
    nhiều hơn. Đã verify: khoảng cách trung bình đến tâm crop (chuẩn hoá
    0=tâm, ~1.4=góc) của cụm ĐÚNG luôn thấp hơn cụm SAI trên cả 2 case
    (xe trắng: 0.60 vs 0.81; xe máy đen: 0.59 vs 0.88) -- NHẤT QUÁN theo
    2 hướng ngược nhau, khác với các heuristic thuần màu sắc ở trên.
    Cách chọn: nếu có center_dist, chọn cụm có (count / (1 + mean_dist)^4)
    lớn nhất -- số mũ 4 được chọn THỰC NGHIỆM: đã thử power=1 và power=2,
    cả 2 vẫn chọn SAI cả 2 case thật (chênh lệch count giữa cụm đúng/sai
    chỉ ~1.3-1.5x, không đủ để phạt lệch tâm ở mức phạt yếu vượt qua) --
    power=3 sửa được ca xe máy đen nhưng CHƯA sửa được ca xe trắng (vẫn
    chọn nhầm cụm kính+bóng tối), power=4 sửa đúng CẢ 2 case. Nếu không có
    center_dist (gọi không kèm toạ độ), fallback về "cụm lớn nhất" như cũ.

    ĐÃ KIỂM TRA THÊM: chạy so sánh "cụm lớn nhất" so với "power=4" trên 24
    crop thật khác nhau đã thu thập trong quá trình điều tra (nhiều track,
    nhiều tình huống xe/máy/gần/xa) -- CHỈ 4/24 case cho kết quả khác nhau,
    3/4 case khác đó được xác nhận qua ảnh thật là ĐÚNG HƠN với power=4
    (xe trắng track145 x2 lần chọn frame khác nhau đều đúng ra trắng thay
    vì xám; 1 xe bạc/xám khác trước bị nhận nhầm "đen" nay đúng "bạc"), và
    trường hợp còn lại là chính ca xe máy đen mục tiêu (được sửa đúng).
    20/24 case không đổi kết quả (an toàn, không có dấu hiệu thoái lui).

    GIỚI HẠN: mới verify trên 24 crop có sẵn (do thời gian), CHƯA chạy lại
    toàn bộ pipeline trên mọi track trong video để xác nhận tuyệt đối không
    có thoái lui nào -- vẫn có thể còn case khác center-distance không giúp
    được (vd xe chiếm gần hết bbox, không còn nền ở rìa để phân biệt, hoặc
    2 cụm màu đều cách tâm gần bằng nhau). Ngoài ra: màu be/vàng đồng/gold
    (dom_s thấp dưới ngưỡng 45 do ánh sáng camera) vẫn bị xếp nhầm vào
    "grey" vì bảng 9 màu (đen/xanh dương/nâu/xanh lá/xám/đỏ/bạc/trắng/vàng)
    không phân biệt được be/gold khỏi xám ở mức bão hoà thấp -- đây là GIỚI
    HẠN CẤU TRÚC của cách phân loại theo Hue/Saturation/Value, KHÔNG liên
    quan tới việc chọn cụm, và không được sửa bởi thay đổi này (xem
    track184 trong ghi chú điều tra: xe màu be/gold nhận nhầm "grey",
    dom_s=23 dưới ngưỡng 45 nên rơi vào nhánh trung tính bất kể chọn cụm
    nào).
    """
    v = hsv_pixels[:, 2]
    bright_mask = v > 40  # loại pixel quá tối (bóng, kính đen, gầm xe)
    if np.sum(bright_mask) > 20:
        pixels = hsv_pixels[bright_mask]
        dists = center_dist[bright_mask] if center_dist is not None else None
    else:
        pixels = hsv_pixels
        dists = center_dist
    if len(pixels) < k:
        return np.median(pixels, axis=0) if len(pixels) else np.median(hsv_pixels, axis=0)

    # ĐÃ SỬA (phát hiện lỗi khi test thật): dùng RNG không seed +
    # KMEANS_RANDOM_CENTERS khiến kết quả THAY ĐỔI giữa các lần gọi trên
    # CÙNG 1 ảnh (đã verify: Hue nhảy từ ~150 sang ~10 giữa các lần chạy) --
    # không chấp nhận được cho 1 pipeline cần kết quả tái lập được. Cố định
    # seed cho phần lấy mẫu (đủ để ổn định mà vẫn nhanh) + dùng
    # KMEANS_PP_CENTERS (khởi tạo theo thuật toán K-means++, ổn định hơn
    # nhiều so với random centers) thay vì random thuần.
    rng = np.random.RandomState(42)
    sample_size = min(800, len(pixels))
    idx = rng.choice(len(pixels), sample_size, replace=False)
    sample = pixels[idx].astype(np.float32)
    sample_dist = dists[idx] if dists is not None else None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    cv2.setRNGSeed(42)
    _, labels, centers = cv2.kmeans(sample, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()
    counts = np.bincount(labels, minlength=k)

    if sample_dist is not None:
        scores = np.zeros(k)
        for i in range(k):
            cluster_mask = labels == i
            if not np.any(cluster_mask):
                continue
            mean_dist = float(sample_dist[cluster_mask].mean())
            scores[i] = counts[i] / (1.0 + mean_dist) ** 4
        best_cluster = int(np.argmax(scores))
    else:
        best_cluster = int(np.argmax(counts))
    return centers[best_cluster]


def _classify_neutral_or_hue(hsv_pixels: np.ndarray, center_dist: Optional[np.ndarray] = None) -> str:
    dom_h, dom_s, dom_v = _dominant_hsv_via_kmeans(hsv_pixels, center_dist=center_dist)

    # Bão hoà thấp -- xe màu trung tính (đen/trắng/xám/bạc), phân biệt theo độ sáng.
    # ĐÃ THỬ nâng ngưỡng 70->76 để sửa 1 ca xe máy đen cho dom_v=71.4 (rất
    # sát ngưỡng cũ) nhưng ROLLBACK NGAY vì làm 2 ca xe xám/bạc đậm khác
    # (confirmed đúng là xám/bạc qua ảnh thật) bị xếp NHẦM thành "black" --
    # đổi 1 lỗi biên (đen->xám) lấy 2 lỗi khác (xám->đen) là không đáng.
    # Giữ nguyên ngưỡng 70: ca xe máy đen dom_v=71.4 vẫn có thể lệch 1 bậc
    # thành "grey" dù CỤM MÀU đã được chọn đúng (xem _dominant_hsv_via_kmeans)
    # -- đây là nhiễu ở đúng ranh giới rời rạc hoá liên tục thành 4 mức,
    # chấp nhận là giới hạn còn lại thay vì chỉnh ngưỡng theo 1 ảnh.
    if dom_s < 45:
        if dom_v < 70:
            return "black"
        if dom_v < 140:
            return "grey"
        if dom_v < 200:
            return "silver"
        return "white"

    # Bão hoà đủ cao -- có màu sắc rõ, phân loại theo Hue của cụm màu chủ đạo.
    for name, ranges in _HSV_COLOR_RANGES:
        for lo, hi in ranges:
            if lo[0] <= dom_h <= hi[0]:
                return name
    return "grey"


@dataclass
class VehicleColorResult:
    color: str
    confidence: float  # 0.0-1.0; với fallback HSV đây là tỉ lệ pixel đồng thuận, không phải xác suất model thật
    method: str  # "model" | "hsv_fallback"


class VehicleColorClassifier:
    def __init__(self, model_path: Optional[str] = None):
        """
        model_path: đường dẫn tới model.tflite đã tải sẵn (xem
        vehicle_color_model.py::download_color_model hoặc notebook Kaggle).
        Nếu None hoặc tải thất bại, TỰ ĐỘNG dùng fallback HSV -- không raise
        lỗi, vì thiếu model màu xe không nên làm sập cả pipeline.
        """
        self._interpreter = None
        self._input_details = None
        self._output_details = None

        if model_path and os.path.exists(model_path):
            try:
                self._load_tflite(model_path)
                logger.info(f"Đã load model màu xe TFLite: {model_path}")
            except Exception as exc:
                logger.warning(f"Không load được model màu xe ({exc}) -- dùng fallback HSV cho toàn bộ.")
        else:
            logger.warning("Không có model màu xe (model_path=None hoặc file không tồn tại) "
                            "-- dùng fallback HSV cho toàn bộ.")

    def _load_tflite(self, model_path: str):
        try:
            import tflite_runtime.interpreter as tflite
            self._interpreter = tflite.Interpreter(model_path=model_path)
        except ImportError:
            import tensorflow as tf
            self._interpreter = tf.lite.Interpreter(model_path=model_path)
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

    def _classify_with_model(self, crop_bgr: np.ndarray) -> Optional[VehicleColorResult]:
        if self._interpreter is None:
            return None
        try:
            img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            self._interpreter.set_tensor(self._input_details[0]["index"], img)
            self._interpreter.invoke()
            output = self._interpreter.get_tensor(self._output_details[0]["index"])[0]
            idx = int(np.argmax(output))
            conf = float(output[idx])
            label = _TFLITE_CLASSES[idx] if idx < len(_TFLITE_CLASSES) else "unknown"
            return VehicleColorResult(color=label, confidence=conf, method="model")
        except Exception as exc:
            logger.warning(f"Model màu xe lỗi khi chạy ({exc}) -- fallback HSV cho lần này.")
            return None

    def _classify_with_hsv(self, crop_bgr: np.ndarray) -> VehicleColorResult:
        # bỏ viền 15% quanh bbox để giảm lẫn nền đường/vỉa hè vào vùng tính
        # màu -- bbox object detection thường rộng hơn thân xe thật 1 chút.
        h, w = crop_bgr.shape[:2]
        my, mx = int(h * 0.15), int(w * 0.15)
        core = crop_bgr[my:h - my, mx:w - mx] if h > 2 * my and w > 2 * mx else crop_bgr
        if core.size == 0:
            core = crop_bgr
        ch, cw = core.shape[:2]
        hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV).reshape(-1, 3)

        # ĐÃ THÊM (xem docstring _dominant_hsv_via_kmeans): khoảng cách
        # chuẩn hoá từ mỗi pixel tới TÂM crop -- dùng để phạt các cụm màu
        # nằm nhiều ở rìa/góc (thường là nền: nhựa đường, kính xe ở góc)
        # khi chọn cụm "chủ đạo" đại diện màu xe.
        if ch > 1 and cw > 1:
            ys, xs = np.mgrid[0:ch, 0:cw]
            cy, cx = ch / 2.0, cw / 2.0
            center_dist = np.sqrt(((ys - cy) / cy) ** 2 + ((xs - cx) / cx) ** 2).reshape(-1)
        else:
            center_dist = None

        color = _classify_neutral_or_hue(hsv, center_dist=center_dist)

        # confidence xấp xỉ = tỉ lệ pixel (đã loại pixel quá tối, cùng cách
        # lọc như lúc phân cụm) "đồng thuận" với màu được chọn -- không
        # phải xác suất mô hình thật, chỉ để tham khảo độ đồng nhất màu.
        v_all = hsv[:, 2]
        bright = hsv[v_all > 40]
        ref = bright if len(bright) > 20 else hsv
        s_ref, v_ref, h_ref = ref[:, 1], ref[:, 2], ref[:, 0]
        if color in ("black", "grey", "silver", "white"):
            thresholds = {"black": (v_ref < 70), "grey": (v_ref >= 70) & (v_ref < 140),
                          "silver": (v_ref >= 140) & (v_ref < 200), "white": (v_ref >= 200)}
            mask = thresholds.get(color, np.zeros(len(ref), dtype=bool)) & (s_ref < 45)
        else:
            mask = np.zeros(len(ref), dtype=bool)
            for name, ranges in _HSV_COLOR_RANGES:
                if name != color:
                    continue
                for lo, hi in ranges:
                    mask |= (h_ref >= lo[0]) & (h_ref <= hi[0])
        agreement = float(np.mean(mask)) if len(mask) else 0.0
        return VehicleColorResult(color=color, confidence=round(agreement, 3), method="hsv_fallback")

    def classify(self, frame_bgr: np.ndarray, bbox: tuple) -> Optional[VehicleColorResult]:
        """bbox: (x1, y1, x2, y2) trong toạ độ frame gốc. Trả None nếu crop
        rỗng/lỗi (không raise, để không làm sập batch xử lý)."""
        x1, y1, x2, y2 = (int(v) for v in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        result = self._classify_with_model(crop)
        if result is not None:
            return result
        return self._classify_with_hsv(crop)
