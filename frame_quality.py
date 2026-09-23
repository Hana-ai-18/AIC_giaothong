"""
Đánh giá "chất lượng" 1 detection (frame + bbox) để chọn representative_frame
THÔNG MINH hơn thay vì cứng nhắc lấy frame đầu/giữa/cuối track.

Tiêu chí (kết hợp có trọng số, đã cân nhắc theo đặc thù camera giao thông
tĩnh -- xe càng gần & rõ nét thì frame càng hữu ích cho search bằng mắt
hoặc bằng embedding sau này):
  1. Kích thước bbox (xe càng gần camera, bbox càng lớn, càng nhiều chi
     tiết -- quan trọng nhất vì ảnh hưởng trực tiếp đến việc đọc được biển
     số/nhận diện chi tiết nhỏ).
  2. Độ nét (variance of Laplacian trên vùng bbox) -- loại các frame bị mờ
     do chuyển động nhanh/lấy nét kém, vốn rất phổ biến với xe đang chạy.
  3. Độ tin cậy detect (conf) -- model tự tin hơn thường tương ứng với ảnh
     rõ ràng hơn.
  4. Vị trí trong khung hình -- ưu tiên object không bị cắt ở rìa ảnh
     (object dính biên thường bị che 1 phần, ít hữu ích để xem/OCR).

Đây là hàm THUẦN (không cần model AI thêm) nên rất rẻ để chạy trên toàn bộ
detection của track, không ảnh hưởng đáng kể thời gian pipeline.
"""
from __future__ import annotations

import cv2
import numpy as np


def sharpness_score(frame_bgr: np.ndarray, bbox: tuple) -> float:
    """Variance of Laplacian trong vùng bbox -- giá trị càng cao càng nét.
    Trả 0.0 nếu vùng crop rỗng/lỗi (không raise, vì 1 frame lỗi không nên
    làm sập cả pipeline)."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def edge_penalty(bbox: tuple, frame_width: int, frame_height: int, margin: int = 15) -> float:
    """Phạt nếu bbox dính sát biên khung hình (object bị cắt, khả năng bị
    che 1 phần) -- trả về hệ số nhân trong khoảng (0.5, 1.0], 1.0 nghĩa là
    hoàn toàn không bị cắt."""
    x1, y1, x2, y2 = bbox
    near_edge = (x1 <= margin or y1 <= margin
                 or x2 >= frame_width - margin or y2 >= frame_height - margin)
    return 0.6 if near_edge else 1.0


def quality_score(
    frame_bgr,
    bbox: tuple,
    det_conf: float,
    frame_width: int,
    frame_height: int,
    sharpness: float = None,
) -> float:
    """
    Điểm tổng hợp (càng cao càng tốt), KHÔNG chuẩn hoá cố định về [0,1] vì
    chỉ dùng để SO SÁNH tương đối giữa các detection trong CÙNG 1 track
    (không so sánh giữa các track khác nhau).

    sharpness: truyền sẵn nếu đã tính trước (tránh tính Laplacian lại nhiều
    lần khi frame_bgr là None -- xem cách dùng trong event_timeline.py, nơi
    ta KHÔNG đọc lại từng frame gốc cho mọi detection vì quá tốn kém, chỉ
    tính full quality_score khi có sẵn frame_bgr thật).
    """
    x1, y1, x2, y2 = bbox
    area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    if sharpness is None:
        sharpness = sharpness_score(frame_bgr, bbox) if frame_bgr is not None else 0.0
    penalty = edge_penalty(bbox, frame_width, frame_height)

    # ĐÃ TEST: Laplacian variance TỰ NHIÊN tăng vọt trên bbox nhỏ (cạnh nét
    # chiếm tỉ trọng cao hơn trong ít pixel), nên sharpness thô không so
    # sánh được trực tiếp giữa bbox to/nhỏ khác nhau -- phải CHUẨN HOÁ theo
    # diện tích trước khi cộng vào điểm tổng, nếu không bbox nhỏ luôn thắng
    # dù ít chi tiết hữu ích hơn hẳn cho search/OCR.
    sharpness_density = sharpness / max(area, 1.0)

    size_term = np.log1p(area)              # ưu tiên chính: bbox càng lớn càng nhiều chi tiết
    sharp_term = np.log1p(sharpness_density * 1000)  # phạt ảnh mờ, không để lấn át size_term
    conf_term = det_conf * 5
    return (size_term * 1.0 + sharp_term * 0.5 + conf_term * 0.3) * penalty
