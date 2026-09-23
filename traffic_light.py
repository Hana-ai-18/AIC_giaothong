"""
Phân loại màu đèn giao thông bằng ngưỡng HSV đơn giản trên vùng crop cố
định -- KHÔNG cần model, vì camera tĩnh nên vị trí đèn không đổi.

Cách hoạt động: đếm số pixel "sáng" (giá trị cao ở kênh V) rơi vào dải màu
đỏ/vàng/xanh trong vùng crop, chọn màu có nhiều pixel sáng nhất. Đèn giao
thông LED thường rất sáng (V cao) so với nền xung quanh nên ngưỡng đơn giản
này đủ tin cậy.
"""
from __future__ import annotations

import cv2
import numpy as np

from config import CROP_TRAFFIC_LIGHT

# Ngưỡng HSV (OpenCV: H 0-179, S 0-255, V 0-255). Đỏ nằm ở 2 đầu dải Hue nên
# cần 2 khoảng.
_HSV_RANGES = {
    "red": [((0, 100, 150), (10, 255, 255)), ((170, 100, 150), (179, 255, 255))],
    "yellow": [((15, 100, 150), (35, 255, 255))],
    "green": [((40, 60, 120), (90, 255, 255))],
}

_MIN_LIT_PIXELS = 15  # dưới ngưỡng này coi là "không xác định" (đèn tắt/bị che khuất)


def _crop(frame_bgr: np.ndarray) -> np.ndarray:
    l, t, r, b = CROP_TRAFFIC_LIGHT
    return frame_bgr[t:b, l:r]


def detect_light_color(frame_bgr: np.ndarray) -> str:
    """Trả về 'red' | 'yellow' | 'green' | 'unknown'."""
    crop = _crop(frame_bgr)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    counts = {}
    for color, ranges in _HSV_RANGES.items():
        total = 0
        for lo, hi in ranges:
            mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
            total += int(mask.sum() / 255)
        counts[color] = total

    best_color = max(counts, key=counts.get)
    if counts[best_color] < _MIN_LIT_PIXELS:
        return "unknown"
    return best_color
