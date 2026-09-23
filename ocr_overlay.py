"""
OCR vùng overlay cố định (tên giao lộ + ngày giờ) -- ĐÃ TEST THẬT trên frame
của video N001-V001.mov, OCR ra đúng 100% với psm=7 (single text line).

Chỉ cần chạy 1 lần MỖI GIÂY (không phải mỗi frame) vì đồng hồ chỉ đổi theo
giây -- tiết kiệm rất nhiều so với OCR toàn bộ frame.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytesseract

from config import CROP_LOCATION, CROP_DATETIME


@dataclass
class OverlayInfo:
    location: Optional[str]
    date_str: Optional[str]
    time_str: Optional[str]
    raw_datetime: Optional[str]


_DATETIME_RE = re.compile(
    r"(\d{1,2}[.\-/][A-Za-z]{3}[.\-/ ]?\d{4})\s+(\d{1,2}:\d{2}:\d{2})"
)


def _crop(frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    l, t, r, b = box
    return frame_bgr[t:b, l:r]


def ocr_overlay(frame_bgr: np.ndarray) -> OverlayInfo:
    """OCR 2 vùng overlay cố định trên 1 frame BGR (OpenCV). Trả về
    OverlayInfo -- các field None nếu OCR không đọc được (không raise lỗi,
    vì OCR có thể trượt lác đác 1-2 frame, không nên làm sập cả pipeline)."""
    loc_crop = _crop(frame_bgr, CROP_LOCATION)
    dt_crop = _crop(frame_bgr, CROP_DATETIME)

    loc_text = pytesseract.image_to_string(loc_crop, config="--psm 7").strip()
    dt_text = pytesseract.image_to_string(dt_crop, config="--psm 7").strip()

    location = loc_text if loc_text else None

    date_str, time_str = None, None
    m = _DATETIME_RE.search(dt_text)
    if m:
        date_str, time_str = m.group(1), m.group(2)

    return OverlayInfo(
        location=location, date_str=date_str, time_str=time_str,
        raw_datetime=dt_text if dt_text else None,
    )
