"""
Tự động đo RAM thật của tiến trình + hệ thống (qua psutil) để QUYẾT ĐỊNH khi
nào cần dọn bộ nhớ, thay vì đếm số lần cố định (cách cũ: cứ 300 lần đọc lại
đóng/mở VideoCapture, bất kể RAM thực tế đang cao hay thấp -- không thích
nghi được với video ngắn/dài khác nhau trong dataset AIC26, hoặc với các
session Kaggle có cấu hình RAM khác nhau).

SỐ LIỆU RAM KAGGLE THẬT (đã tra cứu, không đoán): notebook Kaggle bật GPU
hiện tại (bản cập nhật gần đây) cấp **29GB RAM CPU + 4 vCPU** -- xem
https://www.kaggle.com/discussions/product-feedback/448251 (bản cũ hơn chỉ
13GB, nên nếu chạy trên session cũ/chưa cập nhật, hàm below tự đo RAM_TOTAL
thật của máy đang chạy chứ KHÔNG hardcode 29GB, vẫn đúng trong cả 2 trường
hợp).

CHIẾN LƯỢC NGƯỠNG: dùng % RAM hệ thống thay vì số MB cố định, vì  hoạt động
đúng bất kể máy có 13GB hay 29GB hay bất kỳ con số nào khác:
  - SOFT_LIMIT_RATIO = 0.70: RSS tiến trình vượt 70% RAM hệ thống -> chủ
    động dọn (đóng/mở lại VideoCapture, gc.collect()) NGAY, không đợi hết
    REOPEN_EVERY lần đọc.
  - HARD_LIMIT_RATIO = 0.85: vượt 85% -> cảnh báo to (log ERROR), vì ở mức
    này rủi ro bị Kaggle OOM-kill hoặc "tried to use more disk space" (khi
    RAM tràn sang swap/disk) là rất cao -- pipeline vẫn cố chạy tiếp (không
    tự sập), nhưng người dùng cần biết để cân nhắc giảm STRIDE/video ngắn
    hơn nếu thấy log này lặp lại nhiều.
"""
from __future__ import annotations

import gc
import logging
from typing import Optional

logger = logging.getLogger("traffic_pipeline.mem_guard")

try:
    import psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False
    logger.warning(
        "Không có psutil -- KHÔNG thể tự động đo RAM thật để quyết định dọn "
        "bộ nhớ theo ngưỡng động. Cài bằng `pip install psutil` (đã có sẵn "
        "trên hầu hết môi trường Kaggle). Pipeline vẫn chạy được, nhưng chỉ "
        "dùng lưới an toàn cố định (REOPEN_EVERY) làm phòng ngừa, không "
        "thích nghi theo RAM thực tế."
    )

SOFT_LIMIT_RATIO = 0.70
HARD_LIMIT_RATIO = 0.85

_last_hard_warn_pct: Optional[float] = None  # tránh spam log ERROR liên tục ở mức HARD


def get_ram_usage_ratio() -> Optional[float]:
    """Trả về RSS tiến trình hiện tại / tổng RAM hệ thống (0.0-1.0+), hoặc
    None nếu không đo được (thiếu psutil). Đo TIẾN TRÌNH hiện tại (không
    phải toàn hệ thống) vì đây là process Python đang giữ buffer decoder/
    danh sách Detection/... -- đúng đối tượng cần theo dõi để quyết định
    dọn RAM của CHÍNH pipeline này, không lẫn với RAM tiến trình khác (vd
    Jupyter kernel driver) đang chạy song song trên cùng máy Kaggle."""
    if not _PSUTIL_OK:
        return None
    try:
        proc = psutil.Process()
        rss = proc.memory_info().rss
        total = psutil.virtual_memory().total
        if total <= 0:
            return None
        return rss / total
    except Exception:
        return None


def should_free_memory(soft_ratio: float = SOFT_LIMIT_RATIO) -> bool:
    """True nếu RAM tiến trình đã vượt ngưỡng an toàn -- gọi hàm này ở các
    điểm có thể dọn được (đóng/mở lại VideoCapture, gc.collect(), xoá cache
    tạm) để quyết định NGAY LÚC ĐÓ có cần dọn hay chưa, thay vì đợi tới mốc
    cố định (số lần đọc/số giây) như trước. Nếu không đo được RAM (thiếu
    psutil), trả về False -- pipeline vẫn chạy tiếp, dựa vào lưới an toàn
    cố định (REOPEN_EVERY) làm phòng ngừa duy nhất."""
    ratio = get_ram_usage_ratio()
    if ratio is None:
        return False
    return ratio >= soft_ratio


def check_and_warn_if_critical(context: str = "") -> None:
    """Log CẢNH BÁO (throttled, không spam) nếu RAM đã vượt ngưỡng nguy
    hiểm (HARD_LIMIT_RATIO) -- không tự dừng pipeline (không phải lúc nào
    RAM cao cũng dẫn tới crash), chỉ để người theo dõi log biết sớm thay vì
    chỉ phát hiện khi Kaggle đã OOM-kill/báo tràn disk."""
    global _last_hard_warn_pct
    ratio = get_ram_usage_ratio()
    if ratio is None or ratio < HARD_LIMIT_RATIO:
        return
    pct = round(ratio * 100, 1)
    # chỉ log lại nếu tăng thêm >=5 điểm % so với lần cảnh báo trước, tránh
    # log ERROR dồn dập mỗi vài giây khi RAM dao động quanh ngưỡng.
    if _last_hard_warn_pct is not None and pct - _last_hard_warn_pct < 5.0:
        return
    _last_hard_warn_pct = pct
    logger.error(
        f"[mem_guard]{(' ' + context) if context else ''} RAM tiến trình đã dùng "
        f"{pct}% tổng RAM hệ thống (ngưỡng nguy hiểm {HARD_LIMIT_RATIO*100:.0f}%) -- "
        f"rủi ro cao bị Kaggle OOM-kill hoặc tràn sang swap/disk. Cân nhắc giảm "
        f"STRIDE (xử lý ít frame hơn) hoặc xử lý video ngắn hơn nếu log này lặp lại."
    )


def free_memory_if_needed(reopen_fn, context: str = "") -> bool:
    """Điểm gọi DUY NHẤT nơi pipeline muốn "có thể dọn RAM ở đây" -- tự
    quyết định có cần gọi reopen_fn() (hàm đóng/mở lại resource nặng, vd
    VideoCapture) hay không dựa trên RAM thực tế đo được. Luôn gọi
    check_and_warn_if_critical() sau đó để không bỏ sót cảnh báo mức nguy
    hiểm dù có dọn hay không. Trả về True nếu đã thực sự dọn."""
    freed = False
    if should_free_memory():
        logger.info(f"[mem_guard]{(' ' + context) if context else ''} RAM tiến trình vượt "
                    f"{SOFT_LIMIT_RATIO*100:.0f}% RAM hệ thống -- chủ động dọn ngay.")
        reopen_fn()
        gc.collect()
        freed = True
    check_and_warn_if_critical(context)
    return freed
