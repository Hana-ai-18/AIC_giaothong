"""
mem_guard.py -- theo dõi RAM THẬT của tiến trình (qua psutil) và cung cấp 2
hàm được main_pipeline.py / detect_track.py gọi để tự động dọn bộ nhớ /
cảnh báo sớm khi RAM lên cao, thay vì chỉ đoán 1 con số đếm cố định
(REOPEN_EVERY / mỗi 100 frame) không thích nghi theo video dài/ngắn hay RAM
thật khác nhau giữa các session (13GB / 29GB Kaggle...).

Nếu KHÔNG có psutil (import lỗi), 2 hàm dưới đây trở thành no-op an toàn --
code gọi chúng (main_pipeline.py) đã có sẵn "lưới an toàn cố định"
(REOPEN_EVERY / REOPEN_EVERY_SEC) làm phương án dự phòng, nên pipeline vẫn
chạy được, chỉ là không tự thích nghi theo RAM thật nữa.

API:
    free_memory_if_needed(reopen_fn, context="") -> bool
        Gọi khi ở 1 điểm AN TOÀN để "dọn" (ví dụ trước khi đọc frame kế
        tiếp). Nếu RAM đang vượt ngưỡng CRITICAL, gọi reopen_fn() (hàm
        không đối số, ví dụ đóng/mở lại VideoCapture) + gc.collect(), rồi
        trả True. Nếu RAM còn ổn, không làm gì và trả False. Có cooldown
        tối thiểu giữa 2 lần dọn liên tiếp (dù RAM vẫn cao) để tránh
        đóng/mở VideoCapture liên tục gây chậm thêm thay vì tiết kiệm.

    check_and_warn_if_critical(context="") -> None
        Gọi ở những nơi KHÔNG có cách dọn giữa chừng (ví dụ giữa vòng lặp
        detect+track, dữ liệu detections phải giữ nguyên tới cuối). CHỈ
        log cảnh báo (throttle theo thời gian), không tự hành động.
"""
from __future__ import annotations

import gc
import logging
import time

logger = logging.getLogger("traffic_pipeline.mem_guard")

# Ngưỡng % RAM HỆ THỐNG (không phải riêng RSS tiến trình) mà từ đó coi là
# "cao" (free_memory_if_needed sẽ chủ động dọn) / "nguy cấp" (chỉ cảnh báo,
# dùng ở những chỗ không dọn được giữa chừng). Whitepaper trong
# main_pipeline.py nói "vượt 70%" -- dùng chung 1 ngưỡng CRITICAL, thêm 1
# ngưỡng WARN thấp hơn 1 chút để cảnh báo sớm hơn 1 nhịp.
WARN_PCT = 65.0
CRITICAL_PCT = 70.0

# Không đóng/mở lại quá dày (dù RAM vẫn cao) -- tốn thời gian I/O mở lại
# VideoCapture, có thể làm chậm hơn là lợi nếu gọi liên tục.
MIN_SECONDS_BETWEEN_REOPEN = 5.0
# Cảnh báo (log) cũng throttle theo thời gian để không spam log mỗi 100 frame.
MIN_SECONDS_BETWEEN_WARN = 15.0

try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except Exception:  # noqa: BLE001
    psutil = None  # type: ignore
    _HAS_PSUTIL = False
    logger.warning(
        "mem_guard: KHÔNG có psutil -- tự động theo RAM thật bị TẮT, chỉ còn "
        "lưới an toàn cố định (REOPEN_EVERY/REOPEN_EVERY_SEC) trong "
        "main_pipeline.py. Cài `pip install psutil` để bật lại tính năng này."
    )

_last_reopen_ts = 0.0
_last_warn_ts = 0.0


def _ram_percent() -> float | None:
    """% RAM hệ thống đang dùng (0-100), None nếu không đo được."""
    if not _HAS_PSUTIL:
        return None
    try:
        return float(psutil.virtual_memory().percent)
    except Exception:  # noqa: BLE001
        return None


def free_memory_if_needed(reopen_fn, context: str = "") -> bool:
    """Nếu RAM hệ thống đang >= CRITICAL_PCT VÀ đã đủ lâu kể từ lần dọn
    trước, gọi reopen_fn() + gc.collect() rồi trả True. Ngược lại trả False
    (không làm gì, kể cả khi không đo được RAM -- caller sẽ tự rơi về lưới
    an toàn cố định của nó)."""
    global _last_reopen_ts

    pct = _ram_percent()
    if pct is None or pct < CRITICAL_PCT:
        return False

    now = time.time()
    if now - _last_reopen_ts < MIN_SECONDS_BETWEEN_REOPEN:
        return False

    logger.warning(
        f"mem_guard[{context}]: RAM hệ thống {pct:.1f}% >= {CRITICAL_PCT:.0f}% "
        f"-- chủ động dọn (đóng/mở lại + gc.collect())."
    )
    try:
        reopen_fn()
    finally:
        gc.collect()
        _last_reopen_ts = now
    return True


def check_and_warn_if_critical(context: str = "") -> None:
    """Chỉ cảnh báo (log), KHÔNG tự hành động -- dùng ở chỗ không có cách
    dọn an toàn giữa chừng (ví dụ trong lúc gom detections cho ByteTrack)."""
    global _last_warn_ts

    pct = _ram_percent()
    if pct is None or pct < WARN_PCT:
        return

    now = time.time()
    if now - _last_warn_ts < MIN_SECONDS_BETWEEN_WARN:
        return
    _last_warn_ts = now

    level = logger.error if pct >= CRITICAL_PCT else logger.warning
    level(
        f"mem_guard[{context}]: RAM hệ thống {pct:.1f}% "
        f"({'>= CRITICAL ' + str(CRITICAL_PCT) if pct >= CRITICAL_PCT else '>= WARN ' + str(WARN_PCT)}%) "
        f"-- cân nhắc giảm STRIDE hoặc xử lý video ngắn hơn nếu tiếp tục tăng."
    )
