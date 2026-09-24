"""
Mở rộng SỐ LOẠI OBJECT nhận diện được, đối chiếu mẫu BTC N001-V001.zip (dùng
model OpenImages, phát hiện thêm cả Tree/Building/Window/Wheel/Tire/Clothing/
Toy/House ngoài Car/Person/Van/Truck/Bus/Motorcycle/Bicycle) -- pipeline này
đã bao phủ đầy đủ nhóm phương tiện/người (xem vehicle_color.py,
vehicle_type_refine.py), phần còn thiếu là nhóm "bối cảnh tĩnh" (cây, nhà,
cửa sổ...) không phải mục tiêu track theo track_id.

TẠI SAO KHÔNG TRACK CÁC CLASS NÀY NHƯ VEHICLE/PERSON:
  Tree/Building/Window/House là vật thể TĨNH (không di chuyển, không có
  "khoảnh khắc xuất hiện/biến mất" như xe/người) -- gắn track_id, t_start/
  t_end cho chúng không có ý nghĩa và làm phình event index vô ích (1 cái cây
  đứng yên suốt video sẽ tạo hàng trăm detection trùng lặp nếu track như xe).
  Wheel/Tire/Clothing/Toy cũng KHÔNG nên tách thành entity riêng -- chúng là
  BỘ PHẬN của 1 track đã có (bánh xe là 1 phần của Event xe, áo quần là 1
  phần của Event người, đã có shirt_color riêng) -- track thêm sẽ trùng lặp
  dữ liệu, không thêm thông tin tìm kiếm mới.

CÁCH LÀM (nhẹ, an toàn, KHÔNG lặp lại vấn đề false-positive-tự-tin-cao đã
gặp ở vehicle_type_refine.py vì đây chỉ là liệt kê BỐI CẢNH tổng quan, không
gán nhãn cho từng track cụ thể -- sai 1 class bối cảnh ít rủi ro hơn nhiều so
với gán nhầm loại xe cho 1 track cụ thể):
  Lấy 1 mẫu nhỏ (mặc định 5) frame RẢI ĐỀU theo thời gian trong video (không
  cần chạy mọi frame -- bối cảnh tĩnh không đổi nhanh), chạy 1 LẦN YOLOE
  open-vocab (tái dùng đúng model yoloe-11s-seg.pt đã có trong
  vehicle_type_refine.py, tránh tải thêm model mới) với danh sách prompt
  riêng cho nhóm bối cảnh, gộp kết quả thành 1 dict tổng hợp
  {class_name: {"count": n lần thấy trong các frame mẫu, "avg_confidence":
  ...}} -- KHÔNG gắn vào bất kỳ Event/track nào, xuất riêng thành 1 file
  "<video_id>_scene_context.json" độc lập.

GIỚI HẠN (nói rõ):
  - Đây là ƯỚC LƯỢNG BỐI CẢNH dựa trên vài frame mẫu, KHÔNG phải xác nhận
    100% có/không có -- 1 cái cây khuất sau xe ở 5 frame mẫu vẫn có thể có
    trong video, chỉ là không lọt vào mẫu.
  - KHÔNG có vị trí/bbox cụ thể lưu lại (chỉ đếm số lần thấy trong mẫu) --
    nếu cần bbox thật của từng object bối cảnh, phải mở rộng thêm, hiện tại
    mục tiêu chỉ là "phủ được nhiều loại object hơn" để so khớp với phạm vi
    class của BTC, không phải để dùng cho query lọc theo vị trí bối cảnh.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional

logger = logging.getLogger("traffic_pipeline.scene_context_classes")

# Nhóm bối cảnh tĩnh -- đối chiếu trực tiếp các class tần suất cao nhất quan
# sát được trong mẫu BTC N001-V001.zip mà pipeline hiện tại CHƯA phủ tới
# (Land vehicle/Vehicle/Car/Person/Motorcycle/Bus/Truck/Bicycle đã có sẵn qua
# YOLOv8, không lặp lại ở đây).
SCENE_CONTEXT_PROMPTS = [
    "tree", "building", "window", "house", "wheel", "tire", "clothing", "toy",
]

_CONTEXT_VI = {
    "tree": "cây xanh", "building": "toà nhà", "window": "cửa sổ", "house": "nhà",
    "wheel": "bánh xe", "tire": "lốp xe", "clothing": "quần áo", "toy": "đồ chơi",
}

MIN_CONTEXT_CONFIDENCE = 0.3
DEFAULT_N_SAMPLE_FRAMES = 5


def sample_frame_indices(n_total_frames: int, n_samples: int = DEFAULT_N_SAMPLE_FRAMES) -> List[int]:
    """Rải đều n_samples frame_idx trong khoảng [0, n_total_frames), tránh
    2 mẫu trùng nhau khi video quá ngắn."""
    if n_total_frames <= 0:
        return []
    n_samples = min(n_samples, n_total_frames)
    step = n_total_frames / n_samples
    return sorted(set(min(n_total_frames - 1, int(i * step)) for i in range(n_samples)))


def detect_scene_context(
    refiner,  # VehicleTypeRefiner đã khởi tạo (vehicle_type_refine.py) -- TÁI DÙNG model, không load thêm
    frame_reader,
    n_total_frames: int,
    n_samples: int = DEFAULT_N_SAMPLE_FRAMES,
) -> Dict[str, dict]:
    """
    Trả về dict {class_name_en: {"class_name_vi": ..., "n_frames_seen": int,
    "n_sample_frames": int, "avg_confidence": float}} -- rỗng nếu refiner
    không khả dụng (model YOLOE không tải được, xem vehicle_type_refine.py's
    GHI CHÚ THỰC NGHIỆM về lý do có thể không tải được).
    """
    if refiner is None or not getattr(refiner, "available", False):
        logger.warning("scene_context_classes: YOLOE không khả dụng, bỏ qua bước phát hiện bối cảnh.")
        return {}

    frame_indices = sample_frame_indices(n_total_frames, n_samples)
    if not frame_indices:
        return {}

    try:
        from ultralytics import YOLOE  # noqa: F401 -- chỉ để xác nhận cùng gói đã import ở refiner
    except Exception:
        return {}

    # Chạy YOLOE trực tiếp trên CẢ FRAME (không crop theo bbox track như
    # vehicle_type_refine.py, vì bối cảnh không gắn với 1 track cụ thể nào).
    # Dùng lại đúng model đã load trong refiner (self.model), chỉ đổi prompt
    # tạm thời rồi khôi phục lại prompt gốc để không ảnh hưởng các lần gọi
    # classify_crop() khác dùng chung refiner này.
    model = refiner.model
    original_prompts = refiner.prompts
    try:
        model.set_classes(SCENE_CONTEXT_PROMPTS, model.get_text_pe(SCENE_CONTEXT_PROMPTS))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"scene_context_classes: lỗi set_classes ({type(exc).__name__}: {exc}), bỏ qua.")
        return {}

    counts: Dict[str, int] = defaultdict(int)
    conf_sums: Dict[str, float] = defaultdict(float)
    n_valid_frames = 0

    try:
        for idx in frame_indices:
            frame = frame_reader(idx)
            if frame is None:
                continue
            n_valid_frames += 1
            try:
                results = model.predict(frame, conf=MIN_CONTEXT_CONFIDENCE, verbose=False)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"scene_context_classes: lỗi predict frame {idx} ({type(exc).__name__}: {exc}).")
                continue
            r = results[0]
            seen_this_frame = set()
            for box in r.boxes:
                conf = float(box.conf[0])
                name = SCENE_CONTEXT_PROMPTS[int(box.cls[0])]
                conf_sums[name] += conf
                if name not in seen_this_frame:
                    counts[name] += 1
                    seen_this_frame.add(name)
    finally:
        # Khôi phục lại prompt gốc (nhóm loại xe chi tiết) để refiner dùng
        # tiếp cho classify_crop() bình thường ở nơi khác trong pipeline.
        try:
            model.set_classes(original_prompts, model.get_text_pe(original_prompts))
        except Exception:  # noqa: BLE001
            pass

    out = {}
    for name, n_frames_seen in counts.items():
        out[name] = {
            "class_name_vi": _CONTEXT_VI.get(name, name),
            "n_frames_seen": n_frames_seen,
            "n_sample_frames": n_valid_frames,
            "avg_confidence": round(conf_sums[name] / max(1, n_frames_seen), 3) if name in conf_sums else None,
        }
    logger.info(f"scene_context_classes: phát hiện {len(out)} loại bối cảnh trong {n_valid_frames} frame mẫu.")
    return out
