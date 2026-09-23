"""
Nhận diện LOẠI PHƯƠNG TIỆN chi tiết hơn 5 lớp COCO gốc (car/truck/bus/
motorcycle/bicycle) bằng model OPEN-VOCABULARY (YOLOE, gói `ultralytics`) --
KHÔNG cần train lại, chỉ cần đặt tên lớp tuỳ ý bằng text prompt tiếng Anh
(model open-vocab hiện tại hiểu tiếng Anh tốt hơn tiếng Việt, xem GHI CHÚ).

CÁCH LÀM (đã kiểm chứng thực tế, xem ghi chú thực nghiệm bên dưới):
  Không chạy YOLOE trên MỌI frame của video (quá chậm: ~0.28s/frame trên CPU
  với ~10 lớp tuỳ chỉnh -- video dài hàng nghìn frame sẽ mất hàng chục phút
  chỉ riêng bước này). Thay vào đó, TÁI SỬ DỤNG cơ chế đã có: mỗi Event/track
  đã được build_events_from_tracks() (event_timeline.py) chọn sẵn 1-3 frame
  đại diện CHẤT LƯỢNG CAO NHẤT (frame_quality.py). Ta chỉ cần chạy YOLOE 1
  LẦN trên frame đại diện TỐT NHẤT (đầu tiên trong danh sách) của MỖI track
  phương tiện đã có -- số lần chạy = số track phương tiện, không phải số
  frame video, giảm chi phí hàng trăm lần.

  Với mỗi track, crop đúng vùng bbox gốc (đã lưu trong trajectory) trên frame
  đại diện, chạy YOLOE.predict() trên crop đó (hoặc cả frame rồi chỉ giữ box
  overlap cao nhất với bbox gốc), lấy nhãn open-vocab có conf cao nhất TRONG
  SỐ CÁC NHÃN "hiếm" đã định nghĩa (VEHICLE_TYPE_PROMPTS) làm `vehicle_type`
  bổ sung -- KHÔNG ghi đè `cls_name` gốc (vẫn giữ nguyên car/truck/... từ
  YOLOv8+ByteTrack để không phá vỡ tương thích ngược với mọi module khác đã
  dùng cls_name), chỉ thêm field mới `vehicle_type_detail` (None nếu không
  phát hiện được loại nào đặc biệt/độ tin cậy quá thấp).

GHI CHÚ THỰC NGHIỆM -- ĐÃ TÌM RA VẤN ĐỀ NGHIÊM TRỌNG, MẶC ĐỊNH TẮT TÍNH NĂNG
NÀY (xem config.ENABLE_VEHICLE_TYPE_DETAIL = False):
  Đã kiểm chứng bằng frame thật từ N001-V001.mov (sandbox này KHÔNG bị chặn
  GitHub nên tải được model+text-encoder của YOLOE, khác với HuggingFace bị
  chặn -- xem vehicle_color.py). `ultralytics.YOLOE` (`yoloe-11s-seg.pt`)
  chạy được, ~0.28s/frame CPU, và PHÂN BIỆT ĐÚNG được vài trường hợp có hình
  dạng rõ ràng khác biệt (vd "van" tách khỏi "car" đúng cho 1 xe MPV/Innova
  thực tế trong video test).

  NHƯNG khi thử lại TRÊN CHÍNH DỮ LIỆU THẬT của pipeline này (chạy hàng loạt
  qua mọi track phương tiện đã track được), phát hiện: một tỉ lệ rất lớn xe
  BÌNH THƯỜNG (ô tô con, xe máy chở hàng thông thường) bị gán nhãn "hiếm" SAI
  với độ tin cậy CAO -- vd 1 xe con màu bạc/trắng bình thường bị gán "police
  car" (conf 0.895), 1 xe máy chở thùng hàng phía sau (loại rất phổ biến ở
  VN, giao hàng) bị gán "cement mixer truck" (conf 0.941). Đã kiểm tra kỹ:
    - Đây KHÔNG phải lỗi tích hợp (crop/toạ độ đúng, đã xem trực tiếp ảnh
      crop bằng mắt) -- KIỂM CHỨNG BẰNG HÌNH THẬT, không suy đoán.
    - Đây KHÔNG phải hiện tượng "vocab đông làm rối model" -- test riêng
      2-lớp (vd chỉ "car" vs "police car") vẫn cho "police car" thắng dù ảnh
      rõ ràng là xe thường không có gì giống police car.
    - Chỉ 1 vài prompt bị lỗi kiểu này ("police car", "cement mixer truck"
      trong thử nghiệm) trong khi phần lớn prompt khác (ambulance, garbage
      truck, tuk tuk, cargo tricycle, tanker truck...) hoạt động ĐÚNG (thua
      "car"/"motorcycle" đúng khi ảnh không phải loại đó) -- tức đây là vấn
      đề HIỆU CHỈNH ĐIỂM SỐ (calibration) của TỪNG text-embedding cụ thể
      trong model YOLOE bản hiện tại, không phải lỗi cách dùng API hay lỗi
      thiết kế min-confidence-threshold đơn thuần (ngưỡng nào cũng có nguy
      cơ dính false positive tự tin cao kiểu này).
  KẾT LUẬN: open-vocabulary YOLOE, dùng "as-is" không fine-tune, KHÔNG ĐỦ
  TIN CẬY để tự động gán nhãn loại xe hiếm cho pipeline này -- rủi ro tạo ra
  metadata SAI (police car/cement mixer truck giả) còn tệ hơn không có
  thông tin gì, vì người dùng cuối/hệ thống search phía sau sẽ tin vào nhãn
  đó.

  CẬP NHẬT (sau khi loại "police car"/"cement mixer truck" khỏi danh sách và
  tăng ngưỡng lên 0.5): tỉ lệ false positive giảm mạnh (157 -> 23 track bị
  gán/157 track tổng trên video test) nhưng KHÔNG về 0 -- vẫn phát hiện thêm
  các prompt khác cũng bị lỗi tương tự trên dữ liệu này (vd "tanker truck",
  "tow truck", "fire truck" cũng gán sai cho vài xe máy/ô tô thường). Đồng
  thời, MỘT SỐ kết quả VẪN ĐÚNG và hữu ích thật (vd 1 xe van chở hàng thật đã
  được xác nhận bằng mắt qua chữ "NHẬN CHỞ HÀNG" trên xe, gán "van" đúng conf
  0.649). Tức đây KHÔNG phải model hoàn toàn vô dụng, mà là ĐỘ TIN CẬY KHÔNG
  ĐỀU giữa các prompt/loại xe -- không có cách rẻ tiền (chỉ chỉnh ngưỡng số)
  để tách hoàn toàn đúng/sai mà không bỏ sót nhiều kết quả đúng. Do đó:
    - `ENABLE_VEHICLE_TYPE_DETAIL` mặc định = False trong config.py --
      TÍNH NĂNG NÀY TỒN TẠI TRONG MÃ NGUỒN NHƯNG BỊ TẮT MẶC ĐỊNH.
    - Nếu muốn bật, PHẢI tự kiểm tra lại bằng mắt (frame_names) MỌI kết quả
      trước khi tin tưởng, và nên cân nhắc fine-tune model trên vài trăm ảnh
      xe VN thật (ambulance/xe cứu hỏa/xe ba gác) thay vì dùng open-vocab
      "as-is" nếu cần độ chính xác thật cho việc này.
    - Nhãn tiếng Anh cho kết quả embedding tốt hơn nhãn tiếng Việt có dấu
      (model CLIP-based train chủ yếu trên text tiếng Anh).
"""
from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger("traffic_pipeline.vehicle_type_refine")

# Tên lớp tiếng Anh dùng làm text-prompt cho YOLOE -- xem GHI CHÚ THỰC
# NGHIỆM ở trên về lý do dùng tiếng Anh, cụm ngắn.
#
# QUAN TRỌNG: danh sách này đã được RÚT GỌN sau khi kiểm chứng thực nghiệm
# (xem GHI CHÚ THỰC NGHIỆM ở đầu file) phát hiện "police car" và "cement
# mixer truck" bị model đánh giá điểm SAI/quá cao một cách có hệ thống (false
# positive tự tin cao trên xe thường) -- ĐÃ LOẠI BỎ 2 PROMPT NÀY khỏi danh
# sách mặc định. Các prompt còn lại đã kiểm tra pairwise-vs-"car"/"motorcycle"
# trên vài crop thật và cho kết quả ĐÚNG (thua nhãn gốc khi ảnh không phải
# loại đó) -- nhưng đây vẫn chỉ là vài mẫu, KHÔNG phải bảo chứng cho mọi
# trường hợp. Nếu thêm prompt mới, PHẢI tự kiểm tra pairwise tương tự trước
# khi thêm vào danh sách mặc định (xem cách làm trong lịch sử phát triển
# module này), tránh lặp lại lỗi "police car"/"cement mixer truck".
VEHICLE_TYPE_PROMPTS = [
    "ambulance", "fire truck", "garbage truck",
    "cargo tricycle", "tuk tuk", "pickup truck", "van", "tanker truck",
    "tow truck", "school bus",
]

# Các lớp COCO gốc pipeline đã có sẵn -- luôn đưa vào cùng lúc predict() để
# YOLOE có lựa chọn "bình thường" cạnh tranh với nhãn hiếm, tránh việc model
# bị ép chọn nhãn hiếm chỉ vì đó là lựa chọn duy nhất trong danh sách.
_BASE_PROMPTS = ["car", "truck", "bus", "motorcycle", "bicycle"]

_TYPE_VI = {
    "ambulance": "xe cứu thương", "fire truck": "xe cứu hỏa",
    "garbage truck": "xe rác", "cargo tricycle": "xe ba gác", "tuk tuk": "xe lôi/tuk tuk",
    "pickup truck": "xe bán tải", "van": "xe van", "tanker truck": "xe bồn",
    "tow truck": "xe cứu hộ", "school bus": "xe buýt trường học",
}

# Ngưỡng CAO hơn nhiều so với thiết kế ban đầu (0.25) -- đã hạ xuống mức thận
# trọng hơn sau khi phát hiện 1 số prompt cho điểm quá tự tin dù sai (xem GHI
# CHÚ THỰC NGHIỆM). 0.5 KHÔNG loại bỏ hoàn toàn rủi ro (police car false
# positive lên tới 0.895) -- chỉ giảm bớt các trường hợp yếu, LUÔN cần xem
# lại bằng mắt trước khi tin.
MIN_TYPE_CONFIDENCE = 0.5


class VehicleTypeRefiner:
    """Bọc quanh ultralytics.YOLOE -- load 1 LẦN, dùng lại cho toàn bộ video/
    batch (load model + tải text-encoder mất vài giây tới vài chục giây lần
    đầu, không nên load lại mỗi track)."""

    def __init__(self, model_name: str = "yoloe-11s-seg.pt",
                 extra_prompts: Optional[List[str]] = None):
        self.available = False
        self.model = None
        self.prompts = _BASE_PROMPTS + (extra_prompts or VEHICLE_TYPE_PROMPTS)
        try:
            from ultralytics import YOLOE
            self.model = YOLOE(model_name)
            self.model.set_classes(self.prompts, self.model.get_text_pe(self.prompts))
            self.available = True
            logger.info(f"VehicleTypeRefiner: đã load {model_name} với {len(self.prompts)} nhãn "
                        f"({len(_BASE_PROMPTS)} lớp gốc + {len(self.prompts) - len(_BASE_PROMPTS)} loại chi tiết).")
        except Exception as exc:  # noqa: BLE001 -- cố tình bắt rộng: mọi lỗi (thiếu mạng, thiếu gói, lỗi model) đều fallback về None thay vì crash pipeline
            logger.warning(f"VehicleTypeRefiner: KHÔNG dùng được YOLOE ({type(exc).__name__}: {exc}) "
                           f"-- sẽ bỏ qua bước nhận diện loại xe chi tiết, giữ nguyên cls_name gốc từ YOLOv8.")
            self.available = False

    def classify_crop(self, frame_bgr, bbox: dict, conf_thresh: float = 0.05) -> Optional[dict]:
        """
        frame_bgr: frame gốc (numpy array, BGR -- từ cv2).
        bbox: dict {cx, cy, w, h} theo toạ độ frame gốc (giống trajectory
        point trong Event) -- dùng để: (1) crop vùng lân cận (nới rộng nhẹ
        để model có ngữ cảnh) trước khi predict, giúp nhanh hơn + tập trung
        hơn chạy trên cả frame; (2) chọn trong các box YOLOE trả về, box nào
        overlap (IoU) cao nhất với bbox gốc -- vì YOLOE có thể phát hiện
        thêm các object khác trong crop không liên quan tới xe đang xét.

        Trả về None nếu: model không sẵn sàng, không có box nào đủ overlap,
        hoặc nhãn tốt nhất tìm được chính là 1 trong 5 lớp COCO gốc (nghĩa là
        "không phát hiện gì mới hơn những gì YOLOv8 đã biết").
        Ngược lại trả {"vehicle_type": <tên_anh>, "vehicle_type_vi":
        <tên_việt>, "confidence": <float>}.
        """
        if not self.available:
            return None
        import numpy as np

        h_frame, w_frame = frame_bgr.shape[:2]
        cx, cy, w, h = bbox["cx"], bbox["cy"], bbox["w"], bbox["h"]
        # nới rộng vùng crop 40% mỗi chiều để model thấy thêm ngữ cảnh xung
        # quanh xe (biển số, hình dạng thùng xe...) thay vì chỉ đúng bbox khít
        pad_w, pad_h = w * 0.4, h * 0.4
        x1 = max(0, int(cx - w / 2 - pad_w))
        y1 = max(0, int(cy - h / 2 - pad_h))
        x2 = min(w_frame, int(cx + w / 2 + pad_w))
        y2 = min(h_frame, int(cy + h / 2 + pad_h))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame_bgr[y1:y2, x1:x2]

        try:
            results = self.model.predict(crop, conf=conf_thresh, verbose=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"VehicleTypeRefiner: lỗi khi predict ({type(exc).__name__}: {exc}), bỏ qua track này.")
            return None

        r = results[0]
        if len(r.boxes) == 0:
            return None

        # bbox gốc trong hệ toạ độ crop (để tính overlap)
        orig_in_crop = (cx - w / 2 - x1, cy - h / 2 - y1, cx + w / 2 - x1, cy + h / 2 - y1)

        best = None  # (iou, conf, name)
        for box in r.boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            iou = _iou((bx1, by1, bx2, by2), orig_in_crop)
            if iou < 0.2:  # box không liên quan tới xe đang xét (vd xe khác lọt vào vùng crop mở rộng)
                continue
            conf = float(box.conf[0])
            name = self.prompts[int(box.cls[0])]
            if best is None or conf > best[1]:
                best = (iou, conf, name)

        if best is None:
            return None
        _, conf, name = best
        if name in _BASE_PROMPTS:
            return None  # không có gì mới hơn cls_name gốc
        if conf < MIN_TYPE_CONFIDENCE:
            return None
        return {
            "vehicle_type": name,
            "vehicle_type_vi": _TYPE_VI.get(name, name),
            "confidence": round(conf, 3),
        }


def _iou(box_a: tuple, box_b: tuple) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def refine_vehicle_types(events: List, refiner: "VehicleTypeRefiner", frame_reader) -> int:
    """
    events: danh sách Event (từ event_timeline.py) -- sẽ gán thêm field
    `vehicle_type_detail` (dict hoặc None) TRỰC TIẾP lên từng Event có
    cls_name thuộc nhóm phương tiện (dùng setattr vì Event là dataclass,
    field này KHÔNG bắt buộc phải khai báo sẵn trong class để tương thích
    ngược -- nhưng field vehicle_type_detail ĐÃ được thêm vào Event trong
    event_timeline.py, xem ở đó).
    refiner: VehicleTypeRefiner đã khởi tạo (dùng chung cho mọi track, KHÔNG
    tạo mới mỗi lần gọi hàm này).
    frame_reader: hàm (frame_idx) -> frame ảnh gốc, giống frame_reader dùng
    trong build_events_from_tracks() (event_timeline.py) -- dùng lại đúng
    frame đại diện đã chọn (representative_frame_idxs[0], chất lượng cao
    nhất theo frame_quality.py) để crop và phân loại, tránh phải seek video
    lại từ đầu.

    Trả về số track ĐÃ gán được vehicle_type_detail (khác None) -- tiện log.
    """
    from vehicle_relations import _VEHICLE_CLASSES

    if not refiner.available:
        for e in events:
            e.vehicle_type_detail = None
        return 0

    n_detected = 0
    for e in events:
        e.vehicle_type_detail = None
        if e.cls_name not in _VEHICLE_CLASSES or not e.representative_frame_idxs or not e.trajectory:
            continue
        frame = frame_reader(e.representative_frame_idxs[0])
        if frame is None:
            continue
        # dùng trajectory point GẦN NHẤT với thời điểm của frame đại diện đã
        # chọn (không phải luôn là điểm đầu/cuối) để bbox khớp đúng vị trí xe
        # trong đúng frame đó.
        target_t = e.representative_timestamps[0] if e.representative_timestamps else e.t_start
        bbox = min(e.trajectory, key=lambda p: abs(p["t"] - target_t))
        result = refiner.classify_crop(frame, bbox)
        if result is not None:
            e.vehicle_type_detail = result
            n_detected += 1
    return n_detected
