"""
Cấu hình vùng crop cố định cho video camera giao thông tĩnh (N001-V001.mov,
1920x1080). ĐÃ ĐO THẬT bằng cách crop + xem trực tiếp trên frame thật của
video, KHÔNG đoán tọa độ.

Nếu đổi sang video/camera khác (góc quay khác), phải đo lại các vùng này --
chúng gắn chặt với overlay/vị trí đèn của TỪNG CAMERA, không tổng quát.
"""

VIDEO_PATH = "/mnt/user-data/uploads/N001-V001.mov"
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# Vùng OCR overlay -- đo từ frame thật, khớp chính xác dải chữ trắng nền đen
# ở mép trên cùng video.
CROP_LOCATION = (0, 0, 660, 40)      # (left, top, right, bottom) -- tên giao lộ
CROP_DATETIME = (900, 0, 1310, 40)   # ngày giờ

# Vùng đèn giao thông (cột đèn bên phải khung hình, nhìn thấy 2 đèn: đỏ ở
# trên dành cho hướng ngang, xanh ở dưới dành cho hướng dọc/rẽ -- xem ảnh đã
# xác nhận). Nếu camera khác, phải đo lại toạ độ này.
CROP_TRAFFIC_LIGHT = (1400, 70, 1490, 330)

# Model detect chính: đã đổi từ "yolov8n.pt" (nano) sang "yolov8s.pt" (small)
# trong detect_track.py để tăng độ chính xác (mAP cao hơn rõ rệt trên COCO
# val), đối chiếu yêu cầu "mạnh hơn BTC" -- xem giải thích đầy đủ trong
# docstring đầu detect_track.py. Không cần đổi gì ở đây, model tự tải qua
# ultralytics khi chạy lần đầu, giống cách "yolov8n.pt" đã tự tải trước đó.

# Class ID trong YOLOv8 (COCO) liên quan tới giao thông.
VEHICLE_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PERSON_CLASS = {0: "person"}
RELEVANT_CLASSES = {**VEHICLE_CLASSES, **PERSON_CLASS}

# Tham số event segmentation.
MIN_EVENT_DURATION_SEC = 1.0     # sự kiện quá ngắn (nhiễu/detect chớp nhoáng) bị bỏ qua
TRACK_GAP_TOLERANCE_SEC = 1.0    # cho phép track "mất dấu" tối đa bao lâu vẫn coi là cùng 1 xe (occlusion ngắn)

# Nhận diện màu xe -- xem vehicle_color.py. Model chính (TFLite, 9 màu cơ
# bản) tải từ HuggingFace trong notebook Kaggle; nếu model không tải
# được/lỗi, TỰ ĐỘNG fallback sang thuật toán HSV (không cần cấu hình gì
# thêm, luôn chạy được).
ENABLE_VEHICLE_COLOR = True
VEHICLE_COLOR_MODEL_PATH = "./vehicle_color_model.tflite"  # None hoặc file không tồn tại => dùng fallback HSV toàn bộ

# Nhận diện MÀU ÁO cho người đi bộ VÀ người ngồi trên xe máy/xe đạp -- xem
# person_pose.py (dùng YOLO-pose xác định vùng vai-hông) + vehicle_color.py
# (tái dùng đúng HSV/K-means classifier, chỉ khác vùng crop đưa vào).
#
# ĐÃ KIỂM CHỨNG THỰC NGHIỆM (chạy trên chính video N001-V001.mov): model
# `yolov8s-pose.pt` (KHÔNG dùng bản nano -- đã thử, conf phát hiện người quá
# thấp 0.06-0.14, không đủ tin cậy) cho kết quả tốt trên người đi bộ đứng
# thẳng (2/2 case đúng, conf 0.8-1.0), nhưng với người NGỒI TRÊN XE MÁY có
# rủi ro bbox áo bị khớp NHẦM vào mũ bảo hiểm/áo khoác trùm đầu khi tư thế
# lái xe che khuất vai thật (1 case sai đã bắt được: áo vàng bị nhận nhầm
# "blue" vì bbox trúng mũ trùm đầu, confidence CHỈ 0.42) -- đã thêm ngưỡng
# lọc SHIRT_COLOR_MIN_CONF=0.5 trong event_timeline.py để loại các case
# confidence thấp kiểu này, NHƯNG không đảm bảo loại được MỌI case sai (chỉ
# lọc được case có confidence thấp rõ rệt như đã quan sát).
ENABLE_SHIRT_COLOR = True
POSE_MODEL_NAME = "yolov8s-pose.pt"

# Nhận diện loại phương tiện CHI TIẾT hơn 5 lớp COCO gốc (ambulance, cargo
# tricycle, tuk tuk, xe bồn...) -- xem vehicle_type_refine.py. Dùng model
# open-vocabulary YOLOE (gói `ultralytics`, tải từ GitHub releases -- KHÔNG
# phụ thuộc HuggingFace nên chạy được cả trong sandbox dev bị chặn
# huggingface.co).
#
# MẶC ĐỊNH TẮT (False) -- ĐÃ KIỂM CHỨNG THỰC NGHIỆM (chạy trên chính dữ liệu
# thật của N001-V001.mov) rằng model open-vocab YOLOE "as-is" (chưa fine-tune)
# gán SAI nhãn hiếm với độ tin cậy CAO cho một tỉ lệ đáng kể xe bình thường
# (vd 1 xe con thường bị gán "police car" conf 0.895, 1 xe máy chở hàng
# thường bị gán "cement mixer truck" conf 0.941 -- xem GHI CHÚ THỰC NGHIỆM
# trong vehicle_type_refine.py). Đã loại các prompt gây lỗi rõ rệt nhất khỏi
# danh sách mặc định và tăng ngưỡng tin cậy, nhưng CHƯA đủ để coi là an toàn
# bật mặc định -- rủi ro tạo metadata sai còn tệ hơn không có thông tin.
# Bật tính năng này (=True) CHỈ KHI đã tự xem lại kết quả bằng mắt
# (representative_frame_names) và chấp nhận rủi ro false positive còn lại.
ENABLE_VEHICLE_TYPE_DETAIL = False

# Mở rộng SỐ LOẠI OBJECT nhận diện được (cây/nhà/cửa sổ/bánh xe/lốp/quần áo/
# đồ chơi...) -- xem scene_context_classes.py. AN TOÀN HƠN nhiều so với
# ENABLE_VEHICLE_TYPE_DETAIL ở trên: đây chỉ là liệt kê BỐI CẢNH tổng quan
# của cả video (lấy mẫu vài frame, không gắn nhãn cho track cụ thể nào), nên
# không có rủi ro "gán sai loại xe cho 1 track thật" như vehicle_type_refine.py
# đã gặp phải -- MẶC ĐỊNH BẬT. Vẫn dùng chung model YOLOE với
# ENABLE_VEHICLE_TYPE_DETAIL (VehicleTypeRefiner) nên PHẢI để refiner load
# được (không phụ thuộc ENABLE_VEHICLE_TYPE_DETAIL đang bật hay tắt -- nếu
# tắt cả 2, main_pipeline.py vẫn tự load refiner riêng cho scene context nếu
# cần, xem main_pipeline.py).
ENABLE_SCENE_CONTEXT_CLASSES = True
SCENE_CONTEXT_N_SAMPLE_FRAMES = 5  # số frame mẫu rải đều theo thời gian, tăng lên nếu muốn chắc chắn hơn (chậm hơn)

# GHI CHÚ: đã BỎ nhận diện biển số (ALPR) theo yêu cầu -- file plate_ocr.py
# vẫn còn trong mã nguồn (không xoá) nếu sau này cần bật lại, nhưng KHÔNG
# được gọi từ main_pipeline.py/event_timeline.py nữa.

# Polygon vạch kẻ đường dành cho người đi bộ (crosswalk) -- ĐÃ ĐO THẬT bằng
# cách vẽ polygon lên frame thật rồi xem lại (không đoán), khớp chính xác
# vạch sọc trắng trong khung hình camera N001-V001.mov. 4 điểm theo thứ tự
# (trên-trái, trên-phải, dưới-phải, dưới-trái) -- vẽ theo hình thang vì góc
# camera nghiêng khiến vạch kẻ đường không phải hình chữ nhật thẳng trong
# ảnh 2D. Nếu đổi camera khác, PHẢI đo lại (xem "Đo lại vùng crop" trong
# notebook, cùng cách làm với CROP_LOCATION/CROP_TRAFFIC_LIGHT).
CROSSWALK_POLYGON = [(560, 395), (1350, 360), (1400, 420), (500, 440)]
