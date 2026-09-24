"""
Chuyển danh sách Detection (phẳng, theo frame) thành EVENT TIMELINE -- đúng
thuật toán đã thống nhất: mỗi track_id là 1 sự kiện tiềm năng (track-to-event
conversion), gắn kèm metadata OCR overlay (location/giờ) + trạng thái đèn +
màu xe tại thời điểm track hoạt động.

Đây là bước GIẢM SỐ ĐƠN VỊ PHẢI INDEX: từ hàng nghìn Detection (1 dòng/đối
tượng/frame) xuống hàng chục Event (1 dòng/track, có t_start-t_end).

GHI CHÚ: đã BỎ nhận diện biển số (ALPR) theo yêu cầu -- xem plate_ocr.py nếu
cần bật lại sau này (file vẫn giữ nguyên, chỉ không gọi từ đây nữa). Trọng
tâm hiện tại: thuộc tính xe (màu sắc qua vehicle_color.py) + quan hệ giữa
các xe (thứ tự trước/sau, vượt, va chạm qua vehicle_relations.py).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from detect_track import Detection
from config import MIN_EVENT_DURATION_SEC, FRAME_WIDTH, FRAME_HEIGHT, CROSSWALK_POLYGON
from frame_quality import quality_score
from frame_position import describe_position

logger = logging.getLogger("traffic_pipeline.event_timeline")

# Ngưỡng confidence tối thiểu để CHẤP NHẬN kết quả màu áo (person_pose.py +
# vehicle_color.py's HSV) -- ĐÃ CHỌN THỰC NGHIỆM: verify trên 4 case thật, 3
# case ĐÚNG có confidence 0.8-1.0, 1 case SAI (bbox áo khớp nhầm mũ bảo
# hiểm/áo trùm đầu) có confidence chỉ 0.42 -- đặt ngưỡng ở giữa 2 nhóm này để
# lọc bớt case sai rõ rệt mà không loại các case đúng đã quan sát được.
SHIRT_COLOR_MIN_CONF = 0.5


@dataclass
class Event:
    video_id: str
    track_id: int
    cls_name: str
    t_start: float
    t_end: float
    n_detections: int
    representative_frame_idxs: List[int] = field(default_factory=list)
    representative_timestamps: List[float] = field(default_factory=list)
    # frame_name đúng quy ước "{video_id}_{timeMs}.jpg" mà backend PixelPals
    # dùng làm primary key trong Milvus (xem database_beit3.py::frame_name =
    # os.path.basename(path) và rrf_provider.py::_frame_name_to_video_frame()
    # -- 2 quy ước frame_name thật đang tồn tại: "{video_id}/{filename}" hoặc
    # "{video_id}_{timeMs}.jpg"). Xuất sẵn field này để sau này ghép events
    # vào backend không phải viết lại logic đặt tên từ đầu.
    representative_frame_names: List[str] = field(default_factory=list)
    location: Optional[str] = None       # từ OCR overlay, lấy tại t_start (không đổi trong video)
    light_at_start: Optional[str] = None  # màu đèn tại thời điểm track xuất hiện
    light_at_end: Optional[str] = None    # màu đèn tại thời điểm track biến mất
    avg_conf: float = 0.0
    # Màu xe (vehicle_color.py) -- None với class "person" hoặc khi không
    # nhận diện được (model lỗi/ảnh quá nhỏ/mờ).
    vehicle_color: Optional[str] = None
    vehicle_color_conf: Optional[float] = None
    vehicle_color_method: Optional[str] = None  # "model" hoặc "hsv_fallback" -- minh bạch nguồn kết quả
    # Quỹ đạo tâm bbox theo thời gian, cần cho vehicle_relations.py (thứ tự
    # trước/sau, vượt, va chạm) -- KHÔNG lưu mọi điểm mà chỉ lưu đã downsample
    # (tối đa vài chục điểm) để không làm phình JSON output không cần thiết.
    trajectory: List[dict] = field(default_factory=list)  # [{"t":.., "cx":.., "cy":.., "w":.., "h":..}]
    # Loại phương tiện CHI TIẾT hơn 5 lớp COCO gốc (vd ambulance, cargo
    # tricycle...) -- do vehicle_type_refine.py gán SAU KHI build_events đã
    # chạy xong (không gán trong hàm build_events_from_tracks() bên dưới vì
    # cần chạy model open-vocab RIÊNG, tốn thời gian hơn nhiều so với các
    # bước khác -- xem main_pipeline.py). None nếu: chưa chạy bước này, model
    # không khả dụng, hoặc không phát hiện loại nào khác 5 lớp gốc. Khi có
    # giá trị là dict {"vehicle_type": <tên tiếng Anh>, "vehicle_type_vi":
    # <tên tiếng Việt>, "confidence": <float>}.
    vehicle_type_detail: Optional[dict] = None
    # Gán bởi track_dedup.py nếu track này là kết quả GỘP từ nhiều track_id
    # trùng lặp (cùng 1 vật thể vật lý bị track nhầm nhiều lần) -- None (mặc
    # định) nghĩa là track KHÔNG bị gộp. Xem track_dedup.py để biết chi tiết
    # phát hiện thực nghiệm dẫn tới bước lọc này.
    merged_from_track_ids: Optional[List[int]] = None
    merge_note: Optional[str] = None
    # Màu áo (person_pose.py + vehicle_color.py's HSV) -- áp dụng cho class
    # "person" (người đi bộ) VÀ người NGỒI TRÊN "motorcycle"/"bicycle" (lấy
    # từ người có bbox pose trùng khít nhất với bbox xe, xem
    # build_events_from_tracks()). None nếu: class không phải người/xe 2
    # bánh, pose_estimator không khả dụng, hoặc không đủ keypoint tin cậy để
    # xác định vùng áo (xem person_pose.py GIỚI HẠN) -- KHÔNG suy đoán màu áo
    # từ bbox toàn thân khi thiếu keypoint, vì đó chính là vấn đề đang cố
    # tránh (lẫn nền/quần/mũ bảo hiểm vào vùng tính màu).
    shirt_color: Optional[str] = None
    shirt_color_conf: Optional[float] = None
    shirt_color_method: Optional[str] = None
    # Vị trí ngữ nghĩa trong khung hình (frame_position.py) -- bổ sung theo
    # yêu cầu "nhận diện vị trí vật kỹ càng", đối chiếu mẫu BTC N001-V001.zip
    # (chỉ có bbox thô, không có mô tả vị trí gì thêm). Tính từ trajectory đã
    # downsample ở trên, KHÔNG cần frame_reader/model nào thêm -- luôn có giá
    # trị (trừ khi track không có điểm quỹ đạo nào, cực hiếm).
    position_grid_vi: Optional[str] = None   # vd "chính giữa khung hình", "góc trên bên trái"
    position_grid_en: Optional[str] = None   # vd "center", "top-left"
    crosswalk_fraction: Optional[float] = None  # % điểm quỹ đạo nằm trong vùng vạch sang đường
    in_crosswalk: bool = False
    position_description: Optional[str] = None  # câu tiếng Việt gộp cả 2 lớp thông tin trên


def _frame_name(video_id: str, timestamp_sec: float) -> str:
    """Đúng quy ước frame_name "{video_id}_{timeMs}.jpg" backend PixelPals
    đang dùng làm primary key Milvus (xem comment trong Event ở trên)."""
    time_ms = int(round(timestamp_sec * 1000))
    return f"{video_id}_{time_ms}.jpg"


def _select_representative(dets: List[Detection], frame_reader) -> List[Detection]:
    """Chọn tối đa 3 frame đại diện THEO CHẤT LƯỢNG (frame_quality.py: kích
    thước bbox + độ nét + conf + không dính biên) thay vì cứng nhắc lấy
    frame đầu/giữa/cuối track -- đảm bảo representative_frame thực sự là
    những khoảnh khắc RÕ NHẤT của track, hữu ích hơn cho search bằng mắt
    lẫn embedding sau này."""
    n = len(dets)
    if frame_reader is None:
        mid_idx = n // 2
        rep = [dets[0], dets[mid_idx], dets[-1]] if n >= 3 else list(dets)
    else:
        by_size = sorted(dets, key=lambda d: (d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1]), reverse=True)
        top_candidates = by_size[:min(8, n)]
        scored = []
        for d in top_candidates:
            frame = frame_reader(d.frame_idx)
            score = quality_score(frame, d.bbox, d.conf, FRAME_WIDTH, FRAME_HEIGHT) if frame is not None else 0.0
            scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        rep = [d for _, d in scored[:3]]
        if not rep:
            mid_idx = n // 2
            rep = [dets[0], dets[mid_idx], dets[-1]] if n >= 3 else list(dets)

    seen_idx = set()
    rep_unique = []
    for d in rep:
        if d.frame_idx not in seen_idx:
            rep_unique.append(d)
            seen_idx.add(d.frame_idx)
    rep_unique.sort(key=lambda d: d.timestamp)
    return rep_unique


def _downsample_trajectory(dets: List[Detection], max_points: int = 20) -> List[dict]:
    """Lấy tối đa max_points điểm quỹ đạo (tâm bbox theo thời gian), rải
    đều theo index thay vì giữ mọi điểm -- đủ để suy luận hướng/thứ tự
    trước-sau ở vehicle_relations.py mà không phình JSON."""
    n = len(dets)
    if n <= max_points:
        chosen = dets
    else:
        step = n / max_points
        chosen = [dets[int(i * step)] for i in range(max_points)]
    return [
        {
            "t": round(d.timestamp, 2),
            "cx": round((d.bbox[0] + d.bbox[2]) / 2, 1),
            "cy": round((d.bbox[1] + d.bbox[3]) / 2, 1),
            "w": round(d.bbox[2] - d.bbox[0], 1),
            "h": round(d.bbox[3] - d.bbox[1], 1),
        }
        for d in chosen
    ]


def build_events_from_tracks(
    detections: List[Detection],
    video_id: str,
    ocr_lookup=None,        # callable(timestamp) -> OverlayInfo | None, optional
    light_lookup=None,      # callable(timestamp) -> str | None, optional
    frame_reader=None,      # callable(frame_idx) -> np.ndarray | None -- dùng để chọn representative_frame chất lượng + màu xe
    color_classifier=None,  # VehicleColorClassifier | None -- xem vehicle_color.py
    pose_estimator=None,    # PersonPoseEstimator | None -- xem person_pose.py, dùng để xác định vùng áo
    vehicle_classes=("car", "motorcycle", "bus", "truck", "bicycle"),
    two_wheeler_classes=("motorcycle", "bicycle"),  # class có thể có người NGỒI TRÊN cần lấy màu áo
    min_event_duration: float = MIN_EVENT_DURATION_SEC,
) -> List[Event]:
    """
    Nhóm Detection theo track_id -> mỗi track_id thành 1 Event với
    t_start/t_end thật (KHÔNG phải mọi frame trong khoảng đó -- chỉ giữ tối
    đa 3 frame đại diện CHẤT LƯỢNG CAO NHẤT, xem _select_representative).

    ocr_lookup/light_lookup: hàm tra cứu theo timestamp, để gắn metadata vào
    từng event mà không phải OCR/detect đèn lại từ đầu (nên tính toán trước
    và cache theo giây, xem main_pipeline.py).

    color_classifier + frame_reader: nếu event là phương tiện, nhận diện
    màu xe trên frame CHẤT LƯỢNG CAO NHẤT của track (bbox lớn, nét, không
    dính biên) -- xem vehicle_color.py.

    pose_estimator + frame_reader: nếu event là "person" (người đi bộ) hoặc
    xe 2 bánh (two_wheeler_classes -- lấy màu áo của NGƯỜI NGỒI TRÊN xe, nếu
    khớp được), chạy YOLO-pose trên frame đại diện để xác định vùng áo (vai
    tới hông, xem person_pose.py) rồi đưa đúng vùng đó vào color_classifier
    (dùng lại chính HSV/K-means classifier của vehicle_color.py, chỉ khác
    vùng crop đưa vào). Pose model chạy theo TỪNG FRAME (không theo bbox
    riêng lẻ như màu xe) nên kết quả được CACHE theo frame_idx trong hàm này
    để không chạy lại nhiều lần nếu nhiều track dùng chung representative
    frame.
    """
    pose_cache: Dict[int, list] = {}

    def _get_pose_results(frame_idx: int, frame):
        if frame_idx not in pose_cache:
            pose_cache[frame_idx] = pose_estimator.estimate(frame) if frame is not None else []
        return pose_cache[frame_idx]
    by_track: Dict[int, List[Detection]] = defaultdict(list)
    for d in detections:
        by_track[d.track_id].append(d)

    events: List[Event] = []
    for track_id, dets in by_track.items():
        dets.sort(key=lambda d: d.timestamp)
        t_start, t_end = dets[0].timestamp, dets[-1].timestamp
        duration = t_end - t_start

        if duration < min_event_duration:
            continue  # track quá ngắn -- khả năng cao là nhiễu detect chớp nhoáng

        n = len(dets)
        rep_dets_unique = _select_representative(dets, frame_reader)

        cls_name = max(set(d.cls_name for d in dets), key=lambda c: sum(1 for d in dets if d.cls_name == c))

        location = None
        if ocr_lookup is not None:
            info = ocr_lookup(t_start)
            location = info.location if info else None

        light_start = light_lookup(t_start) if light_lookup is not None else None
        light_end = light_lookup(t_end) if light_lookup is not None else None

        vcolor = vcolor_conf = vcolor_method = None
        if (color_classifier is not None and frame_reader is not None
                and cls_name in vehicle_classes and rep_dets_unique):
            # ĐÃ SỬA (phát hiện lỗi khi test thật): chọn bbox LỚN NHẤT thôi
            # là chưa đủ -- khi xe ở CỰC GẦN camera (đặc biệt camera nhìn
            # chéo từ trên xuống như N001-V001.mov), bbox có thể bị MÉO
            # (tỉ lệ w/h bất thường, vd 3.4:1 thay vì ~1-1.5:1 của ô tô nhìn
            # chéo thông thường) vì chỉ chụp được phần cản sau/gầm xe --
            # phần này thường tối màu (bóng, nhựa đen) và làm HSV fallback
            # đoán sai màu sơn thật (đã verify: 1 xe TRẮNG thật bị đoán
            # thành "grey" chỉ vì bbox méo chụp trúng cản sau tối). Ưu tiên
            # bbox có tỉ lệ khung hình "bình thường" (không quá dẹt) trong
            # số các representative frame đã chọn, CHỈ dùng bbox lớn nhất
            # làm phương án fallback nếu không có bbox nào tỉ lệ hợp lý.
            def _aspect_ratio_ok(d):
                bw, bh = d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]
                ar = bw / bh if bh > 0 else 999
                return 0.5 <= ar <= 2.2  # phạm vi hợp lý cho xe nhìn từ trên/chéo, loại bbox quá dẹt/quá cao

            normal_reps = [d for d in rep_dets_unique if _aspect_ratio_ok(d)]
            candidates = normal_reps if normal_reps else rep_dets_unique
            best_rep = max(candidates, key=lambda d: (d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1]))
            frame = frame_reader(best_rep.frame_idx)
            if frame is not None:
                result = color_classifier.classify(frame, best_rep.bbox)
                if result is not None:
                    vcolor, vcolor_conf, vcolor_method = result.color, result.confidence, result.method

        scolor = scolor_conf = scolor_method = None
        if (pose_estimator is not None and pose_estimator.available and color_classifier is not None
                and frame_reader is not None
                and cls_name in ("person",) + tuple(two_wheeler_classes) and rep_dets_unique):
            from person_pose import match_shirt_box_to_bbox
            # Thử lần lượt qua các representative frame (đã chọn chất lượng
            # cao) tới khi tìm được 1 frame mà pose model khớp được vùng áo VÀ
            # cho kết quả màu đủ tin cậy -- KHÔNG dừng ở frame đầu tiên khớp
            # được như màu xe, vì ĐÃ PHÁT HIỆN THỰC NGHIỆM: pose model có thể
            # khớp "vùng áo" NHẦM vào MŨ BẢO HIỂM/ÁO KHOÁC TRÙM ĐẦU khi người
            # lái xe máy cúi người/nghiêng đầu che khuất vai thật (verify qua
            # ảnh: 1 người mặc áo khoác chống nắng màu VÀNG bị nhận nhầm
            # "blue" vì bbox áo tính ra trúng ngay mũ trùm đầu màu tối, không
            # phải vùng áo) -- kết quả sai này có confidence THẤP RÕ RỆT
            # (0.42) so với các case đúng đã verify (0.8-1.0), nên lọc bằng
            # ngưỡng SCOLOR_MIN_CONF thay vì chỉ tin kết quả đầu tiên tìm
            # được. Nếu representative frame đầu không đủ tin cậy, thử frame
            # đại diện khác của CÙNG track (góc/tư thế khác có thể cho vùng
            # áo đúng hơn) trước khi bỏ cuộc (trả None).
            best_result, best_conf = None, 0.0
            for rep in sorted(rep_dets_unique, key=lambda d: (d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1]), reverse=True):
                frame = frame_reader(rep.frame_idx)
                if frame is None:
                    continue
                pose_results = _get_pose_results(rep.frame_idx, frame)
                if not pose_results:
                    continue
                shirt_box = match_shirt_box_to_bbox(pose_results, rep.bbox)
                if shirt_box is None:
                    continue
                result = color_classifier.classify(frame, shirt_box.bbox)
                if result is not None and result.confidence > best_conf:
                    best_result, best_conf = result, result.confidence
                if best_conf >= SHIRT_COLOR_MIN_CONF:
                    break
            if best_result is not None and best_conf >= SHIRT_COLOR_MIN_CONF:
                scolor, scolor_conf, scolor_method = best_result.color, best_result.confidence, best_result.method

        trajectory = _downsample_trajectory(dets)

        pos = describe_position(trajectory, FRAME_WIDTH, FRAME_HEIGHT, CROSSWALK_POLYGON)

        events.append(Event(
            video_id=video_id, track_id=track_id, cls_name=cls_name,
            t_start=round(t_start, 2), t_end=round(t_end, 2),
            n_detections=n,
            representative_frame_idxs=[d.frame_idx for d in rep_dets_unique],
            representative_timestamps=[round(d.timestamp, 2) for d in rep_dets_unique],
            representative_frame_names=[_frame_name(video_id, d.timestamp) for d in rep_dets_unique],
            location=location, light_at_start=light_start, light_at_end=light_end,
            avg_conf=round(sum(d.conf for d in dets) / n, 3),
            vehicle_color=vcolor, vehicle_color_conf=vcolor_conf, vehicle_color_method=vcolor_method,
            shirt_color=scolor, shirt_color_conf=scolor_conf, shirt_color_method=scolor_method,
            trajectory=trajectory,
            position_grid_vi=pos["grid_position_vi"], position_grid_en=pos["grid_position_en"],
            crosswalk_fraction=pos["crosswalk_fraction"], in_crosswalk=pos["in_crosswalk"],
            position_description=pos["position_description"],
        ))

    events.sort(key=lambda e: e.t_start)
    logger.info(f"Đã dựng {len(events)} event từ {len(by_track)} track thô "
                f"({len(by_track) - len(events)} track bị loại vì quá ngắn <{min_event_duration}s).")
    return events


_VEHICLE_CLASSES = ("car", "motorcycle", "bus", "truck", "bicycle")


def detect_composite_events(
    events: List[Event],
    jam_window_sec: float = 5.0,
    jam_min_vehicles: int = 6,
    abnormal_stop_sec: float = 20.0,
    max_runner_duration: float = 4.0,
) -> List[dict]:
    """
    Sinh các 'composite event' -- mẫu hình cấp cao hơn 1 track đơn lẻ, giúp
    trả lời trực tiếp các câu query ngôn ngữ tự nhiên phổ biến mà KHÔNG cần
    embedding đoán mò (vì các mẫu này định nghĩa bằng RULE trên metadata có
    cấu trúc, chính xác 100% theo đúng dữ liệu track/đèn đã có).

    Đã cài các loại:
      1. vehicle_present_during_red_light -- xe xuất hiện lúc đèn đỏ.
      2. red_light_runner -- xe có track NGẮN (<= max_runner_duration, dấu
         hiệu đang di chuyển qua chứ không đứng chờ) VÀ đèn chuyển từ
         xanh/vàng sang đỏ trong lúc track hoạt động. ĐÃ SỬA LỖI: ban đầu
         chỉ so light_at_start != 'red' và light_at_end == 'red' mà không
         giới hạn duration, gây RẤT NHIỀU false positive vì bất kỳ track
         nào dài hơn 1 chu kỳ đèn (rất phổ biến -- xe đứng chờ dài) đều
         tình cờ khớp điều kiện dù xe không hề "vượt đèn", chỉ đơn giản có
         mặt xuyên suốt lúc đèn đổi màu. Test thật trên N001-V001.mov cho
         thấy 11/31 track bị gắn nhầm composite này trước khi sửa.
      3. abnormal_long_stop -- track có duration bất thường dài (đứng yên
         lâu hơn abnormal_stop_sec).
      4. traffic_jam -- nhiều track phương tiện (>= jam_min_vehicles) có
         khoảng thời gian hoạt động CHỒNG LẤN trong 1 cửa sổ ngắn.

    Đây là danh sách MẪU NỀN TẢNG -- có thể mở rộng thêm rule khác tuỳ nhu
    cầu query thực tế. Các loại liên quan tới QUAN HỆ GIỮA 2 XE (thứ tự
    trước/sau, vượt, va chạm) nằm ở vehicle_relations.py vì cần so sánh
    quỹ đạo giữa nhiều track cùng lúc, không chỉ 1 track đơn lẻ như ở đây.
    """
    composites: List[dict] = []
    vehicle_events = [e for e in events if e.cls_name in _VEHICLE_CLASSES]

    for e in vehicle_events:
        duration = e.t_end - e.t_start
        if e.light_at_start == "red":
            composites.append({
                "event_type": "vehicle_present_during_red_light",
                "track_id": e.track_id, "cls_name": e.cls_name,
                "t_start": e.t_start, "t_end": e.t_end, "location": e.location,
            })
        if (e.light_at_start is not None and e.light_at_start != "red"
                and e.light_at_end == "red" and duration <= max_runner_duration):
            composites.append({
                "event_type": "red_light_runner",
                "track_id": e.track_id, "cls_name": e.cls_name,
                "t_start": e.t_start, "t_end": e.t_end, "location": e.location,
                "note": "Đèn chuyển sang đỏ trong lúc track đang hoạt động, track khá ngắn "
                        "(gợi ý xe đang di chuyển qua chứ không đứng chờ) -- VẪN CẦN xem lại "
                        "video/crop thật để xác nhận, đây chỉ là gợi ý ứng viên dựa trên rule, "
                        "không phải xác nhận vi phạm.",
            })
        if duration >= abnormal_stop_sec:
            composites.append({
                "event_type": "abnormal_long_stop",
                "track_id": e.track_id, "cls_name": e.cls_name,
                "t_start": e.t_start, "t_end": e.t_end, "duration_sec": round(duration, 1),
                "location": e.location,
            })

    if vehicle_events:
        t_min = min(e.t_start for e in vehicle_events)
        t_max = max(e.t_end for e in vehicle_events)
        t = t_min
        reported_windows = []
        while t <= t_max:
            window_end = t + jam_window_sec
            overlapping = [e for e in vehicle_events if e.t_start < window_end and e.t_end > t]
            if len(overlapping) >= jam_min_vehicles:
                if not reported_windows or t - reported_windows[-1] >= jam_window_sec:
                    composites.append({
                        "event_type": "traffic_jam",
                        "t_start": round(t, 1), "t_end": round(window_end, 1),
                        "n_vehicles": len(overlapping),
                        "location": overlapping[0].location,
                    })
                    reported_windows.append(t)
            t += jam_window_sec / 2

    return composites
