"""
Mô tả VỊ TRÍ ngữ nghĩa của 1 track trong khung hình -- bổ sung theo yêu cầu
"nhận diện vị trí vật kỹ càng" (đối chiếu với file mẫu BTC N001-V001.zip,
vốn chỉ có bbox thô [ymin,xmin,ymax,xmax] không kèm mô tả gì thêm -- module
này đi XA HƠN bbox số, sinh ra mô tả có thể đọc/tìm kiếm được).

2 lớp thông tin, tính từ trajectory đã có sẵn (event_timeline.py's
_downsample_trajectory, mỗi điểm {"t","cx","cy","w","h"} theo toạ độ PIXEL
thật của khung hình):

  1. VỊ TRÍ THEO LƯỚI 3x3 (grid) của khung hình -- chia khung hình thành 9
     ô (trái/giữa/phải x trên/giữa/dưới), gán track vào ô chứa VỊ TRÍ
     TRUNG VỊ (median, không phải trung bình -- bền hơn với outlier khi xe
     đi qua nhiều vùng) của quỹ đạo. Cho câu mô tả kiểu "góc trên bên
     trái", "chính giữa khung hình", "giữa bên phải"...

  2. VỊ TRÍ THEO VÙNG NGỮ NGHĨA ĐÃ BIẾT -- tái dùng CROSSWALK_POLYGON
     (config.py, đã có sẵn cho zone_events.py) để biết track có ĐI QUA/ĐỖ
     TRONG vùng vạch sang đường hay không (tỷ lệ % số điểm quỹ đạo nằm
     trong vùng đó).

GIỚI HẠN (nói rõ, không tạo ảo giác chính xác tuyệt đối):
  - Toạ độ 2D chiếu từ góc camera, KHÔNG phải toạ độ thế giới thực -- "góc
    trên bên trái khung hình" không nhất thiết là "phía Bắc" hay "làn
    trong" ngoài đời thật, chỉ mô tả đúng những gì NHÌN THẤY trên video.
  - Lưới 3x3 cố định theo % chiều rộng/cao khung hình, không thích nghi
    theo góc nghiêng camera -- với camera nhìn chéo mạnh, "giữa khung
    hình" có thể lệch xa "giữa giao lộ thật" -- vẫn hữu ích để tìm kiếm/so
    khớp lại video gốc bằng mắt, không dùng để định vị chính xác.
"""
from __future__ import annotations

import statistics
from typing import List, Optional

try:
    from zone_events import _point_in_polygon
except Exception:  # pragma: no cover -- cho phép module này đứng độc lập nếu cần
    def _point_in_polygon(x: float, y: float, polygon: List[tuple]) -> bool:
        n = len(polygon)
        inside = False
        px, py = polygon[-1]
        for cx, cy in polygon:
            if ((cy > y) != (py > y)) and (x < (px - cx) * (y - cy) / (py - cy + 1e-9) + cx):
                inside = not inside
            px, py = cx, cy
        return inside

# Nhãn tiếng Việt cho lưới 3x3, đánh chỉ số (col, row): col 0/1/2 =
# trái/giữa/phải, row 0/1/2 = trên/giữa/dưới.
_GRID_LABELS_VI = {
    (0, 0): "góc trên bên trái", (1, 0): "chính giữa phía trên", (2, 0): "góc trên bên phải",
    (0, 1): "giữa bên trái", (1, 1): "chính giữa khung hình", (2, 1): "giữa bên phải",
    (0, 2): "góc dưới bên trái", (1, 2): "chính giữa phía dưới", (2, 2): "góc dưới bên phải",
}
_GRID_LABELS_EN = {
    (0, 0): "top-left", (1, 0): "top-center", (2, 0): "top-right",
    (0, 1): "middle-left", (1, 1): "center", (2, 1): "middle-right",
    (0, 2): "bottom-left", (1, 2): "bottom-center", (2, 2): "bottom-right",
}


def _grid_cell(cx: float, cy: float, frame_width: int, frame_height: int) -> tuple:
    """Trả về (col, row) 0..2 theo vị trí (cx, cy) pixel thật trong khung
    hình frame_width x frame_height. Chia đều 3 phần theo mỗi trục."""
    if frame_width <= 0 or frame_height <= 0:
        return (1, 1)
    col = min(2, max(0, int(cx / frame_width * 3)))
    row = min(2, max(0, int(cy / frame_height * 3)))
    return (col, row)


def _fraction_in_polygon(trajectory: List[dict], polygon: Optional[List[tuple]]) -> float:
    if not trajectory or not polygon:
        return 0.0
    n_in = sum(1 for p in trajectory if _point_in_polygon(p["cx"], p["cy"], polygon))
    return n_in / len(trajectory)


# Tỷ lệ tối thiểu số điểm quỹ đạo nằm trong crosswalk để coi là "có đi
# qua/dừng trong vùng vạch sang đường" -- 0.3 nghĩa là ít nhất 30% quỹ đạo
# track nằm trong vùng đó, tránh báo dương tính giả chỉ vì 1-2 điểm rìa quỹ
# đạo chạm nhẹ vào polygon.
CROSSWALK_FRACTION_THRESHOLD = 0.3


def describe_position(
    trajectory: List[dict],
    frame_width: int,
    frame_height: int,
    crosswalk_polygon: Optional[List[tuple]] = None,
) -> dict:
    """
    Trả về dict:
      {
        "grid_col": int, "grid_row": int,
        "grid_position_vi": str, "grid_position_en": str,
        "crosswalk_fraction": float (0.0-1.0),
        "in_crosswalk": bool,
        "position_description": str (câu tiếng Việt gộp cả 2 lớp thông tin),
      }
    Trả về dict với các field None/0.0 nếu trajectory rỗng (track không có
    điểm quỹ đạo nào -- hiếm, nhưng không nên crash pipeline vì lý do này).
    """
    if not trajectory:
        return {
            "grid_col": None, "grid_row": None,
            "grid_position_vi": None, "grid_position_en": None,
            "crosswalk_fraction": 0.0, "in_crosswalk": False,
            "position_description": None,
        }

    # Trung vị (median), không phải trung bình -- bền hơn nếu track đi
    # xuyên nhiều vùng khung hình (vd xe chạy từ trên xuống dưới suốt track
    # dài), median phản ánh đúng "phần lớn thời gian track ở đâu" hơn.
    cxs = [p["cx"] for p in trajectory]
    cys = [p["cy"] for p in trajectory]
    median_cx = statistics.median(cxs)
    median_cy = statistics.median(cys)

    col, row = _grid_cell(median_cx, median_cy, frame_width, frame_height)
    grid_vi = _GRID_LABELS_VI[(col, row)]
    grid_en = _GRID_LABELS_EN[(col, row)]

    crosswalk_fraction = _fraction_in_polygon(trajectory, crosswalk_polygon)
    in_crosswalk = crosswalk_fraction >= CROSSWALK_FRACTION_THRESHOLD

    desc = f"xuất hiện ở {grid_vi} của khung hình"
    if in_crosswalk:
        desc += ", đi qua khu vực vạch sang đường"

    return {
        "grid_col": col, "grid_row": row,
        "grid_position_vi": grid_vi, "grid_position_en": grid_en,
        "crosswalk_fraction": round(crosswalk_fraction, 3), "in_crosswalk": in_crosswalk,
        "position_description": desc,
    }
