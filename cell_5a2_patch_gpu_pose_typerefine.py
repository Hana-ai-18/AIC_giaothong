# 5a-2. VÁ THÊM: ép YOLO-pose (person_pose.py) và YOLOE (vehicle_type_refine.py)
# chạy GPU -- speedup_patch.py gốc (cell 5a) CHỈ patch detect_track (BƯỚC 1).
#
# TẠI SAO CẦN THÊM PATCH NÀY:
# person_pose.py dòng ~103:      self._model.predict(frame_bgr, conf=..., verbose=False)
# vehicle_type_refine.py dòng ~181: self.model.predict(crop, conf=..., verbose=False)
# Cả 2 lệnh predict() đều KHÔNG truyền device=... -> ultralytics chạy CPU.
# Chính vehicle_type_refine.py tự đo được ~0.28s/frame TRÊN CPU cho YOLOE --
# với hàng trăm/nghìn track/video, đây là bottleneck to ngang hoặc hơn cả
# detect+track chính, và speedup_patch.py cell 5a KHÔNG hề đụng tới 2 model
# này. Patch ở đây chỉ thêm device=DEVICE vào predict(), không đổi bất kỳ
# logic/ngưỡng/tham số nào khác.
import os

with open(os.path.join(CODE_DIR, "speedup_patch_gpu2.py"), "w", encoding="utf-8") as _f:
    _f.write(r'''"""
speedup_patch_gpu2.py -- ép YOLO-pose (person_pose.py) và YOLOE
(vehicle_type_refine.py) chạy GPU. KHÔNG đổi logic/ngưỡng/đầu ra, chỉ tiêm
device=DEVICE vào lệnh model.predict() đã tồn tại sẵn của 2 class này.

Gọi speedup_patch_gpu2.apply() sau khi import person_pose và
vehicle_type_refine (thứ tự so với speedup_patch.apply() không quan trọng,
2 patch không đụng chạm nhau).
"""
from __future__ import annotations

import logging
import torch

import person_pose
import vehicle_type_refine

_log = logging.getLogger("traffic_pipeline.speedup_gpu2")

DEVICE = 0 if torch.cuda.is_available() else "cpu"


def _wrap_predict(model):
    """Bọc model.predict hiện có để LUÔN chèn device=DEVICE trừ khi caller đã
    tự truyền device khác. Idempotent: gọi nhiều lần không bọc lồng nhau."""
    if getattr(model.predict, "_gpu2_wrapped", False):
        return
    _orig_predict = model.predict

    def _predict_gpu(*args, **kwargs):
        kwargs.setdefault("device", DEVICE)
        return _orig_predict(*args, **kwargs)

    _predict_gpu._gpu2_wrapped = True
    model.predict = _predict_gpu


def apply():
    # ---- PersonPoseEstimator: patch __init__ để bọc self._model.predict ----
    Pose = person_pose.PersonPoseEstimator
    if not hasattr(Pose, "_gpu2_patched"):
        _orig_init = Pose.__init__

        def _init_patched(self, *a, **kw):
            _orig_init(self, *a, **kw)
            if getattr(self, "_model", None) is not None:
                _wrap_predict(self._model)

        Pose.__init__ = _init_patched
        Pose._gpu2_patched = True

    # ---- VehicleTypeRefiner: patch __init__ để bọc self.model.predict ----
    Refiner = vehicle_type_refine.VehicleTypeRefiner
    if not hasattr(Refiner, "_gpu2_patched"):
        _orig_init2 = Refiner.__init__

        def _init_patched2(self, *a, **kw):
            _orig_init2(self, *a, **kw)
            if getattr(self, "model", None) is not None:
                _wrap_predict(self.model)

        Refiner.__init__ = _init_patched2
        Refiner._gpu2_patched = True

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "KHÔNG CÓ GPU"
    _log.info(f"speedup_patch_gpu2: pose+YOLOE sẽ chạy device={DEVICE!r} ({gpu})")
''')
print("Đã ghi", os.path.join(CODE_DIR, "speedup_patch_gpu2.py"))
