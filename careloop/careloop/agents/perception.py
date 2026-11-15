"""Agent 2 - Perception workers (computer vision).

One interface: run(scan) -> list[Finding]. Backends:
  - mock: reads scan.mock_model_output (so the whole loop runs with no GPU/images)
  - torchxrayvision: pretrained DenseNet on a real chest X-ray (pip install torchxrayvision scikit-image)
  - brain MRI: an open-source model (e.g. a MONAI/BraTS bundle) would plug into run_brain_mri()
"""
from __future__ import annotations

from typing import Optional

from ..knowledge import TXRV_TO_CANON, normalize
from ..runtime import Runtime
from ..schema import Case, CaseState, Finding, Scan

_TXRV = {}  # cached model


def _threshold(rt: Runtime, label: str) -> float:
    th = rt.config["cv"]["cxr_thresholds"]
    return th.get(label, th.get("default", 0.6))


def _from_mock(scan: Scan, rt: Runtime, source: str) -> list[Finding]:
    out = []
    for m in scan.mock_model_output:
        label, prob = m["label"], float(m["prob"])
        if prob < _threshold(rt, label):
            continue
        out.append(Finding(
            finding=TXRV_TO_CANON.get(label, normalize(label)),
            location=m.get("location", ""),
            size_mm=m.get("size_mm"),
            count=m.get("count", 1),
            confidence=round(prob, 2),
            sources=[source],
            scan_id=scan.scan_id,
            date=scan.date,
            heatmap_path=m.get("heatmap_path"),
        ))
    return out


def run_cxr(scan: Scan, rt: Runtime) -> list[Finding]:
    if rt.config["cv"]["cxr_backend"] != "torchxrayvision":
        return _from_mock(scan, rt, "cxr_model")
    if not scan.image_path:  # real-model mode: no image means no image read (never fall back to made-up output)
        return []
    from pathlib import Path

    from ..runtime import ROOT

    path = Path(scan.image_path)
    path = path if path.is_absolute() else ROOT / path
    if not path.exists():  # real-model mode: missing file means no image read
        return []
    scan.image_path = str(path)
    import numpy as np
    import skimage.io
    import torch
    import torchxrayvision as xrv

    if "model" not in _TXRV:
        _TXRV["model"] = xrv.models.DenseNet(weights=rt.config["cv"]["cxr_weights"]).eval()
    model = _TXRV["model"]
    img = skimage.io.imread(scan.image_path)
    img = xrv.datasets.normalize(img, 255)
    if img.ndim == 3:
        img = img.mean(2)
    img = img[None, ...]
    import torchvision

    tf = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
    img = torch.from_numpy(tf(img)).float()[None, ...]
    with torch.no_grad():
        probs = model(img)[0].numpy()
    out = []
    for label, p in zip(model.pathologies, probs):
        if not label or label not in TXRV_TO_CANON or p < _threshold(rt, label):
            continue
        out.append(Finding(finding=TXRV_TO_CANON[label], confidence=round(float(p), 2), sources=["cxr_model"],
                           scan_id=scan.scan_id, date=scan.date))
        # TODO(heatmap): Grad-CAM on model.features for this class -> save PNG -> heatmap_path
    return out


def run_brain_mri(scan: Scan, rt: Runtime) -> list[Finding]:
    # No open-source brain MRI model is wired yet; an open model (e.g. a BraTS/MONAI segmentation bundle)
    # would plug in here: segment -> lesion count, volume, location words.
    return _from_mock(scan, rt, "brain_mri_model")


def run_fallback(scan: Scan, rt: Runtime) -> list[Finding]:
    # General medical VLM (e.g. MedGemma) would go here. Capped at low confidence by design.
    fs = _from_mock(scan, rt, "fallback_model")
    for f in fs:
        f.confidence = min(f.confidence, 0.5)
    return fs


WORKERS = {"cxr_model": run_cxr, "brain_mri_model": run_brain_mri, "fallback_model": run_fallback}


def run(case: Case, rt: Runtime) -> Case:
    for sid in case.pending_scan_ids:
        scan = next(s for s in case.scans if s.scan_id == sid)
        for r in scan.route:
            if r in WORKERS:
                found = WORKERS[r](scan, rt)
                case.raw_findings.extend(found)
                case.log("perception", f"{r} on {sid}: {len(found)} finding(s) over threshold", rt.clock.iso())
    case.state = CaseState.PERCEIVED
    return case
