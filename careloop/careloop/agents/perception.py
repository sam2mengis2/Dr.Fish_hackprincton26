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
        heat = None
        try:
            heat = _gradcam_png(model, img, model.pathologies.index(label), scan, label)
        except Exception:  # a heatmap is a nice-to-have; never lose the finding over it
            heat = None
        out.append(Finding(finding=TXRV_TO_CANON[label], confidence=round(float(p), 2), sources=["cxr_model"],
                           scan_id=scan.scan_id, date=scan.date, heatmap_path=heat))
    return out


def _jet(v):
    """Tiny jet colormap (no matplotlib needed): v in [0,1] -> RGB uint8."""
    import numpy as np

    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype("uint8")


def _gradcam_png(model, x, idx: int, scan: Scan, label: str) -> str:
    """Grad-CAM on the last dense block: where in the image the model's evidence for `label` comes from.
    Saved as an overlay PNG next to the case store. It explains the model's attention, not the diagnosis."""
    import numpy as np
    import torch
    from PIL import Image

    from ..runtime import ROOT

    acts = {}
    hook = model.features.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    try:
        x = x.clone()
        with torch.enable_grad():
            out = model(x)
            a = acts["a"]
            a.retain_grad()
            model.zero_grad()
            out[0, idx].backward()
            w = a.grad.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((w * a).sum(1))[0].detach().numpy()
    finally:
        hook.remove()
    if cam.max() <= 0:
        return None
    cam = cam / cam.max()
    cam = np.array(Image.fromarray((cam * 255).astype("uint8")).resize((224, 224), Image.BILINEAR)) / 255.0
    base = x[0, 0].detach().numpy()
    base = ((base - base.min()) / (np.ptp(base) or 1) * 255).astype("uint8")
    gray = np.stack([base] * 3, -1).astype(float)
    alpha = (0.55 * cam)[..., None]
    over = (gray * (1 - alpha) + _jet(cam).astype(float) * alpha).astype("uint8")
    out_dir = ROOT / "runs" / "heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{scan.scan_id}_{label.replace(' ', '_')}.png"
    Image.fromarray(over).resize((448, 448), Image.BILINEAR).save(path)
    return str(path)


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
