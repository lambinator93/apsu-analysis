"""Mask, multitau g2, two-time, and HDF5 save for selected peaks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import h5py
import numpy as np

from autocorrelations import g2calc, getdelayinfo, normalize_g2, twotime
from preprocess import PeakFit, crop_stack, peak_roi
from xpcs import const_int_mask

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **_kwargs):
        return it


def build_masks(
    det_crop: np.ndarray,
    fit: PeakFit,
    roi: Sequence[int],
    *,
    mask_size: float = 3,
    num_rings: int = 10,
    num_slices: int = 1,
    tilt: float = 0.0,
    tol: float = 0.02,
    res: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-intensity elliptical partitions. Center and widths come from the Gaussian fit."""
    x0 = fit.col - roi[2]
    y0 = fit.row - roi[0]
    return const_int_mask(
        det_crop,
        fit.sig_col,
        fit.sig_row,
        x0=x0,
        y0=y0,
        sigAll=mask_size,
        num_rings=num_rings,
        num_slices=num_slices,
        tilt=0.0 if tilt is None else tilt,
        tol=tol,
        res=res,
    )


def partition_dq(masks: np.ndarray, Qx_c: np.ndarray, Qy_c: np.ndarray, Qx0: float, Qy0: float) -> np.ndarray:
    dQ = np.full(masks.shape[0], np.nan)
    rad = np.sqrt((Qx_c - Qx0) ** 2 + (Qy_c - Qy0) ** 2)
    for i, mask in enumerate(masks):
        sel = mask == 1
        if np.any(sel):
            dQ[i] = float(np.mean(rad[sel]))
    return dQ


def run_g2(det_crop: np.ndarray, masks: np.ndarray, frame_spacing: float, dpl: int = 6):
    ones = np.ones(det_crop.shape[1:], dtype=np.float32)
    _sumI, G2, IF, IP, _gc = g2calc(det_crop, ones, dpl)
    delay = getdelayinfo(det_crop.shape[0], dpl)[0]
    delay_s = np.asarray(delay, dtype=float) * float(frame_spacing)

    n_part, n_delay = masks.shape[0], delay_s.shape[1]
    g2 = np.full((n_part, n_delay), np.nan)
    g2_err = np.full((n_part, n_delay), np.nan)
    for i, mask in enumerate(masks):
        if not np.any(mask == 1):
            continue
        mean, err, last = normalize_g2(G2, IF, IP, mask, delay)
        n = min(last, n_delay, len(mean))
        g2[i, :n] = mean[:n]
        g2_err[i, :n] = err[:n]
    return g2, g2_err, delay_s


def run_twotime(det_crop: np.ndarray, masks: np.ndarray) -> np.ndarray:
    n_part = masks.shape[0]
    nfr = det_crop.shape[0]
    out = np.full((n_part, nfr, nfr), np.nan, dtype=np.float32)
    for i, mask in enumerate(tqdm(masks, desc="two-time partitions")):
        pix = det_crop[:, mask == 1]
        if pix.shape[1] < 2:
            continue
        out[i] = twotime(pix.reshape(nfr, pix.shape[1], 1))
    return out


def analyze_peak(
    scan,
    Qx: np.ndarray,
    Qy: np.ndarray,
    fit: PeakFit,
    *,
    xv: float = 5,
    xh: float = 3.5,
    mask_size: float = 3,
    num_rings: int = 10,
    num_slices: int = 1,
    tilt: Optional[float] = None,
    tol: float = 0.02,
    res: float = 1e-4,
    dpl: int = 6,
    do_twotime: bool = True,
) -> dict:
    """Crop around one fitted peak, build masks, run g2 and two-time."""
    roi = peak_roi(fit, xv, xh, scan.det.shape[1:])
    det_crop = crop_stack(scan.det, roi)
    Qx_c = Qx[roi[0]:roi[1], roi[2]:roi[3]]
    Qy_c = Qy[roi[0]:roi[1], roi[2]:roi[3]]

    use_tilt = fit.tilt if tilt is None else tilt
    # const_int_mask rotates CCW; gaussian_2d theta has the opposite sense
    # (same minus sign as notebooks/xpcs.ipynb)
    masks, ring_widths = build_masks(
        det_crop, fit, roi,
        mask_size=mask_size, num_rings=num_rings, num_slices=num_slices,
        tilt=-use_tilt, tol=tol, res=res,
    )

    r = int(np.clip(np.rint(fit.row), 0, Qy.shape[0] - 1))
    c = int(np.clip(np.rint(fit.col), 0, Qx.shape[1] - 1))
    Qx0, Qy0 = float(Qx[r, c]), float(Qy[r, c])
    dQ = partition_dq(masks, Qx_c, Qy_c, Qx0, Qy0)

    g2, g2_err, delay_s = run_g2(det_crop, masks, scan.frame_spacing, dpl=dpl)
    two_time = run_twotime(det_crop, masks) if do_twotime else None

    return {
        "popt": fit.popt,
        "pcov": fit.pcov,
        "seed": np.asarray(fit.seed),
        "roi": np.asarray(roi),
        "det_crop": det_crop,
        "masks": masks,
        "ring_widths": ring_widths,
        "dQ": dQ,
        "G2": g2,
        "G2_error": g2_err,
        "delays": delay_s,
        "TwoTime": two_time,
        "Qx0": Qx0,
        "Qy0": Qy0,
        "tilt": use_tilt,
    }


def analyze_peaks(scan, Qx, Qy, fits: Sequence[PeakFit], **kwargs) -> list[dict]:
    return [analyze_peak(scan, Qx, Qy, fit, **kwargs) for fit in fits]


def mask_overlay(masks: np.ndarray) -> np.ndarray:
    """Integer label image for plotting all partitions."""
    labeled = np.zeros(masks.shape[1:], dtype=float)
    for i, mask in enumerate(masks):
        labeled[mask == 1] = i + 1
    return labeled


def _write_ds(group, name, data, **kwargs):
    if data is None:
        return
    arr = np.asarray(data)
    if arr.shape == ():
        arr = np.array([arr])
    if arr.size <= 1:
        group.create_dataset(name, data=arr, **kwargs)
        return
    group.create_dataset(name, data=arr, compression="gzip", compression_opts=9, **kwargs)


def save_results(
    path,
    scan,
    Qx: np.ndarray,
    Qy: np.ndarray,
    peak_results: Sequence[dict],
    *,
    xv: float,
    xh: float,
    mask_size: float,
    num_rings: int,
    num_slices: int,
    dpl: int,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as h5:
        exp = h5.create_group("experimental_parameters")
        _write_ds(exp, "D", scan.D)
        _write_ds(exp, "D_meta", scan.D_meta)
        _write_ds(exp, "X0", scan.X0)
        _write_ds(exp, "Y0", scan.Y0)
        _write_ds(exp, "X0_meta", scan.X0_meta)
        _write_ds(exp, "Y0_meta", scan.Y0_meta)
        _write_ds(exp, "wavelength", scan.wavelength)
        _write_ds(exp, "energy", scan.energy)
        _write_ds(exp, "ttheta", scan.ttheta)
        _write_ds(exp, "pixel_size", scan.pixel_size)
        _write_ds(exp, "frame_spacing", scan.frame_spacing)
        _write_ds(exp, "det_dimensions", scan.det_dims)
        _write_ds(exp, "nFrames", scan.nframes)
        _write_ds(exp, "samx", scan.samx)
        _write_ds(exp, "samz", scan.samz)
        exp.create_dataset("geometry", data=str(scan.geometry))
        exp.create_dataset("beamline", data=str(scan.beamline))
        exp.create_dataset("particle", data=str(scan.particle))
        exp.create_dataset("temperature", data=np.array([scan.temperature], dtype=float))
        exp.create_dataset("scan", data=np.array([scan.scan], dtype=int))
        _write_ds(exp, "Xv", xv)
        _write_ds(exp, "Xh", xh)
        _write_ds(exp, "maskSize", mask_size)
        _write_ds(exp, "numRings", num_rings)
        _write_ds(exp, "numSlices", num_slices)
        _write_ds(exp, "dpl", dpl)

        qmap = h5.create_group("qmap")
        _write_ds(qmap, "Qx", Qx)
        _write_ds(qmap, "Qy", Qy)

        peaks_grp = h5.create_group("peaks")
        for i, result in enumerate(peak_results):
            pk = peaks_grp.create_group(f"peak_{i}")
            _write_ds(pk, "popt", result["popt"])
            _write_ds(pk, "pcov", result["pcov"])
            _write_ds(pk, "seed", result["seed"])
            _write_ds(pk, "roi", result["roi"])
            _write_ds(pk, "Qx0", result["Qx0"])
            _write_ds(pk, "Qy0", result["Qy0"])

            xp = pk.create_group("XPCS")
            _write_ds(xp, "Mask", result["masks"])
            _write_ds(xp, "ringWidths", result["ring_widths"])
            _write_ds(xp, "dQ", result["dQ"])
            _write_ds(xp, "G2_Results", result["G2"])
            _write_ds(xp, "G2_error", result["G2_error"])
            _write_ds(xp, "delays", result["delays"])
            _write_ds(xp, "TwoTime", result["TwoTime"])

    return path


def default_results_path(scan, mask_size, num_rings, num_slices) -> Path:
    name = (
        f"particle{scan.particle}_temp{scan.temperature}K_scan{scan.scan}"
        f"_ms{mask_size}_nr{num_rings}ns{num_slices}.h5"
    )
    return scan.results_path / name
