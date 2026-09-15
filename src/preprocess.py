"""Peak selection, windowed Gaussian fitting, and full-detector Q-maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

from xpcs import gaussian, gaussian_2d, q_to_tth, reciprocal_space_map

PeakLike = Union[Sequence[float], dict]


def parse_peaks(peaks: Iterable[PeakLike]) -> list[tuple[float, float]]:
    """Normalize typed peaks to (col, row).

    Accepts ``(col, row)`` tuples or dicts with ``col``/``row`` (or ``x``/``y``).
    """
    out: list[tuple[float, float]] = []
    for peak in peaks:
        if isinstance(peak, dict):
            col = peak.get("col", peak.get("x"))
            row = peak.get("row", peak.get("y"))
            if col is None or row is None:
                raise ValueError(f"Peak dict needs col/row (or x/y): {peak}")
            out.append((float(col), float(row)))
        else:
            col, row = peak
            out.append((float(col), float(row)))
    return out


class PeakPicker:
    """Click seeds on a detector image. Each click is stored as (col, row)."""

    def __init__(self, image: np.ndarray, ax=None, **imshow_kwargs):
        self.peaks: list[tuple[float, float]] = []
        self.image = np.asarray(image)
        created_fig = ax is None
        if ax is None:
            _, ax = plt.subplots()
        self.ax = ax
        kwargs = {"cmap": "nipy_spectral", "norm": LogNorm(vmin=1, vmax=35)}
        kwargs.update(imshow_kwargs)
        self.ax.imshow(self.image + 1, **kwargs)
        self.ax.set_title("Click peaks to fit  (col, row)")
        self.ax.set_xlabel("column")
        self.ax.set_ylabel("row")
        self._cid = self.ax.figure.canvas.mpl_connect("button_press_event", self._on_click)
        if created_fig:
            plt.show()

    def _on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        col, row = float(event.xdata), float(event.ydata)
        self.add(col, row)

    def add(self, col: float, row: float) -> None:
        self.peaks.append((float(col), float(row)))
        self.ax.plot(col, row, "x", color="white", markersize=10, markeredgewidth=2)
        self.ax.annotate(
            f"{len(self.peaks)}",
            (col, row),
            color="white",
            fontsize=9,
            xytext=(6, 6),
            textcoords="offset points",
        )
        self.ax.figure.canvas.draw_idle()

    def disconnect(self) -> list[tuple[float, float]]:
        self.ax.figure.canvas.mpl_disconnect(self._cid)
        return list(self.peaks)


@dataclass
class PeakFit:
    popt: np.ndarray
    pcov: np.ndarray
    seed: tuple[float, float]
    window: tuple[int, int, int, int]  # r0, r1, c0, c1
    fitted: np.ndarray

    @property
    def col(self) -> float:
        return float(self.popt[1])

    @property
    def row(self) -> float:
        return float(self.popt[2])

    @property
    def sig_col(self) -> float:
        return float(abs(self.popt[3]))

    @property
    def sig_row(self) -> float:
        return float(abs(self.popt[4]))

    @property
    def tilt(self) -> float:
        return float(self.popt[5])


def _window_bounds(center: float, half: float, limit: int) -> tuple[int, int]:
    lo = max(0, int(np.floor(center - half)))
    hi = min(limit, int(np.ceil(center + half)))
    if hi - lo < 4:
        lo = max(0, int(center) - 2)
        hi = min(limit, lo + 4)
    return lo, hi


def fit_peak(
    det_sum: np.ndarray,
    col: float,
    row: float,
    half_window: float = 80,
    sig_col0: float = 50,
    sig_row0: float = 50,
) -> PeakFit:
    """Fit a 2D Gaussian in a window around (col, row) so a brighter neighbor cannot steal it."""
    det_sum = np.asarray(det_sum, dtype=float)
    y_len, x_len = det_sum.shape
    c0, c1 = _window_bounds(col, half_window, x_len)
    r0, r1 = _window_bounds(row, half_window, y_len)
    win = det_sum[r0:r1, c0:c1]
    if win.size == 0 or not np.any(win):
        raise ValueError(f"Empty fit window around col={col}, row={row}")

    y1 = np.sum(win, axis=0)
    y2 = np.sum(win, axis=1)
    x1 = np.arange(c0, c1, dtype=float)
    x2 = np.arange(r0, r1, dtype=float)
    bg1 = float(np.min(y1[np.nonzero(y1)])) if np.any(y1) else 0.0
    bg2 = float(np.min(y2[np.nonzero(y2)])) if np.any(y2) else 0.0
    sig_col_guess = min(sig_col0, max(2.0, (c1 - c0) / 6))
    sig_row_guess = min(sig_row0, max(2.0, (r1 - r0) / 6))

    def _fit1d(x, y, mu0, sig0, bg):
        try:
            popt, _ = curve_fit(gaussian, x, y, p0=[np.max(y), mu0, sig0, bg], maxfev=4000)
            return popt
        except (RuntimeError, ValueError):
            return np.array([np.max(y), mu0, sig0, bg])

    popt1 = _fit1d(x1, y1, x1[np.argmax(y1)], sig_col_guess, bg1)
    popt2 = _fit1d(x2, y2, x2[np.argmax(y2)], sig_row_guess, bg2)

    xw, yw = np.meshgrid(x1, x2)
    amp0 = float(det_sum[int(np.clip(np.rint(popt2[1]), 0, y_len - 1)),
                         int(np.clip(np.rint(popt1[1]), 0, x_len - 1))])
    bg0 = float(np.min(win[np.nonzero(win)])) if np.any(win) else 0.0
    sigx = float(np.clip(abs(popt1[2]), 2.0, (c1 - c0) / 3))
    sigy = float(np.clip(abs(popt2[2]), 2.0, (r1 - r0) / 3))
    mu_col = float(np.clip(popt1[1], c0 + 1, c1 - 1))
    mu_row = float(np.clip(popt2[1], r0 + 1, r1 - 1))
    guess = [max(amp0, 0.0), mu_col, mu_row, sigx, sigy, 0.0, bg0]

    # Module gaps are zeros; fitting them lets σ and tilt explode into a stripe
    valid = np.isfinite(win) & (win > 0)
    lower = [0.0, c0, r0, 1.0, 1.0, -np.pi / 2, 0.0]
    upper = [np.inf, c1, r1, (c1 - c0) / 2, (r1 - r0) / 2, np.pi / 2, float(np.max(win))]
    popt, pcov = curve_fit(
        gaussian_2d,
        (xw[valid], yw[valid]),
        win[valid],
        p0=guess,
        bounds=(lower, upper),
        maxfev=12000,
    )

    fitted = np.full(det_sum.shape, np.nan)
    fitted[r0:r1, c0:c1] = gaussian_2d((xw, yw), *popt).reshape(win.shape)
    return PeakFit(popt=np.asarray(popt), pcov=np.asarray(pcov), seed=(col, row),
                   window=(r0, r1, c0, c1), fitted=fitted)


def fit_peaks(det_sum: np.ndarray, peaks: Iterable[PeakLike], **kwargs) -> list[PeakFit]:
    return [fit_peak(det_sum, col, row, **kwargs) for col, row in parse_peaks(peaks)]


def peak_roi(fit: PeakFit, xv: float, xh: float, shape_yx: tuple[int, int]) -> list[int]:
    """Bounding box [r0, r1, c0, c1] in units of fitted σ (Xv vertical, Xh horizontal).

    Uses the axis-aligned box of the *rotated* ellipse so a horizontal streak
    is cropped wide, not tall, when tilt is near 90°.
    """
    y_len, x_len = shape_yx
    ct, st = np.cos(fit.tilt), np.sin(fit.tilt)
    half_col = xh * float(np.hypot(fit.sig_col * ct, fit.sig_row * st))
    half_row = xv * float(np.hypot(fit.sig_col * st, fit.sig_row * ct))
    r0 = max(0, int(np.floor(fit.row - half_row)))
    r1 = min(y_len, int(np.ceil(fit.row + half_row)))
    c0 = max(0, int(np.floor(fit.col - half_col)))
    c1 = min(x_len, int(np.ceil(fit.col + half_col)))
    if r1 <= r0 or c1 <= c0:
        raise ValueError("ROI collapsed; increase Xv/Xh or check the Gaussian fit.")
    return [r0, r1, c0, c1]


def crop_stack(det: np.ndarray, roi: Sequence[int]) -> np.ndarray:
    r0, r1, c0, c1 = roi
    return det[:, r0:r1, c0:c1]


def build_qmap(scan) -> tuple[np.ndarray, np.ndarray]:
    """Full-detector Qx, Qy from the current X0, Y0, D, ttheta."""
    qx, qy = reciprocal_space_map(
        scan.wavelength, scan.ttheta, scan.X0, scan.Y0,
        scan.pixel_size, scan.D, scan.det_dims, scan.geometry,
    )
    Qx, Qy = np.meshgrid(qx, qy)
    return Qx, Qy


def tth_profile(
    Qx: np.ndarray,
    Qy: np.ndarray,
    intensity: np.ndarray,
    wavelength: float,
    geometry: str,
    col: Optional[int] = None,
    row: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """1D 2θ vs integrated intensity along the scan axis through the peak."""
    if intensity.ndim == 3:
        img = np.sum(intensity, axis=0)
    else:
        img = intensity
    if geometry == "Horizontal":
        sl = img.shape[0] // 2 if row is None else int(row)
        tth = q_to_tth(np.sqrt(Qx[sl, :] ** 2 + np.mean(Qy, axis=0) ** 2), wavelength)
        integ = np.sum(img, axis=0)
    else:
        sl = img.shape[1] // 2 if col is None else int(col)
        tth = q_to_tth(np.sqrt(Qx[:, sl] ** 2 + np.mean(Qy, axis=1) ** 2), wavelength)
        integ = np.sum(img, axis=1)
    return tth, integ


def refine_beam_center(scan, fits: Sequence[PeakFit], tth_targets: Sequence[float]) -> dict:
    """Shift X0/Y0 so fitted peak pixels map to known Bragg 2θ values.

    Vertical geometry: X0 from the mean peak column (Qx ~ 0), Y0 from the 2θ
    constraint. Horizontal geometry swaps the roles.
    """
    if len(fits) != len(tth_targets):
        raise ValueError("tth_targets must have one 2θ per fitted peak.")

    pix = float(scan.pixel_size)
    dist = float(scan.D)
    tth = float(scan.ttheta)
    X0_old, Y0_old = float(scan.X0), float(scan.Y0)
    estimates = []

    if scan.geometry == "Vertical":
        scan.X0 = float(np.mean([fit.col for fit in fits]))
        for fit, tth_pk in zip(fits, tth_targets):
            estimates.append(fit.row - (dist / pix) * np.tan(np.deg2rad(tth - tth_pk)))
        scan.Y0 = float(np.mean(estimates))
    else:
        scan.Y0 = float(np.mean([fit.row for fit in fits]))
        for fit, tth_pk in zip(fits, tth_targets):
            estimates.append(fit.col - (dist / pix) * np.tan(np.deg2rad(tth - tth_pk)))
        scan.X0 = float(np.mean(estimates))

    return {
        "X0_old": X0_old,
        "Y0_old": Y0_old,
        "X0": scan.X0,
        "Y0": scan.Y0,
        "per_peak": estimates,
    }
