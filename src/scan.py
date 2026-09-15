"""Load XPCS scans and beamline metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401  # registers HDF5 filters used by 8-ID-E files
except ImportError:
    hdf5plugin = None

from xpcs import load_batchinfo


def _scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _clean_listing(work_path: Path) -> list[str]:
    names = sorted(p.name for p in work_path.iterdir() if not p.name.startswith("."))
    return [n for n in names if not n.startswith("._")]


def resolve_work_path(
    compound: str,
    scan: int,
    temperature: int | float,
    attenuation: int = 20,
    base_dir: Optional[Path] = None,
    config: Optional[dict] = None,
) -> Path:
    """Build the scan directory for the NBH / 8-ID-E layout."""
    if base_dir is None:
        base_dir = Path(config["Base"]) if config and config.get("Base") else Path(
            "/Users/eriklamb/data/APS/8_ID_E/Na2B10H10/"
        )
    if compound in ("NBH", "b10", "B10"):
        dataset = f"A{scan}_NaBH_att0000{attenuation}_{temperature}K_001"
        return Path(base_dir) / dataset
    raise ValueError(f"Unknown compound '{compound}'. Add a path rule or pass work_path.")


def find_scan_files(work_path: Path) -> dict[str, Path]:
    """Locate batchinfo / data / metadata by suffix instead of list index."""
    names = _clean_listing(work_path)
    batch = next((n for n in names if n.endswith(".batchinfo")), None)
    data = next((n for n in names if n.endswith(".h5")), None)
    meta = next((n for n in names if n.endswith(".hdf")), None)
    missing = [k for k, v in (("batchinfo", batch), ("h5", data), ("hdf", meta)) if v is None]
    if missing:
        raise FileNotFoundError(f"Missing {missing} in {work_path}. Contains: {names}")
    return {
        "batchinfo": work_path / batch,
        "data": work_path / data,
        "metadata": work_path / meta,
    }


@dataclass
class Scan:
    det: np.ndarray
    det_corr: np.ndarray
    wavelength: float
    energy: float
    ttheta: float
    X0: float
    Y0: float
    pixel_size: float
    D: float
    frame_spacing: float
    det_dims: np.ndarray
    geometry: str
    beamline: str
    work_path: Path
    particle: Any = None
    temperature: Any = None
    scan: Any = None
    attenuation: Any = None
    samx: float = 0.0
    samz: float = 0.0
    X0_meta: float = 0.0
    Y0_meta: float = 0.0
    D_meta: float = 0.0
    batchinfo: dict = field(default_factory=dict)

    @property
    def nframes(self) -> int:
        return int(self.det.shape[0])

    @property
    def processed_path(self) -> Path:
        return self.work_path / "processed"

    @property
    def results_path(self) -> Path:
        return self.work_path / "results"


def load_8ide(
    work_path: Path,
    *,
    geometry: str = "Vertical",
    D: Optional[float] = None,
    X0: Optional[float] = None,
    Y0: Optional[float] = None,
    hot_pixel: float = 100.0,
    particle=None,
    temperature=None,
    scan=None,
    attenuation=None,
) -> Scan:
    """Load an 8-ID-E hdf5 + batchinfo scan. D, X0, Y0 override metadata when given."""
    work_path = Path(work_path)
    files = find_scan_files(work_path)
    batchinfo = load_batchinfo(files["batchinfo"], "=")

    with h5py.File(files["metadata"], "r") as metadata:
        frame_spacing = _scalar(metadata["/entry/instrument/detector_1/acquire_period"])
        X0_meta = _scalar(metadata["/entry/instrument/bluesky/metadata/bcx"])
        Y0_meta = _scalar(metadata["/entry/instrument/bluesky/metadata/bcy"])
        pixel_size = _scalar(metadata["/entry/instrument/bluesky/metadata/pix_dim_x"])
        D_meta = _scalar(metadata["/entry/instrument/bluesky/metadata/det_dist"])
        energy = _scalar(metadata["/entry/instrument/bluesky/metadata/X_energy"])
        wavelength = 10.0 * 1.2398 / energy

    with h5py.File(files["data"], "r") as data:
        det = np.array(data["entry/data/data/"])

    hor_beg, hor_end = batchinfo["col_beg"], batchinfo["col_end"]
    ver_beg, ver_end = batchinfo["row_beg"], batchinfo["row_end"]
    det_dims = np.array([[hor_beg, hor_end], [ver_beg, ver_end]], dtype=float)
    n_row = int(ver_end - ver_beg + 1)
    n_col = int(hor_end - hor_beg + 1)
    if (n_row, n_col) != det.shape[1:]:
        det_dims = np.array([[0, det.shape[2] - 1], [0, det.shape[1] - 1]], dtype=float)

    det_corr = np.copy(det)
    if hot_pixel is not None:
        det = det.copy()
        det[det > hot_pixel] = 0

    return Scan(
        det=det,
        det_corr=det_corr,
        wavelength=float(wavelength),
        energy=float(energy),
        ttheta=_scalar(batchinfo["th"]),
        X0=float(X0) if X0 is not None else X0_meta,
        Y0=float(Y0) if Y0 is not None else Y0_meta,
        pixel_size=float(pixel_size),
        D=float(D) if D is not None else D_meta,
        frame_spacing=float(frame_spacing),
        det_dims=det_dims,
        geometry=geometry,
        beamline="8IDE",
        work_path=work_path,
        particle=particle,
        temperature=temperature,
        scan=scan,
        attenuation=attenuation,
        samx=_scalar(batchinfo.get("samx", 0.0)),
        samz=_scalar(batchinfo.get("samz", 0.0)),
        X0_meta=X0_meta,
        Y0_meta=Y0_meta,
        D_meta=D_meta,
        batchinfo=batchinfo,
    )


def load_p10(*_args, **_kwargs) -> Scan:
    raise NotImplementedError(
        "P10 loading is a stub. Use notebooks/xpcs.ipynb for P10, or extend scan.load_p10."
    )


def load_scan(
    beamline: str,
    work_path: Optional[Path] = None,
    **kwargs,
) -> Scan:
    beamline = beamline.upper().replace("-", "")
    if beamline in ("8IDE", "8ID_E", "8ID"):
        if work_path is None:
            work_path = resolve_work_path(
                kwargs.get("compound"),
                kwargs.get("scan"),
                kwargs.get("temperature"),
                attenuation=kwargs.get("attenuation", 20),
                base_dir=kwargs.get("base_dir"),
                config=kwargs.get("config"),
            )
        kwargs.pop("compound", None)
        kwargs.pop("base_dir", None)
        kwargs.pop("config", None)
        return load_8ide(work_path, **kwargs)
    if beamline == "P10":
        return load_p10(work_path, **kwargs)
    raise ValueError(f"Unsupported beamline '{beamline}'.")
