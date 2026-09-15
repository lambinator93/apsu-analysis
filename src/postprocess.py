"""g2 fitting and Q-dependence summaries from saved XPCS results."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

import h5py
import numpy as np
from scipy.optimize import curve_fit


def g2_two_tau(t, A, a, tt1, tt2, b1, b2):
    """Same double KWW used in xpcs.g2_two_tau."""
    return 1 + A * np.abs(a * np.exp(-(t / tt1) ** b1) + (1 - a) * np.exp(-(t / tt2) ** b2)) ** 2


REFLECTION_ORDER = ("011", "10-1", "120", "101")
SINGLE_PEAK = ("111",)
FOUR_PEAK_SCANS = (
    (320, 343),
    (329, 344),
    (339, 349),
    (350, 350),
)
SINGLE_PEAK_SCANS = (
    (360, 353),
    (370, 354),
    (380, 358),
    (399, 362),
    (420, 363),
)
TRACKED_SCANS = tuple((T, scan, "120") for T, scan in FOUR_PEAK_SCANS) + tuple(
    (T, scan, "111") for T, scan in SINGLE_PEAK_SCANS
)
TEMPERATURES = tuple(T for T, *_ in TRACKED_SCANS)
TEMP_CMAP_NAME = "turbo"
REFLECTION_RE = re.compile(r"\(([^)]+)\)")
QDEP_NAME = re.compile(r"^qdep_T(?P<temp>.+)K\.h5$")


def _scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _h5_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
        return item.decode() if isinstance(item, bytes) else str(item)
    return str(value)


def parse_reflection(path: Path | str) -> Optional[str]:
    match = REFLECTION_RE.search(Path(path).name)
    return match.group(1) if match else None


def reflection_sort_key(label: str) -> tuple:
    try:
        return (0, REFLECTION_ORDER.index(label))
    except ValueError:
        return (1, label)


def peak_q0(qx0: float, qy0: float) -> float:
    return float(np.hypot(qx0, qy0))


def discover_results(
    results_dir: Path | str,
    reflections: Optional[Sequence[str]] = REFLECTION_ORDER,
) -> dict[str, Path]:
    """Map reflection label -> analysis HDF5 in a scan results/ folder.

    Pass reflections=None to take every (hkl) file in the folder.
    """
    results_dir = Path(results_dir)
    allowed = None if reflections is None else set(reflections)
    found: dict[str, Path] = {}
    for path in sorted(results_dir.glob("*.h5")):
        if path.name.startswith("qdep_"):
            continue
        label = parse_reflection(path)
        if label is None or (allowed is not None and label not in allowed):
            continue
        found[label] = path
    return dict(sorted(found.items(), key=lambda kv: reflection_sort_key(kv[0])))


def load_peak_g2(path: Path | str, peak: str = "peak_0") -> dict:
    path = Path(path)
    with h5py.File(path, "r") as h5:
        exp = h5["experimental_parameters"]
        pk = h5[f"peaks/{peak}"]
        xp = pk["XPCS"]
        qx0 = _scalar(pk["Qx0"])
        qy0 = _scalar(pk["Qy0"])
        delays = np.asarray(xp["delays"], dtype=float)
        if delays.ndim == 2:
            delays = delays[0]
        out = {
            "path": path,
            "reflection": parse_reflection(path),
            "temperature": _scalar(exp["temperature"]),
            "scan": int(_scalar(exp["scan"])),
            "particle": _h5_str(exp["particle"][()]),
            "num_rings": int(_scalar(exp["numRings"])),
            "num_slices": int(_scalar(exp["numSlices"])),
            "Qx0": qx0,
            "Qy0": qy0,
            "Q0": peak_q0(qx0, qy0),
            "dQ": np.asarray(xp["dQ"], dtype=float),
            "G2": np.asarray(xp["G2_Results"], dtype=float),
            "G2_error": np.asarray(xp["G2_error"], dtype=float),
            "delays": delays,
        }
        if "g2_fit" in xp:
            fit = xp["g2_fit"]
            out["A"] = _scalar(fit["A"])
            out["central_index"] = int(fit.attrs.get("central_index", _central_index(out["dQ"])))
            if "popt" in fit:
                out["popt"] = np.asarray(fit["popt"], dtype=float)
        return out


def _slice_delay(n: int, trunc) -> slice:
    if trunc is None or trunc == 0:
        return slice(None)
    return slice(None, int(trunc))


def _valid_xy(delay: np.ndarray, g2: np.ndarray, g2_err: Optional[np.ndarray], trunc):
    sl = _slice_delay(delay.size, trunc)
    t = np.asarray(delay[sl], dtype=float)
    y = np.asarray(g2[sl], dtype=float)
    ok = np.isfinite(t) & np.isfinite(y) & (t > 0)
    if g2_err is not None:
        e = np.asarray(g2_err[sl], dtype=float)
        ok &= np.isfinite(e) & (e > 0)
    else:
        e = None
    if e is None:
        return t[ok], y[ok], None
    return t[ok], y[ok], e[ok]


def _ordered_two_tau(popt: np.ndarray) -> dict[str, float]:
    """Map (A, a, tt1, tt2, b1, b2) onto slow/fast by tau magnitude."""
    A, mix, tt1, tt2, b1, b2 = [float(v) for v in popt]
    if tt1 <= tt2:
        tau_fast, tau_slow = tt1, tt2
        beta_fast, beta_slow = b1, b2
        amp_fast = mix
    else:
        tau_fast, tau_slow = tt2, tt1
        beta_fast, beta_slow = b2, b1
        amp_fast = 1.0 - mix
    return {
        "A": A,
        "amp_fast": amp_fast,
        "amp_slow": 1.0 - amp_fast,
        "tau_fast": tau_fast,
        "tau_slow": tau_slow,
        "beta_fast": beta_fast,
        "beta_slow": beta_slow,
    }


def fit_g2_two_tau(
    delay: np.ndarray,
    g2: np.ndarray,
    g2_err: Optional[np.ndarray] = None,
    *,
    p0: Optional[Sequence[float]] = None,
    trunc=None,
    bounds: Optional[tuple] = None,
) -> dict:
    t, y, sigma = _valid_xy(delay, g2, g2_err, trunc)
    if t.size < 8:
        raise RuntimeError(f"too few valid g2 points ({t.size})")

    contrast = float(np.nanmax(y) - 1.0)
    seed = np.array(p0 if p0 is not None else [max(contrast, 0.05), 0.5, 10.0, 200.0, 1.0, 1.5], dtype=float)
    if bounds is None:
        tmin, tmax = float(t.min()), float(t.max())
        lower = np.array([0.0, 0.0, tmin / 20.0, tmin / 20.0, 0.1, 0.1])
        upper = np.array([10.0, 1.0, tmax * 20.0, tmax * 20.0, 3.0, 3.0])
        bounds = (lower, upper)

    popt, pcov = curve_fit(
        g2_two_tau,
        t,
        y,
        p0=seed,
        bounds=bounds,
        sigma=sigma,
        absolute_sigma=sigma is not None,
        maxfev=20000,
    )
    yhat = g2_two_tau(t, *popt)
    resid = y - yhat
    chi2 = float(np.sum((resid / sigma) ** 2)) if sigma is not None else float(np.sum(resid**2))
    lo, hi = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
    at_bound = (popt <= lo + 1e-8) | (popt >= hi - 1e-8)
    out = _ordered_two_tau(popt)
    out.update(
        {
            "popt": np.asarray(popt, dtype=float),
            "pcov": np.asarray(pcov, dtype=float),
            "success": True,
            "chi2": chi2,
            "n_fit": int(t.size),
            "at_bound": at_bound,
        }
    )
    return out


def _central_index(dQ: np.ndarray, index: Optional[int] = None) -> int:
    if index is not None:
        return int(index)
    finite = np.where(np.isfinite(dQ))[0]
    if finite.size == 0:
        return 0
    return int(finite[np.argmin(dQ[finite])])


def temperature_norm(vmin: Optional[float] = None, vmax: Optional[float] = None):
    """Locked T scale for every temperature-colored figure."""
    from matplotlib.colors import Normalize

    temps = np.asarray(TEMPERATURES, dtype=float)
    return Normalize(
        vmin=float(temps.min()) if vmin is None else vmin,
        vmax=float(temps.max()) if vmax is None else vmax,
    )


def temperature_cmap(name: str = TEMP_CMAP_NAME):
    import matplotlib.pyplot as plt

    return plt.get_cmap(name)


def temperature_color(T, cmap=None, norm=None):
    """Return the turbo color used for this temperature on all later plots."""
    if cmap is None:
        cmap = temperature_cmap()
    if norm is None:
        norm = temperature_norm()
    return cmap(norm(float(T)))


def central_F2(data: dict, A: Optional[float] = None) -> dict:
    """Normalize the central-ring g2 to |F|^2 via g2 = 1 + A |F|^2."""
    mid = int(data.get("central_index", _central_index(data["dQ"])))
    g2 = np.asarray(data["G2"][mid], dtype=float)
    err = np.asarray(data["G2_error"][mid], dtype=float)
    if A is None:
        A = data.get("A")
    if A is None or not np.isfinite(A) or A == 0:
        A = float(np.nanmax(g2) - 1.0)
    A = float(A)
    return {
        **data,
        "central_index": mid,
        "A": A,
        "F2": (g2 - 1.0) / A,
        "F2_error": err / np.abs(A),
        "g2_central": g2,
        "g2_central_error": err,
    }


def fit_scan(
    path: Path | str,
    *,
    peak: str = "peak_0",
    trunc=-18,
    central: Optional[int] = None,
    p0: Optional[Sequence[float]] = None,
    use_g2_error: bool = False,
) -> dict:
    """Fit every Q partition. Representative point is the innermost ring."""
    data = load_peak_g2(path, peak=peak)
    g2 = data["G2"]
    g2_err = data["G2_error"] if use_g2_error else None
    n_part = g2.shape[0]
    seed = None if p0 is None else np.asarray(p0, dtype=float)

    keys = ("A", "amp_fast", "tau_fast", "tau_slow", "beta_fast", "beta_slow")
    parts = {k: np.full(n_part, np.nan) for k in keys}
    success = np.zeros(n_part, dtype=bool)
    popts = np.full((n_part, 6), np.nan)
    chi2 = np.full(n_part, np.nan)
    at_bound = np.zeros(n_part, dtype=bool)

    for i in range(n_part):
        err = None if g2_err is None else g2_err[i]
        try:
            fit = fit_g2_two_tau(data["delays"], g2[i], err, p0=seed, trunc=trunc)
        except Exception:
            continue
        success[i] = True
        popts[i] = fit["popt"]
        chi2[i] = fit["chi2"]
        at_bound[i] = bool(np.any(fit["at_bound"]))
        for k in keys:
            parts[k][i] = fit[k]
        seed = fit["popt"]

    mid = _central_index(data["dQ"], central)
    n_ok = int(success.sum())
    tau_err = float(np.nanstd(parts["tau_fast"], ddof=1)) if n_ok >= 2 else np.nan
    beta_err = float(np.nanstd(parts["beta_fast"], ddof=1)) if n_ok >= 2 else np.nan

    return {
        **data,
        "trunc": trunc,
        "central_index": mid,
        "success": success,
        "at_bound": at_bound,
        "popt": popts,
        "chi2": chi2,
        "partitions": parts,
        "tau_fast": float(parts["tau_fast"][mid]),
        "beta_fast": float(parts["beta_fast"][mid]),
        "tau_slow": float(parts["tau_slow"][mid]),
        "beta_slow": float(parts["beta_slow"][mid]),
        "A": float(parts["A"][mid]),
        "amp_fast": float(parts["amp_fast"][mid]),
        "tau_fast_err": tau_err,
        "beta_fast_err": beta_err,
    }


def _write_ds(group, name, data):
    arr = np.asarray(data)
    if name in group:
        del group[name]
    if arr.dtype == object or arr.dtype.kind in {"U", "S"}:
        group.create_dataset(name, data=np.asarray(arr, dtype=h5py.string_dtype()))
        return
    if np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(float)
    group.create_dataset(name, data=arr)


def write_fit_to_results(fit: dict, path: Optional[Path] = None, peak: str = "peak_0") -> Path:
    """Append the g2 fit group onto the original analysis file."""
    path = Path(path or fit["path"])
    with h5py.File(path, "a") as h5:
        parent = h5[f"peaks/{peak}/XPCS"]
        if "g2_fit" in parent:
            del parent["g2_fit"]
        grp = parent.create_group("g2_fit")
        grp.attrs["model"] = "g2_two_tau"
        grp.attrs["trunc"] = int(fit["trunc"] or 0)
        grp.attrs["central_index"] = int(fit["central_index"])
        _write_ds(grp, "success", fit["success"].astype(np.int8))
        _write_ds(grp, "at_bound", fit["at_bound"].astype(np.int8))
        _write_ds(grp, "popt", fit["popt"])
        _write_ds(grp, "chi2", fit["chi2"])
        parts = grp.create_group("partitions")
        for key, val in fit["partitions"].items():
            _write_ds(parts, key, val)
        for key in (
            "tau_fast",
            "tau_fast_err",
            "beta_fast",
            "beta_fast_err",
            "tau_slow",
            "beta_slow",
            "A",
            "amp_fast",
            "Q0",
            "Qx0",
            "Qy0",
        ):
            _write_ds(grp, key, fit[key])
        _write_ds(grp, "dQ", fit["dQ"])
    return path


def save_temperature_qdep(fits: Sequence[dict], path: Path | str) -> Path:
    """One small HDF5 for all reflections at a temperature."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fits = sorted(fits, key=lambda f: (f["Q0"], reflection_sort_key(f["reflection"] or "")))
    if not fits:
        raise ValueError("no fits to save")

    with h5py.File(path, "w") as h5:
        h5.attrs["temperature"] = float(fits[0]["temperature"])
        h5.attrs["scan"] = int(fits[0]["scan"])
        h5.attrs["particle"] = str(fits[0]["particle"])
        h5.attrs["model"] = "g2_two_tau"
        h5.attrs["n_reflections"] = len(fits)

        labels = [f["reflection"] or f"peak{i}" for i, f in enumerate(fits)]
        summary = h5.create_group("summary")
        _write_ds(summary, "reflection", labels)
        for key in ("Q0", "Qx0", "Qy0", "tau_fast", "tau_fast_err", "beta_fast", "beta_fast_err", "temperature"):
            _write_ds(summary, key, [f[key] for f in fits])
        _write_ds(summary, "scan", [f["scan"] for f in fits])
        _write_ds(summary, "central_index", [f["central_index"] for f in fits])

        refs = h5.create_group("reflections")
        for label, fit in zip(labels, fits):
            grp = refs.create_group(label)
            grp.attrs["source"] = str(fit["path"])
            _write_ds(grp, "Q0", fit["Q0"])
            _write_ds(grp, "Qx0", fit["Qx0"])
            _write_ds(grp, "Qy0", fit["Qy0"])
            _write_ds(grp, "dQ", fit["dQ"])
            _write_ds(grp, "delays", fit["delays"])
            _write_ds(grp, "G2", fit["G2"])
            _write_ds(grp, "success", fit["success"].astype(np.int8))
            _write_ds(grp, "at_bound", fit["at_bound"].astype(np.int8))
            _write_ds(grp, "popt", fit["popt"])
            _write_ds(grp, "chi2", fit["chi2"])
            _write_ds(grp, "central_index", fit["central_index"])
            parts = grp.create_group("partitions")
            for key, val in fit["partitions"].items():
                _write_ds(parts, key, val)
            for key in ("tau_fast", "tau_fast_err", "beta_fast", "beta_fast_err", "tau_slow", "beta_slow", "A", "amp_fast"):
                _write_ds(grp, key, fit[key])
    return path


def fit_temperature(
    results_dir: Path | str,
    *,
    reflections: Optional[Sequence[str]] = REFLECTION_ORDER,
    trunc=-18,
    write_back: bool = True,
    save: bool = True,
    out_path: Optional[Path] = None,
    require_all: bool = True,
) -> list[dict]:
    results_dir = Path(results_dir)
    files = discover_results(results_dir, reflections=reflections)
    if not files:
        raise FileNotFoundError(f"no matching reflection files in {results_dir}")
    if require_all and reflections is not None:
        missing = [r for r in reflections if r not in files]
        if missing:
            raise FileNotFoundError(f"missing reflections {missing} in {results_dir}")

    fits = []
    for label, path in files.items():
        fit = fit_scan(path, trunc=trunc)
        if write_back:
            write_fit_to_results(fit, path)
        fits.append(fit)

    if save:
        temp = fits[0]["temperature"]
        dest = Path(out_path) if out_path is not None else results_dir / f"qdep_T{temp:g}K.h5"
        save_temperature_qdep(fits, dest)
        for fit in fits:
            fit["qdep_path"] = dest
    return fits


def load_temperature_qdep(path: Path | str) -> dict:
    path = Path(path)
    with h5py.File(path, "r") as h5:
        s = h5["summary"]
        return {
            "path": path,
            "temperature": float(h5.attrs["temperature"]),
            "scan": int(h5.attrs["scan"]),
            "particle": str(h5.attrs.get("particle", "")),
            "reflection": [_h5_str(v) for v in s["reflection"][()]],
            "Q0": np.asarray(s["Q0"], dtype=float),
            "tau_fast": np.asarray(s["tau_fast"], dtype=float),
            "tau_fast_err": np.asarray(s["tau_fast_err"], dtype=float),
            "beta_fast": np.asarray(s["beta_fast"], dtype=float),
            "beta_fast_err": np.asarray(s["beta_fast_err"], dtype=float),
        }


def discover_qdep_files(base_dir: Path | str) -> list[Path]:
    base_dir = Path(base_dir)
    files = [p for p in base_dir.glob("*/results/qdep_T*K.h5") if QDEP_NAME.match(p.name)]

    def _temp_key(path: Path) -> float:
        match = QDEP_NAME.match(path.name)
        return float(match.group("temp")) if match else np.inf

    return sorted(files, key=_temp_key)


def compile_qdep(paths: Iterable[Path | str], out_path: Path | str) -> Path:
    rows = [load_temperature_qdep(p) for p in paths]
    rows.sort(key=lambda r: r["temperature"])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h5:
        h5.attrs["n_temperatures"] = len(rows)
        temps = h5.create_group("temperatures")
        for row in rows:
            key = f"T{row['temperature']:g}K"
            grp = temps.create_group(key)
            grp.attrs["source"] = str(row["path"])
            grp.attrs["temperature"] = row["temperature"]
            grp.attrs["scan"] = row["scan"]
            _write_ds(grp, "reflection", row["reflection"])
            for name in ("Q0", "tau_fast", "tau_fast_err", "beta_fast", "beta_fast_err"):
                _write_ds(grp, name, row[name])
        _write_ds(h5, "temperature", [r["temperature"] for r in rows])
        _write_ds(h5, "scan", [r["scan"] for r in rows])
    return out_path


def print_qdep_table(fits: Sequence[dict]) -> None:
    print(f"{'hkl':>6} {'Q0':>8} {'tau_fast':>12} {'d_tau':>10} {'beta_fast':>10} {'d_beta':>8} {'ok':>5} {'bound':>6}")
    for fit in sorted(fits, key=lambda f: f["Q0"]):
        n_ok = int(np.sum(fit["success"]))
        n_bound = int(np.sum(fit["at_bound"]))
        print(
            f"({fit['reflection']:>4}) {fit['Q0']:8.5f} "
            f"{fit['tau_fast']:12.4g} {fit['tau_fast_err']:10.3g} "
            f"{fit['beta_fast']:10.3g} {fit['beta_fast_err']:8.3g} "
            f"{n_ok:>2}/{len(fit['success']):<2} {n_bound:>6}"
        )


def plot_g2_fits(fit: dict, ax=None, **kwargs):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.2))
    delay = fit["delays"]
    t_line = np.logspace(np.log10(max(delay[delay > 0].min(), 1e-3)), np.log10(delay.max()), 400)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, fit["G2"].shape[0]))
    mid = fit["central_index"]
    for i, g2 in enumerate(fit["G2"]):
        lw = 2.0 if i == mid else 1.0
        ax.semilogx(
            delay, g2, "o", color=cmap[i], ms=3.5, alpha=0.85,
            label=f"ring {i}" + (" (central)" if i == mid else ""),
        )
        if fit["success"][i]:
            ax.semilogx(t_line, g2_two_tau(t_line, *fit["popt"][i]), "-", color=cmap[i], lw=lw)
    ax.set_xlabel("delay (s)")
    ax.set_ylabel(r"$g_2$")
    ax.set_title(f"({fit['reflection']})  Q0={fit['Q0']:.4f} Å$^{{-1}}$  T={fit['temperature']:g} K")
    ax.legend(fontsize=8, loc="best")
    return ax


def plot_qdep(fits: Sequence[dict], axes=None):
    import matplotlib.pyplot as plt

    fits = sorted(fits, key=lambda f: f["Q0"])
    q = np.array([f["Q0"] for f in fits])
    tau = np.array([f["tau_fast"] for f in fits])
    tau_e = np.array([f["tau_fast_err"] for f in fits])
    beta = np.array([f["beta_fast"] for f in fits])
    beta_e = np.array([f["beta_fast_err"] for f in fits])
    labels = [f["reflection"] for f in fits]
    temp = fits[0]["temperature"]

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    ax_t, ax_b = axes
    ax_t.errorbar(q, tau, yerr=tau_e, fmt="o", capsize=4, color="C0", label=r"$\tau_\mathrm{fast}$")
    ok = np.isfinite(q) & np.isfinite(tau) & (q > 0) & (tau > 0)
    if ok.sum() >= 1:
        q_ref, tau_ref = float(q[ok][0]), float(tau[ok][0])
        q_line = np.linspace(float(q[ok].min()) * 0.98, float(q[ok].max()) * 1.02, 200)
        ax_t.plot(q_line, tau_ref * (q_ref / q_line), "--", color="0.45", label=r"$\propto Q^{-1}$")
        ax_t.plot(q_line, tau_ref * (q_ref / q_line) ** 2, ":", color="0.15", label=r"$\propto Q^{-2}$")
    ax_t.legend(fontsize=8)
    ax_b.errorbar(q, beta, yerr=beta_e, fmt="o-", capsize=4, color="C1")
    for qi, ti, bi, lab in zip(q, tau, beta, labels):
        ax_t.annotate(f"({lab})", (qi, ti), textcoords="offset points", xytext=(4, 6), fontsize=8)
        ax_b.annotate(f"({lab})", (qi, bi), textcoords="offset points", xytext=(4, 6), fontsize=8)
    ax_t.set_xlabel(r"$Q_0$ (Å$^{-1}$)")
    ax_t.set_ylabel(r"$\tau_\mathrm{fast}$ (s)")
    ax_t.set_title(f"fast relaxation  T={temp:g} K")
    ax_b.set_xlabel(r"$Q_0$ (Å$^{-1}$)")
    ax_b.set_ylabel(r"$\beta_\mathrm{fast}$")
    ax_b.set_title(f"fast shape  T={temp:g} K")
    return axes


def _as_str_list(values) -> list[str]:
    return [_h5_str(v) for v in np.atleast_1d(values)]


def reflection_series(rows: Sequence[dict], reflection: str) -> dict:
    """Pick one hkl out of compiled per-temperature summaries."""
    temps, q0, tau, tau_e, beta, beta_e, scans = [], [], [], [], [], [], []
    for row in rows:
        labels = _as_str_list(row["reflection"])
        if reflection not in labels:
            continue
        i = labels.index(reflection)
        temps.append(float(row["temperature"]))
        q0.append(float(np.atleast_1d(row["Q0"])[i]))
        tau.append(float(np.atleast_1d(row["tau_fast"])[i]))
        tau_e.append(float(np.atleast_1d(row["tau_fast_err"])[i]))
        beta.append(float(np.atleast_1d(row["beta_fast"])[i]))
        beta_e.append(float(np.atleast_1d(row["beta_fast_err"])[i]))
        scans.append(int(row["scan"]))
    order = np.argsort(temps)
    return {
        "reflection": reflection,
        "temperature": np.asarray(temps, dtype=float)[order],
        "scan": np.asarray(scans, dtype=int)[order],
        "Q0": np.asarray(q0, dtype=float)[order],
        "tau_fast": np.asarray(tau, dtype=float)[order],
        "tau_fast_err": np.asarray(tau_e, dtype=float)[order],
        "beta_fast": np.asarray(beta, dtype=float)[order],
        "beta_fast_err": np.asarray(beta_e, dtype=float)[order],
    }


def plot_tdep(series: dict, axes=None):
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    ax_t, ax_b = axes
    t = series["temperature"]
    lab = series["reflection"]
    ax_t.errorbar(t, series["tau_fast"], yerr=series["tau_fast_err"], fmt="o-", capsize=4, color="C0")
    ax_b.errorbar(t, series["beta_fast"], yerr=series["beta_fast_err"], fmt="o-", capsize=4, color="C1")
    ax_t.set_xlabel("T (K)")
    ax_t.set_ylabel(r"$\tau_\mathrm{fast}$ (s)")
    ax_t.set_title(rf"$({lab})$ fast relaxation vs T")
    ax_b.set_xlabel("T (K)")
    ax_b.set_ylabel(r"$\beta_\mathrm{fast}$")
    ax_b.set_title(rf"$({lab})$ fast shape vs T")
    return axes
