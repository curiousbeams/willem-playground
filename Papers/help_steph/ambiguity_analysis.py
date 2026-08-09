"""Analysis of ptychographic translation ambiguity vs. convergence angle.

Each reconstruction is registered against the ground-truth potential by phase
cross-correlation; the resulting sub-pixel shift is one row of a
``shifts_conv*_{kind}.csv``.  When the diffracted disks do not overlap the
reconstruction is only determined up to a translation within the projected unit
cell, so the shift is a random vector uniformly distributed over that cell.
Once the disks overlap the solution is unique and the shift vanishes.

The overlap condition for the first allowed zero-order-Laue-zone reflection is
``alpha > lambda / (2 d)``, which depends on the zone axis -- see ``SAMPLES``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --- experiment constants ---------------------------------------------------

ENERGY_EV = 300e3
AU_LATTICE_CONSTANT = 4.078  # Angstrom

CONVERGENCE_ANGLES = [1.5, 3, 6, 12]
UNIQUE_ANGLE = 12  # the run used to calibrate the systematic registration bias

# ordinal ramp: convergence angle is an ordered variable, so one hue light->dark
ANGLE_COLORS = {1.5: "#86b6ef", 3: "#3987e5", 6: "#1c5cab", 12: "#0d366b"}
INK = "#0b0b0b"
INK_MUTED = "#52514e"
NULL_GRAY = "#8a8984"


def electron_wavelength(energy_ev: float = ENERGY_EV) -> float:
    """Relativistic electron wavelength in Angstrom."""
    m0c2 = 510998.95  # eV
    hc = 12398.42  # eV Angstrom
    return hc / np.sqrt(energy_ev * (2 * m0c2 + energy_ev))


# --- projected geometry -----------------------------------------------------


def hexagonal_basis(d_nn: float = 1.0) -> np.ndarray:
    """Projected net of an fcc [111] zone: triangular, nearest columns ``d_nn`` apart."""
    return np.array([[d_nn, 0.0], [d_nn / 2, d_nn * np.sqrt(3) / 2]])


def square_basis(d_nn: float = 1.0) -> np.ndarray:
    """Projected net of an fcc [100] zone: square, nearest columns ``d_nn`` apart."""
    return np.array([[d_nn, 0.0], [0.0, d_nn]])


def wigner_seitz_vertices(basis: np.ndarray) -> np.ndarray:
    """Vertices of the Wigner-Seitz cell of a 2D lattice, in order.

    Every WS facet bisects a neighbouring lattice vector; each vertex is where
    two adjacent bisectors meet.  Both cells we need (hexagon, square) come out
    of the 8 shortest neighbours.
    """
    neigh = np.array(
        [i * basis[0] + j * basis[1] for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)]
    )
    neigh = neigh[np.argsort(np.linalg.norm(neigh, axis=1))]
    # keep the shell(s) that actually bound the cell: 4 for square, 6 for hex
    lengths = np.linalg.norm(neigh, axis=1)
    neigh = neigh[lengths <= lengths[0] * 1.01 + 1e-12]
    order = np.argsort(np.arctan2(neigh[:, 1], neigh[:, 0]))
    neigh = neigh[order]

    verts = []
    for a, b in zip(neigh, np.roll(neigh, -1, axis=0)):
        # solve  a.v = |a|^2/2,  b.v = |b|^2/2
        m = np.stack([a, b])
        rhs = 0.5 * np.array([a @ a, b @ b])
        verts.append(np.linalg.solve(m, rhs))
    return np.array(verts)


def wrap_to_cell(xy: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Fold shift vectors into the Wigner-Seitz cell.

    The reconstruction is ambiguous modulo a lattice vector, so a shift of
    ``v + R`` is the same solution as ``v``.  Registration occasionally locks
    onto a neighbouring correlation peak; without folding those rows inflate the
    magnitude tail.
    """
    frac = xy @ np.linalg.inv(basis)
    xy = (frac - np.round(frac)) @ basis
    # rounding fractional coordinates lands in the parallelogram cell, which is
    # not the WS cell for a triangular net -- fix the corners explicitly
    shifts = np.array([i * basis[0] + j * basis[1] for i in (-1, 0, 1) for j in (-1, 0, 1)])
    cand = xy[:, None, :] - shifts[None, :, :]
    best = np.argmin(np.linalg.norm(cand, axis=2), axis=1)
    return cand[np.arange(len(xy)), best]


def uniform_cell_sample(n: int, basis: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample uniformly over the Wigner-Seitz cell (the fully non-unique null)."""
    frac = rng.uniform(-0.5, 0.5, size=(int(n), 2))
    return wrap_to_cell(frac @ basis, basis)


def uniform_cell_radii(basis: np.ndarray, n: int = 400_000, seed: int = 0) -> np.ndarray:
    """Sorted radii under the null -- the reference the data is compared against."""
    pts = uniform_cell_sample(n, basis, np.random.default_rng(seed))
    return np.sort(np.linalg.norm(pts, axis=1))


# --- samples ----------------------------------------------------------------


@dataclass
class Sample:
    """One physical particle: where its CSVs live and what its projected lattice is."""

    key: str
    label: str
    zone: str
    kind: str  # CSV suffix
    two_particles: bool
    d: int
    d_min: float  # spacing of the first allowed ZOLZ reflection, Angstrom
    unit_name: str
    basis_fn: object = field(repr=False)

    @property
    def alpha_critical(self) -> float:
        """Convergence semi-angle (mrad) at which adjacent diffracted disks touch."""
        return 1e3 * electron_wavelength() / (2 * self.d_min)

    def basis(self, scale: float = 1.0) -> np.ndarray:
        return self.basis_fn(scale)


_A = AU_LATTICE_CONSTANT

SAMPLES = {
    "single_np": Sample(
        key="single_np",
        label="single NP [111]",
        zone="111",
        kind="always",
        two_particles=False,
        d=10,
        d_min=_A / np.sqrt(8),  # {220}
        unit_name=r"$d_{\mathrm{nn}} = a/\sqrt{2}$",
        basis_fn=hexagonal_basis,
    ),
    "two_upper": Sample(
        key="two_upper",
        label="two NPs, upper [111]",
        zone="111",
        kind="upper",
        two_particles=True,
        d=20,
        d_min=_A / np.sqrt(8),  # {220}
        unit_name=r"$d_{\mathrm{nn}} = a/\sqrt{2}$",
        basis_fn=hexagonal_basis,
    ),
    "two_lower": Sample(
        key="two_lower",
        label="two NPs, lower [100]",
        zone="100",
        kind="lower",
        two_particles=True,
        d=20,
        d_min=_A / 2,  # {200}
        unit_name=r"$d_{\mathrm{nn}} = a/2$",
        basis_fn=square_basis,
    ),
}


# --- paths and loading ------------------------------------------------------

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = dict(gpts=64, batch_size=8, num_iters=30, dose=1e15, pixels=256)


def run_dir(sample: Sample, conv_angle, on_aC: bool, **kw) -> str:
    p = {**DEFAULTS, **kw}
    name = (
        f"results_conv{_fmt_angle(conv_angle)}_d{sample.d}_gpts{p['gpts']}"
        f"_dose{p['dose']:.1e}_batchsize{p['batch_size']}_iters{p['num_iters']}"
        f"_pixels{p['pixels']}_on_aC{on_aC}"
    )
    if sample.two_particles:
        name = "two_particles_" + name
    return os.path.join(DATA_ROOT, name)


def csv_path(sample: Sample, conv_angle, on_aC: bool, **kw) -> str:
    p = {**DEFAULTS, **kw}
    fname = (
        f"shifts_conv{_fmt_angle(conv_angle)}_batchsize{p['batch_size']}"
        f"_iters{p['num_iters']}_gpts{p['gpts']}_{sample.kind}.csv"
    )
    return os.path.join(run_dir(sample, conv_angle, on_aC, **kw), fname)


def _fmt_angle(conv_angle) -> str:
    """Match the directory naming: 1.5 stays '1.5', 3.0 becomes '3'."""
    return f"{conv_angle:g}"


def load_run(sample: Sample, conv_angle, on_aC: bool = False, clean: bool = True, **kw) -> pd.DataFrame:
    """Raw shifts for one (sample, convergence angle, substrate) run, in pixels."""
    df = pd.read_csv(csv_path(sample, conv_angle, on_aC, **kw))
    df = df.rename(columns={"shift_x_0": "dx_px", "shift_y_0": "dy_px"})
    df["sample"] = sample.key
    df["conv_angle"] = float(conv_angle)
    df["on_aC"] = on_aC
    return clean_run(df) if clean else df


def clean_run(df: pd.DataFrame, dup_tol: float = 0.01) -> pd.DataFrame:
    """Drop rows that are not independent reconstructions.

    The CSVs are appended to rather than overwritten between runs, which leaves
    two artefacts: a self-registration row (exactly zero shift at ncc = 1) in the
    12 mrad files, and seeds that were re-run and landed on the same answer.
    Seeds that repeat with a *different* answer are kept -- those are genuine
    independent samples, since the reconstruction is not seed-deterministic.
    """
    self_reg = (df["dx_px"] == 0) & (df["dy_px"] == 0) & (df["ncc_peak"] > 0.999)
    df = df[~self_reg]

    keep, seen = [], []
    for row in df.itertuples():
        if any(r == row.rngi and np.hypot(x - row.dx_px, y - row.dy_px) < dup_tol
               for r, x, y in seen):
            continue
        seen.append((row.rngi, row.dx_px, row.dy_px))
        keep.append(row.Index)
    return df.loc[keep].reset_index(drop=True)


# --- calibration ------------------------------------------------------------


def calibrate_offset(sample: Sample, on_aC: bool = False, tol: float = 0.05, **kw):
    """Systematic registration bias, measured from the unique (12 mrad) run.

    Replaces the hand-tuned ``correction_x``/``correction_y`` constants: at
    12 mrad the solution is unique, so whatever shift survives is instrumental
    and common to every run of the same configuration.  Raises if that run is
    not in fact deterministic, so the assumption is checked rather than trusted.
    """
    df = load_run(sample, UNIQUE_ANGLE, on_aC, **kw)
    xy = df[["dx_px", "dy_px"]].to_numpy()
    offset = np.median(xy, axis=0)
    spread = np.percentile(np.linalg.norm(xy - offset, axis=1), 95)
    if spread > tol:
        raise ValueError(
            f"{sample.key} on_aC={on_aC}: the {UNIQUE_ANGLE} mrad run scatters by "
            f"{spread:.3f} px (> {tol} px), so it cannot calibrate the offset."
        )
    return offset, spread


def fit_cell_scale(sample: Sample, on_aC: bool = False, angles=(1.5, 3), **kw) -> float:
    """Pixels per nearest-column spacing, fit against the fully non-unique runs.

    Under the null the median radius is a fixed fraction of the cell size, so
    matching the observed median fixes the scale.  Used instead of the magic
    ``4.2`` in the old notebooks, which drifts (4.0 / 4.18 / 4.2) between cells
    and cannot be right for both fields of view.
    """
    offset, _ = calibrate_offset(sample, on_aC, **kw)
    radii = []
    for angle in angles:
        df = load_run(sample, angle, on_aC, **kw)
        radii.append(np.linalg.norm(df[["dx_px", "dy_px"]].to_numpy() - offset, axis=1))
    observed = np.median(np.concatenate(radii))
    null_median = np.median(uniform_cell_radii(sample.basis(1.0)))
    return float(observed / null_median)


def estimate_cell_scale_from_png(png_path: str, object_gpts: int = 256) -> float:
    """Independent scale estimate: lattice period read off a reconstruction PNG.

    Does not assume anything about uniqueness, unlike :func:`fit_cell_scale`.
    The PNG is a rendered figure, so the image area is located first and the
    measured period is converted back to object pixels.

    Only meaningful when the field of view holds a single lattice -- for the
    two-particle images the strongest peak mixes both orientations.
    """
    from PIL import Image

    rgba = np.asarray(Image.open(png_path).convert("L"), dtype=float)
    mask = rgba < 250
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    img = rgba[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    img = img[:, : img.shape[1] * 8 // 10]  # drop the colorbar strip
    img = img - img.mean()

    power = np.abs(np.fft.fftshift(np.fft.fft2(img * np.hanning(img.shape[0])[:, None]
                                               * np.hanning(img.shape[1])[None, :]))) ** 2
    cy, cx = np.array(power.shape) // 2
    ky, kx = np.mgrid[: power.shape[0], : power.shape[1]]
    radius = np.hypot(ky - cy, kx - cx)
    power[radius < 8] = 0
    peak = np.unravel_index(np.argmax(power), power.shape)
    period_render_px = power.shape[0] / radius[peak]
    return period_render_px * object_gpts / img.shape[0]


# --- derived quantities and statistics --------------------------------------


def add_derived(df: pd.DataFrame, sample: Sample, offset, scale: float) -> pd.DataFrame:
    """Offset-corrected, cell-folded shifts in units of the nearest-column spacing."""
    xy = df[["dx_px", "dy_px"]].to_numpy() - np.asarray(offset)
    basis = sample.basis(scale)
    folded = wrap_to_cell(xy, basis)

    out = df.copy()
    out["wrapped"] = np.linalg.norm(folded - xy, axis=1) > 1e-9
    out["dx"] = folded[:, 0] / scale
    out["dy"] = folded[:, 1] / scale
    out["r"] = np.hypot(out["dx"], out["dy"])
    out["theta_deg"] = np.degrees(np.arctan2(out["dy"], out["dx"]))
    return out


def rayleigh_test(theta_deg, r=None, r_min: float = 0.05):
    """Test directional uniformity.  Points at the origin have no direction."""
    theta = np.radians(np.asarray(theta_deg))
    if r is not None:
        theta = theta[np.asarray(r) > r_min]
    n = len(theta)
    if n < 5:
        return n, np.nan, np.nan
    rbar = np.hypot(np.cos(theta).mean(), np.sin(theta).mean())
    z = n * rbar**2
    # Greenwood-Durand approximation, good to ~1e-3 for n >= 10
    p = np.exp(-z) * (1 + (2 * z - z**2) / (4 * n) - (24 * z - 132 * z**2 + 76 * z**3 - 9 * z**4) / (288 * n**2))
    return n, z, float(np.clip(p, 0, 1))


def summarize(df: pd.DataFrame, sample: Sample, unique_below: float = 0.05) -> dict:
    """One row of the summary table for a single run."""
    r = df["r"].to_numpy()
    n_dir, z, p = rayleigh_test(df["theta_deg"], r)
    null = uniform_cell_radii(sample.basis(1.0))
    ks = np.max(np.abs(np.searchsorted(null, np.sort(r)) / len(null)
                       - (np.arange(len(r)) + 1) / len(r))) if len(r) else np.nan
    return {
        "sample": sample.label,
        "on_aC": bool(df["on_aC"].iloc[0]),
        "conv_angle": float(df["conv_angle"].iloc[0]),
        "N": len(r),
        "median_r": float(np.median(r)),
        "p90_r": float(np.percentile(r, 90)),
        "unique_frac": float((r < unique_below).mean()),
        "wrapped_frac": float(df["wrapped"].mean()),
        "rayleigh_n": n_dir,
        "rayleigh_p": p,
        "ks_vs_null": float(ks),
    }


def build(sample: Sample, on_aC: bool = False, angles=None, scale=None, **kw):
    """Load every convergence angle for one sample, calibrated and folded."""
    angles = CONVERGENCE_ANGLES if angles is None else angles
    offset, _ = calibrate_offset(sample, on_aC, **kw)
    if scale is None:
        scale = fit_cell_scale(sample, on_aC, **kw)
    runs = {}
    for angle in angles:
        try:
            df = load_run(sample, angle, on_aC, **kw)
        except FileNotFoundError:
            continue
        runs[angle] = add_derived(df, sample, offset, scale)
    return runs, offset, scale


# --- plotting ---------------------------------------------------------------


def style_axes(ax):
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d5d4d0")
    ax.xaxis.label.set_color(INK_MUTED)
    ax.yaxis.label.set_color(INK_MUTED)


def _square_off(ax, verts):
    """Shared panel geometry, so every cell in the grid is drawn to one scale."""
    lim = np.abs(verts).max() * 1.28
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _blank_panel(ax, verts):
    _square_off(ax, verts)
    ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center", va="center",
            fontsize=8, color="#b9b8b4")


def plot_cell_scatter(ax, df, sample: Sample, color: str, unique_below: float = 0.05,
                      disks_overlap: bool = False, min_seeds: int = 5):
    """One panel of the primary figure: seeds as points inside the unit cell.

    Runs with fewer than ``min_seeds`` seeds say nothing about a distribution, so
    they are left blank rather than inviting the eye to read one dot as a result.
    """
    verts = wigner_seitz_vertices(sample.basis(1.0))
    if len(df) < min_seeds:
        _blank_panel(ax, verts)
        return
    if disks_overlap:
        ax.set_facecolor("#eef3fb")
    closed = np.vstack([verts, verts[:1]])
    ax.plot(closed[:, 0], closed[:, 1], color="#c8c7c3", lw=1.0, zorder=1)
    ax.plot([0], [0], marker="+", color="#a9a8a4", ms=7, mew=1.2, zorder=2)

    # shrink the marks as the panel fills so overlapping seeds stay countable
    size = float(np.clip(300 / np.sqrt(len(df)), 9, 34))
    ax.scatter(
        df["dx"], df["dy"],
        s=size, facecolor=color, alpha=0.75,
        edgecolor="white", linewidth=0.7, zorder=3,
    )
    _square_off(ax, verts)

    frac = (df["r"] < unique_below).mean()
    ax.text(0.02, 0.98, f"N = {len(df)}", transform=ax.transAxes,
            ha="left", va="top", fontsize=7, color=INK_MUTED)
    ax.text(0.5, 0.005, f"{frac:.0%} unique", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=7.5,
            color=INK if frac > 0.5 else INK_MUTED,
            fontweight="bold" if frac > 0.5 else "normal")


def plot_ecdf(ax, runs, sample: Sample, label_null: bool = True):
    """Magnitude distributions without binning, against the two limiting cases."""
    if label_null:
        null = uniform_cell_radii(sample.basis(1.0))
        ax.plot(null, np.linspace(0, 1, len(null)), color=NULL_GRAY, ls="--", lw=1.6,
                zorder=1, label="non-unique limit")
    for angle, df in runs.items():
        r = np.sort(df["r"].to_numpy())
        y = np.arange(1, len(r) + 1) / len(r)
        ax.step(np.concatenate([[0], r]), np.concatenate([[0], y]), where="post",
                color=ANGLE_COLORS[angle], lw=2.0, zorder=3, label=f"{angle:g} mrad")
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1.02)
    style_axes(ax)


def plot_rose(ax, df, color: str, nbins: int = 16, r_min: float = 0.05):
    """Angular histogram -- the explicit check that direction carries no information."""
    theta = np.radians(df.loc[df["r"] > r_min, "theta_deg"].to_numpy())
    edges = np.linspace(-np.pi, np.pi, nbins + 1)
    counts, _ = np.histogram(theta, bins=edges)
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
           color=color, edgecolor="white", linewidth=0.8, alpha=0.85)
    if len(theta):
        ax.plot(np.linspace(-np.pi, np.pi, 200),
                np.full(200, len(theta) / nbins), color=NULL_GRAY, ls="--", lw=1.2)
    ax.set_yticklabels([])
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.tick_params(colors=INK_MUTED, labelsize=7, pad=0)
    ax.grid(color="#e6e5e1", lw=0.6)
    ax.set_axisbelow(True)
