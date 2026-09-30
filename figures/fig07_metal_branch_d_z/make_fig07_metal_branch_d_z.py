"""Generate fig07_metal_branch_d_z.pdf"""

from pathlib import Path
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

try:
    from triqs.gf import MeshImFreq, MeshImTime, Gf, Fourier
    from triqs.gf.tools import make_zero_tail

    HAS_TRIQS = True
except ImportError:
    HAS_TRIQS = False
    warnings.warn("TRIQS not found; using manual DFT for G_tau -> G_loc.", stacklevel=2)


HERE = Path(__file__).resolve().parent
OUTFILE = HERE / "output" / "fig07_metal_branch_d_z.pdf"
BETA = 25.0
BRANCH = "METAL"

DATASETS = [
    (
        "QMC",
        HERE / "data" / "qmc",
        [
            (4.40, "QMC_METAL_DMFT_U4p4_b25.npz"),
            (4.50, "QMC_METAL_DMFT_U4p492307692307692_b25.npz"),
            (4.60, "QMC_METAL_DMFT_U4p6000000000000005_b25.npz"),
            (4.70, "QMC_METAL_DMFT_U4p6923076923076925_b25.npz"),
            (4.74, "QMC_METAL_DMFT_U4p7384615384615385_b25.npz"),
            (4.82, "QMC_METAL_DMFT_U4p815384615384615_b25.npz"),
            (4.90, "QMC_METAL_DMFT_U4p892307692307693_b25.npz"),
            (5.00, "QMC_METAL_DMFT_U5p0_b25.npz"),
        ],
    ),
    (
        "500",
        HERE / "data" / "nn_500",
        [
            (4.40, "NN_METAL_DMFT_U4p4_b25.npz"),
            (4.50, "NN_METAL_DMFT_U4p492307692307692_b25.npz"),
            (4.60, "NN_METAL_DMFT_U4p6000000000000005_b25.npz"),
            (4.70, "NN_METAL_DMFT_U4p6923076923076925_b25.npz"),
            (4.74, "NN_METAL_DMFT_U4p7384615384615385_b25.npz"),
            (4.82, "NN_METAL_DMFT_U4p815384615384615_b25.npz"),
            (4.90, "NN_METAL_DMFT_U4p892307692307693_b25.npz"),
            (5.00, "NN_METAL_DMFT_U5p0_b25.npz"),
        ],
    ),
    (
        "1600",
        HERE / "data" / "nn_1600",
        [
            (4.40, "NN_METAL_DMFT_U4p4_b25.npz"),
            (4.50, "NN_METAL_DMFT_U4p492307692307692_b25.npz"),
            (4.60, "NN_METAL_DMFT_U4p6000000000000005_b25.npz"),
            (4.70, "NN_METAL_DMFT_U4p6923076923076925_b25.npz"),
            (4.74, "NN_METAL_DMFT_U4p7384615384615385_b25.npz"),
            (4.82, "NN_METAL_DMFT_U4p815384615384615_b25.npz"),
            (4.90, "NN_METAL_DMFT_U4p892307692307693_b25.npz"),
            (5.00, "NN_METAL_DMFT_U5p0_b25.npz"),
        ],
    ),
]

FIG_W = 3.4
FIG_H = 4.8
COL_QMC = "#03A2FE"
NN_COLORS = {
    "500": "#FF0000",
    "1000": "#FF8000",
    "1600": "#1C831C",
}
NN_PALETTE_FALLBACK = ["#FF0000", "#FF8000", "#2CA02C"]
MARKER = "o"
QMC_MS = 5.0
NN_MS = 5.0

mpl.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.size": 10.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.5,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "0.65",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "figure.dpi": 150,
    }
)


def nn_edge_color(label):
    if str(label) in NN_COLORS:
        return NN_COLORS[str(label)]
    idx = hash(str(label)) % len(NN_PALETTE_FALLBACK)
    return NN_PALETTE_FALLBACK[idx]


def gtau_triqs(g_tau, beta, n_iw):
    n_tau = len(g_tau)
    iw_m = MeshImFreq(beta=float(beta), S="Fermion", n_iw=int(n_iw))
    tau_m = MeshImTime(beta=float(beta), S="Fermion", n_tau=int(n_tau))
    gt = Gf(mesh=tau_m, target_shape=[1, 1])
    gt.data[:, 0, 0] = g_tau.astype(np.complex128)
    gl = Gf(mesh=iw_m, target_shape=[1, 1])
    gl << Fourier(gt)
    m0 = make_zero_tail(gl, n_moments=2)
    m0[0, 0, 0] = 1.0
    m0[1, 0, 0] = 0.0
    gl.fit_tail(m0)
    return gl.data[:, 0, 0].copy()


def gtau_manual(g_tau, iw_vals, beta):
    tau = np.linspace(0, beta, len(g_tau))
    dtau = beta / (len(g_tau) - 1)
    return dtau * (np.exp(1j * np.outer(iw_vals, tau)) @ g_tau.astype(np.complex128))


def gtau_to_gloc(g_tau, iw_vals, beta, n_iw):
    if HAS_TRIQS:
        return gtau_triqs(g_tau, beta, n_iw)
    return gtau_manual(g_tau, iw_vals, beta)


def compute_z(sigma_iw, iw_vals, beta):
    idx0 = len(iw_vals) // 2
    is0 = float(sigma_iw[idx0].imag)
    omega0 = np.pi / float(beta)
    denom = 1.0 - is0 / omega0
    if abs(denom) <= 1e-12:
        return float("nan")
    return float(np.clip(1.0 / denom, -2.0, 2.0))


def compute_d(sigma_iw, g_tau, iw_vals, beta, u_val, n_iw):
    g_loc = gtau_to_gloc(g_tau, iw_vals, beta, n_iw)
    return 0.25 + float(np.sum(sigma_iw * g_loc).real) / (float(u_val) * float(beta))


def load_obs(path):
    with np.load(path, allow_pickle=True) as data:
        required = ("Sigma_iw_conv", "G_tau_conv", "iw_vals", "meta")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"{path.name} missing keys: {missing}")
        meta = data["meta"].item()
        u_val = float(meta["U"])
        beta = float(meta["beta"])
        n_iw = int(meta.get("n_iw", len(data["iw_vals"]) // 2))
        sigma = np.asarray(data["Sigma_iw_conv"], np.complex128)
        g_tau = np.asarray(data["G_tau_conv"], np.float64)
        iw_vals = np.asarray(data["iw_vals"], np.complex128)

    return (
        compute_d(sigma, g_tau, iw_vals, beta, u_val, n_iw),
        compute_z(sigma, iw_vals, beta),
    )


def collect_data():
    data = {}
    for label, directory, entries in DATASETS:
        series = {}
        for display_u, filename in entries:
            path = directory / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            series[display_u] = load_obs(path)
        data[label] = series
    return data


def style_ax(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5, prune=None))


def fill_panel(ax, data, obs_idx, y_label):
    qmc = data["QMC"]
    u_q = sorted(qmc)
    y_q = [qmc[u][obs_idx] for u in u_q]
    ax.scatter(
        u_q,
        y_q,
        marker=MARKER,
        s=QMC_MS**2,
        facecolors=COL_QMC,
        edgecolors=COL_QMC,
        linewidths=0.8,
        zorder=4,
    )

    for idx, label in enumerate(["500", "1600"]):
        src = data[label]
        u_s = sorted(src)
        y_s = [src[u][obs_idx] for u in u_s]
        ax.scatter(
            u_s,
            y_s,
            marker=MARKER,
            s=NN_MS**2,
            facecolors="none",
            edgecolors=nn_edge_color(label),
            linewidths=1.3,
            zorder=6 + idx,
        )

    ax.margins(y=0.08)
    ax.set_ylabel(y_label, labelpad=4)
    style_ax(ax)


def make_figure(data):
    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(FIG_W, FIG_H),
        sharex=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.10, top=0.95, hspace=0.0)
    ax_top.tick_params(labelbottom=False)

    fill_panel(
        ax_top,
        data,
        obs_idx=0,
        y_label=r"$D = \langle \hat{n}_\uparrow \hat{n}_\downarrow \rangle$",
    )
    fill_panel(ax_bot, data, obs_idx=1, y_label=r"$Z$")

    z_lo, _ = ax_bot.get_ylim()
    ax_bot.set_ylim(z_lo, 0.42)
    ax_bot.yaxis.set_major_locator(mticker.FixedLocator([0.0, 0.1, 0.2, 0.3, 0.4]))

    fig.canvas.draw()
    top_ticks = ax_top.yaxis.get_major_ticks()
    if top_ticks:
        top_ticks[0].label1.set_visible(False)

    ax_bot.set_xlabel(r"$U/t$", labelpad=3)
    for ax, label in zip([ax_top, ax_bot], ["(a)", "(b)"]):
        ax.text(
            0.04,
            0.05,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker=MARKER,
            ls="none",
            ms=QMC_MS,
            markerfacecolor=COL_QMC,
            markeredgecolor=COL_QMC,
            label="CT-HYB",
        )
    ]
    for label in ["500", "1600"]:
        handles.append(
            Line2D(
                [0],
                [0],
                marker=MARKER,
                ls="none",
                ms=NN_MS,
                markerfacecolor="none",
                markeredgecolor=nn_edge_color(label),
                markeredgewidth=1.3,
                label=rf"NN ($N_{{\mathrm{{train}}}}={label}$)",
            )
        )

    ax_top.legend(
        handles=handles,
        loc="best",
        handletextpad=0.25,
        labelspacing=0.33,
        borderpad=0.5,
        handlelength=0.8,
        framealpha=0.0,
        fontsize=9.0,
    )

    fig.savefig(OUTFILE, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    data = collect_data()
    make_figure(data)
    print(f"Saved {OUTFILE.name}")


if __name__ == "__main__":
    main()
