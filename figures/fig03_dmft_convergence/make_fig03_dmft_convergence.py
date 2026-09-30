"""
For rmse_vs_iter_nn_qmc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_NN_NPZ = HERE / "data" / "NN_DMFT_U5p0_b20.npz"
DEFAULT_QMC_NPZ = HERE / "data" / "QMC_DMFT_U5p0_b20.npz"
DEFAULT_OUT = HERE / "output" / "fig03_dmft_convergence.pdf"

COLOR_NN = "#FF0000"
COLOR_QMC = "#03A2FE"
COLOR_TOL = "#888888"


def configure_prb_style() -> None:
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "axes.labelsize": 9,
            "font.size": 9,
            "legend.fontsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "axes.linewidth": 1.0,
        }
    )


def load_err_g(npz_path: Path) -> tuple[np.ndarray, float | None]:
    data = np.load(npz_path, allow_pickle=True)
    err_g = data.get("err_G", np.array([], dtype=float)).astype(float)

    tol = None
    meta = data.get("meta", None)
    if meta is not None:
        try:
            tol = meta.item().get("tol", None)
        except Exception:
            tol = None

    return err_g, tol


def plot_rmse_vs_iter(
    nn_npz_path: Path = DEFAULT_NN_NPZ,
    qmc_npz_path: Path = DEFAULT_QMC_NPZ,
    out_path: Path = DEFAULT_OUT,
) -> tuple[plt.Figure, plt.Axes]:
    err_nn, tol_nn = load_err_g(nn_npz_path)
    err_qmc, tol_qmc = load_err_g(qmc_npz_path)

    iters_nn = np.arange(1, len(err_nn) + 1)
    iters_qmc = np.arange(1, len(err_qmc) + 1)

    fig, ax = plt.subplots(figsize=(3.4, 2.9))

    if len(err_nn):
        ax.plot(
            iters_nn,
            err_nn,
            color=COLOR_NN,
            lw=1.0,
            marker="o",
            ms=4,
            markerfacecolor=COLOR_NN,
            markeredgewidth=0.0,
            label="NN",
        )

    if len(err_qmc):
        ax.plot(
            iters_qmc,
            err_qmc,
            color=COLOR_QMC,
            lw=1.0,
            marker="s",
            ms=4,
            markerfacecolor=COLOR_QMC,
            markeredgewidth=0.0,
            label="QMC",
        )

    tol = tol_qmc if tol_qmc is not None else tol_nn
    if tol is not None:
        ax.axhline(
            tol,
            color=COLOR_TOL,
            lw=1.0,
            ls="--",
            label=rf"$\varepsilon_{{\rm tol}}={tol:g}$",
        )

    ax.set_xlabel(r"DMFT iteration $n$")
    ax.set_ylabel(r"$\mathrm{RMSE}\bigl[G(\tau)\bigr]$")

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=6))
    ax.yaxis.set_major_formatter(mpl.ticker.LogFormatterSciNotation())
    ax.yaxis.set_minor_locator(
        mpl.ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12)
    )
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    all_iters = (
        np.concatenate([iters_nn, iters_qmc])
        if (len(iters_nn) and len(iters_qmc))
        else (iters_nn if len(iters_nn) else iters_qmc)
    )
    if len(all_iters):
        max_iter = int(all_iters.max())
        step = max(1, max_iter // 8)
        ax.set_xticks(np.arange(1, max_iter + 1, step))

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")
    ax.grid(False)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper right",
        framealpha=0.0,
        edgecolor="none",
        handlelength=2.0,
        labelspacing=0.3,
        borderpad=0.3,
    )

    fig.subplots_adjust(left=0.20, right=0.95, bottom=0.18, top=0.96)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600)
    print(f"Saved -> {out_path}")

    return fig, ax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fig03_dmft_convergence.pdf.")
    parser.add_argument("--nn", type=Path, default=DEFAULT_NN_NPZ, help="Local NN NPZ path")
    parser.add_argument("--qmc", type=Path, default=DEFAULT_QMC_NPZ, help="Local QMC NPZ path")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output PDF path")
    parser.add_argument("--show", action="store_true", help="Show the figure after saving")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_prb_style()
    fig, _ = plot_rmse_vs_iter(nn_npz_path=args.nn, qmc_npz_path=args.qmc, out_path=args.out)
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
