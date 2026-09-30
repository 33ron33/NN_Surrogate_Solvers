"""Generate fig02_green_functions.pdf"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "output" / "fig02_green_functions.pdf"

METAL_QMC_NPZ = HERE / "data" / "QMC_DMFT_U2p5_b6.npz"
METAL_NN_NPZ = HERE / "data" / "NN_DMFT_U2p5_b6.npz"
INS_QMC_NPZ = HERE / "data" / "QMC_DMFT_U9p0_b30.npz"
INS_NN_NPZ = HERE / "data" / "NN_DMFT_U9p0_b30.npz"

C_QMC_G = "#03A2FE"
C_NN_G = "#CA0000"
C_QMC_G0 = "#FFB006"
C_NN_G0 = "#000000"

SPINE_LW = 1.2
FIG_W, FIG_H = 3.575 * 2, 3.0 * 2


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{amsmath}",
            "font.size": 18,
            "axes.labelsize": 17,
            "legend.fontsize": 16,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "axes.linewidth": 1.2,
            "lines.linewidth": 1.0,
            "lines.markersize": 10,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.minor.width": 0.8,
            "ytick.minor.width": 0.8,
            "xtick.major.size": 5.0,
            "ytick.major.size": 5.0,
            "xtick.minor.size": 2.5,
            "ytick.minor.size": 2.5,
            "savefig.bbox": "tight",
        }
    )
    mpl.rcParams["axes.xmargin"] = 0.05
    mpl.rcParams["axes.ymargin"] = 0.05


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input file(s): " + ", ".join(missing))


def interp_to_tau(
    values: np.ndarray,
    source_tau: np.ndarray,
    target_tau: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    source_tau = np.asarray(source_tau, dtype=float)
    target_tau = np.asarray(target_tau, dtype=float)

    if values.shape == target_tau.shape:
        return values
    if values.shape != source_tau.shape:
        raise ValueError(
            "Cannot align curve: values shape "
            f"{values.shape} does not match source tau shape {source_tau.shape}."
        )
    return np.interp(target_tau, source_tau, values)


def extract(qmc_npz: Path, nn_npz: Path) -> dict:
    q = np.load(qmc_npz, allow_pickle=True)
    n = np.load(nn_npz, allow_pickle=True)

    tau = np.asarray(q["tau_vals"], dtype=float)
    nn_tau = np.asarray(n["tau_vals"], dtype=float) if "tau_vals" in n.files else tau
    meta = q["meta"].item()
    beta = float(meta["beta"])
    interaction = float(meta["U"])

    g_raw_0 = None
    if "G_tau_raw_iters" in q.files and q["G_tau_raw_iters"].shape[0] > 0:
        g_raw_0 = q["G_tau_raw_iters"][0]

    g_qmc_0 = q["G_tau_iters"][0]
    g_nn_0 = interp_to_tau(n["G_tau_iters"][0], nn_tau, tau)
    g0_qmc_0 = q["G0_tau_iters"][0]
    g0_nn_0 = (
        interp_to_tau(n["G0_tau_iters"][0], nn_tau, tau)
        if "G0_tau_iters" in n.files
        else g0_qmc_0
    )

    g_raw_c = q["G_tau_raw_conv"]
    g_qmc_c = q["G_tau_conv"]
    g_nn_c = interp_to_tau(n["G_tau_conv"], nn_tau, tau)
    g0_qmc_c = q["G0_tau_conv"]
    g0_nn_c = (
        interp_to_tau(n["G0_tau_conv"], nn_tau, tau)
        if "G0_tau_conv" in n.files
        else g0_qmc_c
    )

    return {
        "tau": tau,
        "beta": beta,
        "U": interaction,
        "single": (g_raw_0, g_qmc_0, g_nn_0, g0_qmc_0, g0_nn_0),
        "conv": (g_raw_c, g_qmc_c, g_nn_c, g0_qmc_c, g0_nn_c),
    }


def print_metrics(label: str, regime: str, g_qmc: np.ndarray, g_nn: np.ndarray) -> None:
    diff = np.asarray(g_nn, dtype=float) - np.asarray(g_qmc, dtype=float)
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    print(f"[{label} - {regime}]  MSE = {mse:.6e},  RMSE = {rmse:.6e}")


def style_ax(ax: plt.Axes, col: int) -> None:
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LW)
        spine.set_color("black")

    if col == 0:
        ax.spines["right"].set_visible(False)
        ax.tick_params(which="both", right=False)
    else:
        ax.spines["left"].set_linewidth(SPINE_LW)
        ax.tick_params(which="both", left=False, labelleft=False, right=True)


def plot_panel(
    ax: plt.Axes,
    col: int,
    tau: np.ndarray,
    beta: float,
    g_raw: np.ndarray | None,
    g_qmc: np.ndarray,
    g_nn: np.ndarray,
    g0_qmc: np.ndarray,
    g0_nn: np.ndarray,
    skip_g0: bool = False,
) -> None:
    x = tau / beta

    if g_raw is not None:
        ax.plot(
            x,
            -g_raw,
            ".",
            color=C_QMC_G,
            alpha=0.7,
            markersize=5,
            rasterized=True,
            label=r"QMC $G(\tau)$",
        )

    ax.plot(x, -g_nn, "--", lw=2.0, color=C_NN_G, label=r"NN $G(\tau)$")

    if not skip_g0:
        ax.plot(
            x,
            -g0_qmc,
            "-.",
            lw=2.0,
            color=C_QMC_G0,
            label=r"QMC $G_0(\tau)$",
        )
        ax.plot(x, -g0_nn, ":", lw=2.0, color=C_NN_G0, label=r"NN $G_0(\tau)$")

    style_ax(ax, col)


def plot_metal_insulator_single_vs_converged_grid(
    metal_qmc_npz: Path,
    metal_nn_npz: Path,
    ins_qmc_npz: Path,
    ins_nn_npz: Path,
    out_path: Path,
) -> tuple[plt.Figure, np.ndarray]:
    configure_matplotlib()
    require_files([metal_qmc_npz, metal_nn_npz, ins_qmc_npz, ins_nn_npz])

    met = extract(metal_qmc_npz, metal_nn_npz)
    ins = extract(ins_qmc_npz, ins_nn_npz)

    _, g_qmc_0_m, g_nn_0_m, _, _ = met["single"]
    _, g_qmc_c_m, g_nn_c_m, _, _ = met["conv"]
    _, g_qmc_0_i, g_nn_0_i, _, _ = ins["single"]
    _, g_qmc_c_i, g_nn_c_i, _, _ = ins["conv"]

    print("\n--- RMSE / MSE Diagnostics ---")
    print(f"Metal point:      U = {met['U']:.2f},  beta = {met['beta']:.1f}")
    print_metrics("Metal", "single-step", g_qmc_0_m, g_nn_0_m)
    print_metrics("Metal", "converged", g_qmc_c_m, g_nn_c_m)
    print(f"\nInsulator point:  U = {ins['U']:.2f},  beta = {ins['beta']:.1f}")
    print_metrics("Insulator", "single-step", g_qmc_0_i, g_nn_0_i)
    print_metrics("Insulator", "converged", g_qmc_c_i, g_nn_c_i)

    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, FIG_H), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax in (axes[0], axes[1]):
        ax.tick_params(labelbottom=False)

    plot_panel(axes[0], 0, met["tau"], met["beta"], *met["single"], skip_g0=True)
    plot_panel(axes[1], 1, ins["tau"], ins["beta"], *ins["single"], skip_g0=True)
    plot_panel(axes[2], 0, met["tau"], met["beta"], *met["conv"])
    plot_panel(axes[3], 1, ins["tau"], ins["beta"], *ins["conv"])

    for ax, label in zip(axes, ["(a)", "(b)", "(c)", "(d)"]):
        ax.text(
            0.06,
            0.97,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=17,
            fontweight="bold",
        )

    fig.canvas.draw()
    top_box = axes[0].get_position()
    bottom_box = axes[2].get_position()
    y_single_center = 0.534 * (top_box.y0 + top_box.y1)
    y_conv_center = 0.55 * (bottom_box.y0 + bottom_box.y1)
    x_label = top_box.x0 - 0.10

    for y_pos, text in [
        (y_single_center, "Single-step DMFT"),
        (y_conv_center, "Converged DMFT"),
    ]:
        fig.text(
            x_label,
            y_pos,
            text,
            rotation=90,
            va="center",
            ha="center",
            fontsize=20,
            transform=fig.transFigure,
        )

    for ax in axes[2:4]:
        ax.set_xlabel(r"$\tau/\beta$")
    axes[0].set_ylabel(r"$-G(\tau)$")
    axes[2].set_ylabel(r"$-G(\tau),\,-G_0(\tau)$")

    handles, labels = axes[3].get_legend_handles_labels()
    axes[1].legend(
        handles,
        labels,
        ncol=1,
        loc="upper center",
        framealpha=0.0,
        facecolor="none",
        edgecolor="none",
        handlelength=2.2,
        borderpad=0.3,
    )

    for ax in axes:
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    fig.subplots_adjust(
        left=0.14,
        right=0.98,
        bottom=0.10,
        top=0.95,
        wspace=0.0,
        hspace=0.04,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600)
    print(f"Saved PRB-style 2x2 grid to {out_path}")
    return fig, axes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fig02_green_functions.pdf.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output PDF path. Default: {DEFAULT_OUT}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_metal_insulator_single_vs_converged_grid(
        metal_qmc_npz=METAL_QMC_NPZ,
        metal_nn_npz=METAL_NN_NPZ,
        ins_qmc_npz=INS_QMC_NPZ,
        ins_nn_npz=INS_NN_NPZ,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
