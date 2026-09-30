"""Generate fig09_qmc_acceleration.pdf"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "output" / "fig09_qmc_acceleration.pdf"

METAL_QMC_NPZ = HERE / "data" / "QMC_DMFT_U9p0_b32.npz"
METAL_NN_NPZ = HERE / "data" / "NN_DMFT_U9p0_b32.npz"
METAL_QACC_NPZ = HERE / "data" / "QMCACC_DMFT_U9p0_b32.npz"
INS_QMC_NPZ = HERE / "data" / "QMC_DMFT_U4p5_b32.npz"
INS_NN_NPZ = HERE / "data" / "NN_DMFT_U4p5_b32.npz"
INS_QACC_NPZ = HERE / "data" / "QMCACC_DMFT_U4p5_b32.npz"


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "axes.labelsize": 9,
            "font.size": 14.5,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input file(s): " + ", ".join(missing))


def interp_to_tau(values: np.ndarray, source_tau: np.ndarray, target_tau: np.ndarray) -> np.ndarray:
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


def extract(qmc_npz: Path, nn_npz: Path, qacc_npz: Path) -> dict:
    q = np.load(qmc_npz, allow_pickle=True)
    n = np.load(nn_npz, allow_pickle=True)
    a = np.load(qacc_npz, allow_pickle=True)

    tau = np.asarray(q["tau_vals"], dtype=float)
    nn_tau = np.asarray(n["tau_vals"], dtype=float)
    qacc_tau = np.asarray(a["tau_vals"], dtype=float)
    meta = q["meta"].item()
    beta = float(meta["beta"])
    interaction = float(meta["U"])

    return {
        "tau": tau,
        "beta": beta,
        "U": interaction,
        "conv": (
            np.asarray(q["G_tau_raw_conv"], dtype=float),
            np.asarray(q["G_tau_conv"], dtype=float),
            interp_to_tau(n["G_tau_conv"], nn_tau, tau),
            interp_to_tau(a["G_tau_conv"], qacc_tau, tau),
        ),
    }


def print_metrics(label: str, g_qmc: np.ndarray, g_nn: np.ndarray, g_qacc: np.ndarray) -> None:
    for tag, diff in [
        ("NN vs QMC    ", np.asarray(g_nn, float) - np.asarray(g_qmc, float)),
        ("QMCACC vs QMC", np.asarray(g_qacc, float) - np.asarray(g_qmc, float)),
    ]:
        mse = float(np.mean(diff**2))
        print(f"[{label}]  {tag}  MSE={mse:.6e}  RMSE={np.sqrt(mse):.6e}")


def plot_panel(
    ax: plt.Axes,
    tau: np.ndarray,
    beta: float,
    g_raw: np.ndarray,
    g_qmc: np.ndarray,
    g_nn: np.ndarray,
    g_qacc: np.ndarray,
) -> None:
    x = tau / beta
    if g_raw is not None:
        ax.plot(x, -g_raw, ".", alpha=0.6, ms=4, color="#F8C963", label=r"QMC $G(\tau)$")
    ax.plot(x, -g_qacc, "-", lw=1.6, color="#0072B2", label=r"QMC-Acc $G(\tau)$")
    ax.plot(x, -g_nn, "--", lw=1.6, color="#FF0000", label=r"NN $G(\tau)$")

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")
    ax.minorticks_off()
    ax.grid(False)


def add_inset(
    ax: plt.Axes,
    data: dict,
    box: list[float],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xticks: list[float],
    yticks: list[float],
    tick_fontsize: float,
) -> plt.Axes:
    x_arr = np.asarray(data["tau"], float) / data["beta"]
    g_raw, _, g_nn, g_qacc = data["conv"]
    g_raw_arr = -np.asarray(g_raw, float)
    g_qacc_arr = -np.asarray(g_qacc, float)
    g_nn_arr = -np.asarray(g_nn, float)

    pad = 0.02
    mask = (x_arr >= xlim[0] - pad) & (x_arr <= xlim[1] + pad)

    axins = ax.inset_axes(box)
    axins.plot(x_arr[mask], g_raw_arr[mask], ".", alpha=0.6, ms=3, color="#F8C963")
    axins.plot(x_arr[mask], g_qacc_arr[mask], "-", lw=1.4, color="#0072B2")
    axins.plot(x_arr[mask], g_nn_arr[mask], "--", lw=1.4, color="#FF0000")

    axins.set_xlim(*xlim)
    axins.set_ylim(*ylim)
    axins.set_xticks(xticks)
    axins.set_yticks(yticks)
    axins.tick_params(labelsize=tick_fontsize, pad=1.5)
    axins.minorticks_off()
    axins.grid(False)

    for spine in axins.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("black")

    ax.indicate_inset_zoom(axins, edgecolor="0.4", lw=0.6, alpha=0.8)
    return axins


def plot_metal_insulator_stacked(
    metal_qmc_npz: Path,
    metal_nn_npz: Path,
    metal_qacc_npz: Path,
    ins_qmc_npz: Path,
    ins_nn_npz: Path,
    ins_qacc_npz: Path,
    out_path: Path,
) -> tuple[plt.Figure, np.ndarray]:
    configure_matplotlib()
    require_files(
        [
            metal_qmc_npz,
            metal_nn_npz,
            metal_qacc_npz,
            ins_qmc_npz,
            ins_nn_npz,
            ins_qacc_npz,
        ]
    )

    inset_fontsize = 9
    met = extract(metal_qmc_npz, metal_nn_npz, metal_qacc_npz)
    ins = extract(ins_qmc_npz, ins_nn_npz, ins_qacc_npz)

    _, gq_m, gn_m, ga_m = met["conv"]
    _, gq_i, gn_i, ga_i = ins["conv"]

    print("\n--- RMSE / MSE (converged) ---")
    print(f"Metal:      U={met['U']:.2f}  beta={met['beta']:.1f}")
    print_metrics("Metal", gq_m, gn_m, ga_m)
    print(f"Insulator:  U={ins['U']:.2f}  beta={ins['beta']:.1f}")
    print_metrics("Insulator", gq_i, gn_i, ga_i)

    fig, axes = plt.subplots(2, 1, figsize=(3.4, 5.0), sharex=True, sharey=True)
    axes[0].tick_params(labelbottom=False)
    fig.subplots_adjust(left=0.20, right=0.97, bottom=0.10, top=0.97, hspace=0.0)

    plot_panel(axes[0], met["tau"], met["beta"], *met["conv"])
    plot_panel(axes[1], ins["tau"], ins["beta"], *ins["conv"])

    for ax, label in zip(axes, ["(a)", "(b)"]):
        ax.text(
            0.05,
            0.97,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
        )

    axes[1].set_xlabel(r"$\tau/\beta$")
    fig.text(
        0.04,
        0.54,
        r"$-G(\tau)$",
        rotation=90,
        va="center",
        ha="center",
        fontsize=11,
        transform=fig.transFigure,
    )

    for ax in axes:
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax.margins(x=0.04, y=0.05)

    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(
        handles,
        labels,
        ncol=1,
        loc="lower center",
        bbox_to_anchor=(0.72, 1.65),
        bbox_transform=axes[1].transAxes,
        framealpha=0.0,
        facecolor="white",
        edgecolor="none",
        handlelength=2.2,
        columnspacing=0.8,
        labelspacing=0.35,
        borderpad=0.4,
        fontsize=9.0,
    )

    add_inset(
        axes[0],
        met,
        box=[0.38, 0.30, 0.46, 0.30],
        xlim=(0.25, 0.45),
        ylim=(-0.0001, 0.004),
        xticks=[0.25, 0.35, 0.45],
        yticks=[0.000, 0.002, 0.004],
        tick_fontsize=inset_fontsize,
    )
    add_inset(
        axes[1],
        ins,
        box=[0.38, 0.60, 0.46, 0.30],
        xlim=(0.25, 0.45),
        ylim=(0.020, 0.060),
        xticks=[0.25, 0.35, 0.45],
        yticks=[0.020, 0.040, 0.060],
        tick_fontsize=inset_fontsize,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600)
    print(f"Saved -> {out_path}")
    return fig, axes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fig09_qmc_acceleration.pdf.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output PDF path. Default: {DEFAULT_OUT}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_metal_insulator_stacked(
        metal_qmc_npz=METAL_QMC_NPZ,
        metal_nn_npz=METAL_NN_NPZ,
        metal_qacc_npz=METAL_QACC_NPZ,
        ins_qmc_npz=INS_QMC_NPZ,
        ins_nn_npz=INS_NN_NPZ,
        ins_qacc_npz=INS_QACC_NPZ,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
