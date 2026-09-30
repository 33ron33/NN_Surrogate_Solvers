# Interaction strength vs Temperature plot for co-existance region.

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "output" / "fig05_phase_diagram.pdf"

# These points are extracted for Metal and insulator sweeps for double occupancy and
# derivative of imaginary self energy at \omega=0. The points are extracted
# from the QMC and NN solvers for the Mott transition.

QMC_UC = [
    {"beta": 20.0, "uc1": 4.6385, "uc1_err": 0.0077, "uc2": 4.7462, "uc2_err": 0.0077},
    {"beta": 23.0, "uc1": 4.6564, "uc1_err": 0.0103, "uc2": 4.8000, "uc2_err": 0.0103},
    {"beta": 25.0, "uc1": 4.6538, "uc1_err": 0.0077, "uc2": 4.8385, "uc2_err": 0.0077},
    {"beta": 28.5, "uc1": 4.6487, "uc1_err": 0.0128, "uc2": 4.8795, "uc2_err": 0.0128},
    {"beta": 32.0, "uc1": 4.6564, "uc1_err": 0.0103, "uc2": 4.9436, "uc2_err": 0.0103},
    {"beta": 40.0, "uc1": 4.6460, "uc1_err": 0.0154, "uc2": 5.0460, "uc2_err": 0.0154},
    {"beta": 50.0, "uc1": 4.6370, "uc1_err": 0.0357, "uc2": 5.1308, "uc2_err": 0.0357},
]

NN_UC = [
    {"beta": 20.0, "uc1": 4.6077, "uc1_err": 0.0077, "uc2": 4.7092, "uc2_err": 0.0077},
    {"beta": 23.0, "uc1": 4.6538, "uc1_err": 0.0103, "uc2": 4.7769, "uc2_err": 0.0103},
    {"beta": 25.0, "uc1": 4.6846, "uc1_err": 0.0077, "uc2": 4.8385, "uc2_err": 0.0077},
    {"beta": 28.5, "uc1": 4.6744, "uc1_err": 0.0128, "uc2": 5.0770, "uc2_err": 0.0128},
    {"beta": 32.0, "uc1": 4.7385, "uc1_err": 0.0103, "uc2": 5.1692, "uc2_err": 0.0103},
    {"beta": 40.0, "uc1": 4.7690, "uc1_err": 0.0154, "uc2": 5.4000, "uc2_err": 0.0154},
    {"beta": 50.0, "uc1": 4.6500, "uc1_err": 0.0128, "uc2": 5.2077, "uc2_err": 0.0128},
]

STYLE = {
    "dpi_fig": 1200,
    "dpi_save": 1200,
    "figsize": (6.8, 5.5),
    "xlim": (4.40, 5.60),
    "ylim": (0.00, 0.06),
    "xticks": (4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6),
    "yticks": (0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06),
    "x_minor": 0.05,
    "y_minor": 0.005,
    "x_fmt": "%.1f",
    "y_fmt": "%.2f",
    "xlabel": r"Interaction strength $(U/t)$",
    "ylabel": r"Temperature $(T/t)$",
    "title": r"Mott Metal--Insulator Transition: NN Prediction (500)",
    "spine_lw": 1.6,
    "major_len": 7,
    "minor_len": 4,
    "major_w": 1.3,
    "minor_w": 1.0,
    "col_qmc": "#03A2FE",
    "col_nn": "#FF0000",
    "shade_gray": "0.84",
    "shade_extrap": True,
    "extrap_Tmin": 0.00,
    "extrap_Tmax": 0.04,
    "extrap_label": r"Extrapolated region",
    "shade_alpha": 0.75,
    "legend_loc": "upper right",
    "legend_frame": False,
    "legend_fontsize": 13.5,
    "zorder_shade": 0,
    "zorder_qmc1": 40,
    "zorder_qmc2": 41,
    "zorder_nn1": 60,
    "zorder_nn2": 61,
    "capsize": 5,
    "capthick": 1.4,
    "elinewidth": 1.4,
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "axes.labelsize": 19,
            "font.size": 20.5,
            "legend.fontsize": 14,
            "xtick.labelsize": 19,
            "ytick.labelsize": 19,
        }
    )
    mpl.rcParams["axes.xmargin"] = 0.03
    mpl.rcParams["axes.ymargin"] = 0.05


def set_axes_style(ax: plt.Axes, style: dict) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(style["spine_lw"])
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=style["major_len"], width=style["major_w"])
    ax.tick_params(which="minor", length=style["minor_len"], width=style["minor_w"])
    ax.grid(False)


def unpack_phase_points(points: list[dict[str, float]]) -> tuple[np.ndarray, ...]:
    betas = np.array([d["beta"] for d in points], dtype=float)
    return (
        1.0 / betas,
        np.array([d["uc1"] for d in points], dtype=float),
        np.array([d["uc2"] for d in points], dtype=float),
        np.array([d["uc1_err"] for d in points], dtype=float),
        np.array([d["uc2_err"] for d in points], dtype=float),
    )


def overlay_qmc_uc(ax: plt.Axes, qmc_uc: list[dict[str, float]], style: dict) -> None:
    temperature, uc1, uc2, uc1_err, uc2_err = unpack_phase_points(qmc_uc)

    ax.errorbar(
        uc1,
        temperature,
        xerr=uc1_err,
        marker="D",
        linestyle="None",
        markersize=7.0,
        markerfacecolor="white",
        markeredgecolor=style["col_qmc"],
        markeredgewidth=1.0,
        ecolor=style["col_qmc"],
        elinewidth=style["elinewidth"],
        capsize=style["capsize"],
        capthick=style["capthick"],
        alpha=1.0,
        label=r"QMC Solver $U_{c1}$",
        zorder=style["zorder_qmc1"],
    )
    ax.errorbar(
        uc2,
        temperature,
        xerr=uc2_err,
        marker="D",
        linestyle="None",
        markersize=7.0,
        markerfacecolor=style["col_qmc"],
        markeredgecolor=style["col_qmc"],
        markeredgewidth=1.0,
        ecolor=style["col_qmc"],
        elinewidth=style["elinewidth"],
        capsize=style["capsize"],
        capthick=style["capthick"],
        alpha=1.0,
        label=r"QMC Solver $U_{c2}$",
        zorder=style["zorder_qmc2"],
    )


def overlay_nn_uc(ax: plt.Axes, nn_uc: list[dict[str, float]], style: dict) -> None:
    temperature, uc1, uc2, uc1_err, uc2_err = unpack_phase_points(nn_uc)

    ax.errorbar(
        uc1,
        temperature,
        xerr=uc1_err,
        marker="o",
        linestyle="None",
        markersize=7.0,
        markerfacecolor="white",
        markeredgecolor=style["col_nn"],
        markeredgewidth=1.0,
        ecolor=style["col_nn"],
        elinewidth=style["elinewidth"],
        capsize=style["capsize"],
        capthick=style["capthick"],
        alpha=0.85,
        label=r"NN Solver $U_{c1}$",
        zorder=style["zorder_nn1"],
    )
    ax.errorbar(
        uc2,
        temperature,
        xerr=uc2_err,
        marker="o",
        linestyle="None",
        markersize=7.0,
        markerfacecolor=style["col_nn"],
        markeredgecolor=style["col_nn"],
        markeredgewidth=1.0,
        ecolor=style["col_nn"],
        elinewidth=style["elinewidth"],
        capsize=style["capsize"],
        capthick=style["capthick"],
        alpha=0.85,
        label=r"NN Solver $U_{c2}$",
        zorder=style["zorder_nn2"],
    )


def ordered_legend(ax: plt.Axes, style: dict) -> None:
    handles, labels = ax.get_legend_handles_labels()
    by_label = {label: handle for handle, label in zip(handles, labels)}
    label_order = [
        r"QMC Solver $U_{c1}$",
        r"QMC Solver $U_{c2}$",
        r"NN Solver $U_{c1}$",
        r"NN Solver $U_{c2}$",
        style["extrap_label"],
    ]
    ordered_labels = [label for label in label_order if label in by_label]
    ordered_handles = [by_label[label] for label in ordered_labels]

    ax.legend(
        ordered_handles,
        ordered_labels,
        loc=style["legend_loc"],
        frameon=style["legend_frame"],
        fontsize=style["legend_fontsize"],
    )


def plot_fig_5(out_path: Path, show_title: bool = False) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=STYLE["figsize"], dpi=STYLE["dpi_fig"])

    ax.set_xlim(*STYLE["xlim"])
    ax.set_ylim(*STYLE["ylim"])
    ax.set_xticks(STYLE["xticks"])
    ax.set_yticks(STYLE["yticks"])
    ax.xaxis.set_minor_locator(MultipleLocator(STYLE["x_minor"]))
    ax.yaxis.set_minor_locator(MultipleLocator(STYLE["y_minor"]))
    ax.xaxis.set_major_formatter(FormatStrFormatter(STYLE["x_fmt"]))
    ax.yaxis.set_major_formatter(FormatStrFormatter(STYLE["y_fmt"]))
    ax.set_xlabel(STYLE["xlabel"])
    ax.set_ylabel(STYLE["ylabel"])

    if show_title:
        ax.set_title(STYLE["title"], fontsize=20, pad=12)

    set_axes_style(ax, STYLE)

    if STYLE["shade_extrap"]:
        ax.axhspan(
            STYLE["extrap_Tmin"],
            STYLE["extrap_Tmax"],
            color=STYLE["shade_gray"],
            alpha=STYLE["shade_alpha"],
            zorder=STYLE["zorder_shade"],
            label=STYLE["extrap_label"],
        )

    overlay_qmc_uc(ax, QMC_UC, STYLE)
    overlay_nn_uc(ax, NN_UC, STYLE)
    ordered_legend(ax, STYLE)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=STYLE["dpi_save"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the standalone Fig. 5 Uc1/Uc2 phase-points PDF."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output PDF path. Default: {DEFAULT_OUT}",
    )
    parser.add_argument(
        "--show-title",
        action="store_true",
        help="Include the plot title. The paper-style default leaves it off.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_fig_5(args.out, show_title=args.show_title)


if __name__ == "__main__":
    main()
