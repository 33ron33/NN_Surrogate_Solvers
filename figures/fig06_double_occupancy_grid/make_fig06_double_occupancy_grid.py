"""Generate fig06_double_occupancy_grid.pdf"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.interpolate import griddata


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "output" / "fig06_double_occupancy_grid.pdf"
METAL_NPZ = HERE / "data" / "TolG_4e-4_D_Metal_Original.npz"
INS_NPZ = HERE / "data" / "TolG_4e-4_D_Ins_Original.npz"
TRAINING_CSV = HERE / "data" / "Features_0500.csv"

QMC_UC = [
    {"beta": 20.0, "uc1": 4.6385, "uc2": 4.7462},
    {"beta": 23.0, "uc1": 4.6564, "uc2": 4.8000},
    {"beta": 25.0, "uc1": 4.6538, "uc2": 4.8385},
    {"beta": 28.5, "uc1": 4.6487, "uc2": 4.8795},
    {"beta": 32.0, "uc1": 4.6564, "uc2": 4.9436},
    {"beta": 40.0, "uc1": 4.6460, "uc2": 5.0460},
    {"beta": 50.0, "uc1": 4.6370, "uc2": 5.1500},
]


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 12.5,
            "axes.labelsize": 12.0,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "legend.fontsize": 10.0,
            "axes.linewidth": 1.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.minor.width": 0.9,
            "ytick.minor.width": 0.9,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "savefig.bbox": "tight",
        }
    )


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input file(s): " + ", ".join(missing))


def load_d_phase_grid_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    u_grid = np.asarray(data["U_grid"], dtype=float)
    beta_grid = np.asarray(data["beta_grid"], dtype=float)
    d_grid = np.asarray(data["D_grid"], dtype=float)
    return u_grid, beta_grid, d_grid


def load_unique_training_grid(features_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.genfromtxt(features_csv, delimiter=",", names=True)
    names = raw.dtype.names or ()
    if "U" not in names or "beta" not in names:
        raise ValueError("Training CSV must contain columns 'U' and 'beta'.")

    pairs = np.column_stack([np.asarray(raw["U"], dtype=float), np.asarray(raw["beta"], dtype=float)])
    pairs = np.unique(pairs, axis=0)
    order = np.lexsort((pairs[:, 0], pairs[:, 1]))
    pairs = pairs[order]
    return pairs[:, 0], pairs[:, 1]


def interp_field(
    u_grid: np.ndarray,
    t_grid: np.ndarray,
    values: np.ndarray,
    n_fine: int,
    method: str,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    uu, tt = np.meshgrid(u_grid, t_grid)
    points = np.column_stack([uu.ravel(), tt.ravel()])
    field = np.asarray(values, dtype=float).ravel()

    good = np.isfinite(field)
    if good.sum() < 5:
        return None, None, None

    u_fine = np.linspace(u_grid.min(), u_grid.max(), n_fine)
    t_fine = np.linspace(t_grid.min(), t_grid.max(), n_fine)
    ui, ti = np.meshgrid(u_fine, t_fine)
    zi = griddata(points[good], field[good], (ui, ti), method=method)
    return u_fine, t_fine, zi


def overlay_training_points(
    ax: plt.Axes,
    u_train: np.ndarray | None,
    beta_train: np.ndarray | None,
    marker: str,
    size: float,
    line_width: float,
    edge: str,
    face: str,
    alpha: float,
    zorder: int = 50,
) -> None:
    if u_train is None or beta_train is None:
        return
    ax.scatter(
        u_train,
        1.0 / beta_train,
        marker=marker,
        s=size,
        facecolors=face,
        edgecolors=edge,
        linewidths=line_width,
        alpha=alpha,
        zorder=zorder,
    )


def draw_qmc_markers(
    ax: plt.Axes,
    qmc_data: list[dict[str, float]],
    uc2_color: str,
    uc1_color: str,
    markersize: float,
    line_width: float,
    zorder: int = 10,
    return_handles: bool = False,
):
    betas = np.array([d["beta"] for d in qmc_data], dtype=float)
    uc1 = np.array([d["uc1"] for d in qmc_data], dtype=float)
    uc2 = np.array([d["uc2"] for d in qmc_data], dtype=float)
    temperature = 1.0 / betas

    h_uc1 = ax.plot(
        uc1,
        temperature,
        marker="^",
        linestyle="None",
        color=uc1_color,
        markersize=markersize,
        markerfacecolor="none",
        markeredgewidth=line_width,
        label=r"$U_{c1}$ (QMC)",
        zorder=zorder,
    )[0]
    h_uc2 = ax.plot(
        uc2,
        temperature,
        marker="o",
        linestyle="None",
        color=uc2_color,
        markersize=markersize,
        markerfacecolor="none",
        markeredgewidth=line_width,
        label=r"$U_{c2}$ (QMC)",
        zorder=zorder,
    )[0]

    if return_handles:
        return h_uc1, h_uc2
    return None


def plot_d_phase_diagram_qmc(
    metal_npz_path: Path,
    ins_npz_path: Path,
    training_csv: Path,
    out_path: Path,
    vmin: float = 0.0,
    vmax: float = 0.18,
    interpolate: bool = True,
    n_fine: int = 1000,
    interp_method: str = "cubic",
    cmap: str = "viridis",
    uc2_color: str = "#FF0000",
    uc1_color: str = "#ED03FE",
    qmc_markersize: float = 3.0,
    qmc_line_width: float = 1.0,
    show_training_points: bool = True,
    train_marker: str = "s",
    train_size: float = 10,
    train_line_width: float = 1.0,
    train_edge: str = "white",
    train_face: str = "none",
    train_alpha: float = 1.0,
    qmc_data: list[dict[str, float]] | None = None,
) -> None:
    configure_matplotlib()
    require_files([metal_npz_path, ins_npz_path, training_csv])

    if qmc_data is None:
        qmc_data = QMC_UC

    u_m, beta_m, d_m = load_d_phase_grid_npz(metal_npz_path)
    u_i, beta_i, d_i = load_d_phase_grid_npz(ins_npz_path)

    if not (np.allclose(u_m, u_i) and np.allclose(beta_m, beta_i)):
        raise ValueError("Metal and insulator grids do not match.")

    u_grid = u_m
    beta_grid = beta_m
    t_grid = 1.0 / beta_grid

    d_all = np.concatenate([d_m.ravel(), d_i.ravel()])
    d_all = d_all[np.isfinite(d_all)]
    print("Colorbar sanity check:")
    print(f"  D_min(data) = {d_all.min():.5f}")
    print(f"  D_max(data) = {d_all.max():.5f}")
    print(f"  Points below vmin={vmin:.3f}: {np.sum(d_all < vmin)}")
    print(f"  Points above vmax={vmax:.3f}: {np.sum(d_all > vmax)}")

    u_train = beta_train = None
    if show_training_points:
        u_train, beta_train = load_unique_training_grid(training_csv)
        print(f"Training points loaded: {len(u_train)} unique (U,beta) pairs from {training_csv.name}")

    fig = plt.figure(figsize=(4.6, 6.4))
    gs = gridspec.GridSpec(
        2,
        2,
        width_ratios=[1.0, 0.08],
        height_ratios=[1.0, 1.0],
        wspace=0.18,
        hspace=0.045,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top, sharey=ax_top)
    cax = fig.add_subplot(gs[:, 1])

    def draw_heatmap(ax: plt.Axes, values: np.ndarray):
        values = np.asarray(values, dtype=float)
        uu, tt = np.meshgrid(u_grid, t_grid)

        if interpolate:
            u_fine, t_fine, zi = interp_field(
                u_grid,
                t_grid,
                values,
                n_fine=n_fine,
                method=interp_method,
            )
            if zi is None:
                im = ax.pcolormesh(uu, tt, values, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                im = ax.imshow(
                    zi,
                    origin="lower",
                    extent=[u_fine.min(), u_fine.max(), t_fine.min(), t_fine.max()],
                    aspect="auto",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
        else:
            im = ax.pcolormesh(uu, tt, values, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_xlim(u_grid.min(), u_grid.max())
        ax.set_ylim(t_grid.min(), t_grid.max())
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        return im

    im_top = draw_heatmap(ax_top, d_m)
    draw_heatmap(ax_bot, d_i)

    if show_training_points and u_train is not None:
        for ax in (ax_top, ax_bot):
            overlay_training_points(
                ax,
                u_train,
                beta_train,
                marker=train_marker,
                size=train_size,
                line_width=train_line_width,
                edge=train_edge,
                face=train_face,
                alpha=train_alpha,
            )

    for ax in (ax_top, ax_bot):
        ax.axhline(0.0115, color="white", linestyle=":", linewidth=0.8, alpha=0.8, zorder=5)

    h_uc1, h_uc2 = draw_qmc_markers(
        ax_top,
        qmc_data=qmc_data,
        uc2_color=uc2_color,
        uc1_color=uc1_color,
        markersize=qmc_markersize,
        line_width=qmc_line_width,
        return_handles=True,
    )
    draw_qmc_markers(
        ax_bot,
        qmc_data=qmc_data,
        uc2_color=uc2_color,
        uc1_color=uc1_color,
        markersize=qmc_markersize,
        line_width=qmc_line_width,
    )

    handles = [h_uc2, h_uc1]
    if show_training_points and u_train is not None:
        handles.append(
            Line2D(
                [0],
                [0],
                marker=train_marker,
                linestyle="None",
                markerfacecolor="none" if train_face == "none" else train_face,
                markeredgecolor=train_edge,
                markeredgewidth=train_line_width,
                markersize=4,
                label=r"NN Training Grid",
            )
        )

    legend = ax_top.legend(handles=handles, loc="upper right", framealpha=0.0, handlelength=1.0, borderpad=0.25)
    for text in legend.get_texts():
        text.set_color("#FFFFFFBF")

    ax_bot.set_xlabel(r"$U/t$")
    ax_top.set_ylabel(r"$T/t$")
    ax_bot.set_ylabel(r"$T/t$")

    y_major = [0.02, 0.05, 0.10, 0.15, 0.20]
    for ax in (ax_top, ax_bot):
        ax.set_yticks(y_major)
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.01))
        ax.tick_params(axis="y", which="minor", left=True, right=True, length=2.0, width=0.8)

    ax_top.set_ylim(0.02, 0.20)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    x_ticks = [2.01, 3.01, 4.01, 5.01, 6.01, 7.01, 8.01]
    ax_bot.set_xticks(x_ticks)
    ax_bot.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, _: f"{int(x)}" if x == int(x) else f"{x:.1f}")
    )
    ax_bot.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))

    ax_top.text(
        0.02,
        0.98,
        r"(a)",
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        color="black",
        alpha=1.0,
        fontweight="bold",
    )
    ax_bot.text(
        0.02,
        0.98,
        r"(b)",
        transform=ax_bot.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        color="black",
        alpha=1.0,
        fontweight="bold",
    )

    cbar = fig.colorbar(im_top, cax=cax)
    cbar.ax.tick_params(direction="in", width=1.0, length=4)
    cbar.set_label(r"$D=\langle n_\uparrow n_\downarrow\rangle$")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600)
    plt.close(fig)
    print(f"Saved: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fig06_double_occupancy_grid.pdf.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output PDF path. Default: {DEFAULT_OUT}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_d_phase_diagram_qmc(
        metal_npz_path=METAL_NPZ,
        ins_npz_path=INS_NPZ,
        training_csv=TRAINING_CSV,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
