"""Generate fig10_pca_training_coverage.pdf from local PCA projection CSV files.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "output" / "fig10_pca_training_coverage.pdf"
TRAINING_CSV = HERE / "data" / "pca_g0_training_projection_first10_raw_independent.csv"
TRAJECTORY_CSV = HERE / "data" / "pca_g0_dmft_trajectories_first10_raw_independent.csv"
REPORT_TXT = HERE / "data" / "pca_g0_first10_raw_independent_report.txt"


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{amsmath}",
            "font.size": 15,
            "axes.labelsize": 15,
            "legend.fontsize": 13,
            "xtick.labelsize": 13.5,
            "ytick.labelsize": 13.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "axes.linewidth": 1.5,
            "lines.linewidth": 1.5,
            "lines.markersize": 5.5,
        }
    )


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input file(s): " + ", ".join(missing))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_explained_variance(report_path: Path) -> tuple[float, float]:
    text = report_path.read_text()
    pc1 = re.search(r"PC1 explained variance\s*=\s*([0-9.]+)%", text)
    pc2 = re.search(r"PC2 explained variance\s*=\s*([0-9.]+)%", text)
    if not pc1 or not pc2:
        return 93.48348024, 6.25013962
    return float(pc1.group(1)), float(pc2.group(1))


def load_training_projection(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv_rows(path)
    pc1 = np.array([float(row["PC1"]) for row in rows], dtype=float)
    pc2 = np.array([float(row["PC2"]) for row in rows], dtype=float)
    return pc1, pc2


def load_trajectory_rows(path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for row in read_csv_rows(path):
        rows.append(
            {
                "U": float(row["U"]),
                "beta": float(row["beta"]),
                "iteration": int(row["iteration"]),
                "PC1": float(row["PC1"]),
                "PC2": float(row["PC2"]),
            }
        )
    return rows


def plot_fig10(
    training_csv: Path,
    trajectory_csv: Path,
    report_path: Path,
    out_path: Path,
    annotate_all: bool = False,
) -> None:
    configure_matplotlib()
    require_files([training_csv, trajectory_csv, report_path])

    train_pc1, train_pc2 = load_training_projection(training_csv)
    trajectory_rows = load_trajectory_rows(trajectory_csv)
    pc1_variance, pc2_variance = parse_explained_variance(report_path)

    beta_values = sorted({float(row["beta"]) for row in trajectory_rows})

    fig, ax = plt.subplots(figsize=(6.2, 5.0))

    ax.scatter(
        train_pc1,
        train_pc2,
        s=13,
        color="0.60",
        alpha=0.24,
        marker="o",
        edgecolors="none",
        label=r"$G_0$ Training Samples",
        zorder=1,
    )

    markers = ["o", "s", "^", "D", "v", "P", "X"]
    color_map = plt.get_cmap("tab10")

    for beta_index, beta in enumerate(beta_values):
        selected = sorted(
            [row for row in trajectory_rows if np.isclose(float(row["beta"]), float(beta))],
            key=lambda row: int(row["iteration"]),
        )
        if not selected:
            continue

        pc1 = np.array([float(row["PC1"]) for row in selected], dtype=float)
        pc2 = np.array([float(row["PC2"]) for row in selected], dtype=float)
        iterations = np.array([int(row["iteration"]) for row in selected], dtype=int)

        marker = markers[beta_index % len(markers)]
        color = color_map(beta_index % 10)

        ax.plot(
            pc1,
            pc2,
            linestyle="-",
            marker=marker,
            fillstyle="none",
            markeredgewidth=1.15,
            color=color,
            label=rf"$\beta t={float(beta):g}$",
            zorder=3 + beta_index,
        )

        ax.scatter([pc1[0]], [pc2[0]], marker=marker, s=60, color=color, zorder=10)
        ax.scatter(
            [pc1[-1]],
            [pc2[-1]],
            marker="*",
            s=108,
            color=color,
            edgecolors="black",
            linewidths=0.45,
            zorder=11,
        )

        if annotate_all:
            annotation_indices = range(len(iterations))
        else:
            annotation_indices = [0, len(iterations) - 1]

        for local_index in annotation_indices:
            ax.annotate(
                str(iterations[local_index]),
                (pc1[local_index], pc2[local_index]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )

    ax.set_xlabel(rf"PC1 ({pc1_variance:.1f}\% Dataset Variance)")
    ax.set_ylabel(rf"PC2 ({pc2_variance:.1f}\% Dataset Variance)")
    ax.legend(frameon=False, loc="best", ncol=1)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fig10_pca_training_coverage.pdf.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output PDF path. Default: {DEFAULT_OUT}",
    )
    parser.add_argument(
        "--annotate-all-iterations",
        action="store_true",
        help="Label every trajectory point instead of only the first and last point.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_fig10(
        training_csv=TRAINING_CSV,
        trajectory_csv=TRAJECTORY_CSV,
        report_path=REPORT_TXT,
        out_path=args.out,
        annotate_all=bool(args.annotate_all_iterations),
    )


if __name__ == "__main__":
    main()
