#!/usr/bin/env python3
"""Generate fig08_mae_dataset_scaling.pdf"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker


HERE = Path(__file__).resolve().parent
OUTFILE = HERE / "output" / "fig08_mae_dataset_scaling.pdf"

CONFIGS = {
    25: {
        "qmc": HERE / "data" / "beta25" / "qmc",
        "500": HERE / "data" / "beta25" / "nn_500",
        "1600": HERE / "data" / "beta25" / "nn_1600",
    },
    23: {
        "qmc": HERE / "data" / "beta23" / "qmc",
        "500": HERE / "data" / "beta23" / "nn_500",
        "1600": HERE / "data" / "beta23" / "nn_1600",
    },
}

MODEL_LABELS = ["500", "1600"]
BRANCHES = ["METAL", "INS"]

BRANCH_COLOR = {
    "METAL": "#4991d8",
    "INS": "#ff725d",
}
BRANCH_HATCH = {
    "METAL": "",
    "INS": "//////",
}

mpl.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.size": 13.5,
        "axes.labelsize": 13.5,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
        "legend.fontsize": 12.5,
        "legend.framealpha": 0.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "axes.linewidth": 1.0,
        "figure.dpi": 600,
    }
)


def load_npz_dir(folder, branch):
    rows = []
    if not folder.is_dir():
        warnings.warn(f"Not found: {folder.name}", stacklevel=2)
        return pd.DataFrame(columns=["U", "Z_final", "D_final", "converged"])

    for path in sorted(folder.glob("*.npz")):
        if f"_{branch}_" not in path.name:
            continue
        try:
            with np.load(path, allow_pickle=True) as data:
                meta = data["meta"].item()
        except Exception as exc:
            warnings.warn(f"Could not read {path.name}: {exc}", stacklevel=2)
            continue

        rows.append(
            {
                "U": float(meta["U"]),
                "Z_final": float(meta.get("Z_final", np.nan)),
                "D_final": float(meta.get("D_final", np.nan)),
                "converged": bool(meta.get("converged", False)),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["U", "Z_final", "D_final", "converged"])

    df = pd.DataFrame(rows).sort_values("U").reset_index(drop=True)
    return df[df["converged"]].copy()


def mae(df_nn, df_qmc, obs):
    col = f"{obs}_final"
    merged = pd.merge(
        df_nn[["U", col]].rename(columns={col: "nn"}),
        df_qmc[["U", col]].rename(columns={col: "qmc"}),
        on="U",
        how="inner",
    )
    if len(merged) == 0:
        return np.nan
    return float(np.abs(merged["nn"] - merged["qmc"]).mean())


def compute_for_beta(beta):
    cfg = CONFIGS[beta]
    qmc = {branch: load_npz_dir(cfg["qmc"], branch) for branch in BRANCHES}
    results = {"D": {}, "Z": {}}

    for model in MODEL_LABELS:
        results["D"][model] = {}
        results["Z"][model] = {}
        for branch in BRANCHES:
            df_nn = load_npz_dir(cfg[model], branch)
            results["D"][model][branch] = mae(df_nn, qmc[branch], "D")
            results["Z"][model][branch] = mae(df_nn, qmc[branch], "Z")

    return results


def draw_panel(ax, obs_key, ylabel, res23, res25, panel_label):
    x = np.arange(len(MODEL_LABELS))
    width = 0.14
    offsets = [-1.8 * width, -0.6 * width, 0.6 * width, 1.8 * width]
    bar_defs = [
        ("METAL", 25, offsets[0]),
        ("INS", 25, offsets[1]),
        ("METAL", 23, offsets[2]),
        ("INS", 23, offsets[3]),
    ]

    all_vals = []
    for i, model in enumerate(MODEL_LABELS):
        for branch, beta, offset in bar_defs:
            val = res25[obs_key][model][branch] if beta == 25 else res23[obs_key][model][branch]
            all_vals.append(val)
            ax.bar(
                x[i] + offset,
                val,
                width,
                color=BRANCH_COLOR[branch],
                hatch=BRANCH_HATCH[branch],
                alpha=0.82,
                edgecolor="k",
                linewidth=0.5,
                zorder=3,
            )

    finite_vals = [val for val in all_vals if np.isfinite(val)]
    ymax = max(finite_vals) if finite_vals else 1.0
    beta_pad = 0.0005 if obs_key == "D" else 0.002

    for i, model in enumerate(MODEL_LABELS):
        xs_this = []
        vals_this = []
        for branch, beta, offset in bar_defs:
            val = res25[obs_key][model][branch] if beta == 25 else res23[obs_key][model][branch]
            xs_this.append(x[i] + offset)
            vals_this.append(val)

        pair1_x = 0.5 * (xs_this[0] + xs_this[1])
        pair2_x = 0.5 * (xs_this[2] + xs_this[3])
        pair1_y = max([v for v in vals_this[:2] if np.isfinite(v)] or [0.0])
        pair2_y = max([v for v in vals_this[2:] if np.isfinite(v)] or [0.0])

        ax.text(
            pair1_x,
            pair1_y + beta_pad,
            r"$\beta = 25$",
            ha="center",
            va="bottom",
            fontsize=10.0,
            color="0.15",
            clip_on=False,
        )
        ax.text(
            pair2_x,
            pair2_y + beta_pad,
            r"$\beta = 23$",
            ha="center",
            va="bottom",
            fontsize=10.0,
            color="0.15",
            clip_on=False,
        )

    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, max(yhi, ymax * 1.25))
    ax.set_ylabel(ylabel, labelpad=4)
    ax.set_xticks(x)
    ax.set_xticklabels([rf"NN-{model}" for model in MODEL_LABELS], fontsize=9)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    ax.text(
        0.02,
        0.97,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
    )


def make_figure(res23, res25):
    fig, (ax_d, ax_z) = plt.subplots(
        2,
        1,
        figsize=(4.2, 6.1),
        sharex=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.11, top=0.96, hspace=0.03)

    draw_panel(
        ax_d,
        obs_key="D",
        ylabel=r"MAE in $\langle\hat{n}_\uparrow\hat{n}_\downarrow\rangle$",
        res23=res23,
        res25=res25,
        panel_label="(a)",
    )
    draw_panel(
        ax_z,
        obs_key="Z",
        ylabel=r"MAE in $Z$",
        res23=res23,
        res25=res25,
        panel_label="(b)",
    )

    ax_d.tick_params(labelbottom=False)
    ax_z.set_xlabel(r"Training Set Size", labelpad=3)

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=BRANCH_COLOR["METAL"],
            edgecolor="k",
            hatch=BRANCH_HATCH["METAL"],
            linewidth=0.5,
            alpha=0.82,
            label="Metal",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=BRANCH_COLOR["INS"],
            edgecolor="k",
            hatch=BRANCH_HATCH["INS"],
            linewidth=0.5,
            alpha=0.82,
            label="Insulator",
        ),
    ]
    ax_d.legend(handles=handles, loc="upper right", handlelength=1.0, borderpad=0.4, labelspacing=0.3)

    fig.savefig(OUTFILE, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    res23 = compute_for_beta(23)
    res25 = compute_for_beta(25)
    make_figure(res23, res25)
    print(f"Saved {OUTFILE.name}")


if __name__ == "__main__":
    main()
