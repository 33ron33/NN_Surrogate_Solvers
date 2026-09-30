"""Generate fig04_spinodal_beta25.pdf"""

from pathlib import Path
import glob
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl


HERE = Path(__file__).resolve().parent
OUTFILE = HERE / "output" / "fig04_spinodal_beta25.pdf"
BETA = 25.0
UMIN = 4.55
UMAX = 4.95
N_AVG = 3

mpl.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 9.0,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
    }
)

SOLVERS = {
    "QMC": {
        "label": "QMC",
        "dir": HERE / "data" / "qmc",
        "prefix": "QMC",
        "color": "#03A2FE",
        "face": "#3EB8FF8F",
    },
    "NN500": {
        "label": r"NN ($N_\mathrm{train}=500$)",
        "dir": HERE / "data" / "nn_500",
        "prefix": "NN",
        "color": "#FF0000",
        "face": "#FF41418D",
    },
    "NN1600": {
        "label": r"NN ($N_\mathrm{train}=1600$)",
        "dir": HERE / "data" / "nn_1600",
        "prefix": "NN",
        "color": "#2CA02C",
        "face": "#2CA02C8D",
    },
}

LAB_X = {"a": 0.04, "b": 0.04}
LAB_Y = {"a": 0.96, "b": 0.10}

ARROW_Y_UC1 = 0.85
ARROW_Y_UC2 = 0.25
ARROW_X_OFFSET_UC1 = 0.120
ARROW_X_OFFSET_UC2 = -0.10
ARROW_HALF_SPAN = 0.05
ARROW_FONTSIZE = 8.0
ARROW_LW = 1.0
ARROW_TEXT_GAP = 0.001
ARROW_MARKER_SIZE = 4.0
ARROW_MARKER_EW = 1.0

MARKER = {"METAL": "o", "INS": "s"}
LINESTYLE = {"METAL": "-", "INS": "--"}
VERTICAL_LINESTYLE = {"Uc2": "-", "Uc1": "--"}


def discover(solver_key, branch, beta):
    solver = SOLVERS[solver_key]
    pattern = solver["dir"] / f"{solver['prefix']}_{branch}_DMFT_U*_b{int(beta)}.npz"
    return sorted(glob.glob(str(pattern)))


def load_slope_point(path, beta):
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as exc:
        print(f"  [WARN] {os.path.basename(path)}: {exc}")
        return None

    meta = {}
    if "meta" in data:
        try:
            meta = data["meta"].item()
        except Exception:
            pass

    u_val = float(meta.get("U", np.nan))
    if np.isnan(u_val):
        try:
            u_val = float(os.path.basename(path).split("_U")[1].split("_b")[0].replace("p", "."))
        except Exception:
            return None

    im_sigma0 = float(meta.get("ImSigma0_final", np.nan))
    im_sigma1 = float(meta.get("ImSigma1_final", np.nan))

    if np.isnan(im_sigma0) or np.isnan(im_sigma1):
        if "Sigma_iw_conv" in data:
            sigma = np.asarray(data["Sigma_iw_conv"])
            mid = len(sigma) // 2
            im_sigma0 = float(sigma[mid].imag)
            im_sigma1 = float(sigma[mid + 1].imag)
        elif "ImSigma0" in data:
            im_sigma0 = float(np.asarray(data["ImSigma0"])[-1])
            im_sigma1 = float(np.asarray(data.get("ImSigma1", [np.nan]))[-1])

    denom = 2.0 * np.pi / beta
    slope = (
        (im_sigma1 - im_sigma0) / denom
        if np.isfinite(im_sigma0) and np.isfinite(im_sigma1)
        else np.nan
    )
    return {
        "U": u_val,
        "slope": slope,
        "converged": bool(meta.get("converged", True)),
    }


def load_slope_branch(solver_key, branch, beta, umin, umax):
    rows = [
        row
        for path in discover(solver_key, branch, beta)
        if (row := load_slope_point(path, beta)) is not None
    ]
    if not rows:
        return None

    u_vals = np.array([row["U"] for row in rows])
    slopes = np.array([row["slope"] for row in rows])
    converged = np.array([row["converged"] for row in rows], dtype=bool)
    idx = np.argsort(u_vals)
    u_vals, slopes, converged = u_vals[idx], slopes[idx], converged[idx]
    mask = (u_vals >= umin) & (u_vals <= umax)
    if not mask.any():
        return None
    return {"U": u_vals[mask], "slope": slopes[mask], "conv": converged[mask]}


def d_gm(u_val, beta, sigma_iw, g_iw):
    return 0.25 + float(np.sum(sigma_iw * g_iw).real) / (float(u_val) * float(beta))


def g_dyson(g0_iw, sigma_iw):
    return 1.0 / (1.0 / g0_iw - sigma_iw)


def load_d_point(path, n_avg=3):
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as exc:
        print(f"  [WARN] {os.path.basename(path)}: {exc}")
        return None

    meta = data["meta"].item() if "meta" in data else {}
    u_val = float(meta.get("U", np.nan))
    beta = float(meta.get("beta", np.nan))
    converged = bool(meta.get("converged", False))

    have_sigma = (
        "Sigma_iw_iters" in data
        and data["Sigma_iw_iters"].ndim == 2
        and data["Sigma_iw_iters"].shape[0] > 0
    )
    have_g = (
        "G_iw_iters" in data
        and data["G_iw_iters"].ndim == 2
        and data["G_iw_iters"].shape[0] > 0
    )

    if not have_sigma:
        return {"U": u_val, "beta": beta, "converged": converged, "D_GM": np.nan}

    sigma_iters = data["Sigma_iw_iters"]
    if have_g:
        g_iters = data["G_iw_iters"]
    elif "G0_iw_iters" in data:
        g0_iters = data["G0_iw_iters"]
        n_iter = min(g0_iters.shape[0], sigma_iters.shape[0])
        g_iters = np.array([g_dyson(g0_iters[i], sigma_iters[i]) for i in range(n_iter)])
        sigma_iters = sigma_iters[:n_iter]
    else:
        return {"U": u_val, "beta": beta, "converged": converged, "D_GM": np.nan}

    n_iter = min(g_iters.shape[0], sigma_iters.shape[0])
    k = min(n_avg, n_iter)
    vals = [
        d_gm(u_val, beta, sigma_iters[i], g_iters[i])
        for i in range(n_iter - k, n_iter)
    ]
    return {
        "U": u_val,
        "beta": beta,
        "converged": converged,
        "D_GM": float(np.mean(vals)),
    }


def load_d_branch(solver_key, branch, beta, umin, umax, n_avg=3):
    rows = [
        row
        for path in discover(solver_key, branch, beta)
        if (row := load_d_point(path, n_avg=n_avg)) is not None
    ]
    if not rows:
        return None

    rows.sort(key=lambda row: row["U"])
    rows = [row for row in rows if umin <= row["U"] <= umax]
    if not rows:
        return None

    return {
        "U": np.array([row["U"] for row in rows]),
        "D_GM": np.array([row["D_GM"] for row in rows]),
        "conv": np.array([row["converged"] for row in rows], dtype=bool),
    }


def first_sign_change(u_vals, y_vals, direction, crossing):
    u_ordered = np.array(u_vals)
    y_ordered = np.array(y_vals)
    if direction == "desc":
        u_ordered, y_ordered = u_ordered[::-1], y_ordered[::-1]

    for i in range(len(u_ordered) - 1):
        y0, y1 = y_ordered[i], y_ordered[i + 1]
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue
        hit = (
            crossing == "neg_to_pos"
            and y0 < 0
            and y1 >= 0
        ) or (
            crossing == "pos_to_neg"
            and y0 > 0
            and y1 <= 0
        )
        if hit:
            mid = 0.5 * (u_ordered[i] + u_ordered[i + 1])
            err = 0.5 * abs(u_ordered[i + 1] - u_ordered[i])
            return float(mid), float(err)

    dy = np.abs(np.diff(y_ordered))
    if len(dy) and np.any(np.isfinite(dy)):
        i = int(np.nanargmax(dy))
        mid = 0.5 * (u_ordered[i] + u_ordered[i + 1])
        err = 0.5 * abs(u_ordered[i + 1] - u_ordered[i])
        return float(mid), float(err)
    return np.nan, np.nan


def extract_spinodals(d_metal, d_ins):
    uc2 = uc2_err = np.nan
    if d_metal is not None:
        mask = d_metal["conv"] & np.isfinite(d_metal["slope"])
        if mask.sum() >= 2:
            uc2, uc2_err = first_sign_change(
                d_metal["U"][mask],
                d_metal["slope"][mask],
                direction="asc",
                crossing="neg_to_pos",
            )

    uc1 = uc1_err = np.nan
    if d_ins is not None:
        mask = d_ins["conv"] & np.isfinite(d_ins["slope"])
        if mask.sum() >= 2:
            uc1, uc1_err = first_sign_change(
                d_ins["U"][mask],
                d_ins["slope"][mask],
                direction="desc",
                crossing="pos_to_neg",
            )

    return uc1, uc1_err, uc2, uc2_err


def style_ax(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")
    ax.grid(False)


def draw_spinodal_lines(ax, spinodals):
    for label, idx_v, idx_e in [("Uc1", 0, 1), ("Uc2", 2, 3)]:
        ls = VERTICAL_LINESTYLE[label]
        for solver_key, vals in spinodals.items():
            value, err = vals[idx_v], vals[idx_e]
            if not np.isfinite(value):
                continue
            color = SOLVERS[solver_key]["color"]
            ax.axvline(value, color=color, ls=ls, lw=1.2, zorder=4, alpha=0.85)
            if np.isfinite(err):
                ax.axvspan(value - err, value + err, color=color, alpha=0.12, lw=0, zorder=3)


def panel_label(ax, letter):
    ax.text(
        LAB_X[letter],
        LAB_Y[letter],
        f"({letter})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )


def plot_kw(solver_key, branch):
    solver = SOLVERS[solver_key]
    return {
        "ls": LINESTYLE[branch],
        "marker": MARKER[branch],
        "ms": 3.0,
        "lw": 1.2,
        "markerfacecolor": solver["face"],
        "markeredgecolor": solver["color"],
        "markeredgewidth": 1.0,
        "color": solver["color"],
    }


def draw_sweep_arrows(ax, spinodals):
    uc1_vals = [spinodals[key][0] for key in spinodals if np.isfinite(spinodals[key][0])]
    uc2_vals = [spinodals[key][2] for key in spinodals if np.isfinite(spinodals[key][2])]
    if not uc1_vals or not uc2_vals:
        return

    uc1_ctr = float(np.mean(uc1_vals)) + ARROW_X_OFFSET_UC1
    uc2_ctr = float(np.mean(uc2_vals)) + ARROW_X_OFFSET_UC2
    ylo, yhi = ax.get_ylim()

    def ydata(frac):
        return ylo + frac * (yhi - ylo)

    col_arrow = "0.30"
    col_text = "0.25"
    col_marker = "0.30"

    y1 = ydata(ARROW_Y_UC1)
    y1_txt = y1 + ARROW_TEXT_GAP
    ax.annotate(
        "",
        xy=(uc1_ctr - ARROW_HALF_SPAN, y1),
        xytext=(uc1_ctr + ARROW_HALF_SPAN, y1),
        arrowprops={"arrowstyle": "-|>", "lw": ARROW_LW, "color": col_arrow},
        annotation_clip=False,
        zorder=6,
    )
    ax.plot(
        uc1_ctr + ARROW_HALF_SPAN - 0.1428,
        y1_txt + 0.0012,
        marker="s",
        ms=ARROW_MARKER_SIZE,
        markerfacecolor="white",
        markeredgecolor=col_marker,
        markeredgewidth=ARROW_MARKER_EW,
        ls="none",
        zorder=8,
        transform=ax.transData,
    )
    ax.text(
        uc1_ctr + ARROW_HALF_SPAN - 0.005,
        y1_txt,
        r"$U_{c1}$ down sweep",
        ha="right",
        va="bottom",
        fontsize=ARROW_FONTSIZE,
        color=col_text,
        transform=ax.transData,
        zorder=7,
    )

    y2 = ydata(ARROW_Y_UC2)
    y2_txt = y2 + ARROW_TEXT_GAP
    ax.annotate(
        "",
        xy=(uc2_ctr + ARROW_HALF_SPAN, y2),
        xytext=(uc2_ctr - ARROW_HALF_SPAN, y2),
        arrowprops={"arrowstyle": "-|>", "lw": ARROW_LW, "color": col_arrow},
        annotation_clip=False,
        zorder=6,
    )
    ax.plot(
        uc2_ctr - ARROW_HALF_SPAN - 0.005,
        y2_txt + 0.001,
        marker="o",
        ms=ARROW_MARKER_SIZE,
        markerfacecolor="white",
        markeredgecolor=col_marker,
        markeredgewidth=ARROW_MARKER_EW,
        ls="none",
        zorder=8,
        transform=ax.transData,
    )
    ax.text(
        uc2_ctr - ARROW_HALF_SPAN + 0.005,
        y2_txt,
        r"$U_{c2}$ up sweep",
        ha="left",
        va="bottom",
        fontsize=ARROW_FONTSIZE,
        color=col_text,
        transform=ax.transData,
        zorder=7,
    )


def make_combined_plot():
    solver_keys = list(SOLVERS.keys())
    branches = ["METAL", "INS"]
    slope_data = {}
    d_data = {}
    spinodals = {}

    for solver_key in solver_keys:
        for branch in branches:
            slope_data[(solver_key, branch)] = load_slope_branch(
                solver_key, branch, BETA, UMIN, UMAX
            )
            d_data[(solver_key, branch)] = load_d_branch(
                solver_key, branch, BETA, UMIN, UMAX, n_avg=N_AVG
            )

        uc1, uc1_err, uc2, uc2_err = extract_spinodals(
            slope_data[(solver_key, "METAL")],
            slope_data[(solver_key, "INS")],
        )
        spinodals[solver_key] = (uc1, uc1_err, uc2, uc2_err)
        print(
            f"  {solver_key:<8}  Uc1={uc1:.4f} +/- {uc1_err:.4f}  "
            f"Uc2={uc2:.4f} +/- {uc2_err:.4f}"
        )

    for label, idx in [("Uc1", 0), ("Uc2", 2)]:
        vals = [spinodals[key][idx] for key in solver_keys if np.isfinite(spinodals[key][idx])]
        if vals:
            print(f"  Mean {label} = {np.mean(vals):.4f}  (spread = {max(vals) - min(vals):.4f})")

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(3.4, 5.2), sharex=True)
    fig.subplots_adjust(left=0.20, right=0.96, bottom=0.10, top=0.97, hspace=0.0)
    ax_top.tick_params(labelbottom=False)

    ax_top.axhline(0.0, color="0.50", ls=":", lw=1.0, zorder=0)
    for solver_key in solver_keys:
        for branch in branches:
            data = slope_data[(solver_key, branch)]
            if data is None:
                continue
            u_vals, slopes, converged = data["U"], data["slope"], data["conv"]
            kw = plot_kw(solver_key, branch)
            if converged.any():
                ax_top.plot(u_vals[converged], slopes[converged], **kw)
            if (~converged).any():
                ax_top.plot(
                    u_vals[~converged],
                    slopes[~converged],
                    marker="x",
                    ls="none",
                    color=SOLVERS[solver_key]["color"],
                    ms=6,
                    mew=1.5,
                    alpha=0.35,
                )

    draw_spinodal_lines(ax_top, spinodals)
    style_ax(ax_top)
    ax_top.set_ylabel(
        r"$\partial_\omega\,\mathrm{Im}\,\Sigma(i\omega_n)\big|_{\omega\to 0}$",
        fontsize=9,
    )
    panel_label(ax_top, "a")

    for solver_key in solver_keys:
        for branch in branches:
            data = d_data[(solver_key, branch)]
            if data is None:
                continue
            u_vals, d_vals, converged = data["U"], data["D_GM"], data["conv"]
            kw = plot_kw(solver_key, branch)
            if converged.any():
                ax_bot.plot(u_vals[converged], d_vals[converged], **kw)
            if (~converged).any():
                ax_bot.plot(
                    u_vals[~converged],
                    d_vals[~converged],
                    marker="x",
                    ls="none",
                    color=SOLVERS[solver_key]["color"],
                    ms=6,
                    mew=1.5,
                    alpha=0.35,
                )

    draw_spinodal_lines(ax_bot, spinodals)
    ax_bot.set_xlim(UMIN, UMAX)
    ax_bot.margins(x=0.04)
    ax_bot.autoscale_view()
    draw_sweep_arrows(ax_bot, spinodals)
    style_ax(ax_bot)
    ax_bot.set_xlabel(r"$U/t$")
    ax_bot.set_ylabel(r"$D = \langle n_{\uparrow} n_{\downarrow} \rangle$", fontsize=10)
    panel_label(ax_bot, "b")

    fig.savefig(OUTFILE, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved -> {OUTFILE.name}")


def main():
    make_combined_plot()


if __name__ == "__main__":
    main()
