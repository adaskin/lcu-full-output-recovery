"""Publication figures for the recovery experiments (reads data/*.npz written by lcu_recovery*.py).
All text (titles, axis labels, ticks, legend) is 14 pt at the printed size: the figures are
produced at 6.3 in width.
Run:  python3 plot_recovery_figures.py   ->  figures/fig_recovery_exact.pdf, figures/fig_recovery_noise.pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS = 14
plt.rcParams.update({
    "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS, "xtick.labelsize": FS,
    "ytick.labelsize": FS, "legend.fontsize": FS, "font.family": "serif", "mathtext.fontset": "stix",
    "font.serif": ["STIXGeneral", "DejaVu Serif"], "lines.linewidth": 1.8, "lines.markersize": 6,
    "savefig.dpi": 300, "pdf.fonttype": 42,
})
STYLE = [("svp_Phi", r"SVP $\Phi$", "o-"), ("svp_phi", r"SVP $\varphi$", "s--"),
         ("fac_Phi", r"Factorized $\Phi$", "d-"), ("fac_phi", r"Factorized $\varphi$", "x--")]
COL = ["C0", "C1", "C2", "C3"]


def make(tag, xlabel, out, logx, ticks):
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 4.6), sharey=True)
    for ax, N in zip(axs, (256, 1024)):
        d = np.load(f"data/results_{tag}_K4_N{N}.npz")
        for (key, lab, fmt), c in zip(STYLE, COL):
            ax.errorbar(d["x"], d[key], yerr=d[key + "_std"], fmt=fmt, color=c, capsize=2, label=lab)
        ax.set_yscale("log")
        if logx:
            ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"$10^{{{int(np.log10(t))}}}$" for t in ticks] if logx else [f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_title(rf"$N={N}$")
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
    axs[0].set_ylabel("Relative error")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(out)
    plt.close(fig)


os.makedirs("figures", exist_ok=True)
make("completion", "Observed fraction", "figures/fig_recovery_exact.pdf", False, [0.2, 0.4, 0.6, 0.8])
make("noise", r"Noise $\sigma$", "figures/fig_recovery_noise.pdf", True, [1e-5, 1e-4, 1e-3, 1e-2])
