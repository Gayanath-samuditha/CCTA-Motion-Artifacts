#plot_cnr_histogram.py
#plots CNR distribution across 1000 ImageCAS cases
#caps x-axis at 12, annotates median and min as vertical lines without legend
#usage: python plot_cnr_histogram.py [--csv PATH] [--out PATH] [--cap FLOAT]

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

DEFAULT_CSV = "/wrk-vakka/users/mohogaya/caisa/results/s1_p92_cnr.csv"
DEFAULT_OUT = "/wrk-vakka/users/mohogaya/caisa/results/figures/cnr_histogram_capped.png"
DEFAULT_CAP = 12.0

def plot_cnr(csv_path, out_path, x_cap):
    df = pd.read_csv(csv_path)
    cnr = df[df["status"] == "ok"]["cnr"].dropna()

    n_capped = (cnr > x_cap).sum()
    median   = cnr.median()
    cnr_min  = cnr.min()

    print(f"Total ok: {len(cnr)} | Capped (>{x_cap}): {n_capped}")
    print(f"Mean={cnr.mean():.2f} Std={cnr.std():.2f} Median={median:.2f} Min={cnr_min:.2f} Max={cnr.max():.2f}")

    cnr_plot = cnr[cnr <= x_cap]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(cnr_plot, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.5)

    #vertical lines without legend entries
    ax.axvline(median,  color="crimson",    linestyle="--", linewidth=1.5)
    ax.axvline(cnr_min, color="darkorange", linestyle=":",  linewidth=1.5)

    #inline labels directly on the lines
    ylim = ax.get_ylim()
    ax.text(median  + 0.05, ylim[1] * 0.92, f"Median = {median:.2f}",
            color="crimson",    fontsize=13, va="top")
    ax.text(cnr_min + 0.05, ylim[1] * 0.82, f"Min = {cnr_min:.2f}",
            color="darkorange", fontsize=13, va="top")

    if n_capped > 0:
        ax.text(x_cap - 0.05, ylim[1] * 0.98,
                f"{n_capped} case(s) > {x_cap:.0f}\n(not shown)",
                ha="right", va="top", fontsize=13, color="gray")

    ax.set_xlabel("CNR", fontsize=16)
    ax.set_ylabel("Number of cases", fontsize=16)
    ax.tick_params(axis='both', labelsize=13)
    ax.tick_params(axis='both', labelsize=13)
    ax.set_xlim(left=cnr_min - 0.2, right=x_cap)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot CNR histogram")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--cap", type=float, default=DEFAULT_CAP)
    args = parser.parse_args()
    plot_cnr(args.csv, args.out, args.cap)

if __name__ == "__main__":
    main()
