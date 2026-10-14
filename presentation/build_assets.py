#!/usr/bin/env python3
"""
Render presentation assets at print resolution.

The six analytical figures are re-rendered from visualization/build_visuals.py —
the same code that reads the pipeline's own reports — so the values on the slides
are the values in the repository. Nothing is retyped.

One additional diagram is created here because the review needs it and the
visualisation package does not contain it: the split-strategy timeline.

    python presentation/build_assets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UE = ROOT.parent
sys.path.insert(0, str(UE / "visualization"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import build_visuals as bv          # noqa: E402  reads the repository's reports

ASSETS = ROOT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
DPI = 200


def render_existing() -> None:
    """Re-render the six validated figures as high-DPI PNGs."""
    original = bv.save

    def save_png(fig, name):
        fig.savefig(ASSETS / f"{name}.png", format="png", dpi=DPI,
                    bbox_inches="tight", pad_inches=0.12,
                    facecolor=bv.SURFACE)
        plt.close(fig)
        return f"{name}.png"

    bv.save = save_png
    for fn in (bv.fig_architecture, bv.fig_status, bv.fig_datasets,
               bv.fig_resolution, bv.fig_sla, bv.fig_hotspot):
        fn()
    bv.save = original


def fig_split_strategy() -> None:
    """
    Why two split columns exist, drawn on the corpus's real date ranges.

    The three sources occupy disjoint windows, which is exactly why a single
    global chronological cut produces Chicago-only evaluation folds. Dates come
    from all_incidents; the cut points come from the split metadata written by
    the pipeline.
    """
    inc = pd.read_parquet(UE / "UrbanEye_ML_Data_Pipeline" / "data" / "processed" /
                          "incidents" / "all_incidents.parquet",
                          columns=["source_dataset", "city", "reported_at"])
    rng = inc.groupby("city")["reported_at"].agg(["min", "max"])

    res = pd.read_parquet(UE / "UrbanEye_ML_Data_Pipeline" / "data" / "processed" /
                          "ml" / "resolution_ml.parquet",
                          columns=["city", "reported_at", "split", "split_global"])
    ts = pd.to_datetime(res["reported_at"], utc=True)

    fig, axes = plt.subplots(2, 1, figsize=(13.2, 5.2), sharex=True,
                             gridspec_kw={"hspace": 0.78, "top": 0.86, "bottom": 0.14})
    cities = ["New York City", "San Francisco", "Chicago"]
    fold_colour = {"train": bv.O_DONE, "val": bv.O_PART, "test": bv.O_PLAN}

    for ax, col, title, note in (
            (axes[0], "split",
             "A.   split   —   per-source chronological (each city cut on its own timeline)",
             "Every city appears in train, validation and test — but the folds are not "
             "ordered on one clock."),
            (axes[1], "split_global",
             "B.   split_global   —   one global chronological cut",
             "No evaluation row precedes a training row. The cost: validation and test "
             "contain Chicago only.")):
        for i, city in enumerate(cities):
            sub = res[res["city"] == city]
            t = ts.loc[sub.index]
            spans = []
            for fold in ("train", "val", "test"):
                f = sub[col] == fold
                if not f.any():
                    continue
                lo, hi = t[f].min(), t[f].max()
                spans.append((fold, lo, hi))
                ax.barh(i, max((hi - lo).days, 6), left=lo, height=0.5,
                        color=fold_colour[fold], edgecolor=bv.SURFACE, linewidth=1.2)
            if spans:
                lo = min(s[1] for s in spans); hi = max(s[2] for s in spans)
                folds = "/".join(s[0][:2] for s in spans)
                ax.text(hi + pd.Timedelta(days=70), i,
                        f"{lo:%b %Y} – {hi:%b %Y}   ({folds})",
                        va="center", fontsize=8.5, color=bv.INK2)
        ax.set_yticks(range(len(cities)))
        ax.set_yticklabels(cities, fontsize=10, color=bv.INK)
        ax.set_ylim(-0.65, len(cities) - 0.35)
        ax.xaxis.grid(True, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold",
                     color=bv.INK, pad=8)
        ax.text(0.0, -0.40, note, transform=ax.transAxes, fontsize=9,
                color=bv.INK2, va="top")
    axes[1].set_xlim(pd.Timestamp("2009-09-01", tz="UTC"),
                     pd.Timestamp("2021-04-01", tz="UTC"))

    handles = [plt.Rectangle((0, 0), 1, 1, color=fold_colour[f])
               for f in ("train", "val", "test")]
    fig.legend(handles, ["train", "validation", "test"], frameon=False, ncol=3,
               loc="upper right", bbox_to_anchor=(0.995, 1.01), fontsize=9.5)
    fig.savefig(ASSETS / "07_split_strategy.png", format="png", dpi=DPI,
                bbox_inches="tight", pad_inches=0.12, facecolor=bv.SURFACE)
    plt.close(fig)


def main() -> int:
    render_existing()
    fig_split_strategy()
    for p in sorted(ASSETS.glob("*.png")):
        print(f"  {p.name:34s} {p.stat().st_size // 1024:>5} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
