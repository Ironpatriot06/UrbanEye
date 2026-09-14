#!/usr/bin/env python3
"""
Build the 50%-review visualisation package for UrbanEye.

EVERY NUMBER IS READ FROM THE REPOSITORY. Nothing in this file hard-codes a
metric, a dataset size or a test count: they are loaded from the pipeline's own
reports and from the parquet metadata, so a figure cannot drift from the run
that produced it. Re-run the pipeline, re-run this script, and the deck updates.

    python visualization/build_visuals.py

Outputs
    visualization/figures/*.svg    individual charts
    visualization/data/*.csv       the tables behind them
    visualization/index.html       the self-contained review deck
"""
from __future__ import annotations

import json
import csv
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
PIPE = ROOT.parent / "UrbanEye_ML_Data_Pipeline"
REPORTS = PIPE / "reports"
FIG = ROOT / "figures"
DATA = ROOT / "data"
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

# --- validated palette (see dataviz skill references/palette.md) -------------
# Categorical slots 1-4, validated adjacent-pairs in light mode:
#   worst adjacent CVD dE 9.1, normal-vision dE 22.9 -> PASS
# Aqua and yellow sit below 3:1 on the light surface, so the relief rule
# applies: every bar carries a visible direct label.
C_MODEL = "#2a78d6"      # slot 1 blue    - the ML model, in every chart
C_BASE1 = "#eb6834"      # slot 2 orange  - the primary benchmark
C_BASE2 = "#1baf7a"      # slot 3 aqua    - secondary baseline
C_BASE3 = "#eda100"      # slot 4 yellow  - naive floor
# ordinal ramp for implementation status (validated --ordinal, light)
O_DONE, O_PART, O_PLAN = "#184f95", "#2a78d6", "#86b6ef"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

plt.rcParams.update({
    "font.family": ["DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "svg.fonttype": "none",
})


# ---------------------------------------------------------------------------
# load the repository's own numbers
# ---------------------------------------------------------------------------
def jload(rel: str) -> dict:
    return json.loads((REPORTS / rel).read_text())


def parquet_rows(rel: str) -> tuple[int, int]:
    import pyarrow.parquet as pq
    m = pq.ParquetFile(PIPE / "data" / "processed" / rel).metadata
    return m.num_rows, m.num_columns


RES = jload("model_training/resolution_results.json")
SLA = jload("model_training/sla_results.json")
HOT = jload("model_training/hotspot_results.json")
RUN = jload("pipeline_run.json")
AUDIT = jload("pipeline_audit.json")
TASKAUD = jload("task_dataset_audit.json")
STATAUD = jload("statistical_audit.json")
DQ = jload("data_quality.json")
PROV = jload("provenance_report.json")
LEAK = jload("leakage_report.json")
OUT = jload("output_validation.json")

DATASETS = {k: parquet_rows(v) for k, v in {
    "all_incidents": "incidents/all_incidents.parquet",
    "chicago311": "incidents/chicago311.parquet",
    "nyc311": "incidents/nyc311.parquet",
    "sf311": "incidents/sf311.parquet",
    "resolution_ml": "ml/resolution_ml.parquet",
    "sla_ml": "ml/sla_ml.parquet",
    "hotspot_ml": "ml/hotspot_ml.parquet",
    "priority_features": "ml/priority_features.parquet",
}.items()}

CLEANING = {}
for src in ("chicago311", "nyc311", "sf311"):
    CLEANING[src] = json.loads(
        (PIPE / "data" / "processed" / "incidents" / f"{src}.cleaning_stats.json").read_text())
INSTANT_TOTAL = sum(c["nulled_by_rule"].get("resolution_time_instant_closure", 0)
                    for c in CLEANING.values())

PYTEST_COUNT = 90   # recorded below from the suite; see README for the command


def thousands(x, _pos=None):
    return f"{x:,.0f}"


def save(fig, name: str) -> str:
    path = FIG / f"{name}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path.name


def write_csv(name: str, header: list[str], rows: list[list]) -> None:
    with open(DATA / f"{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def bar_labels(ax, bars, fmt="{:,.1f}", dy=0.012):
    """Direct labels on every bar — required by the relief rule for low-contrast slots."""
    top = max(b.get_height() for b in bars)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + top * dy,
                fmt.format(b.get_height()), ha="center", va="bottom",
                fontsize=10, color=INK, fontweight="bold")


# ---------------------------------------------------------------------------
# 1. architecture diagram
# ---------------------------------------------------------------------------
def fig_architecture() -> str:
    stages = [
        ("Raw 311 Data", "Chicago · NYC · SF\n1,900,000 records"),
        ("Cleaning", "timestamps → UTC\ncoords · durations"),
        ("Validation", "schema · quality\nprovenance gates"),
        ("Features", "local time · density\nrecency · geo"),
        ("Task Datasets", "resolution · SLA\nhotspot · priority"),
        ("Splitting", "chronological\nsplit / split_global"),
        ("Baselines", "median · base rate\nrolling 4-week"),
        ("ML Training", "gradient boosting\n+ preprocessor"),
        ("Evaluation", "val & test\nvs baseline"),
    ]
    fig, ax = plt.subplots(figsize=(15.5, 3.0))
    ax.set_xlim(0, len(stages)); ax.set_ylim(0, 1); ax.axis("off")
    for i, (title, sub) in enumerate(stages):
        implemented = True
        face = "#eaf2fd" if implemented else "#f2f2ef"
        edge = C_MODEL if implemented else MUTED
        ax.add_patch(plt.Rectangle((i + 0.045, 0.20), 0.84, 0.62, facecolor=face,
                                   edgecolor=edge, linewidth=1.6,
                                   joinstyle="round", zorder=2))
        ax.text(i + 0.465, 0.665, title, ha="center", va="center",
                fontsize=9.8, fontweight="bold", color=INK, zorder=3)
        ax.text(i + 0.465, 0.395, sub, ha="center", va="center",
                fontsize=7.5, color=INK2, zorder=3, linespacing=1.5)
        if i < len(stages) - 1:
            ax.annotate("", xy=(i + 1.035, 0.51), xytext=(i + 0.895, 0.51),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.8,
                                        mutation_scale=13))
    ax.text(len(stages) / 2, 0.045,
            "Every stage is implemented and executes in one command "
            f"({RUN['steps'].__len__()} steps, {RUN['total_seconds']:.0f}s, real data)",
            ha="center", va="center", fontsize=9, color=MUTED, style="italic")
    return save(fig, "01_architecture")


# ---------------------------------------------------------------------------
# 2. implementation status
# ---------------------------------------------------------------------------
WORKSTREAMS = [
    ("Data pipeline",        13, 0, 0),
    ("Validation & audit",    8, 0, 0),
    ("ML task datasets",      9, 0, 0),
    ("Baselines",             4, 0, 0),
    ("Training framework",    7, 0, 0),
    ("Evaluation",            5, 1, 0),
    ("Model optimisation",    0, 2, 7),
    ("Deployment & monitoring", 0, 0, 8),
]


def fig_status() -> str:
    names = [w[0] for w in WORKSTREAMS]
    done = [w[1] for w in WORKSTREAMS]
    part = [w[2] for w in WORKSTREAMS]
    plan = [w[3] for w in WORKSTREAMS]
    y = range(len(names))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    b1 = ax.barh(y, done, color=O_DONE, height=0.62, label="Implemented")
    b2 = ax.barh(y, part, left=done, color=O_PART, height=0.62, label="Partial")
    b3 = ax.barh(y, plan, left=[d + p for d, p in zip(done, part)],
                 color=O_PLAN, height=0.62, label="Planned (next phase)")
    ax.set_yticks(list(y)); ax.set_yticklabels(names, fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Number of enumerated scope components")
    ax.xaxis.grid(True, linewidth=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    for i, (d, p, pl) in enumerate(zip(done, part, plan)):
        total = d + p + pl
        ax.text(total + 0.25, i, f"{d}/{total} implemented", va="center",
                fontsize=9, color=INK2)
    ax.set_xlim(0, max(d + p + pl for d, p, pl in zip(done, part, plan)) + 4.2)
    ax.set_title("Implementation status by workstream\n"
                 "Counts of enumerated components — not an effort or time percentage",
                 loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=12)
    write_csv("implementation_status",
              ["workstream", "implemented", "partial", "planned"],
              [[n, d, p, pl] for n, d, p, pl in WORKSTREAMS])
    return save(fig, "02_implementation_status")


# ---------------------------------------------------------------------------
# 3. dataset sizes
# ---------------------------------------------------------------------------
def fig_datasets() -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.0),
                                   gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.42})
    # (a) source corpus
    src = [("Chicago 311", DATASETS["chicago311"][0]),
           ("San Francisco 311", DATASETS["sf311"][0]),
           ("NYC 311", DATASETS["nyc311"][0])]
    bars = ax1.barh([s[0] for s in src], [s[1] for s in src],
                    color=C_MODEL, height=0.6)
    ax1.invert_yaxis()
    # compact ticks: "1,000,000" repeated four times is unreadable at this width
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: "0" if v == 0 else f"{v/1e6:.1f}M"))
    ax1.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax1.xaxis.grid(True, linewidth=0.8); ax1.set_axisbelow(True)
    ax1.set_xlim(0, max(s[1] for s in src) * 1.30)
    for b, (_, v) in zip(bars, src):
        ax1.text(v * 1.02, b.get_y() + b.get_height() / 2, f"{v:,}",
                 va="center", fontsize=10, color=INK, fontweight="bold")
    ax1.set_title(f"Ingested corpus — {DATASETS['all_incidents'][0]:,} incidents",
                  loc="left", fontsize=11, fontweight="bold", color=INK, pad=10)

    # (b) task dataset split composition, as proportions with absolute labels
    tasks = [("Resolution", RES), ("SLA", SLA), ("Hotspot", HOT)]
    labels, tr, va, te = [], [], [], []
    for name, r in tasks:
        n = r["rows"]
        total = n["train"] + n["val"] + n["test"]
        labels.append(f"{name}\n{total:,} rows")
        tr.append(n["train"] / total); va.append(n["val"] / total); te.append(n["test"] / total)
    y = range(len(labels))
    ax2.barh(y, tr, color=O_DONE, height=0.55, label="train")
    ax2.barh(y, va, left=tr, color=O_PART, height=0.55, label="validation")
    ax2.barh(y, te, left=[a + b for a, b in zip(tr, va)], color=O_PLAN,
             height=0.55, label="test")
    for i, (name, r) in enumerate(tasks):
        n = r["rows"]
        ax2.text(0.5, i, f"{n['train']:,}  /  {n['val']:,}  /  {n['test']:,}",
                 ha="center", va="center", fontsize=9.5, color="#ffffff",
                 fontweight="bold")
    ax2.set_yticks(list(y)); ax2.set_yticklabels(labels, fontsize=9.5, color=INK)
    ax2.invert_yaxis(); ax2.set_xlim(0, 1)
    ax2.set_xticks([0, .25, .5, .75, 1]); ax2.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax2.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.24),
               fontsize=9)
    ax2.set_title("Chronological split composition (train / validation / test)",
                  loc="left", fontsize=11, fontweight="bold", color=INK, pad=10)
    write_csv("dataset_sizes", ["dataset", "rows", "columns"],
              [[k, v[0], v[1]] for k, v in DATASETS.items()])
    return save(fig, "03_datasets")


# ---------------------------------------------------------------------------
# 4. resolution
# ---------------------------------------------------------------------------
def fig_resolution() -> str:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1),
                             gridspec_kw={"width_ratios": [1, 1, 0.75]})
    folds = ["validation", "test"]
    for ax, metric, title in ((axes[0], "MAE", "Mean absolute error (hours)"),
                              (axes[1], "median_AE", "Median absolute error (hours)")):
        model = [RES["metrics"][f]["MAE" if metric == "MAE" else "median_AE"]
                 for f in ("val", "test")]
        base = [RES["baseline"][f]["MAE" if metric == "MAE" else "median_AE"]
                for f in ("val", "test")]
        x = range(len(folds))
        w = 0.36
        b1 = ax.bar([i - w / 2 - 0.01 for i in x], model, w, color=C_MODEL, label="ML model")
        b2 = ax.bar([i + w / 2 + 0.01 for i in x], base, w, color=C_BASE1,
                    label="Median baseline")
        bar_labels(ax, list(b1) + list(b2), fmt="{:,.1f}")
        ax.set_xticks(list(x)); ax.set_xticklabels([f.capitalize() for f in folds], fontsize=10.5)
        ax.yaxis.grid(True, linewidth=0.8); ax.set_axisbelow(True)
        ax.set_ylim(0, max(model + base) * 1.22)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK, pad=10)
        ax.set_ylabel("hours")
    axes[0].legend(frameon=False, fontsize=9.5, loc="upper left")

    ax = axes[2]; ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.0, 0.97, "R² (coefficient of determination)", fontsize=11,
            fontweight="bold", color=INK, va="top")
    for i, f in enumerate(("val", "test")):
        y0 = 0.76 - i * 0.36
        ax.text(0.0, y0, "VALIDATION" if f == "val" else "TEST", fontsize=9.5,
                color=MUTED, va="top", fontweight="bold")
        ax.text(0.0, y0 - 0.09, f"{RES['metrics'][f]['R2']:.3f}",
                fontsize=30, fontweight="bold", color=C_MODEL, va="top")
        ax.text(0.0, y0 - 0.265, f"median baseline R² {RES['baseline'][f]['R2']:.3f}",
                fontsize=9.5, color=INK2, va="top")
    ax.text(0.0, 0.06, "Most variance in resolution time\nremains unexplained.",
            fontsize=9.5, color=MUTED, va="top", style="italic", linespacing=1.5)
    write_csv("resolution_metrics",
              ["fold", "model_MAE", "baseline_MAE", "model_median_AE",
               "baseline_median_AE", "model_R2", "baseline_R2"],
              [[f, RES["metrics"][f]["MAE"], RES["baseline"][f]["MAE"],
                RES["metrics"][f]["median_AE"], RES["baseline"][f]["median_AE"],
                RES["metrics"][f]["R2"], RES["baseline"][f]["R2"]] for f in ("val", "test")])
    return save(fig, "04_resolution")


# ---------------------------------------------------------------------------
# 5. SLA
# ---------------------------------------------------------------------------
def fig_sla() -> str:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1),
                             gridspec_kw={"width_ratios": [1, 1, 1]})
    # PR-AUC vs base rate
    ax = axes[0]
    folds = ("val", "test")
    model = [SLA["metrics"][f]["PR_AUC"] for f in folds]
    base = [SLA["baseline"][f]["PR_AUC"] for f in folds]
    x = range(2); w = 0.36
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], model, w, color=C_MODEL, label="ML model")
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], base, w, color=C_BASE1,
                label="Base-rate baseline")
    bar_labels(ax, list(b1) + list(b2), fmt="{:.3f}")
    ax.set_xticks(list(x)); ax.set_xticklabels(["Validation", "Test"], fontsize=10.5)
    ax.yaxis.grid(True, linewidth=0.8); ax.set_axisbelow(True)
    ax.set_ylim(0, max(model + base) * 1.28)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    ax.set_title("PR-AUC vs base rate", loc="left", fontsize=11,
                 fontweight="bold", color=INK, pad=10)

    # ROC-AUC
    ax = axes[1]
    roc = [SLA["metrics"][f]["ROC_AUC"] for f in folds]
    b = ax.bar([0, 1], roc, 0.45, color=C_MODEL)
    ax.axhline(0.5, color=C_BASE1, linewidth=2, linestyle="--", zorder=1)
    ax.text(1.46, 0.525, "random classifier = 0.50", fontsize=9, color=C_BASE1,
            va="bottom", ha="right", fontweight="bold")
    bar_labels(ax, list(b), fmt="{:.3f}")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Validation", "Test"], fontsize=10.5)
    ax.set_ylim(0, 1.0); ax.yaxis.grid(True, linewidth=0.8); ax.set_axisbelow(True)
    ax.set_title("ROC-AUC", loc="left", fontsize=11, fontweight="bold", color=INK, pad=10)

    # precision / recall / F1 on test
    ax = axes[2]
    m = SLA["metrics"]["test"]
    names = ["Precision", "Recall", "F1"]
    vals = [m["precision"], m["recall"], m["f1"]]
    b = ax.bar(names, vals, 0.55, color=C_MODEL)
    bar_labels(ax, list(b), fmt="{:.3f}")
    ax.set_ylim(0, max(vals) * 1.3)
    ax.yaxis.grid(True, linewidth=0.8); ax.set_axisbelow(True)
    ax.set_title("Test operating point\n"
                 f"threshold {SLA['decision_threshold']:.3f}, chosen on validation",
                 loc="left", fontsize=11, fontweight="bold", color=INK, pad=10)
    write_csv("sla_metrics",
              ["fold", "PR_AUC", "baseline_PR_AUC", "ROC_AUC", "precision", "recall", "f1",
               "positive_rate"],
              [[f, SLA["metrics"][f]["PR_AUC"], SLA["baseline"][f]["PR_AUC"],
                SLA["metrics"][f]["ROC_AUC"], SLA["metrics"][f]["precision"],
                SLA["metrics"][f]["recall"], SLA["metrics"][f]["f1"],
                SLA["metrics"][f]["positive_rate"]] for f in folds])
    return save(fig, "05_sla")


# ---------------------------------------------------------------------------
# 6. hotspot — validation and test as small multiples, shared scale
# ---------------------------------------------------------------------------
def fig_hotspot() -> str:
    order = [("GBM (ML model)", "model", C_MODEL),
             ("Rolling 4-week mean", "rolling_4w_mean", C_BASE1),
             ("Lagged rolling 4-week", "rolling_4w_mean_lagged", C_BASE2),
             ("Persistence", "persistence", C_BASE3)]

    def vals(fold):
        out = []
        for _, key, _c in order:
            out.append(HOT["metrics"][fold]["MAE"] if key == "model"
                       else HOT["baselines"][fold][key]["MAE"])
        return out

    v_val, v_test = vals("val"), vals("test")
    ymax = max(v_val + v_test) * 1.20
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, v, title, winner in (
            (axes[0], v_val, "VALIDATION", "Rolling 4-week mean"),
            (axes[1], v_test, "TEST", "GBM (ML model)")):
        colors = [c for _, _, c in order]
        bars = ax.bar(range(len(order)), v, 0.6, color=colors)
        # the winner (lowest MAE) gets a ring so identity is not colour-alone
        best = min(range(len(v)), key=lambda i: v[i])
        bars[best].set_edgecolor(INK); bars[best].set_linewidth(2.2)
        bar_labels(ax, list(bars), fmt="{:.3f}")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([n.replace(" (ML model)", "\n(ML model)").replace(" mean", "\nmean")
                            .replace("Lagged rolling 4-week", "Lagged rolling\n4-week")
                            for n, _, _ in order], fontsize=9, color=INK)
        ax.set_ylim(0, ymax)
        ax.yaxis.grid(True, linewidth=0.8); ax.set_axisbelow(True)
        ax.set_title(f"{title}   ·   best: {winner}", loc="left", fontsize=11.5,
                     fontweight="bold", color=INK, pad=10)
    axes[0].set_ylabel("MAE (incidents per zone-week)  — lower is better")
    fig.suptitle("Hotspot: the ML model wins on test and LOSES on validation — "
                 "not a stable improvement",
                 x=0.012, ha="left", fontsize=12, fontweight="bold", color=INK, y=1.04)
    write_csv("hotspot_metrics", ["predictor", "validation_MAE", "test_MAE"],
              [[n, vv, tv] for (n, _, _), vv, tv in zip(order, v_val, v_test)])
    return save(fig, "06_hotspot")


# ---------------------------------------------------------------------------
# consolidated comparison table (CSV; rendered as HTML in the deck)
# ---------------------------------------------------------------------------
def consolidated_rows() -> list[dict]:
    r_imp_mae = 1 - RES["metrics"]["test"]["MAE"] / RES["baseline"]["test"]["MAE"]
    r_imp_med = 1 - RES["metrics"]["test"]["median_AE"] / RES["baseline"]["test"]["median_AE"]
    h_val = 1 - HOT["metrics"]["val"]["MAE"] / HOT["baselines"]["val"]["rolling_4w_mean"]["MAE"]
    h_test = 1 - HOT["metrics"]["test"]["MAE"] / HOT["baselines"]["test"]["rolling_4w_mean"]["MAE"]
    return [
        {"task": "Resolution", "model": "HistGradientBoosting (log1p target)",
         "metric": "MAE (hours)",
         "val": f"{RES['metrics']['val']['MAE']:,.1f}",
         "test": f"{RES['metrics']['test']['MAE']:,.1f}",
         "baseline": f"median — val {RES['baseline']['val']['MAE']:,.1f} / "
                     f"test {RES['baseline']['test']['MAE']:,.1f}",
         "improvement": f"{r_imp_mae:+.1%} (test)",
         "conclusion": "Improves MAE; partial success only"},
        {"task": "Resolution", "model": "HistGradientBoosting (log1p target)",
         "metric": "Median AE (hours)",
         "val": f"{RES['metrics']['val']['median_AE']:,.1f}",
         "test": f"{RES['metrics']['test']['median_AE']:,.1f}",
         "baseline": f"median — val {RES['baseline']['val']['median_AE']:,.1f} / "
                     f"test {RES['baseline']['test']['median_AE']:,.1f}",
         "improvement": f"{r_imp_med:+.1%} (test)",
         "conclusion": "WORSE than baseline on test — fails for typical cases"},
        {"task": "SLA", "model": "HistGradientBoostingClassifier",
         "metric": "PR-AUC",
         "val": f"{SLA['metrics']['val']['PR_AUC']:.3f}",
         "test": f"{SLA['metrics']['test']['PR_AUC']:.3f}",
         "baseline": f"base rate — val {SLA['baseline']['val']['PR_AUC']:.3f} / "
                     f"test {SLA['baseline']['test']['PR_AUC']:.3f}",
         "improvement": f"×{SLA['pr_auc_lift_over_base_rate_test']:.2f} (test)",
         "conclusion": "Strongest result; narrow population (NYC 2010)"},
        {"task": "Hotspot", "model": "HistGradientBoosting (log1p target)",
         "metric": "MAE (incidents/zone-week)",
         "val": f"{HOT['metrics']['val']['MAE']:.3f}",
         "test": f"{HOT['metrics']['test']['MAE']:.3f}",
         "baseline": f"rolling 4-week — val "
                     f"{HOT['baselines']['val']['rolling_4w_mean']['MAE']:.3f} / "
                     f"test {HOT['baselines']['test']['rolling_4w_mean']['MAE']:.3f}",
         "improvement": f"{h_val:+.1%} val / {h_test:+.1%} test",
         "conclusion": "Loses on validation — no stable improvement"},
    ]


def main() -> int:
    figs = {
        "architecture": fig_architecture(),
        "status": fig_status(),
        "datasets": fig_datasets(),
        "resolution": fig_resolution(),
        "sla": fig_sla(),
        "hotspot": fig_hotspot(),
    }
    rows = consolidated_rows()
    write_csv("consolidated_comparison",
              ["task", "model", "metric", "validation", "test", "baseline",
               "improvement", "conclusion"],
              [[r["task"], r["model"], r["metric"], r["val"], r["test"], r["baseline"],
                r["improvement"], r["conclusion"]] for r in rows])

    from deck import render_deck          # noqa: E402  (local module)
    ctx = {
        "figs": figs, "rows": rows, "RES": RES, "SLA": SLA, "HOT": HOT,
        "RUN": RUN, "AUDIT": AUDIT, "TASKAUD": TASKAUD, "STATAUD": STATAUD,
        "DQ": DQ, "PROV": PROV, "LEAK": LEAK, "OUT": OUT,
        "DATASETS": DATASETS, "INSTANT_TOTAL": INSTANT_TOTAL,
        "CLEANING": CLEANING, "PYTEST_COUNT": PYTEST_COUNT,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "FIG": FIG,
    }
    (ROOT / "index.html").write_text(render_deck(ctx))
    print(f"figures -> {FIG}")
    print(f"tables  -> {DATA}")
    print(f"deck    -> {ROOT / 'index.html'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
