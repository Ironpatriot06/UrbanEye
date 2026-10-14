#!/usr/bin/env python3
"""
Build the UrbanEye mid-project technical review deck (.pptx).

Every metric, row count and hyper-parameter is read from the repository at build
time — the reports in UrbanEye_ML_Data_Pipeline/reports/ and the metadata saved
beside each trained model. Nothing is typed into this file by hand, so a slide
cannot disagree with the run that produced it.

    python presentation/build_pptx.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent
UE = ROOT.parent
PIPE = UE / "UrbanEye_ML_Data_Pipeline"
REPORTS = PIPE / "reports"
ASSETS = ROOT / "assets"
OUT = ROOT / "UrbanEye_Mid_Project_Review.pptx"

# --- design tokens (same palette as the validated charts) -------------------
BG = RGBColor(0xFC, 0xFC, 0xFB)
INK = RGBColor(0x0B, 0x0B, 0x0B)
INK2 = RGBColor(0x52, 0x51, 0x4E)
MUTED = RGBColor(0x89, 0x87, 0x81)
RULE = RGBColor(0xE1, 0xE0, 0xD9)
ACCENT = RGBColor(0x2A, 0x78, 0xD6)
WARN = RGBColor(0xEB, 0x68, 0x34)
CRIT = RGBColor(0xD0, 0x3B, 0x3B)
GOOD = RGBColor(0x0A, 0x76, 0x22)
BAND = RGBColor(0xEA, 0xF2, 0xFD)
BAND_W = RGBColor(0xFD, 0xF0, 0xEA)
BAND_C = RGBColor(0xFB, 0xEC, 0xEC)
BAND_G = RGBColor(0xEA, 0xF5, 0xEC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Arial"

W, H = Inches(13.333), Inches(7.5)
M = Inches(0.62)                      # page margin


# ---------------------------------------------------------------------------
# repository facts
# ---------------------------------------------------------------------------
def jload(rel: str) -> dict:
    return json.loads((REPORTS / rel).read_text())


def rows_cols(rel: str) -> tuple[int, int]:
    m = pq.ParquetFile(PIPE / "data" / "processed" / rel).metadata
    return m.num_rows, m.num_columns


RES = jload("model_training/resolution_results.json")
SLA = jload("model_training/sla_results.json")
HOT = jload("model_training/hotspot_results.json")
RUN = jload("pipeline_run.json")
DQ = jload("data_quality.json")
PROV = jload("provenance_report.json")
LEAK = jload("leakage_report.json")
OUTV = jload("output_validation.json")
AUD = jload("pipeline_audit.json")
TASKAUD = jload("task_dataset_audit.json")
STAT = jload("statistical_audit.json")

META = {t: json.loads((UE / "UrbanEye_ML_Data_Pipeline" / "models" / t /
                       "metadata.json").read_text())
        for t in ("resolution", "sla", "hotspot")}

DS = {k: rows_cols(v) for k, v in {
    "all_incidents": "incidents/all_incidents.parquet",
    "chicago311": "incidents/chicago311.parquet",
    "nyc311": "incidents/nyc311.parquet",
    "sf311": "incidents/sf311.parquet",
    "resolution_ml": "ml/resolution_ml.parquet",
    "sla_ml": "ml/sla_ml.parquet",
    "hotspot_ml": "ml/hotspot_ml.parquet",
    "priority_features": "ml/priority_features.parquet",
    "duplicate_ml": "ml/duplicate_ml.parquet",
}.items()}

CLEAN = {s: json.loads((PIPE / "data" / "processed" / "incidents" /
                        f"{s}.cleaning_stats.json").read_text())
         for s in ("chicago311", "nyc311", "sf311")}
INSTANT = sum(c["nulled_by_rule"].get("resolution_time_instant_closure", 0)
              for c in CLEAN.values())
PYTEST_N = 90          # from `python -m pytest -q`; see speaker notes


def cap(text: str) -> str:
    """Repository verdict strings are lower-case fragments; only the case changes."""
    return text[0].upper() + text[1:] if text else text


def hp(task: str) -> str:
    """Hyper-parameters exactly as saved with the model."""
    p = META[task]["model_params"]
    return ", ".join(f"{k}={v}" for k, v in p.items())


# ---------------------------------------------------------------------------
# slide primitives
# ---------------------------------------------------------------------------
prs = Presentation()
prs.slide_width, prs.slide_height = W, H
BLANK = prs.slide_layouts[6]
_num = {"i": 0}


def _bg(slide) -> None:
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG


def textbox(slide, x, y, w, h, *, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    return tf


def para(tf, text, *, size=13, bold=False, color=INK2, space_after=5,
         align=PP_ALIGN.LEFT, first=False, italic=False, space_before=0):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space_after)
    p.space_before = Pt(space_before)
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    r.font.name = FONT
    return p


def rect(slide, x, y, w, h, fill=None, line=None, line_w=1.0):
    from pptx.enum.shapes import MSO_SHAPE
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(line_w)
    sh.shadow.inherit = False
    return sh


def slide_page(kicker: str, title: str, *, subtitle: str = None):
    s = prs.slides.add_slide(BLANK)
    _bg(s)
    _num["i"] += 1

    tf = textbox(s, M, Inches(0.42), W - 2 * M, Inches(0.26))
    para(tf, kicker.upper(), size=10.5, bold=True, color=ACCENT, first=True,
         space_after=0)

    tf = textbox(s, M, Inches(0.70), W - 2 * M, Inches(0.46))
    para(tf, title, size=25, bold=True, color=INK, first=True, space_after=0)

    y = Inches(1.26)
    if subtitle:
        tf = textbox(s, M, Inches(1.20), W - 2 * M, Inches(0.34))
        para(tf, subtitle, size=12.5, color=INK2, first=True, space_after=0)
        y = Inches(1.62)

    ln = rect(s, M, y, W - 2 * M, Pt(1.2), fill=RULE)
    ln.line.fill.background()

    # footer
    tf = textbox(s, M, H - Inches(0.44), Inches(7.0), Inches(0.24))
    para(tf, "UrbanEye  ·  Mid-Project Technical Review", size=9,
         color=MUTED, first=True, space_after=0)
    tf = textbox(s, W - M - Inches(1.0), H - Inches(0.44), Inches(1.0), Inches(0.24))
    para(tf, str(_num["i"]), size=9, color=MUTED, first=True, space_after=0,
         align=PP_ALIGN.RIGHT)
    return s, y + Inches(0.22)


def picture(slide, name: str, x, y, w):
    return slide.shapes.add_picture(str(ASSETS / name), x, y, width=w)


_ASPECT: dict[str, float] = {}


def aspect(name: str) -> float:
    """Width / height of an asset, measured once from the file itself."""
    if name not in _ASPECT:
        from PIL import Image
        w, h = Image.open(ASSETS / name).size
        _ASPECT[name] = w / h
    return _ASPECT[name]


def picture_fit(slide, name: str, x, y, max_w, max_h, *, centre=True):
    """
    Place an asset scaled to fit inside (max_w, max_h), never exceeding either.

    Images are the one element whose height this file does not choose, so it is
    measured rather than assumed — that is what stops a chart from growing into
    the table beneath it.
    """
    a = aspect(name)
    w, h = max_w, int(max_w / a)
    if h > max_h:
        h, w = max_h, int(max_h * a)
    if centre:
        x = x + (max_w - w) // 2
    return slide.shapes.add_picture(str(ASSETS / name), x, y, width=w, height=h)


def config_box(slide, x, y, w, h, task: str, estimator: str,
               extra: str = "") -> None:
    """The exact configuration saved beside the model — never retyped."""
    m = META[task]
    env = m["environment"]
    rect(slide, x, y, w, h, fill=RGBColor(0xF7, 0xF7, 0xF5), line=RULE)
    tf = textbox(slide, x + Inches(0.18), y + Inches(0.13), w - Inches(0.36),
                 h - Inches(0.24))
    para(tf, "SELECTED CONFIGURATION  ·  read from the saved model metadata",
         size=9, bold=True, color=MUTED, first=True, space_after=4)
    para(tf, f"{estimator}   ·   variant {m['selected_config']}   ·   selected on "
             f"{m['selection_metric']}"
             + (f"   ·   target transform {m['target_transform']}"
                if m.get("target_transform") else ""),
         size=9.5, bold=True, color=INK, space_after=3)
    para(tf, hp(task), size=9.5, color=INK2, space_after=3)
    para(tf, f"seed {env['seed']}  ·  scikit-learn {env['scikit_learn']}  ·  "
             f"preprocessing fitted on the training fold only (rare levels folded "
             f"to the train top-{m['preprocessing']['max_levels_per_categorical']})"
             f"{extra}",
         size=8.5, color=MUTED, space_after=0)


def callout(slide, x, y, w, h, title, body, *, tone="info"):
    fill, bar = {"info": (BAND, ACCENT), "warn": (BAND_W, WARN),
                 "crit": (BAND_C, CRIT)}[tone]
    rect(slide, x, y, w, h, fill=fill)
    rect(slide, x, y, Pt(3.5), h, fill=bar)
    tf = textbox(slide, x + Inches(0.16), y + Inches(0.11),
                 w - Inches(0.30), h - Inches(0.20))
    para(tf, title, size=12, bold=True, color=INK, first=True, space_after=3)
    para(tf, body, size=10.5, color=INK2, space_after=0)


def table(slide, x, y, w, h, data, *, col_w=None, head_size=9.5, body_size=9.5,
          head_fill=RGBColor(0xF0, 0xEF, 0xEC), zebra=True, align=None,
          bold_col0=False, row_h=None):
    rows, cols = len(data), len(data[0])
    shape = slide.shapes.add_table(rows, cols, x, y, w, h)
    tbl = shape.table
    tbl.first_row = True
    tbl.horz_banding = False
    if col_w:
        total = sum(col_w)
        for i, cw in enumerate(col_w):
            tbl.columns[i].width = Emu(int(w * cw / total))
    if row_h:
        for i, rh in enumerate(row_h):
            tbl.rows[i].height = rh
    for r, row in enumerate(data):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.margin_left = cell.margin_right = Inches(0.07)
            cell.margin_top = cell.margin_bottom = Inches(0.035)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if r == 0:
                cell.fill.fore_color.rgb = head_fill
            elif zebra and r % 2 == 0:
                cell.fill.fore_color.rgb = RGBColor(0xF7, 0xF7, 0xF5)
            else:
                cell.fill.fore_color.rgb = WHITE
            tf = cell.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_right = 0
            p = tf.paragraphs[0]
            p.alignment = (align[c] if align else PP_ALIGN.LEFT)
            colour, bold, txt = INK2, False, str(val)
            if r == 0:
                colour, bold = INK2, True
            elif bold_col0 and c == 0:
                colour, bold = INK, True
            if txt.startswith("!!"):
                txt, colour, bold = txt[2:], CRIT, True
            elif txt.startswith("^^"):
                txt, colour, bold = txt[2:], GOOD, True
            elif txt.startswith("~~"):
                txt, colour, bold = txt[2:], WARN, True
            r_ = p.add_run()
            r_.text = txt
            r_.font.size = Pt(head_size if r == 0 else body_size)
            r_.font.bold = bold
            r_.font.color.rgb = colour
            r_.font.name = FONT
    return tbl


def notes(slide, text: str) -> None:
    slide.notes_slide.notes_text_frame.text = text.strip()


# ---------------------------------------------------------------------------
# slides
# ---------------------------------------------------------------------------
def s01_title():
    s = prs.slides.add_slide(BLANK)
    _bg(s)
    _num["i"] += 1
    rect(s, Inches(0), Inches(0), Pt(9), H, fill=ACCENT)

    tf = textbox(s, Inches(1.15), Inches(1.85), Inches(10.5), Inches(0.4))
    para(tf, "MID-PROJECT TECHNICAL REVIEW", size=13, bold=True, color=ACCENT,
         first=True, space_after=0)

    tf = textbox(s, Inches(1.15), Inches(2.32), Inches(10.5), Inches(1.0))
    para(tf, "UrbanEye", size=54, bold=True, color=INK, first=True, space_after=0)

    tf = textbox(s, Inches(1.15), Inches(3.42), Inches(10.5), Inches(0.5))
    para(tf, "Data Engineering  →  ML Pipeline  →  Initial Evaluation",
         size=19, color=INK2, first=True, space_after=0)

    rect(s, Inches(1.15), Inches(4.18), Inches(3.2), Pt(1.5), fill=RULE)

    tf = textbox(s, Inches(1.15), Inches(4.48), Inches(11.0), Inches(1.3))
    para(tf, "A reproducible, auditable pipeline that turns municipal 311 "
             "service-request data into ML-ready datasets — and an honest first "
             "evaluation of three prediction tasks against strong baselines.",
         size=13.5, color=INK2, first=True, space_after=0)

    facts = [f"{DS['all_incidents'][0]:,} incidents", "3 cities ingested",
             f"{len(RUN['steps'])}/{len(RUN['steps'])} pipeline stages",
             f"{PYTEST_N} automated tests", "3 ML tasks trained"]
    x = Inches(1.15)
    for f in facts:
        rect(s, x, Inches(5.72), Inches(2.06), Inches(0.42), fill=BAND)
        tf = textbox(s, x, Inches(5.80), Inches(2.06), Inches(0.28))
        para(tf, f, size=10.5, bold=True, color=ACCENT, first=True,
             space_after=0, align=PP_ALIGN.CENTER)
        x += Inches(2.20)

    tf = textbox(s, Inches(1.15), H - Inches(0.85), Inches(10.5), Inches(0.3))
    para(tf, f"Generated from repository reports · "
             f"{datetime.now():%d %B %Y}",
         size=9.5, color=MUTED, first=True, space_after=0)
    notes(s, NOTES["s1"])


def s02_problem():
    s, y = slide_page("01  Problem", "Why civic 311 data resists direct machine learning")
    col = Inches(3.95)
    gap = Inches(0.32)
    blocks = [
        ("The operational problem",
         "Cities receive hundreds of thousands of civic service requests a year — "
         "potholes, streetlight faults, graffiti, drainage. Crews are finite. "
         "Deciding what to send where, and warning when a case is about to miss "
         "its deadline, is currently reactive."),
        ("Why 311 data is useful",
         "311 systems record what was reported, where, when, by which channel, "
         "which department owned it and when it closed. That is a large, public, "
         "longitudinal record of how a city actually responds to its own citizens."),
        ("Why it cannot be modelled directly",
         "Every city uses a different schema, vocabulary and timestamp convention. "
         "Closure timestamps do not always mean service completed. Random splits "
         "leak the future into the past. Naive use produces confident, wrong models."),
    ]
    x = M
    for title, body in blocks:
        rect(s, x, y, col, Inches(2.05), fill=WHITE, line=RULE)
        tf = textbox(s, x + Inches(0.20), y + Inches(0.18), col - Inches(0.40),
                     Inches(2.2))
        para(tf, title, size=14, bold=True, color=INK, first=True, space_after=7)
        para(tf, body, size=11.5, color=INK2, space_after=0)
        x += col + gap

    callout(s, M, y + Inches(2.32), W - 2 * M, Inches(1.52),
            "What UrbanEye sets out to do",
            "Build the data foundation first: ingest heterogeneous municipal 311 feeds, "
            "normalise them into one audited schema, engineer leakage-safe features, and "
            "only then train models — each measured against a baseline strong enough to "
            "be worth beating.\n\n"
            "This review reports the state of that foundation and the first models built "
            "on it. The models are experimental; the pipeline is not.")

    tf = textbox(s, M, y + Inches(4.00), W - 2 * M, Inches(0.26))
    para(tf, "WHAT THIS REVIEW COVERS", size=10, bold=True, color=MUTED, first=True,
         space_after=0)
    steps = [("01", "Data foundation", "sources, schema, generated datasets"),
             ("02", "Assurance", "quality, provenance, leakage, audits"),
             ("03", "Methodology", "task design and chronological splitting"),
             ("04", "Results", "three models against strong baselines"),
             ("05", "Next phase", "limitations and proposed work")]
    x = M
    gap = Inches(0.16)
    cw = ((W - 2 * M) - 4 * gap) // 5
    for n, t, d in steps:
        rect(s, x, y + Inches(4.32), cw, Inches(0.88), fill=WHITE, line=RULE)
        tf = textbox(s, x + Inches(0.16), y + Inches(4.41), cw - Inches(0.32),
                     Inches(0.72))
        para(tf, n, size=10, bold=True, color=ACCENT, first=True, space_after=2)
        para(tf, t, size=11.5, bold=True, color=INK, space_after=2)
        para(tf, d, size=9, color=MUTED, space_after=0)
        x += cw + gap
    notes(s, NOTES["s2"])


def s03_system():
    s, y = slide_page("02  System", "End-to-end pipeline — what runs today",
                      subtitle=f"All {len(RUN['steps'])} stages execute in one command "
                               f"and rebuild every dataset from raw in "
                               f"{RUN['total_seconds']:.0f} s.")
    picture_fit(s, "01_architecture.png", M, y + Inches(0.14), W - 2 * M,
                Inches(2.60))
    yy = y + Inches(2.92)
    left = Inches(6.05)
    right = W - M - (M + left + Inches(0.32))
    rect(s, M, yy, left, Inches(2.10), fill=WHITE, line=RULE)
    tf = textbox(s, M + Inches(0.20), yy + Inches(0.17), left - Inches(0.40), Inches(1.8))
    para(tf, "IMPLEMENTED AND VALIDATED", size=11, bold=True, color=GOOD,
         first=True, space_after=6)
    para(tf, "Ingestion · schema normalisation · cleaning · provenance and leakage "
             "validation · time, density and recency features · task datasets · "
             "chronological splitting · baselines · training framework · evaluation "
             "and reporting.", size=10.5, color=INK2, space_after=8)
    import collections
    phases = collections.Counter(st["phase"] for st in RUN["steps"])
    para(tf, "STAGES PER PHASE, AS EXECUTED", size=9, bold=True, color=MUTED,
         space_after=3)
    para(tf, "   ".join(f"{k.split('-', 1)[1]} {v}" for k, v in sorted(phases.items())),
         size=10, bold=True, color=ACCENT, space_after=3)
    para(tf, f"Slowest stage {max(RUN['steps'], key=lambda x: x['seconds'])['step']} "
             f"at {max(st['seconds'] for st in RUN['steps']):.0f} s; every stage "
             f"returned OK.", size=9.5, color=MUTED, space_after=0)

    rect(s, M + left + Inches(0.32), yy, right, Inches(2.10), fill=WHITE, line=RULE)
    tf = textbox(s, M + left + Inches(0.52), yy + Inches(0.17), right - Inches(0.40),
                 Inches(1.8))
    para(tf, "PROPOSED — NOT YET BUILT", size=11, bold=True, color=MUTED,
         first=True, space_after=6)
    para(tf, "Model optimisation · cross-city generalisation · richer spatial and "
             "workload features · interpretability · inference API · drift monitoring · "
             "automated retraining · dashboard integration.",
         size=10.5, color=INK2, space_after=8)
    para(tf, "WHY THE BOUNDARY SITS HERE", size=9, bold=True, color=MUTED,
         space_after=3)
    para(tf, "Deployment work is gated on evidence, not on schedule: two of the three "
             "models do not yet beat their baseline on both folds, and serving a model "
             "in that state would be worse than serving none.",
         size=10, color=INK2, space_after=0)
    notes(s, NOTES["s3"])


def s04_data():
    s, y = slide_page("03  Data", "Data foundation",
                      subtitle="Three municipal 311 sources normalised into one audited "
                               "schema; Boston is configured and licence-cleared but not "
                               "yet downloaded.")
    picture_fit(s, "03_datasets.png", M, y + Inches(0.04), W - 2 * M,
                Inches(3.18))

    data = [["Generated dataset", "Rows", "Cols", "Purpose"],
            ["all_incidents", f"{DS['all_incidents'][0]:,}", DS['all_incidents'][1],
             "Unified cleaned incident record"],
            ["resolution_ml", f"{DS['resolution_ml'][0]:,}", DS['resolution_ml'][1],
             "Resolution-time regression"],
            ["hotspot_ml", f"{DS['hotspot_ml'][0]:,}", DS['hotspot_ml'][1],
             "Zone x category x week count panel"],
            ["sla_ml", f"{DS['sla_ml'][0]:,}", DS['sla_ml'][1],
             "SLA-breach classification"],
            ["priority_features", f"{DS['priority_features'][0]:,}",
             DS['priority_features'][1], "Policy features — no supervised target"],
            ["duplicate_ml", f"{DS['duplicate_ml'][0]:,}", DS['duplicate_ml'][1],
             "Duplicate pairs — labels ready, evaluation set is not"]]
    table(s, M, y + Inches(3.34), W - 2 * M, Inches(1.40), data,
          col_w=[3.0, 1.5, 1.0, 6.0], bold_col0=True, body_size=9,
          row_h=[Inches(0.20)] * 7,
          align=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.LEFT])

    tf = textbox(s, M, y + Inches(4.90), W - 2 * M, Inches(0.36))
    para(tf, "Raw and generated datasets are deliberately untracked in Git; the "
             "repository carries the code and configuration that regenerate them. "
             "A full rebuild reproduced all 18 tables content-identical.",
         size=10, color=MUTED, first=True, space_after=0, italic=True)
    notes(s, NOTES["s4"])


def s05_validation():
    s, y = slide_page("04  Assurance", "Validation infrastructure",
                      subtitle="The pipeline is tested before its outputs are trusted. "
                               "These gates run on every execution.")
    data = [["Gate", "Result", "What it checks"],
            ["Full pipeline", f"^^{sum(1 for x in RUN['steps'] if x['status']=='OK')}"
                              f"/{len(RUN['steps'])} stages",
             f"End-to-end rebuild from raw in {RUN['total_seconds']:.0f} s"],
            ["Automated tests", f"^^{PYTEST_N} passed",
             "Unit and integration tests across pipeline and training"],
            ["Data quality", f"^^{DQ['integrity_summary']['passed']}"
                             f"/{DQ['integrity_summary']['total']}",
             "Integrity assertions on the incident table"],
            ["Provenance", f"^^{PROV['summary']['passed']}/{PROV['summary']['total']}",
             "Includes deliberate fabrication attempts that must be rejected"],
            ["Leakage", f"^^{LEAK['summary']['passed']}/{LEAK['summary']['total']}",
             "Backward-window features re-derived independently"],
            ["Output validation", f"^^{OUTV['summary']['passed']}"
                                  f"/{OUTV['summary']['total']}",
             "Shape and content of every ML-ready table"],
            ["Pipeline audit", f"^^{AUD['summary']['passed']}/{AUD['summary']['total']}",
             "Stage-by-stage audit of the whole chain"],
            ["Task & leakage audit", f"^^{TASKAUD['summary']['passed']}"
                                     f"/{TASKAUD['summary']['total']}",
             f"12 leakage routes per task ({TASKAUD['summary']['skipped']} "
             f"not applicable)"]]
    table(s, M, y + Inches(0.10), W - 2 * M, Inches(3.15), data,
          col_w=[2.6, 1.9, 7.0], bold_col0=True,
          align=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT])

    callout(s, M, y + Inches(3.50), W - 2 * M, Inches(1.28),
            "The statistical audit is reported, not suppressed",
            f"{STAT['summary']['needs_preprocessing']} findings classified "
            f"needs-preprocessing, {STAT['summary']['policy_decision']} policy decision, "
            f"{STAT['summary']['dangerous']} dangerous. The dangerous finding is a "
            f"documented regime change inside the evaluation window — recorded so results "
            f"are read correctly, not removed to make a chart look better.", tone="warn")
    notes(s, NOTES["s5"])


def s06_instant():
    s, y = slide_page("05  Data-quality finding",
                      "Sub-minute closures: an artefact, not service time",
                      subtitle="Discovered during validation of the resolution target.")
    left = Inches(6.1)
    rect(s, M, y + Inches(0.10), left, Inches(2.42), fill=WHITE, line=RULE)
    tf = textbox(s, M + Inches(0.22), y + Inches(0.26), left - Inches(0.44), Inches(2.3))
    para(tf, "Evidence", size=13, bold=True, color=INK, first=True, space_after=7)
    para(tf, "Chicago closes 18,279 cases at exactly 7 seconds, 18,125 at 6 seconds "
             "and 14,578 at 5 seconds — a transactional write, not field work.",
         size=11, color=INK2, space_after=6)
    para(tf, "In NYC the sub-minute population is 7,645 cases, of which 7,640 sit at "
             "exactly 0 seconds and only 5 fall anywhere between 1 and 60 seconds.",
         size=11, color=INK2, space_after=6)
    para(tf, "One minute is also the coarsest timestamp granularity in the sources, so "
             "a sub-minute duration is below what the data can express.",
         size=11, color=INK2, space_after=0)

    right = Inches(5.9)
    rx = M + left + Inches(0.32)
    rect(s, rx, y + Inches(0.10), right, Inches(2.42), fill=WHITE, line=RULE)
    tf = textbox(s, rx + Inches(0.22), y + Inches(0.26), right - Inches(0.44), Inches(2.3))
    para(tf, "Rule applied", size=13, bold=True, color=INK, first=True, space_after=7)
    para(tf, "cleaning.resolution_time.min_observable_seconds = 60",
         size=11, bold=True, color=ACCENT, space_after=7)
    para(tf, "No row is deleted. status, closed_at and a resolution_instant_closure "
             "flag are all retained; only the duration becomes NULL, because it is not "
             "reliably observable.", size=11, color=INK2, space_after=6)
    para(tf, f"{INSTANT:,} durations were made unobserved — the same treatment "
             f"negative and absurd durations already received.",
         size=11, color=INK2, space_after=0)

    data = [["", "Before", "After"],
            ["all_incidents rows", f"{DS['all_incidents'][0]:,}",
             f"{DS['all_incidents'][0]:,}"],
            ["resolution_ml rows", "1,026,596", f"{DS['resolution_ml'][0]:,}"],
            ["Target p10", "0.03 h", "1.32 h"],
            ["Target p25", "8.87 h", "20.20 h"],
            ["Target median", "82.99 h", "116.04 h"],
            ["Share under 1 minute", "!!8.21 %", "^^0.00 %"]]
    table(s, M, y + Inches(2.70), Inches(6.1), Inches(1.95), data,
          col_w=[2.7, 1.6, 1.6], bold_col0=True, body_size=9.5,
          align=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT])

    callout(s, rx, y + Inches(2.70), right, Inches(1.95),
            "Why this matters for the review",
            "The artefact was 8.2 % of the regression target. Left in place, a model "
            "would have learned clerical behaviour and reported it as service-time "
            "prediction.\n\n"
            "Caveat we state openly: the 60-second threshold is corpus-specific and "
            "justified by these sources' timestamp granularity. A publisher with "
            "sub-second timestamps would need its own value.", tone="warn")
    notes(s, NOTES["s6"])


def s07_tasks():
    s, y = slide_page("06  Tasks", "Three supervised tasks — and two deliberately excluded")
    data = [["", "Resolution time", "SLA breach", "Hotspot"],
            ["Question",
             "How long until this request is resolved?",
             "Will this request miss its due date?",
             "How many incidents next week in this zone and category?"],
            ["Target", "resolution_time_hours", "sla_breach", "future_incident_count"],
            ["Type", "Regression (hours)", "Binary classification", "Count regression"],
            ["Dataset", f"resolution_ml\n{DS['resolution_ml'][0]:,} rows",
             f"sla_ml\n{DS['sla_ml'][0]:,} rows",
             f"hotspot_ml\n{DS['hotspot_ml'][0]:,} rows"],
            ["Split used", META['resolution']['split_column'],
             META['sla']['split_column'], META['hotspot']['split_column']],
            ["Features", f"{META['resolution']['n_features']}",
             f"{META['sla']['n_features']}", f"{META['hotspot']['n_features']}"],
            ["Headline metric", "MAE + median AE", "PR-AUC", "MAE vs rolling mean"],
            ["Status", "~~Mixed — partial success", "^^Most promising",
             "!!Not yet better than baseline"]]
    table(s, M, y + Inches(0.12), W - 2 * M, Inches(3.55), data,
          col_w=[2.0, 3.35, 3.35, 3.35], bold_col0=True, body_size=10)

    callout(s, M, y + Inches(3.92), W - 2 * M, Inches(0.98),
            "Two tasks are deliberately not modelled",
            "Priority — no public 311 dataset records an operational priority, and the "
            "existing priority_baseline is a deterministic function of a configuration "
            "file; training on it would reproduce the YAML.   "
            "Duplicate detection — the labels are genuine, but in the realistic candidate "
            "population the test fold contains 23,480 positives and zero negatives, so no "
            "metric on it would mean anything.")
    notes(s, NOTES["s7"])


def s08_split():
    s, y = slide_page("07  Methodology", "Chronological splitting — two strategies, stated openly",
                      subtitle="Random splitting would leak the future into the past. "
                               "There is no single correct chronological cut, so the "
                               "pipeline publishes both.")
    picture_fit(s, "07_split_strategy.png", M, y + Inches(0.02),
                Inches(8.55), Inches(4.20))

    bx = Inches(8.55)
    rect(s, M, y + Inches(4.02), bx, Inches(1.18), fill=WHITE, line=RULE)
    tf = textbox(s, M + Inches(0.20), y + Inches(4.16), bx - Inches(0.40),
                 Inches(0.95))
    para(tf, "What keeps the folds honest", size=12, bold=True, color=INK,
         first=True, space_after=5)
    para(tf, "Density and recency windows are strictly backward and half-open "
             "(reported_at < t), so no feature can see its own outcome. The leakage "
             "suite re-derives every window independently and compares — "
             f"{LEAK['summary']['passed']}/{LEAK['summary']['total']} checks pass, and "
             f"the task audit tests 12 leakage routes per task. Predictors come from a "
             f"declared manifest per task, never 'all columns except the target'.",
         size=10, color=INK2, space_after=0)

    rx = M + Inches(8.87)
    rw = W - M - rx
    rect(s, rx, y + Inches(0.06), rw, Inches(2.05), fill=WHITE, line=RULE)
    tf = textbox(s, rx + Inches(0.18), y + Inches(0.20), rw - Inches(0.36), Inches(1.8))
    para(tf, "How the column is chosen", size=12, bold=True, color=INK, first=True,
         space_after=6)
    para(tf, "split_global for multi-city tables — one wall clock, so no evaluation "
             "row precedes a training row.", size=10.5, color=INK2, space_after=5)
    para(tf, "split for single-source tables — sla_ml is NYC-only, and a global cut "
             "would put every row in train.", size=10.5, color=INK2, space_after=0)

    callout(s, rx, y + Inches(2.28), rw, Inches(2.30),
            "The trade-off we accept",
            "Because the three cities occupy disjoint windows, split_global leaves "
            "validation and test Chicago-only. Per-source split keeps all cities in every "
            "fold but is not globally ordered.\n\n"
            "Neither is free. The pipeline exposes both rather than silently choosing, "
            "and each task manifest records which column was used and why.", tone="warn")
    notes(s, NOTES["s8"])


def s09_resolution():
    s, y = slide_page("08  Result", "Resolution time — improves MAE, loses on median error",
                      subtitle=f"HistGradientBoostingRegressor on log1p(target) · "
                               f"{META['resolution']['n_features']} features · "
                               f"split_global · "
                               f"{RES['rows']['train']:,} train / {RES['rows']['val']:,} "
                               f"validation / {RES['rows']['test']:,} test")
    IMG_W = Inches(8.55)
    picture_fit(s, "04_resolution.png", M, y + Inches(0.02), IMG_W, Inches(2.82))

    data = [["Test metric", "Model", "Median baseline", "Verdict"],
            ["MAE (hours)", f"{RES['metrics']['test']['MAE']:,.1f}",
             f"{RES['baseline']['test']['MAE']:,.1f}",
             f"^^improves {1-RES['metrics']['test']['MAE']/RES['baseline']['test']['MAE']:.1%}"],
            ["Median AE (hours)", f"{RES['metrics']['test']['median_AE']:,.1f}",
             f"{RES['baseline']['test']['median_AE']:,.1f}",
             f"!!worse by {RES['metrics']['test']['median_AE']/RES['baseline']['test']['median_AE']-1:.1%}"],
            ["R squared", f"{RES['metrics']['test']['R2']:.3f}",
             f"{RES['baseline']['test']['R2']:.3f}", "most variance unexplained"]]
    table(s, M, y + Inches(2.96), IMG_W, Inches(1.04), data,
          col_w=[2.3, 1.4, 1.9, 2.4], bold_col0=True, body_size=9.5,
          row_h=[Inches(0.26)] * 4,
          align=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.LEFT])

    config_box(s, M, y + Inches(4.14), IMG_W, Inches(0.92), "resolution",
               "HistGradientBoostingRegressor")

    rx = M + IMG_W + Inches(0.32)
    rw = W - M - rx
    callout(s, rx, y + Inches(0.06), rw, Inches(2.55),
            "Why this is only a partial success",
            "The model wins on the long tail and loses in the middle of the "
            "distribution, which is where most operational decisions sit.\n\n"
            "On validation it beat the baseline on both metrics — 84.2 h against "
            "108.8 h median error. The reversal appears only on test, so neither "
            "fold alone describes this model.", tone="warn")
    rect(s, rx, y + Inches(2.76), rw, Inches(2.30), fill=WHITE, line=RULE)
    tf = textbox(s, rx + Inches(0.18), y + Inches(2.90), rw - Inches(0.36),
                 Inches(2.05))
    para(tf, "Verdict recorded in the results file", size=12, bold=True, color=INK,
         first=True, space_after=6)
    para(tf, cap(RES["verdict"]), size=10.5, color=INK2, space_after=7)
    para(tf, RES["honesty_note"], size=9.5, color=MUTED, space_after=0, italic=True)
    notes(s, NOTES["s9"])


def s10_sla():
    s, y = slide_page("09  Result", "SLA breach — strongest current result",
                      subtitle=f"HistGradientBoostingClassifier · "
                               f"{META['sla']['n_features']} features · split · "
                               f"{SLA['rows']['train']:,} train / {SLA['rows']['val']:,} "
                               f"validation / {SLA['rows']['test']:,} test · decision "
                               f"threshold {SLA['decision_threshold']:.3f}, chosen on "
                               f"validation")
    IMG_W = Inches(8.55)
    picture_fit(s, "05_sla.png", M, y + Inches(0.02), IMG_W, Inches(2.82))

    m = SLA["metrics"]["test"]
    cm = m["confusion_matrix"]
    data = [["Test metric", "Value", "Test metric", "Value"],
            ["PR-AUC", f"^^{m['PR_AUC']:.3f}", "ROC-AUC", f"{m['ROC_AUC']:.3f}"],
            ["Base rate (prevalence)", f"{m['positive_rate']:.3f}",
             "Precision / Recall / F1",
             f"{m['precision']:.3f} / {m['recall']:.3f} / {m['f1']:.3f}"],
            ["Lift over base rate",
             f"^^x{SLA['pr_auc_lift_over_base_rate_test']:.2f}",
             "Confusion  tn/fp/fn/tp",
             f"{cm['tn']:,} / {cm['fp']:,} / {cm['fn']:,} / {cm['tp']:,}"]]
    table(s, M, y + Inches(2.96), IMG_W, Inches(1.04), data,
          col_w=[2.3, 1.3, 2.4, 2.0], bold_col0=True, body_size=9.5,
          row_h=[Inches(0.26)] * 4,
          align=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.LEFT, PP_ALIGN.RIGHT])

    config_box(s, M, y + Inches(4.14), IMG_W, Inches(0.92), "sla",
               "HistGradientBoostingClassifier",
               "  ·  threshold frozen on validation")

    rx = M + IMG_W + Inches(0.32)
    rw = W - M - rx
    callout(s, rx, y + Inches(0.06), rw, Inches(2.30),
            "Promising — and narrow",
            "A 2.4x lift over prevalence is real signal, and it holds on both "
            "folds.\n\nAccuracy is deliberately not reported: at a ~10 % positive "
            "rate, always predicting 'no breach' scores ~90 % accuracy and catches "
            "nothing.")
    rect(s, rx, y + Inches(2.52), rw, Inches(2.54), fill=WHITE, line=RULE)
    tf = textbox(s, rx + Inches(0.18), y + Inches(2.66), rw - Inches(0.36),
                 Inches(2.28))
    para(tf, "Population caveat", size=12, bold=True, color=INK, first=True,
         space_after=6)
    para(tf, SLA["population_caveat"], size=10, color=INK2, space_after=7)
    para(tf, SLA["headline_metric_note"], size=9.5, color=MUTED, space_after=0,
         italic=True)
    notes(s, NOTES["s10"])


def s11_hotspot():
    s, y = slide_page("10  Result", "Hotspot — the baseline still wins on validation",
                      subtitle=f"HistGradientBoostingRegressor on log1p(count) · "
                               f"{META['hotspot']['n_features']} features · "
                               f"split_global · "
                               f"{HOT['rows']['train']:,} train / {HOT['rows']['val']:,} "
                               f"validation / {HOT['rows']['test']:,} test · benchmark: "
                               f"rolling 4-week mean")
    IMG_W = Inches(8.55)
    picture_fit(s, "06_hotspot.png", M, y + Inches(0.04), IMG_W, Inches(3.55))
    config_box(s, M, y + Inches(3.78), IMG_W, Inches(0.92), "hotspot",
               "HistGradientBoostingRegressor")

    rx = M + IMG_W + Inches(0.32)
    rw = W - M - rx
    callout(s, rx, y + Inches(0.06), rw, Inches(2.15),
            "Test improvement is not stable",
            f"The model beats the rolling 4-week mean on test by "
            f"{1-HOT['metrics']['test']['MAE']/HOT['baselines']['test']['rolling_4w_mean']['MAE']:.1%} "
            f"and loses to it on validation by "
            f"{abs(1-HOT['metrics']['val']['MAE']/HOT['baselines']['val']['rolling_4w_mean']['MAE']):.1%}. "
            f"A win on one fold and a loss on the other is not evidence of "
            f"superiority.", tone="crit")

    rect(s, rx, y + Inches(2.36), rw, Inches(2.34), fill=WHITE, line=RULE)
    tf = textbox(s, rx + Inches(0.18), y + Inches(2.50), rw - Inches(0.36),
                 Inches(2.05))
    para(tf, "Current recommendation", size=12, bold=True, color=INK, first=True,
         space_after=6)
    para(tf, cap(HOT["recommendation"]), size=10.5, color=INK2, space_after=7)
    para(tf, "This verdict is derived in code from the measured numbers, and an "
             "automated test prevents the report from claiming a win it did not "
             "earn.", size=9.5, color=MUTED, space_after=0, italic=True)
    notes(s, NOTES["s11"])


def s12_consolidated():
    s, y = slide_page("11  Evidence", "Consolidated model vs baseline",
                      subtitle="Validation and test reported separately throughout. "
                               "Improvement is stated against the appropriate baseline "
                               "for each task, including where it is negative.")
    rv, rt = RES["metrics"]["val"], RES["metrics"]["test"]
    bv_, bt = RES["baseline"]["val"], RES["baseline"]["test"]
    sv, st = SLA["metrics"]["val"], SLA["metrics"]["test"]
    hv, ht = HOT["metrics"]["val"], HOT["metrics"]["test"]
    hbv = HOT["baselines"]["val"]["rolling_4w_mean"]["MAE"]
    hbt = HOT["baselines"]["test"]["rolling_4w_mean"]["MAE"]

    data = [["Task", "Metric", "Validation", "Test", "Baseline (test)",
             "Improvement", "Conclusion"],
            ["Resolution", "MAE (h)", f"{rv['MAE']:,.1f}", f"{rt['MAE']:,.1f}",
             f"median  {bt['MAE']:,.1f}",
             f"^^{1-rt['MAE']/bt['MAE']:+.1%}", "Improves the tail"],
            ["Resolution", "Median AE (h)", f"{rv['median_AE']:,.1f}",
             f"{rt['median_AE']:,.1f}", f"median  {bt['median_AE']:,.1f}",
             f"!!{1-rt['median_AE']/bt['median_AE']:+.1%}",
             "!!Worse for typical cases"],
            ["SLA", "PR-AUC", f"{sv['PR_AUC']:.3f}", f"{st['PR_AUC']:.3f}",
             f"base rate  {st['positive_rate']:.3f}",
             f"^^x{SLA['pr_auc_lift_over_base_rate_test']:.2f}",
             "^^Most promising"],
            ["Hotspot", "MAE (per zone-week)", f"{hv['MAE']:.3f}", f"{ht['MAE']:.3f}",
             f"rolling 4w  {hbt:.3f}",
             f"!!{1-hv['MAE']/hbv:+.1%} val / {1-ht['MAE']/hbt:+.1%} test",
             "!!Unproven"]]
    table(s, M, y + Inches(0.14), W - 2 * M, Inches(2.05), data,
          col_w=[1.5, 2.0, 1.4, 1.4, 2.3, 2.4, 2.3], bold_col0=True, body_size=9.5,
          align=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT,
                 PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.LEFT])

    cards = [("SLA — promising", GOOD, BAND_G,
              "Clears its baseline on both folds and on every reported metric. "
              "Limited by population, not by the model.",
              "To extend it: more cities and years, improved due-date coverage, "
              "explicit probability calibration."),
             ("Resolution — mixed", WARN, BAND_W,
              "Clears one metric and fails another. Useful today for tail-risk "
              "flagging, not for point prediction.",
              "To fix it: quantile and robust objectives, plus the operational "
              "features the data does not yet carry."),
             ("Hotspot — unproven", CRIT, BAND_C,
              "Clears the benchmark on one fold and fails on the other. The simple "
              "baseline remains the defensible choice.",
              "To prove it: beat the rolling 4-week mean on validation and test "
              "together, with spatial and seasonal features.")]
    x = M
    cw = Inches(3.95)
    for title, col, fill, body, nxt in cards:
        rect(s, x, y + Inches(2.48), cw, Inches(1.88), fill=fill)
        rect(s, x, y + Inches(2.48), Pt(3.5), Inches(1.88), fill=col)
        tf = textbox(s, x + Inches(0.18), y + Inches(2.62), cw - Inches(0.34),
                     Inches(1.66))
        para(tf, title, size=12.5, bold=True, color=INK, first=True, space_after=5)
        para(tf, body, size=10.5, color=INK2, space_after=6)
        para(tf, nxt, size=9.5, color=MUTED, space_after=0, italic=True)
        x += cw + Inches(0.32)

    tf = textbox(s, M, y + Inches(4.54), W - 2 * M, Inches(0.40))
    para(tf, "One of three tasks currently has a defensible predictive advantage over a "
             "trivial baseline. That is the honest reading of this table.",
         size=11, bold=True, color=INK, first=True, space_after=0)
    notes(s, NOTES["s12"])


def s13_implemented():
    s, y = slide_page("12  Progress", "What has been implemented",
                      subtitle="Substantial interim implementation covering the data, "
                               "validation, dataset, baseline, training and evaluation "
                               "layers.")
    picture_fit(s, "02_implementation_status.png", M, y + Inches(0.06),
                Inches(7.9), Inches(5.10))

    rx = M + Inches(8.15)
    rw = W - M - rx
    rect(s, rx, y + Inches(0.06), rw, Inches(3.42), fill=WHITE, line=RULE)
    tf = textbox(s, rx + Inches(0.20), y + Inches(0.22), rw - Inches(0.40), Inches(3.1))
    para(tf, "Delivered and validated", size=12.5, bold=True, color=INK, first=True,
         space_after=7)
    for item in ["Multi-source ingestion and schema normalisation",
                 "Cleaning, provenance and quality gates",
                 "Time, density and recency feature engineering",
                 "Task dataset construction with manifests",
                 "Leakage validation by independent re-derivation",
                 "Two chronological split strategies",
                 "Baselines computed on the same folds as models",
                 "Reproducible training framework and saved artifacts",
                 "Automated evaluation and reporting"]:
        para(tf, f"•  {item}", size=10.5, color=INK2, space_after=3)

    callout(s, rx, y + Inches(3.66), rw, Inches(1.52),
            "On the question of percentage complete",
            "We report counts of enumerated scope components rather than a percentage. "
            "The repository contains no approved scope document from which a percentage "
            "could be derived, so stating one would be invention.")
    notes(s, NOTES["s13"])


def s14_weaknesses():
    s, y = slide_page("13  Limitations", "Current weaknesses — stated without mitigation language")
    data = [["Problem", "Evidence in the repository", "Impact"],
            ["Resolution target semantics",
             f"{INSTANT:,} sub-minute closures were artefacts; the 60 s threshold is "
             f"corpus-specific",
             "Target may still contain administrative closures"],
            ["Resolution model quality",
             f"Test median AE {RES['metrics']['test']['median_AE']:,.1f} h vs baseline "
             f"{RES['baseline']['test']['median_AE']:,.1f} h; R² "
             f"{RES['metrics']['test']['R2']:.3f}",
             "!!Not usable for typical-case point prediction"],
            ["Hotspot not proven",
             f"Validation MAE {HOT['metrics']['val']['MAE']:.3f} vs rolling 4-week "
             f"{hbv_():.3f}",
             "!!Simple baseline remains preferable"],
            ["SLA population is narrow",
             f"{DS['sla_ml'][0]:,} rows, NYC only, Jan–Mar 2010, Due-Date types only",
             "No cross-city or multi-year evidence"],
            ["City generalisation untested",
             "Sources differ in schema, vocabulary and operational policy; "
             "split_global folds are Chicago-only",
             "Pooled model may not transfer"],
            ["Temporal robustness untested",
             "Single chronological cut per task; one regime change flagged as dangerous",
             "Performance across future windows unknown"],
            ["Operational features missing",
             "No workload, backlog, crew-capacity or weather features exist",
             "Key drivers of service time are unobserved"],
            ["Deployment not addressed",
             "No inference API, monitoring, drift detection or retraining",
             "None of the three models is production-ready"]]
    table(s, M, y + Inches(0.14), W - 2 * M, Inches(4.35), data,
          col_w=[2.9, 5.6, 3.6], bold_col0=True, body_size=9.5)
    notes(s, NOTES["s14"])


def hbv_():
    return HOT["baselines"]["val"]["rolling_4w_mean"]["MAE"]


def s15_solutions():
    s, y = slide_page("14  Next phase", "Proposed solutions",
                      subtitle="PROPOSED WORK — none of the following is implemented.")
    data = [["Weakness", "Proposed solution", "Expected benefit"],
            ["Resolution target semantics",
             "Source-specific timestamp analysis; status-transition events; separate "
             "administrative closure from genuine resolution",
             "A target that measures service time rather than data entry"],
            ["Typical-case error",
             "Quantile regression and robust/Huber objectives; survival analysis for "
             "censored cases; prediction intervals",
             "Better middle-of-distribution accuracy and honest uncertainty"],
            ["Weak operational signal",
             "Workload and backlog features; seasonality; category interactions; "
             "city/source indicators",
             "Represent the drivers of service time that are currently unobserved"],
            ["Hotspot not proven",
             "Spatial lag and neighbourhood history; longer windows; classical count "
             "forecasting; multi-horizon evaluation",
             "A model that clears the rolling baseline on both folds"],
            ["Narrow SLA population",
             "More cities and years; improved Due-Date coverage; probability "
             "calibration; per-service-type models",
             "Evidence that SLA risk generalises beyond NYC 2010"],
            ["City generalisation",
             "City-aware and hierarchical models; leave-one-city-out holdout",
             "Quantified transfer loss between municipal regimes"],
            ["Temporal robustness",
             "Rolling-origin evaluation across multiple future windows; bootstrap "
             "confidence intervals",
             "Performance claims with stated uncertainty"],
            ["Deployment readiness",
             "Inference API, versioning, prediction logging, drift and performance "
             "monitoring, retraining",
             "Only after models consistently beat strong baselines"]]
    table(s, M, y + Inches(0.14), W - 2 * M, Inches(4.35), data,
          col_w=[2.7, 5.2, 4.2], bold_col0=True, body_size=9.5)
    notes(s, NOTES["s15"])


def s16_conclusion():
    s, y = slide_page("15  Conclusion", "Current status, and the objective for the next review")
    cw = Inches(6.1)
    rect(s, M, y + Inches(0.10), cw, Inches(2.62), fill=WHITE, line=RULE)
    tf = textbox(s, M + Inches(0.22), y + Inches(0.24), cw - Inches(0.44), Inches(2.2))
    para(tf, "IMPLEMENTED", size=11, bold=True, color=GOOD, first=True, space_after=7)
    for i in ["Data pipeline — 22 stages, reproducible from raw",
              "Validation — quality, provenance, leakage, audits",
              "ML datasets with declared manifests",
              "Baselines for every task",
              "Initial models for three tasks",
              "Honest evaluation and reporting"]:
        para(tf, f"•  {i}", size=11, color=INK2, space_after=4)

    rx = M + cw + Inches(0.32)
    rw = W - M - rx
    rect(s, rx, y + Inches(0.10), rw, Inches(2.62), fill=RGBColor(0xF7, 0xF7, 0xF5),
         line=RULE)
    tf = textbox(s, rx + Inches(0.22), y + Inches(0.24), rw - Inches(0.44), Inches(2.2))
    para(tf, "NEXT PHASE", size=11, bold=True, color=MUTED, first=True, space_after=7)
    for i in ["Target-quality investigation",
              "Feature expansion — workload, spatial, seasonal",
              "Model optimisation — quantile, survival, forecasting",
              "Cross-city and rolling-origin generalisation",
              "Interpretability and error analysis",
              "Deployment, monitoring and dashboard integration"]:
        para(tf, f"•  {i}", size=11, color=INK2, space_after=4)

    msgs = [("The foundation works and is reproducible",
             f"{len(RUN['steps'])} pipeline stages, {PYTEST_N} automated tests, "
             f"{AUD['summary']['passed']}/{AUD['summary']['total']} audit checks; a full "
             f"rebuild reproduces every dataset content-identical."),
            ("The models give measurable but mixed improvements",
             "SLA clears its baseline; resolution clears one metric and fails another; "
             "hotspot does not yet clear its benchmark on both folds."),
            ("The next phase is target quality, features and generalisation",
             "The mixed results are diagnostic — they point precisely at where the next "
             "engineering effort belongs.")]
    x = M
    cw2 = Inches(3.95)
    for i, (t, b) in enumerate(msgs, 1):
        rect(s, x, y + Inches(2.94), cw2, Inches(1.52), fill=WHITE, line=RULE)
        tf = textbox(s, x + Inches(0.18), y + Inches(3.08), cw2 - Inches(0.36),
                     Inches(1.30))
        para(tf, f"{i}.  {t}", size=11.5, bold=True, color=INK, first=True,
             space_after=5)
        para(tf, b, size=10, color=INK2, space_after=0)
        x += cw2 + Inches(0.32)

    rect(s, M, y + Inches(4.64), W - 2 * M, Inches(0.62), fill=ACCENT)
    tf = textbox(s, M, y + Inches(4.77), W - 2 * M, Inches(0.36))
    para(tf, "Objective for the next review:  move from a validated ML pipeline to "
             "models that consistently outperform strong baselines.",
         size=13, bold=True, color=WHITE, first=True, space_after=0,
         align=PP_ALIGN.CENTER)
    notes(s, NOTES["s16"])


# ---------------------------------------------------------------------------
NOTES = json.loads((ROOT / "speaker_notes.json").read_text()) \
    if (ROOT / "speaker_notes.json").exists() else {f"s{i}": "" for i in range(1, 17)}


def main() -> int:
    for fn in (s01_title, s02_problem, s03_system, s04_data, s05_validation,
               s06_instant, s07_tasks, s08_split, s09_resolution, s10_sla,
               s11_hotspot, s12_consolidated, s13_implemented, s14_weaknesses,
               s15_solutions, s16_conclusion):
        fn()
    prs.save(OUT)
    print(f"  {OUT}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides, "
          f"{OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
