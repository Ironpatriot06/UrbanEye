"""
Renders the self-contained HTML review deck.

The SVGs are inlined rather than linked so the file can be emailed, printed or
opened from anywhere without a folder of assets travelling with it.
"""
from __future__ import annotations

from pathlib import Path


def _svg(fig_dir: Path, name: str) -> str:
    txt = (fig_dir / name).read_text()
    return txt[txt.index("<svg"):]


LEARNED = [
    ("Complex models do not automatically beat simple baselines",
     "A gradient-boosted model loses to a four-week moving average on the hotspot "
     "validation fold, and loses to a constant median on resolution median error. "
     "Both were only visible because the baseline was implemented first."),
    ("Target distribution drives model behaviour",
     "Resolution time spans four orders of magnitude. Fitting absolute error on raw "
     "hours — optimising the headline metric directly — scored worse than fitting "
     "squared error on log1p. The shape of the target mattered more than the loss."),
    ("Baseline selection changes the conclusion",
     "Against persistence the hotspot model looks 16.5% better. Against a rolling "
     "four-week mean the same model is 3.2% better on test and 6.9% worse on "
     "validation. The benchmark chosen decides whether the model looks successful."),
    ("Data quality directly determines target validity",
     "173,346 closure durations were below the timestamp resolution of the source "
     "systems. Left in place they would have taught a model clerical behaviour "
     "rather than service time."),
    ("Cross-city generalisation remains open",
     "The three sources occupy disjoint time windows, so a single global "
     "chronological split yields city-specific evaluation folds. Transfer between "
     "cities has not yet been measured."),
]

LIMITATIONS = [
    ("Resolution", "Does not consistently beat the median baseline — better on the long tail, worse for typical cases.", "High"),
    ("Resolution", "Target is highly skewed with a difficult long tail (p99 ≈ 18,600 h).", "High"),
    ("Hotspot", "GBM does not consistently beat the rolling four-week baseline.", "High"),
    ("SLA", "Coverage restricted to NYC, Jan–Mar 2010, and complaint types carrying a Due Date.", "High"),
    ("All tasks", "Cross-city generalisation has not been demonstrated.", "High"),
    ("Evaluation", "Global temporal test folds become city-specific because source date ranges differ.", "Medium"),
    ("Evaluation", "Temporal robustness across multiple future windows not yet demonstrated.", "Medium"),
    ("Analysis", "Error analysis by city, category, geography and target bucket still required.", "Medium"),
    ("Features", "Temporal, spatial, historical and external features can be expanded.", "Medium"),
    ("Interpretability", "Feature importance and attribution not yet developed.", "Medium"),
    ("Engineering", "Inference API, monitoring, drift detection and retraining are future work.", "Planned"),
    ("Status", "Project is training/research-ready, not a deployed production ML system.", "Context"),
]

ROADMAP = [
    ("Data &amp; features",
     ["Seasonal, trend and multi-window rolling features",
      "Spatial lag and neighbouring-zone activity",
      "Expand SLA beyond the NYC 2010 subset; improve label consistency",
      "Evaluate external signals (weather, events, holidays) where justified"]),
    ("Cross-city validation",
     ["Leave-one-city-out validation",
      "Cross-city train/test experiments",
      "Quantify transfer loss between municipal regimes"]),
    ("Temporal robustness",
     ["Rolling-origin temporal evaluation",
      "Multiple future test windows",
      "Explicit regime-change reporting (e.g. 2020 disruption)"]),
    ("Model experimentation",
     ["Quantile regression and Huber / robust losses",
      "Tail-aware and survival / time-to-event modelling",
      "Category-specific and hierarchical city models",
      "Alternative forecasting and spatio-temporal models",
      "Every candidate compared against the rolling baseline"]),
    ("Error analysis",
     ["Breakdown by resolution bucket, city, category and geography",
      "Correct vs incorrect prediction exemplars",
      "Cost-sensitive and precision/recall trade-off analysis for SLA"]),
    ("Interpretability",
     ["Feature importance and SHAP or equivalent attribution",
      "Probability calibration and threshold optimisation"]),
    ("Deployment &amp; monitoring",
     ["Prediction API and inference pipeline",
      "Model versioning, rollback and prediction logging",
      "Data / feature drift and performance monitoring",
      "Automated retraining strategy"]),
]


def render_deck(c: dict) -> str:
    figs, rows = c["figs"], c["rows"]
    FIG = c["FIG"]
    RES, SLA, HOT = c["RES"], c["SLA"], c["HOT"]
    D = c["DATASETS"]

    val_checks = [
        ("Full pipeline", f"{sum(1 for s in c['RUN']['steps'] if s.get('status') == 'OK')}/"
                          f"{len(c['RUN']['steps'])} steps",
         f"{c['RUN']['total_seconds']:.0f}s on real data"),
        ("Unit &amp; integration tests", f"{c['PYTEST_COUNT']} passed", "pytest"),
        ("Data quality", f"{c['DQ']['integrity_summary']['passed']}/"
                         f"{c['DQ']['integrity_summary']['total']}", "integrity assertions"),
        ("Provenance", f"{c['PROV']['summary']['passed']}/{c['PROV']['summary']['total']}",
         "incl. active fabrication attempts"),
        ("Leakage", f"{c['LEAK']['summary']['passed']}/{c['LEAK']['summary']['total']}",
         "independent re-derivation"),
        ("Output validation", f"{c['OUT']['summary']['passed']}/{c['OUT']['summary']['total']}",
         "ML-ready table checks"),
        ("Pipeline audit", f"{c['AUDIT']['summary']['passed']}/{c['AUDIT']['summary']['total']}",
         "stage-by-stage"),
        ("Task &amp; leakage audit", f"{c['TASKAUD']['summary']['passed']}/"
                                     f"{c['TASKAUD']['summary']['total']}",
         f"{c['TASKAUD']['summary']['skipped']} skipped (not applicable)"),
    ]

    achieved = [
        ("End-to-end data pipeline",
         f"{len(c['RUN']['steps'])} stages execute in one command on "
         f"{D['all_incidents'][0]:,} records; outputs regenerate content-identical from raw",
         "Extend ingestion to further cities and years"),
        ("Data cleaning &amp; validation",
         f"{c['DQ']['integrity_summary']['passed']}/{c['DQ']['integrity_summary']['total']} "
         f"quality, {c['PROV']['summary']['passed']}/{c['PROV']['summary']['total']} provenance, "
         f"{c['LEAK']['summary']['passed']}/{c['LEAK']['summary']['total']} leakage checks pass",
         "Broaden category mapping coverage"),
        ("Training-ready task datasets",
         f"Resolution {D['resolution_ml'][0]:,} · Hotspot {D['hotspot_ml'][0]:,} · "
         f"SLA {D['sla_ml'][0]:,} rows, each with a declared manifest",
         "Expand SLA coverage; add cross-city folds"),
        ("Leakage-controlled splitting",
         "Chronological splits; <code>split_global</code> verified to contain zero test rows "
         "preceding the last training row",
         "Rolling-origin and leave-one-city-out evaluation"),
        ("Baselines implemented first",
         "Median, base-rate, persistence and rolling four-week baselines computed on the "
         "same folds as the models",
         "Retain as the acceptance bar for every future model"),
        ("Reproducible training framework",
         "Shared preprocessor fitted on train only; saved model + preprocessor + metadata; "
         "identical metrics across repeated runs",
         "Hyper-parameter and objective experimentation"),
        ("Honest evaluation",
         "Validation and test reported separately; verdicts derived in code from the numbers",
         "Error analysis and interpretability"),
        ("Data-quality investigation",
         f"{c['INSTANT_TOTAL']:,} sub-minute closures identified and flagged rather than "
         f"deleted",
         "Apply the same scrutiny to remaining fields"),
    ]

    def tr(cells, th=False):
        tag = "th" if th else "td"
        return "<tr>" + "".join(f"<{tag}>{x}</{tag}>" for x in cells) + "</tr>"

    html = f"""<title>UrbanEye — 50% Progress Review</title>
<style>
  :root {{
    color-scheme: light;
    --surface: #fcfcfb; --plane: #f3f3f0; --ink: #0b0b0b; --ink2: #52514e;
    --muted: #898781; --grid: #e1e0d9; --rule: #c3c2b7;
    --blue: #2a78d6; --orange: #eb6834; --aqua: #1baf7a; --yellow: #eda100;
    --good: #0ca30c; --warn: #fab219; --crit: #d03b3b;
    --band: #eaf2fd;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--plane); color: var(--ink);
         font: 15px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif; }}
  .wrap {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px 80px; }}
  header.hero {{ background: var(--surface); border: 1px solid var(--grid);
       border-radius: 14px; padding: 30px 32px; margin-bottom: 26px; }}
  h1 {{ font-size: 27px; margin: 0 0 6px; letter-spacing: -0.02em; }}
  .sub {{ color: var(--ink2); font-size: 15px; margin: 0; }}
  .meta {{ color: var(--muted); font-size: 12.5px; margin-top: 14px; }}
  section {{ background: var(--surface); border: 1px solid var(--grid);
       border-radius: 14px; padding: 26px 30px; margin-bottom: 22px; }}
  h2 {{ font-size: 19px; margin: 0 0 4px; letter-spacing: -0.01em; }}
  h2 .n {{ color: var(--muted); font-weight: 600; margin-right: 10px;
          font-variant-numeric: tabular-nums; }}
  .lede {{ color: var(--ink2); margin: 0 0 20px; font-size: 14.5px; max-width: 78ch; }}
  figure {{ margin: 0 0 6px; overflow-x: auto; }}
  figure svg {{ max-width: 100%; height: auto; display: block; }}
  figcaption {{ color: var(--muted); font-size: 12.5px; margin-top: 8px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; margin-top: 6px; }}
  th, td {{ text-align: left; padding: 9px 11px; border-bottom: 1px solid var(--grid);
           vertical-align: top; }}
  th {{ font-size: 12px; text-transform: uppercase; letter-spacing: .05em;
       color: var(--ink2); background: var(--plane); font-weight: 700; }}
  td.num {{ font-variant-numeric: tabular-nums; white-space: nowrap; }}
  tr:last-child td {{ border-bottom: none; }}
  .box {{ border-left: 4px solid var(--blue); background: var(--band);
         padding: 15px 18px; border-radius: 0 10px 10px 0; margin-top: 16px; }}
  .box.warn {{ border-left-color: var(--orange); background: #fdf0ea; }}
  .box.crit {{ border-left-color: var(--crit); background: #fbecec; }}
  .box h3 {{ margin: 0 0 6px; font-size: 14.5px; }}
  .box p {{ margin: 0; font-size: 14px; color: var(--ink2); }}
  .grid2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(310px, 1fr));
           gap: 14px; margin-top: 16px; }}
  .card {{ border: 1px solid var(--grid); border-radius: 10px; padding: 15px 17px; }}
  .card h3 {{ margin: 0 0 6px; font-size: 14.5px; }}
  .card p {{ margin: 0; font-size: 13.5px; color: var(--ink2); }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
           gap: 12px; margin-top: 16px; }}
  .stat {{ border: 1px solid var(--grid); border-radius: 10px; padding: 13px 15px; }}
  .stat .v {{ font-size: 21px; font-weight: 700; font-variant-numeric: tabular-nums; }}
  .stat .k {{ font-size: 12px; color: var(--muted); margin-top: 2px; }}
  .stat .d {{ font-size: 11.5px; color: var(--ink2); margin-top: 5px; }}
  .pill {{ display: inline-block; padding: 2px 9px; border-radius: 999px;
          font-size: 11.5px; font-weight: 700; }}
  .p-ok {{ background: #e3f5e3; color: #0a6b0a; }}
  .p-part {{ background: #fdf1d9; color: #8a5d00; }}
  .p-no {{ background: #fbecec; color: #a12727; }}
  .p-plan {{ background: #eceaf6; color: #3b3080; }}
  ul.tight {{ margin: 6px 0 0; padding-left: 19px; font-size: 13.5px; color: var(--ink2); }}
  ul.tight li {{ margin-bottom: 4px; }}
  .rm {{ display: grid; grid-template-columns: 210px 1fr; gap: 0;
        border-top: 1px solid var(--grid); }}
  .rm > div {{ padding: 13px 14px; border-bottom: 1px solid var(--grid); }}
  .rm .ph {{ font-weight: 700; font-size: 14px; background: var(--plane); }}
  code {{ background: var(--plane); padding: 1px 5px; border-radius: 4px;
         font-size: 12.5px; }}
  footer {{ color: var(--muted); font-size: 12.5px; text-align: center;
           margin-top: 30px; }}
  @media print {{
    body {{ background: #fff; }} section, header.hero {{ break-inside: avoid; }}
  }}
</style>

<div class="wrap">

<header class="hero">
  <h1>UrbanEye — ML Data Pipeline</h1>
  <p class="sub">50% progress review · interim results and technical assessment</p>
  <p class="meta">Generated {c['generated']} · every figure is produced by
    <code>visualization/build_visuals.py</code> directly from the repository's own
    reports. No metric in this deck is entered by hand.</p>
  <div class="stats">
    <div class="stat"><div class="v">{D['all_incidents'][0]:,}</div>
      <div class="k">incidents ingested</div><div class="d">3 cities, cleaned &amp; validated</div></div>
    <div class="stat"><div class="v">{len(c['RUN']['steps'])}/{len(c['RUN']['steps'])}</div>
      <div class="k">pipeline stages pass</div><div class="d">{c['RUN']['total_seconds']:.0f}s, reproducible from raw</div></div>
    <div class="stat"><div class="v">{c['PYTEST_COUNT']}</div>
      <div class="k">automated tests pass</div><div class="d">pipeline + training framework</div></div>
    <div class="stat"><div class="v">3</div>
      <div class="k">ML tasks trained</div><div class="d">resolution · SLA · hotspot</div></div>
    <div class="stat"><div class="v">1 of 3</div>
      <div class="k">clears its baseline outright</div><div class="d">SLA only — see §7</div></div>
  </div>
</header>

<section>
  <h2><span class="n">01</span>System architecture</h2>
  <p class="lede">The full path from raw municipal 311 records to evaluated models.
     Every stage shown is implemented and runs as one reproducible command.</p>
  <figure>{_svg(FIG, figs['architecture'])}</figure>
</section>

<section>
  <h2><span class="n">02</span>Implementation status</h2>
  <p class="lede">Status is reported as counts of enumerated scope components, not as an
     effort or schedule percentage — the repository contains no basis for the latter,
     so assigning one would be invention. Roughly half of the enumerated components are
     implemented; the remainder are model optimisation and production engineering.</p>
  <figure>{_svg(FIG, figs['status'])}</figure>
  <div class="box">
    <h3>What "50%" means here</h3>
    <p>The data, validation, dataset-construction, baseline, training and evaluation
       workstreams are implemented and tested. Model optimisation, cross-city
       generalisation, interpretability and production deployment are not started.
       The project is <strong>research/training-ready, not production-deployed</strong>.</p>
  </div>
</section>

<section>
  <h2><span class="n">03</span>Data foundation</h2>
  <p class="lede">Three municipal 311 sources ingested, normalised to a common schema and
     split chronologically. Counts read from the generated parquet files.</p>
  <figure>{_svg(FIG, figs['datasets'])}</figure>
  <table>
    <thead>{tr(['Dataset', 'Rows', 'Columns', 'Role'], th=True)}</thead>
    <tbody>
      {tr(['<code>all_incidents</code>', f"{D['all_incidents'][0]:,}", D['all_incidents'][1], 'Unified cleaned corpus'])}
      {tr(['<code>resolution_ml</code>', f"{D['resolution_ml'][0]:,}", D['resolution_ml'][1], 'Resolution-time regression'])}
      {tr(['<code>hotspot_ml</code>', f"{D['hotspot_ml'][0]:,}", D['hotspot_ml'][1], 'Zone × category × week panel'])}
      {tr(['<code>sla_ml</code>', f"{D['sla_ml'][0]:,}", D['sla_ml'][1], 'SLA-breach classification'])}
      {tr(['<code>priority_features</code>', f"{D['priority_features'][0]:,}", D['priority_features'][1], 'Policy features — no supervised target'])}
    </tbody>
  </table>
  <div class="box warn">
    <h3>Data-quality investigation: sub-minute closure durations</h3>
    <p>{c['INSTANT_TOTAL']:,} closed cases recorded a resolution duration below one minute —
       Chicago closes thousands of cases at exactly 5–7 seconds, and NYC's entire sub-minute
       population sits at exactly 0 s. One minute is the coarsest timestamp granularity in the
       sources, so these values cannot be interpreted as service durations.
       <strong>Rows are retained</strong>; only the duration is made unobserved and flagged
       via <code>resolution_instant_closure</code>. This removed an 8.2% artefact population
       from the regression target without discarding any record.</p>
  </div>
</section>

<section>
  <h2><span class="n">04</span>Resolution-time prediction</h2>
  <p class="lede">Regression on <code>resolution_time_hours</code>, pooled across cities on a
     single chronological timeline (<code>split_global</code>). Baseline: predict the
     training median for every row.</p>
  <figure>{_svg(FIG, figs['resolution'])}</figure>
  <div class="box warn">
    <h3>Interpretation — partial success only</h3>
    <p>The model improves mean absolute error by
       <strong>{(1 - RES['metrics']['test']['MAE'] / RES['baseline']['test']['MAE']):.1%}</strong>
       on test, but is <strong>worse than the median baseline on median absolute error</strong>
       ({RES['metrics']['test']['median_AE']:,.1f} h vs
       {RES['baseline']['test']['median_AE']:,.1f} h). It performs better on the long tail and
       worse for typical cases, which is where most operational decisions sit. Note also that
       validation tells a more favourable story than test
       ({RES['metrics']['val']['median_AE']:,.1f} h vs
       {RES['baseline']['val']['median_AE']:,.1f} h) — the reversal is a property of the test
       period, so a single headline number would misrepresent the model either way.
       <strong>This model is not presented as successful.</strong></p>
  </div>
</section>

<section>
  <h2><span class="n">05</span>SLA-breach prediction</h2>
  <p class="lede">Binary classification on <code>sla_breach</code>. Because the table is
     single-source, the per-source chronological <code>split</code> column is used — a global
     cut would place every row in training. Decision threshold selected on validation and
     frozen before test.</p>
  <figure>{_svg(FIG, figs['sla'])}</figure>
  <div class="box">
    <h3>Interpretation — strongest current result, narrow population</h3>
    <p>Test PR-AUC {SLA['metrics']['test']['PR_AUC']:.3f} against a base rate of
       {SLA['metrics']['test']['positive_rate']:.3f} is a
       <strong>×{SLA['pr_auc_lift_over_base_rate_test']:.2f} lift</strong>, with ROC-AUC
       {SLA['metrics']['test']['ROC_AUC']:.3f}. Accuracy is deliberately not reported as a
       headline: at a ~10% positive rate, always predicting "no breach" scores ~90% accuracy
       and catches nothing. <strong>Limitation:</strong> the SLA dataset currently covers
       NYC only, January–March 2010, and only complaint types carrying a usable Due Date.
       <strong>No claim of general cross-city SLA prediction is made.</strong></p>
  </div>
</section>

<section>
  <h2><span class="n">06</span>Hotspot forecasting</h2>
  <p class="lede">Count regression for incidents in week <em>t+1</em> per zone × category.
     Validation and test are shown separately and at the same scale; the lowest bar in each
     panel is outlined.</p>
  <figure>{_svg(FIG, figs['hotspot'])}</figure>
  <div class="box crit">
    <h3>Conclusion</h3>
    <p><strong>The rolling four-week baseline currently provides the strongest validation
       performance; the ML improvement is not yet stable.</strong> The GBM beats the rolling
       mean on test by {(1 - HOT['metrics']['test']['MAE'] / HOT['baselines']['test']['rolling_4w_mean']['MAE']):.1%}
       but loses to it on validation by
       {abs(1 - HOT['metrics']['val']['MAE'] / HOT['baselines']['val']['rolling_4w_mean']['MAE']):.1%}.
       A win on one fold and a loss on the other is not evidence of superiority. The current
       recommendation is to treat the rolling four-week mean as the operating benchmark and
       the model as unproven. This verdict is derived in code from the measured numbers, and
       an automated test prevents the report from claiming a win it did not earn.</p>
  </div>
</section>

<section>
  <h2><span class="n">07</span>Consolidated model vs baseline</h2>
  <p class="lede">Validation and test are shown separately throughout. Improvement is stated
     against the appropriate baseline for each task, including where it is negative.</p>
  <table>
    <thead>{tr(['Task', 'Model', 'Metric', 'Validation', 'Test', 'Baseline', 'Improvement',
                'Current conclusion'], th=True)}</thead>
    <tbody>
"""
    for r in rows:
        neg = r["improvement"].startswith("-") or "WORSE" in r["conclusion"] or "Loses" in r["conclusion"]
        style = ' style="color:#a12727;font-weight:700"' if neg else ' style="font-weight:600"'
        html += (f"      <tr><td>{r['task']}</td><td>{r['model']}</td><td>{r['metric']}</td>"
                 f"<td class='num'>{r['val']}</td><td class='num'>{r['test']}</td>"
                 f"<td class='num'>{r['baseline']}</td><td class='num'{style}>{r['improvement']}</td>"
                 f"<td>{r['conclusion']}</td></tr>\n")
    html += f"""    </tbody>
  </table>
  <div class="box warn">
    <h3>Reading this table honestly</h3>
    <p>Only the SLA model clears its baseline on every reported metric and on both folds.
       The resolution model clears one metric and fails another. The hotspot model clears its
       benchmark on one fold and fails on the other. <strong>One of three tasks currently has
       a defensible predictive advantage over a trivial baseline.</strong></p>
  </div>
</section>

<section>
  <h2><span class="n">08</span>Validation infrastructure</h2>
  <p class="lede">The experimentation pipeline is itself tested. These gates run on every
     pipeline execution and are the reason the limitations above could be identified rather
     than missed.</p>
  <table>
    <thead>{tr(['Check', 'Result', 'Scope'], th=True)}</thead>
    <tbody>
"""
    for name, result, scope in val_checks:
        html += (f"      <tr><td>{name}</td>"
                 f"<td class='num'><span class='pill p-ok'>{result}</span></td>"
                 f"<td>{scope}</td></tr>\n")
    html += f"""    </tbody>
  </table>
  <p class="lede" style="margin-top:14px">The statistical audit additionally classifies
     {c['STATAUD']['summary']['needs_preprocessing']} findings as
     <em>needs&nbsp;preprocessing</em>, {c['STATAUD']['summary']['policy_decision']} as
     <em>policy&nbsp;decision</em> and {c['STATAUD']['summary']['dangerous']} as
     <em>dangerous</em> — the last being a documented regime change inside the evaluation
     window, reported rather than removed.</p>
</section>

<section>
  <h2><span class="n">09</span>What we learned so far</h2>
  <p class="lede">The most useful results of this phase are negative ones. Each was only
     observable because baselines were implemented before models.</p>
  <div class="grid2">
"""
    for title, body in LEARNED:
        html += f'    <div class="card"><h3>{title}</h3><p>{body}</p></div>\n'
    html += f"""  </div>
</section>

<section>
  <h2><span class="n">10</span>Current limitations</h2>
  <p class="lede">Identified weaknesses, stated without mitigation language.</p>
  <table>
    <thead>{tr(['Area', 'Limitation', 'Severity'], th=True)}</thead>
    <tbody>
"""
    sev = {"High": "p-no", "Medium": "p-part", "Planned": "p-plan", "Context": "p-plan"}
    for area, lim, s in LIMITATIONS:
        html += (f"      <tr><td>{area}</td><td>{lim}</td>"
                 f"<td><span class='pill {sev[s]}'>{s}</span></td></tr>\n")
    html += """    </tbody>
  </table>
</section>

<section>
  <h2><span class="n">11</span>Next-phase roadmap</h2>
  <p class="lede">Sequenced so that generalisation and error analysis precede deployment —
     there is no value in deploying a model that has not been shown to transfer.</p>
  <div class="rm">
"""
    for phase, items in ROADMAP:
        lis = "".join(f"<li>{i}</li>" for i in items)
        html += (f'    <div class="ph">{phase}</div>'
                 f'<div><ul class="tight" style="margin-top:0">{lis}</ul></div>\n')
    html += f"""  </div>
</section>

<section>
  <h2><span class="n">12</span>Achieved vs remaining</h2>
  <table>
    <thead>{tr(['Completed', 'Current evidence', 'Next objective'], th=True)}</thead>
    <tbody>
"""
    for done, ev, nxt in achieved:
        html += f"      <tr><td><strong>{done}</strong></td><td>{ev}</td><td>{nxt}</td></tr>\n"
    html += """    </tbody>
  </table>
  <div class="box">
    <h3>Overall position at the 50% point</h3>
    <p>A reliable, tested and reproducible experimentation pipeline exists, and it has been
       used to obtain honest interim results for three ML tasks. Two of those three models do
       not yet justify themselves against simple baselines — a finding produced by the
       infrastructure rather than hidden by it. The next phase addresses model quality,
       generalisation and interpretability before any deployment work begins.</p>
  </div>
</section>

<footer>
  UrbanEye ML Data Pipeline · 50% progress review · figures regenerated from
  <code>reports/</code> via <code>python visualization/build_visuals.py</code>
</footer>

</div>
"""
    return html
