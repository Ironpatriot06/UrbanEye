# UrbanEye — 50% progress review visualisation package

Open **`index.html`** in a browser. It is self-contained: the charts are inlined,
so the file can be emailed or printed on its own.

## Regenerating

```bash
python visualization/build_visuals.py
```

Everything is rebuilt from the repository's own outputs — no metric, dataset size
or test count is typed into the chart code. The script reads:

| Source | Used for |
|---|---|
| `UrbanEye_ML_Data_Pipeline/reports/model_training/*.json` | all model and baseline metrics |
| `UrbanEye_ML_Data_Pipeline/reports/{pipeline_run,data_quality,provenance_report,leakage_report,output_validation,pipeline_audit,task_dataset_audit,statistical_audit}.json` | validation status |
| `UrbanEye_ML_Data_Pipeline/data/processed/**/*.parquet` (metadata only) | dataset row and column counts |
| `UrbanEye_ML_Data_Pipeline/data/processed/incidents/*.cleaning_stats.json` | the instant-closure finding |

To refresh the underlying numbers first:

```bash
cd UrbanEye_ML_Data_Pipeline
python scripts/run_pipeline.py          # rebuild datasets + reports
python scripts/baselines/run_baselines.py
python scripts/train/run_all.py         # retrain + re-evaluate
python -m pytest -q                     # test count quoted in the deck
```

## Contents

```
visualization/
├── index.html                 the review deck (self-contained)
├── build_visuals.py           generator — reads reports, writes everything here
├── deck.py                    HTML rendering
├── figures/*.svg              individual charts
└── data/*.csv                 the tables behind the charts
```

## Note on the one hand-maintained number

`PYTEST_COUNT` in `build_visuals.py` records the passing test count, because
pytest does not emit a machine-readable report in this project. Update it from
`python -m pytest -q` if the suite changes. Every other figure is derived.

## Colour

The palette is the validated default from the `dataviz` reference
(`#2a78d6`, `#eb6834`, `#1baf7a`, `#eda100`), checked with the palette validator
for adjacent-pair CVD separation in light mode. Because two of those slots fall
below 3:1 contrast on the light surface, every bar carries a visible direct
label rather than relying on colour alone.
