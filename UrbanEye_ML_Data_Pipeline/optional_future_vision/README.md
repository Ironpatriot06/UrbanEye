# Optional / future: vision datasets

**Nothing in this folder is part of the UrbanEye+ ML data pipeline.**

The citizen selects the incident category in the app, so there is no
image -> category task, and severity is out of the product's current UI scope,
so there is no image -> severity task. These scripts are preserved for the day
that changes.

| Script | Dataset | Licence status |
|---|---|---|
| `download_rdd2022.py` / `preprocess_rdd2022.py` | RDD2022 road damage | Unverified — the download script reads it from the Figshare API and writes it to the manifest |
| `download_smartathon.py` / `preprocess_smartathon.py` | SDAIA Smartathon visual pollution | **Unverified at the original source.** The download script refuses to run without an explicit, recorded risk acknowledgement. |
| `build_vision_samples.py` | builds a `vision_samples` table | — |
| `build_text_dataset.py` | text -> category dataset | Out of scope: the category is citizen-provided |

They reference `scripts.utils.*` and would need import paths and the vision
schema restored before running. See `DATASET_SOURCES.md` for full provenance.

One correctness note worth preserving: RDD2022's **`D50` class marks an INTACT
manhole cover**, not an open manhole. It must never be mapped to `OPEN_MANHOLE` —
that would invert the label's meaning. `config/category_mapping.csv` sends it to
`REVIEW_REQUIRED` and `validate_mappings.py` asserts this (check `MAP-5`).
