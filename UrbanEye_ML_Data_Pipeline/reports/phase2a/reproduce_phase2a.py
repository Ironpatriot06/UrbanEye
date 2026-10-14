"""Phase 2A: read-only reproduction of saved-model metrics. Writes only to scratchpad."""
import json, sys, os
import numpy as np, pandas as pd, joblib

ROOT = "/Users/ratishkapoor/Desktop/UrbanEye/UrbanEye_ML_Data_Pipeline"
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
from scripts.train._common import regression_metrics, classification_metrics, load_task  # noqa
from scripts.baselines.run_baselines import rolling_mean_baseline  # noqa

res = {}

def load(task):
    d = f"{ROOT}/models/{task}"
    return joblib.load(f"{d}/model.joblib"), joblib.load(f"{d}/preprocessor.joblib"), json.load(open(f"{d}/metadata.json"))

# ---------------- RESOLUTION ----------------
m, pre, meta = load("resolution")
data = load_task("resolution", "resolution_ml", "split_global")
assert data.split_column == meta["split_column"]
tr, va, te = data.fold("train"), data.fold("val"), data.fold("test")
r = {"rows": {k: len(v) for k, v in (("train", tr), ("val", va), ("test", te))}}
med = float(np.median(tr[data.target].astype(float)))
r["train_median"] = med
preds = {}
for k, part in (("val", va), ("test", te)):
    y = part[data.target].astype(float).to_numpy()
    p = np.clip(np.expm1(m.predict(pre.transform(part))), 0, None)
    preds[k] = p
    r[k] = {"model": regression_metrics(y, p), "baseline_median": regression_metrics(y, np.full(len(y), med))}
    # extra baselines: category median, category x city median (train only)
    for keys, name in ((["category"], "baseline_cat_median"),
                       (["category", "department"], "baseline_cat_dept_median"),
                       (["category", "subcategory", "department"], "baseline_cat_sub_dept_median")):
        g = tr.assign(_k=tr[keys].astype("string").fillna("NA").agg("|".join, axis=1)).groupby("_k")[data.target].median()
        kk = part[keys].astype("string").fillna("NA").agg("|".join, axis=1)
        bp = kk.map(g).fillna(med).to_numpy(float)
        r[k][name] = regression_metrics(y, bp)
    # geometric-mean-like baseline: exp(mean(log1p)) in train, = what a log model predicts with no features
    gm = float(np.expm1(np.log1p(tr[data.target].astype(float)).mean()))
    r[k]["baseline_log_mean"] = regression_metrics(y, np.full(len(y), gm))
r["train_log_mean_hours"] = gm
# error decomposition on test
y = te[data.target].astype(float).to_numpy(); p = preds["test"]
ae_m, ae_b = np.abs(y - p), np.abs(y - med)
bins = [0, 1, 24, 72, 168, 720, 2160, 8760, 1e9]
lab = ["<1h", "1-24h", "1-3d", "3-7d", "7-30d", "30-90d", "90d-1y", ">1y"]
cut = pd.cut(y, bins, labels=lab, right=False)
dec = pd.DataFrame({"bin": cut, "ae_m": ae_m, "ae_b": ae_b, "y": y, "p": p}).groupby("bin", observed=False).agg(
    n=("y", "size"), y_med=("y", "median"), pred_med=("p", "median"), mae_model=("ae_m", "mean"), mae_base=("ae_b", "mean"),
    medae_model=("ae_m", "median"), medae_base=("ae_b", "median"), sum_ae_model=("ae_m", "sum"), sum_ae_base=("ae_b", "sum"))
dec["share_n"] = dec["n"] / dec["n"].sum()
dec["share_total_err_model"] = dec["sum_ae_model"] / dec["sum_ae_model"].sum()
r["test_error_by_true_bin"] = dec.reset_index().to_dict("records")
r["test_pred_quantiles"] = {q: float(np.quantile(p, q)) for q in (0.01, .1, .25, .5, .75, .9, .99)}
r["test_true_quantiles"] = {q: float(np.quantile(y, q)) for q in (0.01, .1, .25, .5, .75, .9, .99)}
r["val_true_quantiles"] = {q: float(np.quantile(va[data.target].astype(float), q)) for q in (.1, .25, .5, .75, .9)}
r["train_true_quantiles"] = {q: float(np.quantile(tr[data.target].astype(float), q)) for q in (.1, .25, .5, .75, .9)}
r["share_model_better_than_baseline_test"] = float((ae_m < ae_b).mean())
# log-space metrics: what the model was actually trained to minimise
r["test_log1p_MAE"] = {"model": float(np.mean(np.abs(np.log1p(y) - np.log1p(p)))),
                       "baseline_median": float(np.mean(np.abs(np.log1p(y) - np.log1p(med))))}
# bootstrap CI of MAE difference, test
rng = np.random.default_rng(0)
diffs_mae, diffs_med = [], []
n = len(y)
for _ in range(200):
    i = rng.integers(0, n, n)
    diffs_mae.append(ae_b[i].mean() - ae_m[i].mean())
    diffs_med.append(np.median(ae_b[i]) - np.median(ae_m[i]))
r["bootstrap_test_MAE_improvement_hours_CI95"] = [float(np.quantile(diffs_mae, .025)), float(np.quantile(diffs_mae, .975))]
r["bootstrap_test_medAE_improvement_hours_CI95"] = [float(np.quantile(diffs_med, .025)), float(np.quantile(diffs_med, .975))]
# per-category on test
pc = []
for c, g in te.assign(_p=p, _y=y).groupby("category", observed=True):
    pc.append({"category": c, "n": len(g), "y_med": float(g._y.median()),
               "mae_model": float((g._y - g._p).abs().mean()), "mae_base": float((g._y - med).abs().mean()),
               "medae_model": float((g._y - g._p).abs().median()), "medae_base": float((g._y - med).abs().median())})
r["test_by_category"] = pc
# feature importance (permutation on a test subsample)
from sklearn.inspection import permutation_importance
sub = te.sample(30000, random_state=0)
Xs = pre.transform(sub); ys = np.log1p(sub[data.target].astype(float))
pi = permutation_importance(m, Xs, ys, n_repeats=3, random_state=0, scoring="neg_mean_absolute_error")
r["perm_importance_log_mae_test"] = sorted(({"f": f, "imp": float(v)} for f, v in zip(pre.predictors, pi.importances_mean)), key=lambda d: -d["imp"])
res["resolution"] = r
print("resolution done", flush=True)

# ---------------- SLA ----------------
m, pre, meta = load("sla")
data = load_task("sla", "sla_ml", "split_global")
assert data.split_column == meta["split_column"], (data.split_column, meta["split_column"])
tr, va, te = data.fold("train"), data.fold("val"), data.fold("test")
y_of = lambda part: part[data.target].astype("boolean").fillna(False).to_numpy(bool).astype(int)
s = {"rows": {k: len(v) for k, v in (("train", tr), ("val", va), ("test", te))},
     "threshold": meta["decision_threshold"], "base_rate_train": float(y_of(tr).mean())}
for k, part in (("val", va), ("test", te)):
    yy = y_of(part); pp = m.predict_proba(pre.transform(part))[:, 1]
    s[k] = {"model": classification_metrics(yy, pp, meta["decision_threshold"]),
            "baseline": classification_metrics(yy, np.full(len(yy), s["base_rate_train"]), 0.5)}
    s[k]["thresholds"] = {str(t): {kk: v for kk, v in classification_metrics(yy, pp, t).items()
                                   if kk in ("precision", "recall", "f1", "confusion_matrix")}
                          for t in (0.05, 0.1, 0.1666, 0.2, 0.3, 0.4, 0.5)}
    s[k]["share_flagged_at_thr"] = float((pp >= meta["decision_threshold"]).mean())
    # rule baselines: per-subcategory breach rate from train
    for keys, name in ((["subcategory"], "subcat_rate"), (["category"], "cat_rate"),
                       (["sla_target_hours"], "slatarget_rate"), (["department", "subcategory"], "dept_subcat_rate")):
        kt = tr[keys].astype("string").fillna("NA").agg("|".join, axis=1)
        rate = pd.Series(y_of(tr), index=tr.index).groupby(kt).mean()
        kp = part[keys].astype("string").fillna("NA").agg("|".join, axis=1)
        bp = kp.map(rate).fillna(s["base_rate_train"]).to_numpy(float)
        from sklearn.metrics import average_precision_score, roc_auc_score
        s[k][name] = {"PR_AUC": float(average_precision_score(yy, bp)), "ROC_AUC": float(roc_auc_score(yy, bp))}
sub = te
pi = permutation_importance(m, pre.transform(sub), y_of(sub), n_repeats=5, random_state=0, scoring="average_precision")
s["perm_importance_prauc_test"] = sorted(({"f": f, "imp": float(v)} for f, v in zip(pre.predictors, pi.importances_mean)), key=lambda d: -d["imp"])
res["sla"] = s
print("sla done", flush=True)

# ---------------- HOTSPOT ----------------
m, pre, meta = load("hotspot")
data = load_task("hotspot", "hotspot_ml", "split_global")
panel = data.frame.copy()
panel["_roll4"] = rolling_mean_baseline(panel, 4, lagged=False)
panel["_roll4_lagged"] = rolling_mean_baseline(panel, 4, lagged=True)
data.frame = panel
tr, va, te = data.fold("train"), data.fold("val"), data.fold("test")
h = {"rows": {k: len(v) for k, v in (("train", tr), ("val", va), ("test", te))}}
for k, part in (("val", va), ("test", te)):
    yy = part[data.target].astype(float).to_numpy()
    pp = np.clip(np.expm1(m.predict(pre.transform(part))), 0, None)
    cur = part["incident_count"].astype(float)
    h[k] = {"model": regression_metrics(yy, pp),
            "rolling_4w_mean": regression_metrics(yy, part["_roll4"].fillna(cur).to_numpy(float)),
            "rolling_4w_mean_lagged": regression_metrics(yy, part["_roll4_lagged"].fillna(cur).to_numpy(float)),
            "persistence": regression_metrics(yy, cur.to_numpy(float))}
    # rolling 8/12-week means as extra naive baselines (computed same way)
    h[k]["cities"] = part["city"].value_counts().to_dict()
    h[k]["weeks"] = sorted(part["week_start"].astype(str).unique().tolist())
    ae_m = np.abs(yy - pp); ae_r = np.abs(yy - part["_roll4"].fillna(cur).to_numpy(float))
    h[k]["share_rows_model_better"] = float((ae_m < ae_r).mean())
    h[k]["share_rows_tie"] = float((ae_m == ae_r).mean())
    # per-week MAE comparison
    wk = pd.DataFrame({"w": part["week_start"].astype(str).to_numpy(), "m": ae_m, "r": ae_r}).groupby("w").mean()
    h[k]["per_week_mae"] = wk.round(3).reset_index().to_dict("records")
    rng = np.random.default_rng(0)
    # block bootstrap by series (zone x category) for MAE difference
    sk = (part["zone_key"].astype(str) + "|" + part["category"].astype(str)).to_numpy()
    useries = np.unique(sk); idx_by = {u: np.flatnonzero(sk == u) for u in useries}
    d = []
    for _ in range(500):
        pick = rng.choice(useries, len(useries))
        ii = np.concatenate([idx_by[u] for u in pick])
        d.append(ae_r[ii].mean() - ae_m[ii].mean())
    h[k]["series_bootstrap_MAE_gain_vs_roll4_CI95"] = [float(np.quantile(d, .025)), float(np.quantile(d, .975))]
    # zero-target behaviour
    z = yy == 0
    h[k]["zero_target_share"] = float(z.mean())
res["hotspot"] = h
print("hotspot done", flush=True)

json.dump(res, open(f"{OUT}/reproduced_metrics.json", "w"), indent=1, default=str)
print("wrote", f"{OUT}/reproduced_metrics.json")
