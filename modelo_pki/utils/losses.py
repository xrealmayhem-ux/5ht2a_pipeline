import numpy as np
from scipy.stats import spearmanr, pearsonr

from chemprop import data


ACTIVITY_BINS = [(4.5, 6.0, "<6"), (6.0, 7.0, "6-7"),
                 (7.0, 8.0, "7-8"), (8.0, 9.0, "8-9"), (9.0, 10.5, ">9")]


def compute_metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    residuals = true - pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    pr = float(pearsonr(true, pred)[0]) if len(true) > 1 else 0.0
    sr = float(spearmanr(true, pred)[0]) if len(true) > 1 else 0.0
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Pearson_r": pr, "Spearman": sr}


def compute_bin_metrics(true: np.ndarray, pred: np.ndarray, bins=ACTIVITY_BINS):
    results = {}
    for lo, hi, label in bins:
        mask = (true >= lo) & (true < hi)
        n = mask.sum()
        if n < 3:
            results[label] = {"n": int(n), "RMSE": float("nan"), "MAE": float("nan"), "mean_true": float("nan"), "mean_pred": float("nan")}
            continue
        t, p = true[mask], pred[mask]
        rmse = float(np.sqrt(np.mean((t - p) ** 2)))
        mae = float(np.mean(np.abs(t - p)))
        results[label] = {
            "n": int(n),
            "RMSE": rmse,
            "MAE": mae,
            "mean_true": float(t.mean()),
            "mean_pred": float(p.mean()),
            "bias": float(p.mean() - t.mean()),
        }
    return results


def compute_enrichment(true: np.ndarray, pred: np.ndarray, top_k=0.05):
    n_top = max(1, int(len(true) * top_k))
    idx = np.argsort(pred)[::-1][:n_top]
    mean_top_actual = float(true[idx].mean())
    mean_all_actual = float(true.mean())
    enrichment = mean_top_actual / mean_all_actual if mean_all_actual > 0 else 1.0
    return {"top_k": top_k, "n_top": n_top,
            "mean_top_actual": mean_top_actual,
            "mean_all_actual": mean_all_actual,
            "enrichment_factor": enrichment}


def make_stratified_scaffold_split(mols, ys, split_sizes, bins, seed=42):
    all_train, all_val, all_test = [], [], []
    for bin_id in np.unique(bins):
        mask = np.where(bins == bin_id)[0]
        n = len(mask)
        if n < 10:
            all_train.extend(mask.tolist())
            continue
        bin_mols = [mols[int(i)] for i in mask]
        if n < 20:
            tr, te = data.make_split_indices(
                bin_mols, "scaffold_balanced", (split_sizes[0], split_sizes[2]), seed=seed
            )
            all_train.extend(mask[tr[0]].tolist())
            all_test.extend(mask[te[0]].tolist())
        else:
            tr, va, te = data.make_split_indices(
                bin_mols, "scaffold_balanced", split_sizes, seed=seed
            )
            all_train.extend(mask[tr[0]].tolist())
            all_val.extend(mask[va[0]].tolist())
            all_test.extend(mask[te[0]].tolist())
    return all_train, all_val, all_test


def compute_activity_bins(ys, boundaries=(5, 6, 7, 8, 9)):
    return np.digitize(ys.squeeze(), boundaries)


def print_comparison(current_metrics: dict, new_metrics: dict, label: str = "CheMeleon"):
    print(f"\n{'='*55}")
    print(f"  Performance comparison: Original vs {label}")
    print(f"{'='*55}")
    keys = [k for k in ["RMSE", "MAE", "R2", "Pearson_r", "Spearman"] if k in current_metrics and k in new_metrics]
    print(f"  {'' :<15} {'Original':>10} {label:>10} {'Change':>10}")
    print(f"  {'-'*45}")
    for k in keys:
        old_v = current_metrics.get(k, float("nan"))
        new_v = new_metrics.get(k, float("nan"))
        change = (new_v - old_v) if k in ("R2", "Pearson_r", "Spearman") else (old_v - new_v)
        print(f"  {k:<15} {old_v:>10.4f} {new_v:>10.4f} {change:>+10.4f}")
    print(f"{'='*55}\n")


def print_bin_table(bin_results, title="RMSE by activity bin"):
    print(f"\n  --- {title} ---")
    print(f"  {'Bin':<10} {'n':>6} {'RMSE':>8} {'MAE':>8} {'mean_true':>10} {'mean_pred':>10} {'bias':>8}")
    print(f"  {'-'*60}")
    for label, m in bin_results.items():
        if np.isnan(m["RMSE"]):
            print(f"  {label:<10} {m['n']:>6}  {'N/A':>8}")
        else:
            print(f"  {label:<10} {m['n']:>6} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} {m['mean_true']:>10.3f} {m['mean_pred']:>10.3f} {m['bias']:>+8.3f}")


def print_enrichment_table(enrich_results, title="Enrichment"):
    print(f"\n  --- {title} ---")
    for er in enrich_results:
        print(f"  Top {er['top_k']*100:.0f}%: mean_actual={er['mean_top_actual']:.3f}  "
              f"mean_all={er['mean_all_actual']:.3f}  "
              f"enrichment={er['enrichment_factor']:.3f}")
