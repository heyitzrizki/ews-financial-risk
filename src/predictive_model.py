from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
np.random.seed(42)

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_int_list(name: str, default: list[int]) -> list[int]:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
        return parsed if parsed else default
    except ValueError:
        return default


BASE_DIR = Path(__file__).resolve().parents[1]
TVP_PATH = BASE_DIR / "data" / "processed" / "tvp_var" / "tvp_var_spillover_indices.csv"
HMM_PATH = BASE_DIR / "data" / "processed" / "regime" / "hmm_regimes.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "predictive"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK = _env_int("EWS_PRED_LOOKBACK", 90)
N_SPLITS = _env_int("EWS_PRED_N_SPLITS", 5)
THRESHOLD_FLOOR = float(os.getenv("EWS_PRED_THRESHOLD_FLOOR", "0.05"))
VAL_FRAC = float(os.getenv("EWS_PRED_VAL_FRAC", "0.20"))
HORIZONS = _env_int_list("EWS_PRED_HORIZONS", [40, 60])


def load_inputs(tvp_path: Path, hmm_path: Path) -> pd.DataFrame:
    if not tvp_path.exists():
        raise FileNotFoundError(f"TVP input not found: {tvp_path}")
    if not hmm_path.exists():
        raise FileNotFoundError(f"HMM input not found: {hmm_path}")

    df_tvp = pd.read_csv(tvp_path, parse_dates=["Date"]).set_index("Date").sort_index()
    df_hmm = pd.read_csv(hmm_path, parse_dates=["Date"]).set_index("Date").sort_index()

    if "hmm_state" not in df_hmm.columns:
        raise ValueError("hmm_regimes.csv must contain 'hmm_state' column")
    if "TCI" not in df_tvp.columns:
        raise ValueError("tvp_var_spillover_indices.csv must contain 'TCI' column")

    df = df_tvp.join(df_hmm[["hmm_state"]], how="inner").dropna()
    if df.empty:
        raise ValueError("Merged predictive frame is empty after alignment")

    return df


def _roll_forward_max(arr: np.ndarray, horizon: int) -> np.ndarray:
    s = pd.Series(arr.astype(float))
    return (
        s.shift(-1)
        .rolling(horizon, min_periods=horizon)
        .max()
        .shift(-(horizon - 1))
        .to_numpy()
    )


def make_sequences(X: np.ndarray, y: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(X) < lookback:
        return np.empty((0, lookback, X.shape[1])), np.empty(0), np.empty(0, dtype=int)

    X_seq = []
    y_seq = []
    end_idx = []
    for end in range(lookback - 1, len(X)):
        X_seq.append(X[end - lookback + 1 : end + 1])
        y_seq.append(y[end])
        end_idx.append(end)

    return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq), np.asarray(end_idx, dtype=int)


def build_window_features(X_3d: np.ndarray) -> np.ndarray:
    mean = X_3d.mean(axis=1)
    std = X_3d.std(axis=1)
    last = X_3d[:, -1, :]
    delta = X_3d[:, -1, :] - X_3d[:, 0, :]
    return np.concatenate([mean, std, last, delta], axis=1)


def find_optimal_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    y_true = y_true.astype(int)
    if len(np.unique(y_true)) < 2:
        return 0.5
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.01, 0.99, 99):
        f1 = f1_score(y_true, (prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return float(best_t)


def evaluate_binary(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    y_true = y_true.astype(int)
    y_pred = (prob >= threshold).astype(int)
    metrics = {
        "PR_AUC": np.nan,
        "ROC_AUC": np.nan,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Threshold": threshold,
    }
    if len(np.unique(y_true)) >= 2:
        metrics["PR_AUC"] = average_precision_score(y_true, prob)
        metrics["ROC_AUC"] = roc_auc_score(y_true, prob)
    return metrics


def build_models(class_ratio: float) -> dict:
    models = {
        "Logit": LogisticRegression(
            class_weight="balanced",
            solver="saga",
            max_iter=1200,
            random_state=42,
        ),
        "RF": RandomForestClassifier(
            n_estimators=300,
            max_depth=7,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            scale_pos_weight=max(class_ratio, 1.0),
            eval_metric="logloss",
            random_state=42,
        )

    return models


def run_horizon(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_cols = [col for col in df.columns if col != "hmm_state"]
    X_raw = df[feature_cols].to_numpy()
    hmm_raw = df["hmm_state"].to_numpy().astype(int)
    dates = df.index.to_numpy()

    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=horizon)

    fold_rows = []
    prediction_rows = []

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_raw), start=1):
        tci_train = df.iloc[tr_idx].groupby("hmm_state")["TCI"].mean()
        fold_crisis_state = int(tci_train.idxmax())

        raw_flag = (hmm_raw == fold_crisis_state).astype(float)
        y_all = _roll_forward_max(raw_flag, horizon)
        y_all[raw_flag == 1.0] = np.nan

        scaler = MinMaxScaler()
        scaler.fit(X_raw[tr_idx])
        X_scaled = scaler.transform(X_raw)

        X_seq, y_seq, seq_end_idx = make_sequences(X_scaled, y_all, LOOKBACK)
        valid_mask = ~np.isnan(y_seq)
        X_seq = X_seq[valid_mask]
        y_seq = y_seq[valid_mask].astype(int)
        seq_end_idx = seq_end_idx[valid_mask]

        tr_mask = np.isin(seq_end_idx, tr_idx)
        te_mask = np.isin(seq_end_idx, te_idx)

        X_tr = X_seq[tr_mask]
        y_tr = y_seq[tr_mask]
        end_tr = seq_end_idx[tr_mask]
        X_te = X_seq[te_mask]
        y_te = y_seq[te_mask]
        end_te = seq_end_idx[te_mask]

        if len(X_tr) < 50 or len(X_te) < 10 or len(np.unique(y_tr)) < 2:
            continue

        split_at = int(len(X_tr) * (1 - VAL_FRAC))
        split_at = max(split_at, int(len(X_tr) * 0.7))
        split_at = min(split_at, len(X_tr) - 1)

        X_tr_fit = X_tr[:split_at]
        y_tr_fit = y_tr[:split_at]
        X_val = X_tr[split_at:]
        y_val = y_tr[split_at:]

        if len(np.unique(y_tr_fit)) < 2:
            X_tr_fit = X_tr
            y_tr_fit = y_tr

        if len(np.unique(y_val)) < 2:
            X_tr_fit = X_tr
            y_tr_fit = y_tr
            X_val = X_tr
            y_val = y_tr

        if len(np.unique(y_tr_fit)) < 2:
            continue

        X2d_tr_fit = build_window_features(X_tr_fit)
        X2d_val = build_window_features(X_val)
        X2d_te = build_window_features(X_te)

        n_pos = int((y_tr_fit == 1).sum())
        n_neg = int((y_tr_fit == 0).sum())
        class_ratio = (n_neg / n_pos) if n_pos > 0 else 1.0
        models = build_models(class_ratio=class_ratio)

        for model_name, model in models.items():
            model.fit(X2d_tr_fit, y_tr_fit)
            prob_val = model.predict_proba(X2d_val)[:, 1]
            prob_te = model.predict_proba(X2d_te)[:, 1]
            threshold = max(find_optimal_threshold(y_val, prob_val), THRESHOLD_FLOOR)

            metrics = evaluate_binary(y_te, prob_te, threshold)
            fold_rows.append(
                {
                    "horizon": horizon,
                    "fold": fold,
                    "model": model_name,
                    "crisis_state_fold": fold_crisis_state,
                    **metrics,
                }
            )

            y_pred = (prob_te >= threshold).astype(int)
            for i in range(len(y_te)):
                prediction_rows.append(
                    {
                        "Date": dates[end_te[i]],
                        "horizon": horizon,
                        "fold": fold,
                        "model": model_name,
                        "y_true": int(y_te[i]),
                        "y_prob": float(prob_te[i]),
                        "y_pred": int(y_pred[i]),
                        "threshold": float(threshold),
                    }
                )

        print(
            f"H={horizon:>2}d | Fold {fold}: train_seq={len(X_tr):>4}, test_seq={len(X_te):>4}, crisis_state={fold_crisis_state}"
        )

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(prediction_rows)

    if fold_df.empty:
        return fold_df, pred_df, pd.DataFrame()

    summary_df = (
        fold_df.groupby(["horizon", "model"], as_index=False)
        .agg(
            PR_AUC_mean=("PR_AUC", "mean"),
            PR_AUC_std=("PR_AUC", "std"),
            ROC_AUC_mean=("ROC_AUC", "mean"),
            F1_mean=("F1", "mean"),
            Recall_mean=("Recall", "mean"),
            Precision_mean=("Precision", "mean"),
        )
        .sort_values(["horizon", "PR_AUC_mean"], ascending=[True, False])
    )
    return fold_df, pred_df, summary_df


def train_latest_signal(df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    best_row = summary_df.sort_values("PR_AUC_mean", ascending=False).iloc[0]
    best_horizon = int(best_row["horizon"])
    best_model = str(best_row["model"])

    feature_cols = [col for col in df.columns if col != "hmm_state"]
    X_raw = df[feature_cols].to_numpy()
    hmm_raw = df["hmm_state"].to_numpy().astype(int)
    dates = df.index

    global_crisis_state = int(df.groupby("hmm_state")["TCI"].mean().idxmax())
    raw_flag = (hmm_raw == global_crisis_state).astype(float)
    y_all = _roll_forward_max(raw_flag, best_horizon)
    y_all[raw_flag == 1.0] = np.nan

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_seq, y_seq, seq_end_idx = make_sequences(X_scaled, y_all, LOOKBACK)
    valid_mask = ~np.isnan(y_seq)

    X_seq = X_seq[valid_mask]
    y_seq = y_seq[valid_mask].astype(int)
    seq_end_idx = seq_end_idx[valid_mask]

    if len(X_seq) < 20 or len(np.unique(y_seq)) < 2:
        return pd.DataFrame()

    X2d = build_window_features(X_seq)
    class_ratio = max((y_seq == 0).sum() / max((y_seq == 1).sum(), 1), 1.0)
    models = build_models(class_ratio=class_ratio)
    model = models[best_model]
    model.fit(X2d, y_seq)

    prob_all = model.predict_proba(X2d)[:, 1]
    threshold = max(find_optimal_threshold(y_seq, prob_all), THRESHOLD_FLOOR)
    pred_all = (prob_all >= threshold).astype(int)

    signal_df = pd.DataFrame(
        {
            "Date": dates[seq_end_idx].values,
            "horizon": best_horizon,
            "model": best_model,
            "y_prob": prob_all,
            "y_pred": pred_all,
            "threshold": threshold,
            "global_crisis_state": global_crisis_state,
        }
    )
    return signal_df


def main() -> None:
    print(f"Loading TVP spillover data: {TVP_PATH}")
    print(f"Loading HMM regimes data : {HMM_PATH}")
    df = load_inputs(TVP_PATH, HMM_PATH)
    print(f"Merged predictive frame : {df.shape[0]} rows x {df.shape[1]} cols")

    all_fold = []
    all_pred = []
    all_summary = []

    for horizon in HORIZONS:
        fold_df, pred_df, summary_df = run_horizon(df, horizon)
        if not fold_df.empty:
            all_fold.append(fold_df)
            all_pred.append(pred_df)
            all_summary.append(summary_df)

    if not all_fold:
        raise RuntimeError("No valid fold results produced. Check data length and class balance.")

    fold_metrics = pd.concat(all_fold, ignore_index=True)
    oof_predictions = pd.concat(all_pred, ignore_index=True)
    summary_metrics = pd.concat(all_summary, ignore_index=True).sort_values(
        ["horizon", "PR_AUC_mean"], ascending=[True, False]
    )

    fold_metrics.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
    oof_predictions.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
    summary_metrics.to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False)

    signal_df = train_latest_signal(df, summary_metrics)
    if not signal_df.empty:
        signal_df.to_csv(OUTPUT_DIR / "latest_signal_history.csv", index=False)
        signal_df.tail(1).to_csv(OUTPUT_DIR / "latest_signal.csv", index=False)
        latest = signal_df.iloc[-1]
        print(
            f"Latest signal => date={pd.to_datetime(latest['Date']).date()} | "
            f"model={latest['model']} | horizon={int(latest['horizon'])}d | "
            f"prob={latest['y_prob']:.4f} | pred={int(latest['y_pred'])}"
        )

    print("\nSaved predictive outputs:")
    print(f"- {OUTPUT_DIR / 'fold_metrics.csv'}")
    print(f"- {OUTPUT_DIR / 'oof_predictions.csv'}")
    print(f"- {OUTPUT_DIR / 'summary_metrics.csv'}")
    if (OUTPUT_DIR / "latest_signal.csv").exists():
        print(f"- {OUTPUT_DIR / 'latest_signal_history.csv'}")
        print(f"- {OUTPUT_DIR / 'latest_signal.csv'}")


if __name__ == "__main__":
    main()
