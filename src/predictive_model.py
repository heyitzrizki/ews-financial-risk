from pathlib import Path
import copy
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

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
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
THRESHOLD_FLOOR = _env_float("EWS_PRED_THRESHOLD_FLOOR", 0.05)
VAL_FRAC = _env_float("EWS_PRED_VAL_FRAC", 0.20)
HORIZONS = _env_int_list("EWS_PRED_HORIZONS", [40, 60])

ENABLE_DL = os.getenv("EWS_PRED_ENABLE_DL", "1") == "1"
DL_EPOCHS = _env_int("EWS_PRED_DL_EPOCHS", 25)
DL_BATCH = _env_int("EWS_PRED_DL_BATCH", 128)
DL_PATIENCE = _env_int("EWS_PRED_DL_PATIENCE", 5)
DL_LR = _env_float("EWS_PRED_DL_LR", 1e-3)

ML_CANDIDATES = ["XGBoost", "RF", "Logit"]
DL_CANDIDATES = ["ALSTM", "CausalTCN"]


class AttentionLSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 48, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.attn = nn.Linear(hidden_size, 1)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        seq, _ = self.lstm(x)
        attn_scores = self.attn(seq).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        context = torch.sum(seq * attn_weights, dim=1)
        context = self.drop(context)
        return self.out(context).squeeze(-1)


class CausalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.pad = nn.ConstantPad1d((pad, 0), 0.0)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.proj(x)
        y = self.pad(x)
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.drop(y)
        y = self.pad(y)
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.drop(y)
        return self.relu(y + res)


class CausalTCNClassifier(nn.Module):
    def __init__(self, n_features: int, channels: int = 24, dropout: float = 0.2):
        super().__init__()
        self.block1 = CausalConvBlock(n_features, channels, kernel_size=3, dilation=1, dropout=dropout)
        self.block2 = CausalConvBlock(channels, channels, kernel_size=3, dilation=2, dropout=dropout)
        self.block3 = CausalConvBlock(channels, channels, kernel_size=3, dilation=4, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Linear(channels, 1)

    def forward(self, x):
        y = x.permute(0, 2, 1)
        y = self.block1(y)
        y = self.block2(y)
        y = self.block3(y)
        y = self.pool(y).squeeze(-1)
        return self.out(y).squeeze(-1)


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


def build_ml_models(class_ratio: float) -> dict:
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


def train_dl_model(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, class_ratio: float) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    tr_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    tr_loader = DataLoader(tr_ds, batch_size=DL_BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=DL_BATCH, shuffle=False)

    pos_weight = torch.tensor([max(class_ratio, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=DL_LR)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    patience_count = 0

    for _ in range(DL_EPOCHS):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        n_items = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                n_items += xb.size(0)

        val_loss = val_loss / max(n_items, 1)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= DL_PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def predict_dl_prob(model: nn.Module, X_data: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    ds = TensorDataset(torch.tensor(X_data, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=DL_BATCH, shuffle=False)

    probs = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(probs).astype(np.float32)


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
            X_val = X_tr
            y_val = y_tr
        if len(np.unique(y_tr_fit)) < 2:
            continue

        n_pos = int((y_tr_fit == 1).sum())
        n_neg = int((y_tr_fit == 0).sum())
        class_ratio = (n_neg / n_pos) if n_pos > 0 else 1.0

        X2d_tr_fit = build_window_features(X_tr_fit)
        X2d_val = build_window_features(X_val)
        X2d_te = build_window_features(X_te)

        ml_models = build_ml_models(class_ratio=class_ratio)
        for model_name, model in ml_models.items():
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
                    "family": "ML",
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
                        "family": "ML",
                        "y_true": int(y_te[i]),
                        "y_prob": float(prob_te[i]),
                        "y_pred": int(y_pred[i]),
                        "threshold": float(threshold),
                    }
                )

        if ENABLE_DL and HAS_TORCH:
            n_features = X_tr.shape[2]
            dl_specs = {
                "ALSTM": AttentionLSTMClassifier(n_features=n_features, hidden_size=48, dropout=0.2),
                "CausalTCN": CausalTCNClassifier(n_features=n_features, channels=24, dropout=0.2),
            }

            for model_name, model in dl_specs.items():
                model = train_dl_model(
                    model=model,
                    X_train=X_tr_fit,
                    y_train=y_tr_fit,
                    X_val=X_val,
                    y_val=y_val,
                    class_ratio=class_ratio,
                )
                prob_val = predict_dl_prob(model, X_val)
                prob_te = predict_dl_prob(model, X_te)
                threshold = max(find_optimal_threshold(y_val, prob_val), THRESHOLD_FLOOR)

                metrics = evaluate_binary(y_te, prob_te, threshold)
                fold_rows.append(
                    {
                        "horizon": horizon,
                        "fold": fold,
                        "model": model_name,
                        "family": "DL",
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
                            "family": "DL",
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
        fold_df.groupby(["horizon", "model", "family"], as_index=False)
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


def choose_top_ml_dl(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, dict | None]:
    rows = []
    summary_df = summary_df.copy()
    summary_df["rank_score"] = summary_df["PR_AUC_mean"]
    fallback_mask = summary_df["rank_score"].isna()
    summary_df.loc[fallback_mask, "rank_score"] = summary_df.loc[fallback_mask, "F1_mean"]
    summary_df["rank_score"] = summary_df["rank_score"].fillna(0.0)
    ml_priority = {"XGBoost": 1, "RF": 2, "Logit": 3}
    dl_priority = {"ALSTM": 1, "CausalTCN": 2}

    for horizon in sorted(summary_df["horizon"].unique()):
        h_df = summary_df[summary_df["horizon"] == horizon].copy()

        ml_df = h_df[h_df["model"].isin(ML_CANDIDATES)]
        dl_df = h_df[h_df["model"].isin(DL_CANDIDATES)]

        if ml_df.empty or dl_df.empty:
            continue

        ml_df = ml_df.copy()
        dl_df = dl_df.copy()
        ml_df["priority"] = ml_df["model"].map(ml_priority).fillna(99)
        dl_df["priority"] = dl_df["model"].map(dl_priority).fillna(99)

        best_ml = ml_df.sort_values(["rank_score", "priority"], ascending=[False, True]).iloc[0]
        best_dl = dl_df.sort_values(["rank_score", "priority"], ascending=[False, True]).iloc[0]
        hybrid_score = float((best_ml["rank_score"] + best_dl["rank_score"]) / 2)

        rows.append(
            {
                "horizon": int(horizon),
                "ml_model": str(best_ml["model"]),
                "ml_pr_auc": float(best_ml["PR_AUC_mean"]) if pd.notna(best_ml["PR_AUC_mean"]) else np.nan,
                "dl_model": str(best_dl["model"]),
                "dl_pr_auc": float(best_dl["PR_AUC_mean"]) if pd.notna(best_dl["PR_AUC_mean"]) else np.nan,
                "hybrid_score": hybrid_score,
            }
        )

    top_df = pd.DataFrame(rows)
    if top_df.empty:
        return top_df, None

    selected = top_df.sort_values("hybrid_score", ascending=False).iloc[0].to_dict()
    return top_df.sort_values("hybrid_score", ascending=False), selected


def fit_predict_ml(model_name: str, class_ratio: float, X_train_3d: np.ndarray, y_train: np.ndarray, X_pred_3d: np.ndarray) -> np.ndarray:
    models = build_ml_models(class_ratio=class_ratio)
    if model_name not in models:
        raise ValueError(f"ML model '{model_name}' is not available")

    model = models[model_name]
    model.fit(build_window_features(X_train_3d), y_train)
    return model.predict_proba(build_window_features(X_pred_3d))[:, 1]


def fit_predict_dl(model_name: str, class_ratio: float, X_train_3d: np.ndarray, y_train: np.ndarray, X_pred_3d: np.ndarray) -> np.ndarray:
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for DL models")

    n_features = X_train_3d.shape[2]
    if model_name == "ALSTM":
        model = AttentionLSTMClassifier(n_features=n_features, hidden_size=48, dropout=0.2)
    elif model_name == "CausalTCN":
        model = CausalTCNClassifier(n_features=n_features, channels=24, dropout=0.2)
    else:
        raise ValueError(f"Unknown DL model: {model_name}")

    model = train_dl_model(
        model=model,
        X_train=X_train_3d,
        y_train=y_train,
        X_val=X_train_3d,
        y_val=y_train,
        class_ratio=class_ratio,
    )
    return predict_dl_prob(model, X_pred_3d)


def build_latest_hybrid_signal(df: pd.DataFrame, selected_pair: dict) -> pd.DataFrame:
    horizon = int(selected_pair["horizon"])
    ml_model = str(selected_pair["ml_model"])
    dl_model = str(selected_pair["dl_model"])

    feature_cols = [col for col in df.columns if col != "hmm_state"]
    X_raw = df[feature_cols].to_numpy()
    hmm_raw = df["hmm_state"].to_numpy().astype(int)
    dates = df.index

    global_crisis_state = int(df.groupby("hmm_state")["TCI"].mean().idxmax())
    raw_flag = (hmm_raw == global_crisis_state).astype(float)
    y_all = _roll_forward_max(raw_flag, horizon)
    y_all[raw_flag == 1.0] = np.nan

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_seq, y_seq, seq_end_idx = make_sequences(X_scaled, y_all, LOOKBACK)
    valid_mask = ~np.isnan(y_seq)

    X_seq = X_seq[valid_mask]
    y_seq = y_seq[valid_mask].astype(int)
    seq_end_idx = seq_end_idx[valid_mask]

    if len(X_seq) < 50 or len(np.unique(y_seq)) < 2:
        return pd.DataFrame()

    class_ratio = max((y_seq == 0).sum() / max((y_seq == 1).sum(), 1), 1.0)
    ml_prob = fit_predict_ml(ml_model, class_ratio, X_seq, y_seq, X_seq)
    dl_prob = fit_predict_dl(dl_model, class_ratio, X_seq, y_seq, X_seq)

    hybrid_prob = 0.5 * ml_prob + 0.5 * dl_prob
    threshold = max(find_optimal_threshold(y_seq, hybrid_prob), THRESHOLD_FLOOR)
    hybrid_pred = (hybrid_prob >= threshold).astype(int)

    signal_df = pd.DataFrame(
        {
            "Date": dates[seq_end_idx].values,
            "horizon": horizon,
            "model": "Hybrid",
            "ml_model": ml_model,
            "dl_model": dl_model,
            "ml_prob": ml_prob,
            "dl_prob": dl_prob,
            "y_prob": hybrid_prob,
            "y_pred": hybrid_pred,
            "threshold": threshold,
            "global_crisis_state": global_crisis_state,
        }
    )
    return signal_df


def main() -> None:
    print(f"Loading TVP spillover data: {TVP_PATH}")
    print(f"Loading HMM regimes data : {HMM_PATH}")
    if ENABLE_DL and not HAS_TORCH:
        raise RuntimeError("EWS_PRED_ENABLE_DL=1 but PyTorch is not available.")

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

    top_models_df, selected_pair = choose_top_ml_dl(summary_metrics)
    if selected_pair is None:
        raise RuntimeError("Could not identify top ML and DL pair from summary metrics.")

    fold_metrics.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
    oof_predictions.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
    summary_metrics.to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False)
    top_models_df.to_csv(OUTPUT_DIR / "top_model_selection.csv", index=False)

    signal_df = build_latest_hybrid_signal(df, selected_pair)
    if signal_df.empty:
        raise RuntimeError("Hybrid signal generation failed due to insufficient usable sequence samples.")

    signal_df.to_csv(OUTPUT_DIR / "latest_signal_history.csv", index=False)
    signal_df.tail(1).to_csv(OUTPUT_DIR / "latest_signal.csv", index=False)

    latest = signal_df.iloc[-1]
    print(
        f"Latest hybrid signal => date={pd.to_datetime(latest['Date']).date()} | "
        f"horizon={int(latest['horizon'])}d | "
        f"ML={latest['ml_model']} ({latest['ml_prob']:.4f}) | "
        f"DL={latest['dl_model']} ({latest['dl_prob']:.4f}) | "
        f"Hybrid={latest['y_prob']:.4f} | alert={int(latest['y_pred'])}"
    )

    print("\nSaved predictive outputs:")
    print(f"- {OUTPUT_DIR / 'fold_metrics.csv'}")
    print(f"- {OUTPUT_DIR / 'oof_predictions.csv'}")
    print(f"- {OUTPUT_DIR / 'summary_metrics.csv'}")
    print(f"- {OUTPUT_DIR / 'top_model_selection.csv'}")
    print(f"- {OUTPUT_DIR / 'latest_signal_history.csv'}")
    print(f"- {OUTPUT_DIR / 'latest_signal.csv'}")


if __name__ == "__main__":
    main()
