from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data" / "processed" / "volatility_system.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "tvp_var"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TVP_CONFIG = {
    "lambda_Q": 0.99,
    "lambda_H": 0.94,
    "kappa_Q": 0.01,
    "kappa_H": 0.01,
    "use_forgetting": True,
}

VAR_COLS = ["JCI_RV", "SP500_RV", "SSE_RV", "VIX_Level", "WTI_RV", "Gold_RV", "USDIDR_RV"]
IRF_HORIZON = 10
MAX_LAGS = 10


def load_var_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()

    missing = [col for col in VAR_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_var = df[VAR_COLS].copy().dropna()
    if df_var.empty:
        raise ValueError("VAR input data is empty after dropna()")

    return df_var


def select_optimal_lag(df_var: pd.DataFrame, max_lags: int = MAX_LAGS) -> int:
    model = VAR(df_var)
    lag_results = model.select_order(maxlags=max_lags)

    bic_lag = None
    if hasattr(lag_results, "selected_orders") and isinstance(lag_results.selected_orders, dict):
        bic_lag = lag_results.selected_orders.get("bic")

    if bic_lag is None:
        try:
            bic_lag = int(lag_results.bic)
        except Exception:
            bic_lag = 1

    if bic_lag is None or bic_lag < 1:
        bic_lag = 1

    return int(bic_lag)


def get_ols_initial_state(Y: np.ndarray, p: int) -> np.ndarray:
    n, k = Y.shape

    X = []
    y = []
    for t in range(p, n):
        lags = []
        for lag in range(1, p + 1):
            lags.extend(Y[t - lag, :])
        X.append(lags)
        y.append(Y[t, :])

    X = np.array(X)
    y = np.array(y)

    alpha_0 = []
    for j in range(k):
        X_design = np.column_stack([np.ones(len(X)), X])
        coef, _, _, _ = np.linalg.lstsq(X_design, y[:, j], rcond=None)
        alpha_0.extend(coef)

    return np.array(alpha_0)


def create_design_matrix(Y: np.ndarray, t: int, p: int, k: int) -> np.ndarray:
    if t < p:
        raise ValueError("Need at least p observations")

    Z = []
    for _ in range(k):
        row = [1]
        for lag in range(1, p + 1):
            row.extend(Y[t - lag, :])
        Z.append(row)

    return np.array(Z)


def kalman_filter_tvpvar(
    Y: np.ndarray,
    alpha_0: np.ndarray,
    P_0: np.ndarray,
    p: int,
    config: dict,
    dtype=np.float32,
):
    n, k = Y.shape
    m = 1 + k * p
    state_dim = k * m

    lambda_Q = config["lambda_Q"]
    lambda_H = config["lambda_H"]
    kappa_Q = config["kappa_Q"]
    kappa_H = config["kappa_H"]
    use_forgetting = config.get("use_forgetting", True)

    alpha_filtered = np.zeros((n - p, state_dim), dtype=dtype)
    forecast_errors = np.zeros((n - p, k), dtype=dtype)
    H_series = np.zeros((n - p, k, k), dtype=dtype)

    alpha_t = alpha_0.copy()
    P_t = P_0.copy()

    F = np.eye(state_dim)
    Q_t = np.eye(state_dim) * kappa_Q
    H_t = np.eye(k) * kappa_H

    for t in range(p, n):
        idx = t - p

        alpha_pred = F @ alpha_t
        P_pred = F @ P_t @ F.T + Q_t

        Z_full = create_design_matrix(Y, t, p, k)
        y_t = Y[t, :]

        Z_t = np.zeros((k, state_dim))
        for j in range(k):
            start_idx = j * m
            end_idx = (j + 1) * m
            Z_t[j, start_idx:end_idx] = Z_full[j, :]

        y_pred = Z_t @ alpha_pred
        v_t = y_t - y_pred
        forecast_errors[idx, :] = v_t

        S_t = Z_t @ P_pred @ Z_t.T + H_t
        K_t = P_pred @ Z_t.T @ np.linalg.pinv(S_t)

        alpha_updated = alpha_pred + K_t @ v_t

        I = np.eye(state_dim)
        P_updated = (I - K_t @ Z_t) @ P_pred @ (I - K_t @ Z_t).T + K_t @ H_t @ K_t.T

        if use_forgetting and idx > 0:
            H_t = lambda_H * H_t + (1 - lambda_H) * np.outer(v_t, v_t)
            H_t = (H_t + H_t.T) / 2
            min_eig = np.min(np.linalg.eigvalsh(H_t))
            if min_eig < 1e-8:
                H_t += np.eye(k) * (1e-8 - min_eig)

        H_series[idx, :, :] = H_t

        if use_forgetting:
            Q_t = lambda_Q * Q_t + (1 - lambda_Q) * np.eye(state_dim) * kappa_Q

        alpha_filtered[idx, :] = alpha_updated
        alpha_t = alpha_updated
        P_t = P_updated

    return alpha_filtered, forecast_errors, H_series


def compute_tvp_irf(alpha_t: np.ndarray, H_t: np.ndarray, p: int, k: int, horizon: int = IRF_HORIZON) -> np.ndarray:
    m = 1 + k * p

    A = np.zeros((k, k * p))
    for eq in range(k):
        start_idx = eq * m + 1
        coefs = alpha_t[start_idx:start_idx + k * p]
        A[eq, :] = coefs

    if p > 1:
        F = np.zeros((k * p, k * p))
        F[:k, :] = A
        F[k:, :k * (p - 1)] = np.eye(k * (p - 1))
    else:
        F = A

    H_reg = (H_t + H_t.T) / 2
    min_eig = np.min(np.linalg.eigvalsh(H_reg))
    if min_eig < 1e-10:
        H_reg = H_reg + np.eye(k) * (1e-10 - min_eig)

    irf = np.zeros((horizon, k, k))
    shock_scale = np.sqrt(np.diag(H_reg))

    for j in range(k):
        irf[0, :, j] = H_reg[:, j] / shock_scale[j]

    F_power = np.eye(k * p)
    for h in range(1, horizon):
        F_power = F_power @ F
        Phi_h = F_power[:k, :k]
        for j in range(k):
            irf[h, :, j] = (Phi_h @ H_reg[:, j]) / shock_scale[j]

    return irf


def compute_gfevd(irf: np.ndarray, horizon: int) -> np.ndarray:
    k = irf.shape[1]
    fevd = np.zeros((k, k))

    for i in range(k):
        mse_i = 0
        for h in range(horizon):
            mse_i += np.sum(irf[h, i, :] ** 2)

        for j in range(k):
            numerator = 0
            for h in range(horizon):
                numerator += irf[h, i, j] ** 2
            fevd[i, j] = numerator / mse_i if mse_i > 0 else 0

    row_sums = fevd.sum(axis=1, keepdims=True)
    fevd = np.where(row_sums > 0, fevd / row_sums, 0)

    return fevd * 100


def compute_total_connectedness(fevd: np.ndarray) -> float:
    off_diag = fevd.sum() - np.trace(fevd)
    total = fevd.sum()
    return (off_diag / total) * 100 if total > 0 else 0.0


def compute_directional_to(fevd: np.ndarray) -> np.ndarray:
    k = fevd.shape[0]
    to_spillover = np.zeros(k)
    for j in range(k):
        to_spillover[j] = (fevd[:, j].sum() - fevd[j, j]) / k
    return to_spillover


def compute_directional_from(fevd: np.ndarray) -> np.ndarray:
    k = fevd.shape[0]
    from_spillover = np.zeros(k)
    for i in range(k):
        from_spillover[i] = (fevd[i, :].sum() - fevd[i, i]) / k
    return from_spillover


def run_tvp_var_pipeline() -> pd.DataFrame:
    df_var = load_var_data(INPUT_FILE)
    Y = df_var.values

    p = select_optimal_lag(df_var, MAX_LAGS)
    k = Y.shape[1]
    m = 1 + k * p

    print(f"VAR input shape: {Y.shape}")
    print(f"Selected lag (BIC): {p}")
    print(f"State dimension: {k * m}")

    alpha_0 = get_ols_initial_state(Y, p)
    P_0 = np.eye(len(alpha_0)) * 10.0

    expected_dim = k * m
    if alpha_0.shape[0] != expected_dim:
        raise ValueError(
            f"alpha_0 dimension mismatch: got {alpha_0.shape[0]}, expected {expected_dim}"
        )

    alpha_filt, forecast_errors, H_series = kalman_filter_tvpvar(
        Y=Y,
        alpha_0=alpha_0,
        P_0=P_0,
        p=p,
        config=TVP_CONFIG,
        dtype=np.float32,
    )

    print(f"Filtered states shape: {alpha_filt.shape}")
    print(f"Forecast errors shape: {forecast_errors.shape}")
    print(f"H_t shape: {H_series.shape}")

    fevd_series = np.zeros((len(alpha_filt), k, k))
    for t in range(len(alpha_filt)):
        irf_t = compute_tvp_irf(alpha_filt[t, :], H_series[t, :, :], p, k, IRF_HORIZON)
        fevd_series[t, :, :] = compute_gfevd(irf_t, IRF_HORIZON)

    time_index = df_var.index[p:]
    tci = np.array([compute_total_connectedness(fevd_series[t, :, :]) for t in range(len(fevd_series))])
    to_spillovers = np.array([compute_directional_to(fevd_series[t, :, :]) for t in range(len(fevd_series))])
    from_spillovers = np.array([compute_directional_from(fevd_series[t, :, :]) for t in range(len(fevd_series))])

    variable_names = [v.replace("_RV", "").replace("_Level", "") for v in VAR_COLS]

    tci_df = pd.DataFrame({"TCI": tci}, index=time_index)
    to_df = pd.DataFrame(to_spillovers, index=time_index, columns=variable_names)
    from_df = pd.DataFrame(from_spillovers, index=time_index, columns=variable_names)
    net_df = to_df - from_df

    spillover_data = pd.concat(
        [
            tci_df,
            to_df.add_prefix("TO_"),
            from_df.add_prefix("FROM_"),
            net_df.add_prefix("NET_"),
        ],
        axis=1,
    )

    tci_df.to_csv(OUTPUT_DIR / "tci_filtered.csv", index_label="Date")
    to_df.to_csv(OUTPUT_DIR / "directional_to_filtered.csv", index_label="Date")
    from_df.to_csv(OUTPUT_DIR / "directional_from_filtered.csv", index_label="Date")
    net_df.to_csv(OUTPUT_DIR / "net_spillover_filtered.csv", index_label="Date")
    spillover_data.to_csv(OUTPUT_DIR / "tvp_var_spillover_indices.csv", index_label="Date")

    pd.DataFrame(fevd_series.mean(axis=0), index=variable_names, columns=variable_names).to_csv(
        OUTPUT_DIR / "fevd_average.csv"
    )

    print("\nSaved TVP-VAR outputs:")
    print(f"- {OUTPUT_DIR / 'tci_filtered.csv'}")
    print(f"- {OUTPUT_DIR / 'directional_to_filtered.csv'}")
    print(f"- {OUTPUT_DIR / 'directional_from_filtered.csv'}")
    print(f"- {OUTPUT_DIR / 'net_spillover_filtered.csv'}")
    print(f"- {OUTPUT_DIR / 'tvp_var_spillover_indices.csv'}")
    print(f"- {OUTPUT_DIR / 'fevd_average.csv'}")

    return spillover_data


def main() -> None:
    run_tvp_var_pipeline()


if __name__ == "__main__":
    main()
