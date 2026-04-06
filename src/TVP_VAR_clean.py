# %%
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# # TVP-VAR Spillover Analysis for Indonesia’s Stock Market

# %% [markdown]
# # 1. Setup & Configuration

# %% [markdown]
# ## 1.1 Import Libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import yfinance as yf
import pandas_datareader as pdr
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
import networkx as nx

import os
from pathlib import Path

# %% [markdown]
# ## 1.2 Define Helper Function

# %%
def check_stationarity(series, name='Series'):
    result = adfuller(series.dropna())

    print(f'\n{name}:')
    print('ADF Statistic: {result[0]: .4f}')
    print('p-value: {result[1]: .4f}')

    if result[1] <= 0.05:
        print("The series is stationary.")
    else:
        print("The series is non-stationary.")
    return {'statistic': result[0], 'p_value': result[1]}

# %%
def compute_log_returns(df, columns):
    returns_df = pdf.DataFrame(index=df.index)

    for col in columns:
        returns_df[f'{col}_ret'] = np.log(df[col] / df[col].shift(1))

    return returns_df

# %%
def winsorize_outliers(series, lower=1, upper=99):
    lower_bound = np.percentile(series.dropna(), lower)
    upper_bound = np.percentile(series.dropna(), upper)

    return series.clip(lower=lower_bound, upper=upper_bound)

# %% [markdown]
# ## 1.3 Set Plotting Style

# %%
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')

plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['legend.fontsize'] = 9

print("Plotting style set to 'seaborn-darkgrid' with custom parameters.")

# %% [markdown]
# ## 1.4 Set Working Directory

# %%
# Working directory (Google Colab)
BASE_DIR = Path(r"/content/drive/MyDrive/College Materials/Ajou University/thesis/ Early-Warning System for National Financial Instability/TVP-VAR MODEL")
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(BASE_DIR)
print(f"Working directory set to: {BASE_DIR}")

# %% [markdown]
# ## 2. EDA

# %% [markdown]
# ## 2.1 Define Tickers and Data Sources

# %%
tickers = {
    '^JKSE': 'JCI',
    '^GSPC': 'SP500',
    '000001.SS': 'SSE',
    '^VIX': 'VIX',
    'CL=F': 'WTI',
    'GC=F': 'Gold'
}

start_date = '2001-12-31'
end_date = '2025-01-01'

print(f"Data period: {start_date} to {end_date}")
print(f"Variables: {list(tickers.values())}")

# %% [markdown]
# ## 2.2 Fetch Data from Yahoo Finance

# %%
data = {}

for ticker, name in tickers.items():
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if not df.empty:
            if 'Adj Close' in df.columns and ticker not in ['^VIX', 'CL=F', 'GC=F']:
                data[name] = df['Adj Close']
            else:
                data[name] = df['Close']

            print(f"  {name}: {len(df)} obs")
    except Exception as e:
        print(f"  {name}: Failed - {str(e)}")

print(f"\nDownloaded {len(data)}/{len(tickers)} datasets")

# %% [markdown]
# ## 2.3 Merge All Data into One DataFrame

# %%
df = pd.concat(data.values(), axis=1, keys=data.keys())
df = df.sort_index()

print(f"Merged data shape: {df.shape}")
print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")

df

# %%
# If MultiIndex columns exist → flatten to single level
if isinstance(df.columns, pd.MultiIndex):
    # Flatten to use the second level (ticker symbols) as column names temporarily
    df.columns = df.columns.get_level_values(-1)

# Rename using the original tickers mapping to get desired column names
# The `tickers` dictionary maps ticker symbols (current column names) to display names.
df.rename(columns=tickers, inplace=True)

# %%
df.columns.name = None

# %%
df

# %%
usdidr = pd.read_csv('usdidr.csv')
# set date as index
usdidr.set_index('Date', inplace=True)
usdidr

# %%
# Rename the 'Price' column in usdidr to 'USDIDR'
usdidr = usdidr.rename(columns={'Price': 'USDIDR'})

df.index = pd.to_datetime(df.index)
usdidr.index = pd.to_datetime(usdidr.index)

# Merge usdidr into df based on the index (Date)
df = pd.merge(df, usdidr[['USDIDR']], left_index=True, right_index=True, how='left')

print(f"Merged data shape: {df.shape}")
print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
display(df.head())

# %%
df.tail()

# %% [markdown]
# ## 2.4 Handle Missing Values and Non-Trading Days

# %%
print("Missing values:")
print(df.isnull().sum())

# %%
# Forward fill then backward fill
df = df.fillna(method='ffill').fillna(method='bfill')

print(f"\Clean data: {df.shape[0]} observations")
print(f"Missing values: {df.isnull().sum().sum()}")


# %%
df
# save to csv
df.to_csv(OUTPUT_DIR / 'raw_data.csv')

# %% [markdown]
# ## 2.5 Compute Returns / Transformations

# %%
# 1. Compute Log Returns (for transformation pipeline)
return_vars = ['JCI', 'SP500', 'SSE', 'USDIDR', 'WTI', 'Gold']

df_ret = pd.DataFrame(index=df.index)
for var in return_vars:
    df_ret[f'{var}_ret'] = np.log(df[var] / df[var].shift(1))

# VIX return retained for transformation completeness
df_ret['VIX_ret'] = np.log(df['VIX'] / df['VIX'].shift(1))

# 2. Compute Realized Volatility (RV) for TVP-VAR
# Using 5-day rolling standard deviation of percentage returns, annualized
df_rv = pd.DataFrame(index=df.index)
for var in return_vars:
    daily_pct_ret = df_ret[f'{var}_ret'] * 100
    df_rv[f'{var}_RV'] = daily_pct_ret.rolling(window=5).std() * np.sqrt(252)

# VIX is already an annualized implied volatility (in %), use level directly!
df_rv['VIX_Level'] = df['VIX']

# Combine and drop NaNs
df_transformed = pd.concat([df_ret, df_rv], axis=1).dropna()
df_ret = df_transformed.filter(like='_ret')
df_rv = df_transformed.filter(like='_RV').join(df_transformed['VIX_Level'])

print(f"Transformed data (Volatility System): {df_rv.shape}")
display(df_rv.head())

# %% [markdown]
# ## 2.6 Final Clean Dataset Summary

# %%
print("FINAL DATASET")
print(f"Shape: {df_ret.shape}")
print(f"Period: {df_ret.index.min().date()} to {df_ret.index.max().date()}")
print(f"Variables: {list(df_ret.columns)}")

# %%
df_ret.describe()

# %%
# Correlation matrix
corr = df_ret.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# %%
corr

# %% [markdown]
# # 3. Data Transformations (for TVP-VAR)
# This section prepares stationary series and volatility proxies for the TVP-VAR system.
# No regime labeling (crisis/stress) is performed here.

# %%
# Combine transformed inputs for downstream TVP-VAR preparation
df_final = pd.concat([df_ret, df_rv], axis=1)

print(f"Transformed dataset shape: {df_final.shape}")

# %% [markdown]
# # 4. Preparing Data for TVP-VAR

# %% [markdown]
# ## 4.1 Select Variables for VAR System

# %%
# Use RVs and VIX Level for structurally consistent TVP-VAR
var_cols = ['JCI_RV', 'SP500_RV', 'SSE_RV', 'VIX_Level', 'WTI_RV', 'Gold_RV', 'USDIDR_RV']

df_var = df_final[var_cols].copy()

print(f"VAR system: {len(var_cols)} variables, {len(df_var)} observations")
print(f"Variables: {var_cols}")

# %% [markdown]
# ## 4.2 Align Frequency and Clean Outliers

# %%
print("Checking outliers (Z-score > 4):")

for col in df_var.columns:
    z = np.abs(stats.zscore(df_var[col].dropna()))
    n_outliers = (z > 4).sum()
    print(f"  {col:15s}: {n_outliers:3d} outliers")

# Note: Keep outliers for now (financial crises are real extremes)

# %% [markdown]
# ## 4.3 Stationarity Transformations

# %%
print("Stationarity test (ADF):")

for col in df_var.columns:
    check_stationarity(df_var[col], name=col)

# %% [markdown]
# ## 4.4 Create VAR Input Matrix

# %%
Y = df_var.values

print(f"\nVAR matrix Y: {Y.shape[0]} × {Y.shape[1]}")
print(f"Variables: {list(df_var.columns)}")

# Preview
pd.DataFrame(Y[:5], columns=df_var.columns)

# %% [markdown]
# ## 4.5 Optimal Lag Determination (Classical VAR)

# %%
model = VAR(df_var)
lag_results = model.select_order(maxlags=10)

print("\nLag selection:")
print(lag_results.summary())

# %%
optimal_lag = lag_results.bic

print(f"\Optimal lag (BIC): {optimal_lag}")

# %% [markdown]
# ## 4.6 Why Classical VAR is Not Used Directly

# %% [markdown]
# Classical VAR limitations:
# 
# 1. Assumes constant parameters over the full sample
# 2. Cannot capture structural breaks
# 3. Misses time-varying volatility
# 4. Ignores evolving spillover dynamics
# 
# Solution: Time-Varying Parameter VAR (TVP-VAR)
#    - Allows coefficients to change over time
#    - Captures dynamic transmission
#    - Better suited for nonstationary market interactions

# %%
print(f"Data preparation complete!")
print(f"Ready for TVP-VAR estimation with {optimal_lag} lags")

# %% [markdown]
# # 5. State-Space TVP-VAR Model Setup

# %% [markdown]
# ## 5.1 Mathematical Formulation

# %% [markdown]
# ### 5.1.1 Observation Equation

# %% [markdown]
# **TVP-VAR Observation Equation (Antonakakis 2020):**
# 
# $$y_t = Z_t \alpha_t + \varepsilon_t$$
# 
# where:
# - $y_t$: $k \times 1$ vector of observations at time $t$
# - $Z_t$: $k \times m$ design matrix (constructed from lagged $y$)
# - $\alpha_t$: $m \times 1$ state vector (time-varying VAR coefficients)
# - $\varepsilon_t \sim N(0, H_t)$: observation error ** TIME-VARYING**
# 
# ---
# 
# **Time-Varying Error Covariance**
# 
# $$H_t = \lambda_H H_{t-1} + (1 - \lambda_H) v_t v_t'$$
# 
# where:
# - $v_t = y_t - \hat{y}_t$: forecast error at time $t$
# - $\lambda_H \in (0, 1)$: forgetting factor (0.94 in our implementation)
# - Lower $\lambda_H$ → faster adaptation to volatility spikes
# 
# **Advantages over Fixed $H$:**
# - Captures volatility clustering (GFC 2008, COVID-19 2020)
# - No arbitrary rolling window choice
# - Preserves all observations (no data loss)

# %% [markdown]
# ### 5.1.2 State Equation

# %% [markdown]
# **TVP-VAR State Equation:**
# 
# $$\alpha_t = \alpha_{t-1} + \eta_t$$
# 
# where:
# - $\alpha_t$: time-varying coefficients (random walk)
# - $\eta_t \sim N(0, Q_t)$: state innovation error **TIME-VARYING**
# 
# This allows coefficients to drift smoothly over time.
# 
# ---
# 
# **State Covariance Update (Forgetting Factor):**
# 
# $$Q_t = \lambda_Q Q_{t-1} + (1 - \lambda_Q) \kappa_Q I$$
# 
# where:
# - $\lambda_Q = 0.99$: controls coefficient drift speed
# - Higher $\lambda_Q$ → slower adaptation (more stable coefficients)
# - Follows Antonakakis et al. (2020) specification

# %% [markdown]
# ## 5.2 Prior Specification

# %%
TVP_CONFIG = {
    # Forgetting factors for time-varying variance
    'lambda_Q': 0.99,      # Decay for state covariance Q_t (forgetting old info)
    'lambda_H': 0.94,      # Decay for observation covariance H_t (volatility tracking)

    # Initial variance scales (Minnesota prior-style)
    'kappa_Q': 0.01,       # Base variance for state innovations
    'kappa_H': 0.01,       # Base variance for observation errors

    # For comparison/ablation study
    'use_forgetting': True
}

print("TVP-VAR Hyperparameters (Antonakakis (2020)-style):")
for key, val in TVP_CONFIG.items():
    print(f"  {key}: {val}")

print("\nKey innovations:")
print("lambda_Q: Controls how fast Q_t adapts (0.99 = slow decay)")
print("lambda_H: Controls volatility tracking (0.94 = faster adaptation)")
print("Lower λ → faster forgetting → more adaptive to recent data")

# %% [markdown]
# ## 5.3 Model Initialization

# %%
# Number of variables and lags
k = Y.shape[1]  # number of variables
p = optimal_lag  # number of lags

# Total number of coefficients per equation
# Each equation has: intercept + k*p lagged terms
m = 1 + k * p

print(f"Model dimensions:")
print(f"  k (variables): {k}")
print(f"  p (lags): {p}")
print(f"  m (coefficients per equation): {m}")
print(f"  Total state dimension: {k * m}")

# %%
# Initialize state vector α_0
# Use OLS estimates from classical VAR as starting point
from sklearn.linear_model import LinearRegression

def get_ols_initial_state(Y, p):
    """Get OLS coefficients as initial state"""
    n, k = Y.shape

    # Create lagged matrix
    X = []
    y = []

    for t in range(p, n):
        # Lags
        lags = []
        for lag in range(1, p+1):
            lags.extend(Y[t-lag, :])

        X.append(lags)
        y.append(Y[t, :])

    X = np.array(X)
    y = np.array(y)

    # OLS for each equation
    alpha_0 = []
    for j in range(k):
        reg = LinearRegression(fit_intercept=True)
        reg.fit(X, y[:, j])
        # Coefficients: [intercept, lag1_var1, lag1_var2, ..., lag2_var1, ...]
        coef = np.concatenate([[reg.intercept_], reg.coef_])
        alpha_0.extend(coef)

    return np.array(alpha_0)

alpha_0 = get_ols_initial_state(Y, p)

expected_dim = k * (1 + k * p)
print(f"\nInitial state α_0 shape: {alpha_0.shape}")
print(f"Expected state dimension: {expected_dim}")
if alpha_0.shape[0] != expected_dim:
    raise ValueError("alpha_0 dimension mismatch. Re-run initialization cells and verify k, p, and lag construction.")

print(f"First 10 values: {alpha_0[:10]}")

# %%
# Initialize state covariance P_0
# Use diffuse prior (large variance)
P_0 = np.eye(len(alpha_0)) * 10.0

print(f"Initial state covariance P_0 shape: {P_0.shape}")
print(f"Diagonal values (first 5): {np.diag(P_0)[:5]}")

# %% [markdown]
# # 6. Kalman Filter Estimation

# %% [markdown]
# ## 6.1 Forward Pass: Kalman Filter

# %%
def create_design_matrix(Y, t, p, k):
    """
    Create design matrix Z_t for time t
    Z_t maps state α_t to observation y_t
    """
    if t < p:
        raise ValueError("Need at least p observations")

    # For each variable, create row: [1, y_{t-1}, ..., y_{t-p}]
    Z = []
    for j in range(k):
        row = [1]  # intercept
        for lag in range(1, p+1):
            row.extend(Y[t-lag, :])
        Z.append(row)

    return np.array(Z)

# Test
t_test = p + 10
Z_test = create_design_matrix(Y, t_test, p, k)
print(f"Design matrix Z_t shape: {Z_test.shape}")
print(f"Expected: ({k}, {m})")

# %%
def kalman_filter_tvpvar(Y, alpha_0, P_0, p, config, store_covariances=False, dtype=np.float32):
    n, k = Y.shape
    m = 1 + k * p
    state_dim = k * m

    # Extract hyperparameters
    lambda_Q = config['lambda_Q']
    lambda_H = config['lambda_H']
    kappa_Q = config['kappa_Q']
    kappa_H = config['kappa_H']
    use_forgetting = config.get('use_forgetting', True)

    # Storage
    alpha_filtered = np.zeros((n-p, state_dim), dtype=dtype)
    P_filtered = np.zeros((n-p, state_dim, state_dim), dtype=dtype) if store_covariances else None
    forecast_errors = np.zeros((n-p, k), dtype=dtype)
    H_series = np.zeros((n-p, k, k), dtype=dtype)  # Store time-varying H_t
    Q_series = np.zeros((n-p, state_dim, state_dim), dtype=dtype) if store_covariances else None

    # Initialize
    alpha_t = alpha_0.copy()
    P_t = P_0.copy()

    # Transition matrix (random walk)
    F = np.eye(state_dim)

    # Initialize Q_t (will evolve over time if use_forgetting=True)
    Q_t = np.eye(state_dim) * kappa_Q

    # Initialize H_t (time-varying observation covariance)
    H_t = np.eye(k) * kappa_H

    print(f"Running Kalman filter for {n-p} time points...")
    print(f"  Using forgetting factors: λ_Q={lambda_Q}, λ_H={lambda_H}")
    print(f"  Store full covariances: {store_covariances}")

    for t in range(p, n):
        idx = t - p

        # PREDICTION STEP
        alpha_pred = F @ alpha_t
        P_pred = F @ P_t @ F.T + Q_t  # ✅ Use time-varying Q_t

        # OBSERVATION STEP
        Z_full = create_design_matrix(Y, t, p, k)
        y_t = Y[t, :]

        # Build joint design matrix Z_t (k x state_dim)
        Z_t = np.zeros((k, state_dim))
        for j in range(k):
            start_idx = j * m
            end_idx = (j + 1) * m
            Z_t[j, start_idx:end_idx] = Z_full[j, :]

        # Forecast
        y_pred = Z_t @ alpha_pred

        # Forecast error
        v_t = y_t - y_pred
        forecast_errors[idx, :] = v_t

        # UPDATE STEP (Joint multivariate update)
        S_t = Z_t @ P_pred @ Z_t.T + H_t
        K_t = P_pred @ Z_t.T @ np.linalg.inv(S_t)

        # Update state
        alpha_updated = alpha_pred + K_t @ v_t

        # Joseph form covariance update for numerical stability
        I = np.eye(state_dim)
        P_updated = (I - K_t @ Z_t) @ P_pred @ (I - K_t @ Z_t).T + K_t @ H_t @ K_t.T

        # UPDATE H_t (Exponential Smoothing)
        if use_forgetting and idx > 0:
            # Exponential moving average of squared forecast errors
            # H_t = λ_H * H_{t-1} + (1 - λ_H) * v_t * v_t'
            H_t = lambda_H * H_t + (1 - lambda_H) * np.outer(v_t, v_t)

            # Ensure positive definite (add small ridge if needed)
            H_t = (H_t + H_t.T) / 2  # Symmetrize
            min_eig = np.min(np.linalg.eigvalsh(H_t))
            if min_eig < 1e-8:
                H_t += np.eye(k) * (1e-8 - min_eig)

        # Store H_t and (optionally) Q_t used in prediction
        H_series[idx, :, :] = H_t
        if store_covariances:
            Q_series[idx, :, :] = Q_t

        # UPDATE Q_t (Forgetting Factor)
        if use_forgetting:
            # Q_t = λ_Q * Q_{t-1} + (1 - λ_Q) * κ_Q * I
            Q_t = lambda_Q * Q_t + (1 - lambda_Q) * np.eye(state_dim) * kappa_Q

        # Store filtered estimates
        alpha_filtered[idx, :] = alpha_updated
        if store_covariances:
            P_filtered[idx, :, :] = P_updated

        # Prepare for next iteration
        alpha_t = alpha_updated
        P_t = P_updated

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx+1}/{n-p} observations")

    print("Kalman filter complete (with time-varying H_t and Q_t)")

    return alpha_filtered, P_filtered, forecast_errors, H_series, Q_series

# %%
correct_state_dim = k * m
# Adjust alpha_0 to the correct dimension
alpha_0_adjusted = alpha_0
if alpha_0.shape[0] != correct_state_dim:
    raise ValueError(
        f"Error: alpha_0 size {alpha_0.shape[0]} is not equal to expected state_dim {correct_state_dim}. "
        "Re-run initialization cells for alpha_0 and verify k, p, and lag construction."
    )

# Adjust P_0 to the correct dimension
P_0_adjusted = P_0
if P_0.shape[0] != correct_state_dim:
    raise ValueError(
        f"Error: P_0 size {P_0.shape[0]}x{P_0.shape[1]} is not equal to expected {correct_state_dim}x{correct_state_dim}. "
        "Re-run initialization cells for P_0 and alpha_0."
    )

# Run Kalman filter with TIME-VARYING covariance
alpha_filt, P_filt, fe, H_series, Q_series = kalman_filter_tvpvar(
    Y, alpha_0_adjusted, P_0_adjusted, p,
    TVP_CONFIG,
    store_covariances=False,
    dtype=np.float32
 )

print(f"\nFILTERED STATES:")
print(f"   Shape: {alpha_filt.shape}")
print(f"\nFORECAST ERRORS:")
print(f"   Shape: {fe.shape}")
print(f"\nTIME-VARYING ERROR COVARIANCE (H_t):")
print(f"   Shape: {H_series.shape}")
print(f"   Source: Exponential smoothing with λ_H = {TVP_CONFIG['lambda_H']}")

# Verify positive definite
n_pd = sum([np.all(np.linalg.eigvalsh(H_series[t]) > 0) for t in range(len(H_series))])
print(f"   Positive definite: {n_pd}/{len(H_series)} time points ({n_pd/len(H_series)*100:.1f}%)")

# %% [markdown]
# ## 6.2 Backward Pass: Kalman Smoother

# %%
def kalman_smoother_tvpvar(alpha_filtered, P_filtered, Q_series):
    """
    Rauch-Tung-Striebel smoother using Q_t from the forward pass.

    Returns:
    - alpha_smoothed: smoothed states
    - P_smoothed: smoothed covariances
    """
    if P_filtered is None or Q_series is None:
        raise ValueError("P_filtered/Q_series not stored. Re-run filter with store_covariances=True to enable smoothing.")

    T = alpha_filtered.shape[0]
    state_dim = alpha_filtered.shape[1]

    # Storage
    alpha_smoothed = np.zeros_like(alpha_filtered)
    P_smoothed = np.zeros_like(P_filtered)

    # Initialize with last filtered estimate
    alpha_smoothed[-1, :] = alpha_filtered[-1, :]
    P_smoothed[-1, :, :] = P_filtered[-1, :, :]

    # Transition matrix
    F = np.eye(state_dim)

    print("Running Kalman smoother backward (RTS with stored Q_t)...")

    for t in range(T - 2, -1, -1):
        Q_t = Q_series[t, :, :]
        # Predicted state and covariance for t+1
        alpha_pred = F @ alpha_filtered[t, :]
        P_pred = F @ P_filtered[t, :, :] @ F.T + Q_t

        # Smoother gain
        J_t = P_filtered[t, :, :] @ F.T @ np.linalg.inv(P_pred)

        # Smoothed state
        alpha_smoothed[t, :] = alpha_filtered[t, :] + \
            J_t @ (alpha_smoothed[t + 1, :] - alpha_pred)

        # Smoothed covariance
        P_smoothed[t, :, :] = P_filtered[t, :, :] + \
            J_t @ (P_smoothed[t + 1, :, :] - P_pred) @ J_t.T

        if (T - t) % 500 == 0:
            print(f"  Processed {T - t}/{T} observations")

    print("Kalman smoother complete (RTS)")

    return alpha_smoothed, P_smoothed

# %%
if P_filt is not None and Q_series is not None:
    alpha_smooth, P_smooth = kalman_smoother_tvpvar(alpha_filt, P_filt, Q_series)
    print(f"\nSmoothed states shape: {alpha_smooth.shape}")
else:
    alpha_smooth, P_smooth = None, None
    print("\nSmoother skipped (store_covariances=False).")

# %% [markdown]
# ## 6.3 Extracting Time-Varying Coefficients

# %%
# Extract coefficients for JCI equation (first equation)
# Coefficients: [intercept, JCI_lag1, SP500_lag1, ..., VIX_lag1, ...]

coef_source = alpha_smooth if alpha_smooth is not None else alpha_filt
source_label = "smoothed" if alpha_smooth is not None else "filtered"
jci_coefs = coef_source[:, :m]  # First m coefficients

# Create time index
time_index = df_var.index[p:]

jci_coefs_df = pd.DataFrame(
    jci_coefs,
    index=time_index,
    columns=['Intercept'] + [f'{v}_L{lag}' for lag in range(1, p+1) for v in var_cols]
)

print(f"JCI equation coefficients ({source_label}):")
jci_coefs_df.head()

# %%
# Plot time-varying coefficients for key variables
key_coef_cols = ['Intercept', 'JCI_RV_L1', 'SP500_RV_L1', 'VIX_Level_L1', 'USDIDR_RV_L1']
available_cols = [col for col in key_coef_cols if col in jci_coefs_df.columns]

fig, axes = plt.subplots(len(available_cols), 1, figsize=(14, 2.5*len(available_cols)), sharex=True)

if len(available_cols) == 1:
    axes = [axes]

for i, col in enumerate(available_cols):
    axes[i].plot(jci_coefs_df.index, jci_coefs_df[col], linewidth=1.2)
    axes[i].axhline(0, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[i].set_ylabel('Coefficient')
    axes[i].set_title(f'Time-Varying Coefficient: {col}')
    axes[i].grid(alpha=0.3)

axes[-1].set_xlabel('Date')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'tvp_coefficients.png', dpi=150)
plt.show()

# %% [markdown]
# ## 6.4 Extracting Time-Varying Covariance Matrix H_t

# %%
# Use H_t from Kalman filter output (already computed with exponential smoothing)
H_t = H_series  # Time-varying covariance from Antonakakis-style filtering

print(f"Time-varying covariance H_t shape: {H_t.shape}")
print(f"   Source: Exponential smoothing with λ_H = {TVP_CONFIG['lambda_H']}")

# Verify positive definite
n_pd = sum([np.all(np.linalg.eigvalsh(H_t[t]) > 0) for t in range(len(H_t))])
print(f"   Positive definite: {n_pd}/{len(H_t)} time points ({n_pd/len(H_t)*100:.1f}%)")

# Overall volatility summary from H_t (no labels)
vol_overall = []
for i, var in enumerate(var_cols):
    series = np.sqrt(H_t[:, i, i])
    vol_overall.append({
        "Variable": var,
        "Mean": series.mean(),
        "Std": series.std(),
        "Min": series.min(),
        "Max": series.max()
    })

vol_overall_df = pd.DataFrame(vol_overall)
vol_overall_df.to_csv(OUTPUT_DIR / "H_t_volatility_summary_overall.csv", index=False)
print("Saved: H_t_volatility_summary_overall.csv")
print(vol_overall_df.round(5))

# %%
# Ranked overall volatility summary (highest mean volatility first)
vol_overall_ranked = vol_overall_df.sort_values('Mean', ascending=False).reset_index(drop=True)

print("\nRanked H_t Volatility Summary (Overall):")
print(vol_overall_ranked.round(5))

# %%
# Plot time-varying volatilities (diagonal elements)
fig, axes = plt.subplots(k, 1, figsize=(14, 2*k), sharex=True)

for i in range(k):
    axes[i].plot(time_index, np.sqrt(H_t[:, i, i]), linewidth=1)
    axes[i].set_ylabel('Std Dev')
    axes[i].set_title(f'{var_cols[i]} Time-Varying Volatility')
    axes[i].grid(alpha=0.3)

axes[-1].set_xlabel('Date')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'tvp_volatilities.png', dpi=150)
plt.show()

# %% [markdown]
# ## 6.5 Diagnostic: Stability and Convergence
# 

# %%
# Check forecast error statistics
fe_stats = pd.DataFrame({
    'Mean': fe.mean(axis=0),
    'Std': fe.std(axis=0),
    'Skew': stats.skew(fe, axis=0),
    'Kurt': stats.kurtosis(fe, axis=0)
}, index=var_cols)

print("Forecast Error Diagnostics:")
fe_stats

# %%
# Export to table
fe_stats_export = fe_stats.copy()
fe_stats_export.index.name = 'Variable'
fe_stats_export = fe_stats_export.reset_index()

# Save to CSV
fe_stats_export.to_csv(OUTPUT_DIR / 'Table_6_5_Forecast_Error_Diagnostics.csv', index=False)
print("\nTable saved: Table_6_5_Forecast_Error_Diagnostics.csv")
print("\nForecast Error Summary:")
print(fe_stats_export.round(4))

# %%
# Plot forecast errors
fig, axes = plt.subplots(k, 1, figsize=(14, 2*k), sharex=True)

for i in range(k):
    axes[i].plot(time_index, fe[:, i], linewidth=0.8, alpha=0.7)
    axes[i].axhline(0, color='red', linestyle='--', linewidth=0.8)
    axes[i].set_ylabel('Error')
    axes[i].set_title(f'{var_cols[i]} Forecast Errors')
    axes[i].grid(alpha=0.3)

axes[-1].set_xlabel('Date')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'forecast_errors.png', dpi=150)
plt.show()

# %%
print("\n TVP-VAR estimation complete!")
coef_source = alpha_smooth if alpha_smooth is not None else alpha_filt
coef_label = "smoothed" if alpha_smooth is not None else "filtered"
print(f" Time-varying coefficients extracted ({coef_label}): {coef_source.shape}")
print(f" Time-varying covariances estimated: {H_t.shape}")

# %% [markdown]
# # 7. Time-Varying Impulse Response Functions (TV-IRF)

# %% [markdown]
# ## 7.1 IRF Definition in TVP-VAR

# %% [markdown]
# Time-Varying IRF measures the dynamic response of variable j
# to a shock in variable i at time t.
# 
# IRF(t, h, i→j) = ∂y_{j,t+h} / ∂ε_{i,t}
# 
# where:
# - t: time of shock
# - h: horizon (periods ahead)
# - i: shock variable
# - j: response variable
# 
# In TVP-VAR, IRF changes over time as coefficients α_t evolve.

# %% [markdown]
# ## 7.2 Compute IRF(t) for Each Time Point

# %%
def compute_tvp_irf(alpha_t, H_t, p, k, horizon=10):
    """
    Compute generalized IRF (Pesaran-Shin) at time t

    Parameters:
    - alpha_t: coefficient vector at time t (k*m,)
    - H_t: covariance matrix at time t (k, k)
    - p: lag order
    - k: number of variables
    - horizon: IRF horizon

    Returns:
    - irf: (horizon, k, k) array where irf[h, i, j] is response of j to shock in i at horizon h
    """
    m = 1 + k * p

    # Reshape coefficients into VAR companion form
    # Extract coefficient matrices (exclude intercept)
    A = np.zeros((k, k*p))
    for eq in range(k):
        start_idx = eq * m + 1  # skip intercept
        coefs = alpha_t[start_idx:start_idx + k*p]
        A[eq, :] = coefs

    # Companion matrix
    if p > 1:
        F = np.zeros((k*p, k*p))
        F[:k, :] = A
        F[k:, :k*(p-1)] = np.eye(k*(p-1))
    else:
        F = A

    # Generalized IRF uses reduced-form covariance directly (order-invariant)
    H_reg = (H_t + H_t.T) / 2
    min_eig = np.min(np.linalg.eigvalsh(H_reg))
    if min_eig < 1e-10:
        H_reg = H_reg + np.eye(k) * (1e-10 - min_eig)

    # Compute generalized IRF
    irf = np.zeros((horizon, k, k))
    shock_scale = np.sqrt(np.diag(H_reg))

    # Horizon 0: GIRF_0 = H * e_j / sqrt(sigma_jj)
    for j in range(k):
        irf[0, :, j] = H_reg[:, j] / shock_scale[j]

    # Subsequent horizons
    F_power = np.eye(k*p)
    for h in range(1, horizon):
        F_power = F_power @ F
        Phi_h = F_power[:k, :k]
        for j in range(k):
            irf[h, :, j] = (Phi_h @ H_reg[:, j]) / shock_scale[j]

    return irf

# %%
# Compute IRF for selected time points
irf_horizon = 10
coef_source = alpha_smooth if alpha_smooth is not None else alpha_filt
source_label = "smoothed" if alpha_smooth is not None else "filtered"
sample_times = [0, len(coef_source)//4, len(coef_source)//2, 3*len(coef_source)//4, -1]

irfs = {}

print(f"Computing IRF for {len(sample_times)} time points ({source_label})...")

for idx in sample_times:
    t_date = time_index[idx]
    irf_t = compute_tvp_irf(coef_source[idx, :], H_t[idx, :, :], p, k, irf_horizon)
    irfs[t_date] = irf_t
    print(f"  {t_date.date()}: IRF shape {irf_t.shape}")

print("✓ IRF computation complete")

# %% [markdown]
# ## 7.3 Visualize Structural Changes in IRF

# %%
# Plot IRF: JCI response to SP500 shock over time
jci_idx = 0  # JCI is first variable
sp500_idx = 1  # SP500 is second

fig, axes = plt.subplots(len(sample_times), 1, figsize=(12, 3*len(sample_times)), sharex=True)

if len(sample_times) == 1:
    axes = [axes]

for i, (t_date, irf_t) in enumerate(irfs.items()):
    # Extract JCI response to SP500 shock
    response = irf_t[:, jci_idx, sp500_idx]

    axes[i].plot(range(irf_horizon), response, marker='o', linewidth=2)
    axes[i].axhline(0, color='red', linestyle='--', linewidth=0.8)
    axes[i].set_ylabel('Response')
    axes[i].set_title(f'JCI Response to SP500 Shock - {t_date.date()}')
    axes[i].grid(alpha=0.3)

axes[-1].set_xlabel('Horizon (days)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'tvp_irf_jci_sp500.png', dpi=150)
plt.show()

# %%
# Heatmap of IRF at different time points
# Show IRF at horizon 1 for all variable pairs

fig, axes = plt.subplots(1, len(sample_times), figsize=(5*len(sample_times), 4))

if len(sample_times) == 1:
    axes = [axes]

for i, (t_date, irf_t) in enumerate(irfs.items()):
    # IRF at horizon 1
    irf_h1 = irf_t[1, :, :]

    im = axes[i].imshow(irf_h1, cmap='RdBu_r', aspect='auto', vmin=-np.abs(irf_h1).max(), vmax=np.abs(irf_h1).max())
    axes[i].set_xticks(range(k))
    axes[i].set_yticks(range(k))
    axes[i].set_xticklabels([v.replace('_RV', '').replace('_Level', '') for v in var_cols], rotation=45)
    axes[i].set_yticklabels([v.replace('_RV', '').replace('_Level', '') for v in var_cols])
    axes[i].set_title(f'{t_date.date()}')
    fig.colorbar(im, ax=axes[i], shrink=0.8)

plt.suptitle('Time-Varying IRF Matrix (Horizon=1)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'tvp_irf_heatmap.png', dpi=150)
plt.show()

# %% [markdown]
# ## 7.4 Interpretation

# %% [markdown]
# KEY FINDINGS FROM TV-IRF:
# 
# 1. Time-Varying Spillovers:
#    - IRF changes significantly across different periods
#    - Spillover intensity varies with market conditions
#    - Transmission patterns are not constant through time
# 
# 2. Structural Variation:
#    - Clear shifts visible in IRF evolution
#    - Pre/post episode differences appear in responses
#    - Dynamic sensitivity across horizons
# 
# 3. Dynamic Transmission:
#    - International shocks (SP500, VIX) have time-varying effects on JCI
#    - Commodity shocks (WTI, Gold) show changing importance
#    - Exchange-rate channel effects fluctuate over time
# 
# → Static VAR would miss these dynamics completely!

# %% [markdown]
# # 8. Connectedness Analysis (Spillover Indices)

# %% [markdown]
# ## 8.1 Compute Generalized Forecast Error Variance Decomposition (GFEVD)

# %% [markdown]
# Generalized FEVD (Diebold-Yilmaz approach):
# 
# For each time t, compute the contribution of shocks from variable i
# to the H-step ahead forecast error variance of variable j.
# 
# GFEVD is order-invariant (no need for Cholesky ordering).
# 
# Compute Generalized FEVD from IRF (Pesaran-Shin 1998):
#     
#     Generalized FEVD accounts for contemporaneous correlation via H_t:
#     
#     GFEVD(i←j) = σ_jj^(-1) * Σ_h [e_i' Φ_h Σ e_j]^2 / Σ_h [e_i' Φ_h Σ Φ_h' e_i]
#     
#     where Σ = H_t (time-varying error covariance)
# 
#     Parameters:
#     - irf: (horizon, k, k) generalized impulse responses (Pesaran-Shin, order-invariant)
#     - H_t: (k, k) covariance matrix at time t
#     - horizon: forecast horizon
# 
#     Returns:
#     - fevd: (k, k) matrix where fevd[i,j] = contribution of j to i's variance

# %%
def compute_gfevd(irf, H_t, horizon):
    k = irf.shape[1]

    # GFEVD matrix (generalized IRF, order-invariant)
    fevd = np.zeros((k, k))

    for i in range(k):  # receiving variable
        # Denominator: sum of generalized IRF squares
        mse_i = 0
        for h in range(horizon):
            mse_i += np.sum(irf[h, i, :] ** 2)

        for j in range(k):  # shock variable
            # Numerator: contribution of shock j to forecast error variance of i
            numerator = 0
            for h in range(horizon):
                numerator += irf[h, i, j] ** 2

            # Normalize by MSE
            fevd[i, j] = numerator / mse_i if mse_i > 0 else 0

    # Normalize rows to sum to 100%
    row_sums = fevd.sum(axis=1, keepdims=True)
    fevd = np.where(row_sums > 0, fevd / row_sums, 0)

    return fevd * 100  # percentage

# %%
# Compute GFEVD for all time points
coef_source = alpha_smooth if alpha_smooth is not None else alpha_filt
source_label = "smoothed" if alpha_smooth is not None else "filtered"
fevd_series = np.zeros((len(coef_source), k, k))

for t in range(len(coef_source)):
    # Compute IRF at time t
    irf_t = compute_tvp_irf(coef_source[t, :], H_t[t, :, :], p, k, irf_horizon)

    # Compute FEVD
    fevd_series[t, :, :] = compute_gfevd(irf_t, H_t[t, :, :], irf_horizon)

    if (t + 1) % 500 == 0:
        print(f"  Processed {t+1}/{len(coef_source)}")

print(f"GFEVD computation complete ({source_label})")

# %%
# Quick check: row sums should be ~100
print("\nSample FEVD matrix (time=0):")
fevd_sample = pd.DataFrame(
    fevd_series[0, :, :],
    index=[v.replace('_RV', '').replace('_Level', '') for v in var_cols],
    columns=[v.replace('_RV', '').replace('_Level', '') for v in var_cols]
)
print(fevd_sample.round(2))
print(f"\nRow sums: {fevd_sample.sum(axis=1).values}")

# %% [markdown]
# ## 8.2 Extracting Dynamic Spillover Measures

# %% [markdown]
# ### 8.2.1 Total Connectedness Index (TCI)

# %%
def compute_total_connectedness(fevd):
    """
    Total Connectedness Index (TCI)

    TCI = (sum of off-diagonal elements) / (total sum) × 100
    """
    k = fevd.shape[0]

    # Off-diagonal sum
    off_diag = fevd.sum() - np.trace(fevd)

    # Total sum
    total = fevd.sum()

    return (off_diag / total) * 100

# %%
# Compute TCI over time
tci = np.array([compute_total_connectedness(fevd_series[t, :, :])
                for t in range(len(fevd_series))])

tci_df = pd.DataFrame({
    'Date': time_index,
    'TCI': tci
}).set_index('Date')

print(f"TCI statistics:")
print(tci_df.describe())

# %%
# Export overall TCI statistics (no crisis split)
tci_stats_export = tci_df.describe().reset_index().rename(columns={"index": "Statistic"})
tci_stats_export.to_csv(OUTPUT_DIR / "TCI_Statistics_Overall.csv", index=False)

print("Saved: TCI_Statistics_Overall.csv")
print(tci_stats_export.round(3))

# %%
plt.figure(figsize=(14, 6))
plt.plot(tci_df.index, tci_df['TCI'], linewidth=1.5, color='darkblue', label='Total Connectedness Index')
plt.axhline(tci_df['TCI'].mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean: {tci_df["TCI"].mean():.2f}%')

plt.xlabel('Date')
plt.ylabel('Total Connectedness (%)')
plt.title('Total Connectedness Index (TCI) Over Time')
plt.legend(loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'total_connectedness.png', dpi=150)
plt.show()

# %% [markdown]
# ### 8.2.2 Directional TO spillovers

# %%
def compute_directional_to(fevd):
    """
    Directional TO spillover from variable j to all others

    TO_j = (sum of column j excluding diagonal) / k × 100
    """
    k = fevd.shape[0]
    to_spillover = np.zeros(k)

    for j in range(k):
        # Column sum excluding diagonal
        to_spillover[j] = (fevd[:, j].sum() - fevd[j, j]) / k

    return to_spillover

# %%
# Compute TO spillovers over time
to_spillovers = np.array([compute_directional_to(fevd_series[t, :, :])
                          for t in range(len(fevd_series))])

to_df = pd.DataFrame(
    to_spillovers,
    index=time_index,
    columns=[v.replace('_RV', '').replace('_Level', '') for v in var_cols]
)

print("Directional TO spillovers (first 5 dates):")
print(to_df.head())

# %%
# Plot TO spillovers
fig, axes = plt.subplots(k, 1, figsize=(14, 2*k), sharex=True)

for i, col in enumerate(to_df.columns):
    axes[i].plot(to_df.index, to_df[col], linewidth=1.2)
    axes[i].axhline(to_df[col].mean(), color='red', linestyle='--',
                    linewidth=0.8, alpha=0.5)
    axes[i].set_ylabel('TO (%)')
    axes[i].set_title(f'{col} - Directional TO Others')
    axes[i].grid(alpha=0.3)

axes[-1].set_xlabel('Date')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'directional_to_spillovers.png', dpi=150)
plt.show()

# %% [markdown]
# ### 8.2.3 Directional FROM spillovers

# %%
def compute_directional_from(fevd):
    """
    Directional FROM spillover received by variable i from all others

    FROM_i = (sum of row i excluding diagonal) / k × 100
    """
    k = fevd.shape[0]
    from_spillover = np.zeros(k)

    for i in range(k):
        # Row sum excluding diagonal
        from_spillover[i] = (fevd[i, :].sum() - fevd[i, i]) / k

    return from_spillover

# %%
# Compute FROM spillovers over time
from_spillovers = np.array([compute_directional_from(fevd_series[t, :, :])
                            for t in range(len(fevd_series))])

from_df = pd.DataFrame(
    from_spillovers,
    index=time_index,
    columns=[v.replace('_RV', '').replace('_Level', '') for v in var_cols]
)

print("Directional FROM spillovers (first 5 dates):")
print(from_df.head())

# %%
# Plot FROM spillovers
fig, axes = plt.subplots(k, 1, figsize=(14, 2*k), sharex=True)

for i, col in enumerate(from_df.columns):
    axes[i].plot(from_df.index, from_df[col], linewidth=1.2, color='coral')
    axes[i].axhline(from_df[col].mean(), color='red', linestyle='--',
                    linewidth=0.8, alpha=0.5)
    axes[i].set_ylabel('FROM (%)')
    axes[i].set_title(f'{col} - Directional FROM Others')
    axes[i].grid(alpha=0.3)

axes[-1].set_xlabel('Date')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'directional_from_spillovers.png', dpi=150)
plt.show()

# %% [markdown]
# ### 8.2.4 Net Spillover for each variable

# %%
# Net spillover = TO - FROM
net_spillover = to_df - from_df

print("Net spillovers (first 5 dates):")
print(net_spillover.head())

# %%
# Plot net spillovers
fig, axes = plt.subplots(k, 1, figsize=(14, 2*k), sharex=True)

for i, col in enumerate(net_spillover.columns):
    axes[i].plot(net_spillover.index, net_spillover[col], linewidth=1.2, color='forestgreen')
    axes[i].axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    axes[i].axhline(net_spillover[col].mean(), color='red', linestyle='--',
                    linewidth=0.8, alpha=0.5)
    axes[i].set_ylabel('Net (%)')
    axes[i].set_title(f'{col} - Net Spillover (TO - FROM)')
    axes[i].grid(alpha=0.3)

    # Fill positive/negative areas
    axes[i].fill_between(net_spillover.index, 0, net_spillover[col],
                         where=net_spillover[col]>=0, alpha=0.3, color='green',
                         interpolate=True, label='Net transmitter')
    axes[i].fill_between(net_spillover.index, 0, net_spillover[col],
                         where=net_spillover[col]<0, alpha=0.3, color='red',
                         interpolate=True, label='Net receiver')
    if i == 0:
        axes[i].legend(loc='upper left', fontsize=8)

axes[-1].set_xlabel('Date')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'net_spillovers.png', dpi=150)
plt.show()

# %%
# Summary: Average net spillovers
avg_net = net_spillover.mean().sort_values(ascending=False)

print("\nAverage Net Spillovers (TO - FROM):")
print(avg_net)

plt.figure(figsize=(10, 6))
avg_net.plot(kind='barh', color=['green' if x > 0 else 'red' for x in avg_net])
plt.axvline(0, color='black', linewidth=1)
plt.xlabel('Average Net Spillover (%)')
plt.title('Net Transmitters vs Net Receivers')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'average_net_spillovers.png', dpi=150)
plt.show()

# %% [markdown]
# ### 8.2.5 Pairwise Directional Connectedness Analysis for JCI

# %% [markdown]
# 1. Average FEVD Matrix across time

# %%
# Calculate time-averaged FEVD matrix
fevd_avg = fevd_series.mean(axis=0)  # Shape: (k, k)

print("Average FEVD Matrix (%) - Time-averaged across all periods")

fevd_avg_df = pd.DataFrame(
    fevd_avg,
    index=[v.replace('_RV', '').replace('_Level', '') for v in var_cols],
    columns=[v.replace('_RV', '').replace('_Level', '') for v in var_cols]
)

print(fevd_avg_df.round(2))
print(f"\nRow sums (should be ~100): {fevd_avg_df.sum(axis=1).round(2).values}")


# %% [markdown]
# 2. Table 1: Pairwise Directional Spillover from JCI to others

# %%
# JCI is the first variable (index 0)
jci_idx = 0

# Extract spillovers FROM JCI TO all other variables
spillover_from_jci = {}

for j in range(k):
    if j != jci_idx:
        var_name = var_cols[j].replace('_RV', '').replace('_Level', '')
        # fevd_avg[j, jci_idx] = contribution of JCI shock to variance of variable j
        spillover_from_jci[var_name] = fevd_avg[j, jci_idx]

# Create DataFrame and sort
table1 = pd.DataFrame.from_dict(
    spillover_from_jci,
    orient='index',
    columns=['Spillover FROM JCI (%)'
]
).sort_values('Spillover FROM JCI (%)', ascending=False)

# Add rank
table1['Rank'] = range(1, len(table1) + 1)
table1 = table1[['Rank', 'Spillover FROM JCI (%)']]

print("\n", table1.round(3))

# Identify domestic vs international
domestic_spillover = table1.loc['USDIDR', 'Spillover FROM JCI (%)']
international_spillover = table1.drop('USDIDR')['Spillover FROM JCI (%)'].sum()

# %% [markdown]
# 3. Table 2: Net Pairwise Directional Connectedness (JCI vs Others)

# %%
npdc_results = {}

for j in range(k):
    if j != jci_idx:
        var_name = var_cols[j].replace('_RV', '').replace('_Level', '')

        # Spillover FROM JCI TO variable j
        jci_to_j = fevd_avg[j, jci_idx]

        # Spillover FROM variable j TO JCI
        j_to_jci = fevd_avg[jci_idx, j]

        # Net pairwise directional connectedness
        npdc = jci_to_j - j_to_jci

        npdc_results[var_name] = {
            'JCI → Variable (%)': jci_to_j,
            'Variable → JCI (%)': j_to_jci,
            'NPDC (%)': npdc,
            'Direction': 'JCI transmits' if npdc > 0 else 'JCI receives'
        }

# Create DataFrame
table2 = pd.DataFrame.from_dict(npdc_results, orient='index')
table2 = table2.sort_values('NPDC (%)', ascending=False)

# Add rank
table2.insert(0, 'Rank', range(1, len(table2) + 1))

print("\n", table2.round(3))

# Key insights
print(f"\n KEY FINDINGS:")
print(f"\n1. JCI's Strongest Spillover Target:")
strongest_receiver = table1.index[0]
strongest_value = table1.iloc[0]['Spillover FROM JCI (%)']
print(f"   → {strongest_receiver}: {strongest_value:.2f}% of its variance explained by JCI shocks")

print(f"\n2. Net Transmission Patterns:")
for var in ['USDIDR', 'SP500', 'Gold']:
    if var in table2.index:
        npdc_val = table2.loc[var, 'NPDC (%)']
        direction = "transmits to" if npdc_val > 0 else "receives from"
        print(f"   → JCI {direction} {var}: {abs(npdc_val):.2f}% net spillover")

print(f"\n3. Domestic vs International Spillover:")
if 'USDIDR' in table1.index:
    usdidr_rank = table1.loc['USDIDR', 'Rank']
    print(f"   → USDIDR ranks #{int(usdidr_rank)} among JCI's spillover targets")
    if usdidr_rank == 1:
        print(f"   →  CONFIRMED: JCI primarily transmits to domestic market (USDIDR)")
    else:
        print(f"   →  JCI's spillover is more international than domestic")

# %% [markdown]
# 4. Visualization

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel A: Spillover FROM JCI
ax1 = axes[0]
colors1 = ['darkgreen' if idx == 'USDIDR' else 'steelblue' for idx in table1.index]
table1['Spillover FROM JCI (%)'].plot(
    kind='barh',
    ax=ax1,
    color=colors1,
    edgecolor='black',
    linewidth=1.2
)
ax1.set_xlabel('Spillover (%)', fontsize=11)
ax1.set_title('Panel A: Directional Spillover FROM JCI TO Others\n(Domestic vs International)',
              fontweight='bold', fontsize=12)
ax1.grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, row) in enumerate(table1.iterrows()):
    val = row['Spillover FROM JCI (%)']
    ax1.text(val + 0.1, i, f'{val:.2f}%', va='center', fontsize=9)

# Panel B: Net Pairwise Directional Connectedness
ax2 = axes[1]
colors2 = ['green' if x > 0 else 'red' for x in table2['NPDC (%)']]
table2['NPDC (%)'].plot(
    kind='barh',
    ax=ax2,
    color=colors2,
    edgecolor='black',
    linewidth=1.2
)
ax2.axvline(0, color='black', linestyle='-', linewidth=1.5)
ax2.set_xlabel('Net Spillover (%)', fontsize=11)
ax2.set_title('Panel B: Net Pairwise Directional Connectedness\n(Positive = JCI transmits)',
              fontweight='bold', fontsize=12)
ax2.grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, row) in enumerate(table2.iterrows()):
    val = row['NPDC (%)']
    x_pos = val + (0.15 if val > 0 else -0.15)
    ha = 'left' if val > 0 else 'right'
    ax2.text(x_pos, i, f'{val:.2f}%', va='center', ha=ha, fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Figure_Pairwise_Spillover_JCI.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n Figure saved: Figure_Pairwise_Spillover_JCI.png")

# %%
# ── Step 1: Net pairwise matrix ──
net_pairwise = np.zeros((k, k))
for i in range(k):
    for j in range(k):
        if i != j:
            net_pairwise[i, j] = fevd_avg[j, i] - fevd_avg[i, j]

var_names_clean = [v.replace('_RV', '').replace('_Level', '') for v in var_cols]
net_pairwise_df = pd.DataFrame(net_pairwise, index=var_names_clean, columns=var_names_clean)

print("Net Pairwise Spillover Matrix (i→j, positive = i transmits to j):")
print(net_pairwise_df.round(3))

antisymmetry_error = np.abs(net_pairwise + net_pairwise.T).max()
print(f"\nMax antisymmetry error |A + A'|: {antisymmetry_error:.10f}")

# %%
# ── Gephi Export: Nodes + Edges CSV ─────────────────────────────────────────

# ── NODES ──
# NET  = row sum (valid because matrix is antisymmetric) = TO - FROM
# TO   = sum of positive values in row  → how much this node transmits
# FROM = sum of abs negative values in row → how much this node receives
nodes_rows = []
for node in net_pairwise_df.index:
    row_ = net_pairwise_df.loc[node]
    TO_   = round(row_[row_ > 0].sum(), 4)
    FROM_ = round(row_[row_ < 0].abs().sum(), 4)
    net   = round(row_.sum(), 4)          # = TO - FROM
    role  = "Net Transmitter" if net > 0 else "Net Receiver"
    nodes_rows.append({
        "Id":    node,
        "Label": node,
        "TO":    TO_,
        "FROM":  FROM_,
        "NET":   net,
        "Role":  role
    })

nodes_gephi = pd.DataFrame(nodes_rows)

# Scale abs(NET) to [10, 100] → pakai kolom "Size" ini di Gephi:
#   Appearance → Nodes → Size → Ranking → pilih "Size"
abs_net = nodes_gephi["NET"].abs()
nodes_gephi["Size"] = (
    10 + 90 * (abs_net - abs_net.min()) / (abs_net.max() - abs_net.min())
).round(4)

print("=== NODES ===")
print(nodes_gephi.to_string(index=False))

# ── EDGES ──
# Iterate upper triangle only → each pair appears exactly once, no duplicates
edges_rows = []
nodes_list = list(net_pairwise_df.index)
for i in range(len(nodes_list)):
    for j in range(i + 1, len(nodes_list)):
        u, v = nodes_list[i], nodes_list[j]
        val = net_pairwise_df.loc[u, v]
        if val == 0:
            continue
        if val > 0:
            src, tgt, w = u, v, val      # u net transmits to v
        else:
            src, tgt, w = v, u, -val     # v net transmits to u

        edges_rows.append({
            "Source": src,
            "Target": tgt,
            "Weight": round(w, 4),
            "Label":  f"{w:.4f}%",       # tampil di Gephi sebagai edge label
            "Type":   "Directed"
        })

edges_gephi = (pd.DataFrame(edges_rows)
               .sort_values("Weight", ascending=False)
               .reset_index(drop=True))

print(f"\n=== EDGES ({len(edges_gephi)} total) ===")
print(edges_gephi.to_string(index=False))

# ── SAVE ──
nodes_gephi.to_csv(OUTPUT_DIR / "gephi_nodes.csv", index=False)
edges_gephi.to_csv(OUTPUT_DIR / "gephi_edges.csv", index=False)
print(f"\n✓ gephi_nodes.csv  ({len(nodes_gephi)} rows)  — columns: {list(nodes_gephi.columns)}")
print(f"✓ gephi_edges.csv  ({len(edges_gephi)} rows)  — columns: {list(edges_gephi.columns)}")
print(f"   → saved to: {OUTPUT_DIR}")


# %%
# ── Step 4–9: Network Visualization ──
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Stable fixed positions — scaled out for more inter-node spacing
pos = {
    'JCI':    np.array([ 0.00,  0.00]),
    'SP500':  np.array([ 2.10,  0.00]),
    'SSE':    np.array([ 1.30,  1.80]),
    'VIX':    np.array([-1.05,  1.90]),
    'WTI':    np.array([-2.25,  0.00]),
    'Gold':   np.array([-0.90, -1.85]),
    'USDIDR': np.array([ 1.30, -1.85]),
}
for n in G.nodes():
    if n not in pos:
        angle = 2 * np.pi * list(G.nodes()).index(n) / len(G.nodes())
        pos[n] = np.array([2.0 * np.cos(angle), 2.0 * np.sin(angle)])

node_palette = {'JCI':'#E63946','SP500':'#457B9D','SSE':'#2A9D8F',
                'VIX':'#E9C46A','WTI':'#F4A261','Gold':'#A8DADC','USDIDR':'#6D6875'}

fig, ax = plt.subplots(figsize=(22, 18))
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

role_vals = np.array([node_net_role[n] for n in G.nodes()])
role_norm = mcolors.TwoSlopeNorm(vmin=min(role_vals.min(), -0.01),
                                   vcenter=0, vmax=max(role_vals.max(), 0.01))
node_cmap   = cm.RdYlGn
node_colors = [node_cmap(role_norm(node_net_role[n])) for n in G.nodes()]
node_sizes  = [5500 + abs(node_net_role[n]) * 800 for n in G.nodes()]

nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color=node_colors,
                        edgecolors='#333333', linewidths=2.5, alpha=0.95)

# Node labels — clean, no background box
nx.draw_networkx_labels(G, pos, ax=ax, font_size=17, font_weight='bold', font_color='#111111')

# Variable curvature per edge
def get_edge_rad(u, v, pos):
    p0, p1 = pos[u], pos[v]
    a0, a1 = np.arctan2(p0[1], p0[0]), np.arctan2(p1[1], p1[0])
    delta = a1 - a0
    while delta <= -np.pi: delta += 2 * np.pi
    while delta >  np.pi:  delta -= 2 * np.pi
    sign = 1 if delta >= 0 else -1
    base = 0.16 if 'JCI' in (u, v) else 0.22
    if abs(delta) > 2.2: base = 0.12
    return sign * base

edges   = list(G.edges())
weights = [G[u][v]['weight'] for u, v in edges]
max_w   = max(weights) if weights else 1.0

drawn_edge_info = []
for (u, v) in edges:
    w          = G[u][v]['weight']
    edge_width = 0.8 + (w / max_w) * 5.8
    edge_alpha = 0.35 + (w / max_w) * 0.55
    rad        = get_edge_rad(u, v, pos)
    nx.draw_networkx_edges(
        G, pos, edgelist=[(u, v)], ax=ax,
        width=edge_width, edge_color=[node_palette.get(u, '#999999')],
        alpha=edge_alpha, arrows=True, arrowsize=22, arrowstyle='-|>',
        connectionstyle=f'arc3,rad={rad}',
        min_source_margin=28, min_target_margin=28
    )
    drawn_edge_info.append({'u': u, 'v': v, 'w': w, 'rad': rad})

# Label position using analytic quadratic Bezier — matches arc3 geometry exactly
def bezier_arc_label_pos(p0, p1, rad=0.2, t=0.5, label_offset=0.06):
    """
    Reproduce arc3 control point analytically and evaluate the
    quadratic Bezier at parameter t (0.5 = true geometric midpoint).
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d  = p1 - p0
    L  = np.hypot(d[0], d[1]) + 1e-12
    perp    = np.array([-d[1], d[0]]) / L
    control = 0.5 * (p0 + p1) + rad * L * 0.5 * perp   # arc3 control point

    B  = (1 - t)**2 * p0 + 2 * (1 - t) * t * control + t**2 * p1
    dB = 2 * (1 - t) * (control - p0) + 2 * t * (p1 - control)
    dB_norm = np.hypot(dB[0], dB[1]) + 1e-12
    normal  = np.array([-dB[1], dB[0]]) / dB_norm

    return float(B[0] + label_offset * normal[0]), float(B[1] + label_offset * normal[1])

for info in drawn_edge_info:
    u, v, w, rad = info['u'], info['v'], info['w'], info['rad']
    if w < 0.30:          # skip label for very small edges — keeps figure clean
        continue
    x, y = bezier_arc_label_pos(pos[u], pos[v], rad=rad, t=0.55, label_offset=0.015)
    ax.annotate(f"{w:.2f}%", xy=(x, y), ha='center', va='center',
                fontsize=17, color='#1a1a1a', fontweight='semibold',
                bbox=dict(boxstyle='round,pad=0.28', facecolor='none',
                            edgecolor='none'))

ax.set_title('Figure 4.x: Net Pairwise Directional Connectedness Network\n'
             'Arrow i→j: i is net transmitter to j  |  One dominant direction per pair  |  Thickness ∝ magnitude\n'
             'Node colour: green = net transmitter, red = net receiver  |  Node size ∝ |net role|',
             fontsize=13, fontweight='bold', pad=20)

sm = cm.ScalarMappable(cmap=node_cmap, norm=role_norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.38, pad=0.01)
cbar.set_label('Net Role (%)\n(+ transmitter / − receiver)', fontsize=11)
cbar.ax.tick_params(labelsize=10)

leg_elem = [mpatches.Patch(color=c, label=n) for n, c in node_palette.items()]
ax.legend(handles=leg_elem, loc='lower left', fontsize=11,
          title='Edge Source Node', title_fontsize=11, framealpha=0.95,
          edgecolor='#cccccc')
ax.axis('off')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Figure_Network_Pairwise_Spillover.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: Figure_Network_Pairwise_Spillover.png")

# ── Step 10: Heatmap (separate PNG) ──
fig2, ax2 = plt.subplots(figsize=(8, 6))
mat    = net_pairwise_df.values
nz_abs = np.abs(mat[mat != 0])
absmax = nz_abs.max() if len(nz_abs) > 0 else 1.0
im = ax2.imshow(mat, cmap='RdYlGn', aspect='auto', vmin=-absmax, vmax=absmax)
ax2.set_xticks(range(len(var_names_clean)))
ax2.set_yticks(range(len(var_names_clean)))
ax2.set_xticklabels(var_names_clean, rotation=45, ha='right', fontsize=10)
ax2.set_yticklabels(var_names_clean, fontsize=10)
ax2.set_xlabel('Receiver (j)', fontsize=11)
ax2.set_ylabel('Transmitter (i)', fontsize=11)
ax2.set_title('Net Pairwise Directional Connectedness Matrix\n'
              'Cell (i,j): positive → i net transmits to j | Time-Averaged TVP-VAR',
              fontsize=11, fontweight='bold')
for i in range(len(var_names_clean)):
    for j in range(len(var_names_clean)):
        val = mat[i, j]
        txt_color = 'white' if abs(val) > absmax * 0.55 else 'black'
        ax2.text(j, i, f'{val:.2f}', ha='center', va='center',
                 fontsize=9, color=txt_color, fontweight='bold')
plt.colorbar(im, ax=ax2, shrink=0.75, label='Net Spillover (%)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Figure_Network_Pairwise_Heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: Figure_Network_Pairwise_Heatmap.png")

# %%
# Stable fixed positions — scaled out for more inter-node spacing
pos = {
    'JCI':    np.array([ 0.00,  0.00]),
    'SP500':  np.array([ 2.10,  0.00]),
    'SSE':    np.array([ 1.30,  1.80]),
    'VIX':    np.array([-1.05,  1.90]),
    'WTI':    np.array([-2.25,  0.00]),
    'Gold':   np.array([-0.90, -1.85]),
    'USDIDR': np.array([ 1.30, -1.85]),
}
for n in G.nodes():
    if n not in pos:
        angle = 2 * np.pi * list(G.nodes()).index(n) / len(G.nodes())
        pos[n] = np.array([2.0 * np.cos(angle), 2.0 * np.sin(angle)])

node_palette = {'JCI':'#E63946','SP500':'#457B9D','SSE':'#2A9D8F',
                'VIX':'#E9C46A','WTI':'#F4A261','Gold':'#A8DADC','USDIDR':'#6D6875'}

fig, ax = plt.subplots(figsize=(22, 18))
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

role_vals = np.array([node_net_role[n] for n in G.nodes()])
role_norm = mcolors.TwoSlopeNorm(vmin=min(role_vals.min(), -0.01),
                                   vcenter=0, vmax=max(role_vals.max(), 0.01))
node_cmap   = cm.RdYlGn
node_colors = [node_cmap(role_norm(node_net_role[n])) for n in G.nodes()]
node_sizes  = [5500 + abs(node_net_role[n]) * 800 for n in G.nodes()]

nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color=node_colors,
                        edgecolors='#333333', linewidths=2.5, alpha=0.95)

# Node labels — clean, no background box
nx.draw_networkx_labels(G, pos, ax=ax, font_size=17, font_weight='bold', font_color='#111111')

# Variable curvature per edge
def get_edge_rad(u, v, pos):
    p0, p1 = pos[u], pos[v]
    a0, a1 = np.arctan2(p0[1], p0[0]), np.arctan2(p1[1], p1[0])
    delta = a1 - a0
    while delta <= -np.pi: delta += 2 * np.pi
    while delta >  np.pi:  delta -= 2 * np.pi
    sign = 1 if delta >= 0 else -1
    base = 0.16 if 'JCI' in (u, v) else 0.22
    if abs(delta) > 2.2: base = 0.12
    return sign * base

edges   = list(G.edges())
weights = [G[u][v]['weight'] for u, v in edges]
max_w   = max(weights) if weights else 1.0

# Draw Edges only (labels removed)
for (u, v) in edges:
    w          = G[u][v]['weight']
    edge_width = 0.8 + (w / max_w) * 5.8
    edge_alpha = 0.35 + (w / max_w) * 0.55
    rad        = get_edge_rad(u, v, pos)
    nx.draw_networkx_edges(
        G, pos, edgelist=[(u, v)], ax=ax,
        width=edge_width, edge_color=[node_palette.get(u, '#999999')],
        alpha=edge_alpha, arrows=True, arrowsize=22, arrowstyle='-|>',
        connectionstyle=f'arc3,rad={rad}',
        min_source_margin=28, min_target_margin=28
    )

ax.set_title('Figure 4.x: Net Pairwise Directional Connectedness Network\n'
             'Arrow i→j: i is net transmitter to j  |  One dominant direction per pair  |  Thickness ∝ magnitude\n'
             'Node colour: green = net transmitter, red = net receiver  |  Node size ∝ |net role|',
             fontsize=13, fontweight='bold', pad=20)

sm = cm.ScalarMappable(cmap=node_cmap, norm=role_norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.38, pad=0.01)
cbar.set_label('Net Role (%)\n(+ transmitter / − receiver)', fontsize=11)
cbar.ax.tick_params(labelsize=10)

leg_elem = [mpatches.Patch(color=c, label=n) for n, c in node_palette.items()]
ax.legend(handles=leg_elem, loc='lower left', fontsize=11,
          title='Edge Source Node', title_fontsize=11, framealpha=0.95,
          edgecolor='#cccccc')
ax.axis('off')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Figure_Network_Pairwise_Spillover.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: Figure_Network_Pairwise_Spillover.png")

# %%
# Combine all spillover measures
spillover_data = pd.concat([
    tci_df,
    to_df.add_prefix('TO_'),
    from_df.add_prefix('FROM_'),
    net_spillover.add_prefix('NET_')
], axis=1)

print(f"\nSpillover dataset shape: {spillover_data.shape}")
print("\nColumns:")
print(list(spillover_data.columns))

spillover_data.head()

# %%
# Save spillover indices
spillover_data.to_csv(OUTPUT_DIR / 'tvp_var_spillover_indices.csv')
print("\nSpillover indices saved to 'tvp_var_spillover_indices.csv'")

# %% [markdown]
# ## 8.4 Visualization of Connectedness Dynamics

# %%
# Create comprehensive spillover dashboard
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Total Connectedness
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(tci_df.index, tci_df['TCI'], linewidth=2, color='darkblue')
ax1.axhline(tci_df['TCI'].mean(), color='red', linestyle='--', linewidth=1.5)
ax1.fill_between(tci_df.index, tci_df['TCI'].min(), tci_df['TCI'], alpha=0.2)
ax1.set_ylabel('TCI (%)')
ax1.set_title('Total Connectedness Index', fontweight='bold', fontsize=12)
ax1.grid(alpha=0.3)

# 2. JCI spillovers
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(to_df.index, to_df['JCI'], label='TO', linewidth=1.5)
ax2.plot(from_df.index, from_df['JCI'], label='FROM', linewidth=1.5)
ax2.plot(net_spillover.index, net_spillover['JCI'], label='NET', linewidth=1.5)
ax2.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax2.set_ylabel('Spillover (%)')
ax2.set_title('JCI Spillover Dynamics', fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. SP500 spillovers
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(to_df.index, to_df['SP500'], label='TO', linewidth=1.5)
ax3.plot(from_df.index, from_df['SP500'], label='FROM', linewidth=1.5)
ax3.plot(net_spillover.index, net_spillover['SP500'], label='NET', linewidth=1.5)
ax3.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax3.set_ylabel('Spillover (%)')
ax3.set_title('SP500 Spillover Dynamics', fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Net spillovers comparison
ax4 = fig.add_subplot(gs[2, :])
for col in ['JCI', 'SP500', 'VIX', 'USDIDR', 'SSE']:
    if col in net_spillover.columns:
        ax4.plot(net_spillover.index, net_spillover[col],
                label=col, linewidth=1.5, alpha=0.8)
ax4.axhline(0, color='black', linestyle='-', linewidth=1)
ax4.set_xlabel('Date')
ax4.set_ylabel('Net Spillover (%)')
ax4.set_title('Net Spillover Comparison (Key Variables)', fontweight='bold')
ax4.legend(loc='upper left')
ax4.grid(alpha=0.3)

plt.savefig(OUTPUT_DIR / 'spillover_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# Comprehensive chart and table for thesis

# %%
# Calculate summary statistics for directional spillovers
summary_stats = pd.DataFrame({
    'TO (Mean)': to_df.mean(),
    'TO (Std)': to_df.std(),
    'FROM (Mean)': from_df.mean(),
    'FROM (Std)': from_df.std(),
    'NET (Mean)': net_spillover.mean(),
    'NET (Std)': net_spillover.std(),
    'Role': ['Transmitter' if x > 0 else 'Receiver' for x in net_spillover.mean()]
})

# Add time-varying classification
summary_stats['Transmitter %'] = (net_spillover > 0).sum() / len(net_spillover) * 100

# Sort by NET mean
summary_stats = summary_stats.sort_values('NET (Mean)', ascending=False)

print("\nTable 4.3: Directional Spillover Statistics")
print(summary_stats.round(2))

# Export for LaTeX
summary_stats.to_csv(OUTPUT_DIR / 'table_spillover_summary.csv')

# %%
# Figure 4.5: Net Spillover Dynamics Over Time — 7 rows × 1 col, single PNG
variables = ['SP500', 'JCI', 'VIX', 'Gold', 'WTI', 'USDIDR', 'SSE']
colors = ['darkblue', 'green', 'red', 'gold', 'brown', 'orange', 'purple']

fig, axes = plt.subplots(7, 1, figsize=(14, 28), sharex=True)

for ax, var, color in zip(axes, variables, colors):

    # Plot NET spillover
    ax.plot(net_spillover.index, net_spillover[var],
            linewidth=1.2, color=color, alpha=0.8)

    # Zero line
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    # Mean line
    ax.axhline(net_spillover[var].mean(), color='red',
               linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'Mean: {net_spillover[var].mean():.2f}%')

    # Fill transmitter/receiver zones
    ax.fill_between(net_spillover.index, 0, net_spillover[var],
                    where=net_spillover[var] >= 0,
                    alpha=0.3, color='green',
                    interpolate=True, label='Transmitter')
    ax.fill_between(net_spillover.index, 0, net_spillover[var],
                    where=net_spillover[var] < 0,
                    alpha=0.3, color='red',
                    interpolate=True, label='Receiver')

    ax.set_ylabel('Net Spillover (%)', fontsize=10)
    ax.set_title(f'{var} Net Spillover Dynamics', fontweight='bold', fontsize=11)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

axes[-1].set_xlabel('Date', fontsize=11)
plt.suptitle('Figure 4.5: Time-Varying Net Spillover Roles (2002–2025)',
             fontsize=13, fontweight='bold', y=1.002)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Figure_4_5_Net_Spillover_Dynamics.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Figure 4.5 saved: Net spillover dynamics")

# %%
# Figure 4.3: Total Connectedness Index
fig, ax = plt.subplots(figsize=(14, 6))

# Plot TCI
ax.plot(tci_df.index, tci_df['TCI'], linewidth=2, color='darkblue', label='Total Connectedness Index')
ax.axhline(tci_df['TCI'].mean(), color='red', linestyle='--', linewidth=1.5,
           label=f'Mean: {tci_df["TCI"].mean():.2f}%')

ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Total Connectedness Index (%)', fontsize=11)
ax.set_title('Total Connectedness Index Over Time', fontweight='bold')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Figure_4_3_Total_Connectedness.png', dpi=300, bbox_inches='tight')
plt.show()

print("Figure 4.3 saved: Total Connectedness Index")

# %%
print("\nConnectedness analysis complete!")
print(f"✓ Total observations: {len(spillover_data)}")
print(f"✓ Spillover measures: {spillover_data.shape[1]}")

# %% [markdown]
# ## 8.5 Filtered Spillover Indices (Addressing Look-Ahead Bias)

# %% [markdown]
# **Methodological Note:**
# 
# The spillover indices computed in Section 8.3 use smoothed states (`alpha_smooth`) from the Rauch-Tung-Striebel backward pass, which incorporates future information. While appropriate for ex-post analysis, this creates look-ahead bias when used as predictive features.
# 
# **Solution:** Recompute indices using filtered states (`alpha_filt`) only, which use information available up to time $t$ only. This ensures valid out-of-sample prediction for the EWS model.

# %%
# Recompute GFEVD using filtered states
fevd_filt = np.zeros((len(alpha_filt), k, k))

for t in range(len(alpha_filt)):
    irf_t = compute_tvp_irf(alpha_filt[t, :], H_t[t, :, :], p, k, irf_horizon)
    fevd_filt[t, :, :] = compute_gfevd(irf_t, H_t[t, :, :], irf_horizon)

    if (t + 1) % 1000 == 0:
        print(f"Processed {t+1}/{len(alpha_filt)}")

print(f"Filtered GFEVD computed: {fevd_filt.shape}")

# %%
# Compute TCI from filtered FEVD
tci_filt = np.array([compute_total_connectedness(fevd_filt[t]) for t in range(len(fevd_filt))])
tci_df_filt = pd.DataFrame({'TCI': tci_filt}, index=time_index)

# Compare with smoothed version
print(f"TCI Statistics:")
print(f"  Smoothed: mean={tci_df['TCI'].mean():.2f}%, std={tci_df['TCI'].std():.2f}%")
print(f"  Filtered: mean={tci_df_filt['TCI'].mean():.2f}%, std={tci_df_filt['TCI'].std():.2f}%")
print(f"  Difference: {(tci_df['TCI'] - tci_df_filt['TCI']).abs().mean():.3f}% points")

# %%
# Compute directional spillovers from filtered FEVD
to_filt = np.array([compute_directional_to(fevd_filt[t]) for t in range(len(fevd_filt))])
from_filt = np.array([compute_directional_from(fevd_filt[t]) for t in range(len(fevd_filt))])

to_df_filt = pd.DataFrame(to_filt, index=time_index, columns=[v.replace('_RV', '').replace('_Level', '') for v in var_cols])
from_df_filt = pd.DataFrame(from_filt, index=time_index, columns=[v.replace('_RV', '').replace('_Level', '') for v in var_cols])
net_filt = to_df_filt - from_df_filt

print(f"Directional spillovers computed (filtered): {to_df_filt.shape}")

# %%
# Combine filtered spillover measures
spillover_filt = pd.concat([
    tci_df_filt,
    to_df_filt.add_prefix('TO_'),
    from_df_filt.add_prefix('FROM_'),
    net_filt.add_prefix('NET_')
], axis=1)

print(f"Filtered spillover dataset: {spillover_filt.shape}")

# %%
# Keep filtered spillover in memory for downstream feature export
print("Filtered spillover indices ready in memory: spillover_filt")
print("Export is handled in Section 9 (connectedness features only)")

# %%
# Visual comparison: Smoothed vs Filtered TCI
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

# Panel A: Time series comparison
axes[0].plot(tci_df.index, tci_df['TCI'], linewidth=1.5, color='#E74C3C', alpha=0.8, label='Smoothed')
axes[0].plot(tci_df_filt.index, tci_df_filt['TCI'], linewidth=1.5, color='#3498DB', alpha=0.8, label='Filtered')

axes[0].set_ylabel('TCI (%)')
axes[0].set_title('Total Connectedness Index: Smoothed vs Filtered', fontweight='bold')
axes[0].legend(loc='upper left')
axes[0].grid(alpha=0.3)

# Panel B: Difference (bias from smoothing)
diff = tci_df['TCI'] - tci_df_filt['TCI']
axes[1].plot(diff.index, diff, linewidth=1.2, color='#9B59B6')
axes[1].axhline(0, color='black', linestyle='-', linewidth=1)
axes[1].fill_between(diff.index, 0, diff, alpha=0.3, color='#9B59B6')

axes[1].set_xlabel('Date')
axes[1].set_ylabel('Bias (% points)')
axes[1].set_title('Look-Ahead Bias (Smoothed - Filtered)', fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Figure_Smoothed_vs_Filtered_TCI.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Figure saved: Figure_Smoothed_vs_Filtered_TCI.png")
print(f"Mean absolute bias: {diff.abs().mean():.3f}% points")
print(f"Correlation: {tci_df['TCI'].corr(tci_df_filt['TCI']):.4f}")

# %%
# Summary statistics: Smoothed vs Filtered (overall sample)
stats_summary = pd.DataFrame({
    'Method': ['Smoothed', 'Filtered'],
    'Mean': [tci_df['TCI'].mean(), tci_df_filt['TCI'].mean()],
    'Std': [tci_df['TCI'].std(), tci_df_filt['TCI'].std()],
    'Min': [tci_df['TCI'].min(), tci_df_filt['TCI'].min()],
    'Max': [tci_df['TCI'].max(), tci_df_filt['TCI'].max()]
})

stats_summary.to_csv(OUTPUT_DIR / 'Table_Smoothed_vs_Filtered_Stats.csv', index=False)
print("\nTCI Overall Statistics:")
print(stats_summary.round(3).to_string(index=False))
print("\nTable saved: Table_Smoothed_vs_Filtered_Stats.csv")

# %%
# Summary
print("\n" + "="*70)
print("CONNECTEDNESS FEATURES READY FOR AI NOTEBOOK")
print("="*70)
print("\nPrimary output files:")
print("  1. tvpvar_connectedness_FILTERED.csv")
print("     - No look-ahead bias (uses filtered states)")
print("     - Includes only TCI, TO_*, FROM_*, NET_*")
print("  2. tvpvar_connectedness_feature_list.csv")
print("     - Explicit feature inventory for ML/DL pipeline")
print("\nOptional EDA output:")
print("  - tvpvar_connectedness_SMOOTHED.csv")
print("="*70)

# %% [markdown]
# # 9. Export Connectedness Features Only

# %%
# ============================================================
# 9. EXPORT: TVP-VAR CONNECTEDNESS FEATURES ONLY (CLEAN)
# ============================================================

# Keep ONLY connectedness columns: TCI, TO_, FROM_, NET_
connectedness_cols = [
    c for c in spillover_filt.columns
    if ('TCI' in c.upper()) or ('TO_' in c.upper()) or ('FROM_' in c.upper()) or ('NET_' in c.upper())
]

connectedness_filt = spillover_filt[connectedness_cols].copy()

# Basic checks
assert connectedness_filt.index.is_monotonic_increasing
assert connectedness_filt.index.is_unique
assert connectedness_filt.notnull().all().all()

# Export filtered features (SAFE for ML forecasting)
connectedness_filt.to_csv(OUTPUT_DIR / "tvpvar_connectedness_FILTERED.csv", index_label="Date")
print("Saved: tvpvar_connectedness_FILTERED.csv")
print(f"Rows: {len(connectedness_filt):,} | Cols: {len(connectedness_cols)}")

# Export feature list
pd.Series(connectedness_cols, name="feature").to_csv(OUTPUT_DIR / "tvpvar_connectedness_feature_list.csv", index=False)
print("Saved: tvpvar_connectedness_feature_list.csv")

# OPTIONAL: export smoothed for narrative/EDA only (NOT for ML)
if 'spillover' in globals():
    connectedness_sm = spillover[[c for c in spillover.columns if c in connectedness_cols]].copy()
    connectedness_sm.to_csv(OUTPUT_DIR / "tvpvar_connectedness_SMOOTHED.csv", index_label="Date")
    print("Saved: tvpvar_connectedness_SMOOTHED.csv (EDA only, not for forecasting)")

# %%
connectedness_filt

# %%



