from pathlib import Path
import math
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)


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
INPUT_PATH = BASE_DIR / "data" / "processed" / "tvp_var" / "tvp_var_spillover_indices.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "regime"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "hmm_regimes.csv"
LATENT_PATH = OUTPUT_DIR / "latent_embeddings.csv"
MODEL_PATH = OUTPUT_DIR / "tcn_autoencoder_best.pt"

SMOOTH_WINDOW = 5
LOOKBACK = 60
BATCH_SIZE = 64

NUM_FILTERS = 16
LATENT_DIM = 4
KERNEL_SIZE = 3
DROPOUT = 0.3
DILATIONS = [1, 2, 4, 8]

NUM_EPOCHS = _env_int("EWS_NUM_EPOCHS", 300)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
ES_PATIENCE = _env_int("EWS_ES_PATIENCE", 20)

K_RANGE = _env_int_list("EWS_K_RANGE", [2, 3, 4, 5, 6, 7, 8])
MIN_SPELL = 30


class ConnectednessDataset(Dataset):
    def __init__(self, data: np.ndarray, lookback: int):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.lookback = lookback

    def __len__(self):
        return len(self.data) - self.lookback + 1

    def __getitem__(self, idx):
        seq = self.data[idx: idx + self.lookback]
        return seq, seq


class CausalConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.3):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.pad1 = nn.ConstantPad1d((pad, 0), 0)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.pad2 = nn.ConstantPad1d((pad, 0), 0)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.proj(x)
        out = self.relu(self.norm1(self.conv1(self.pad1(x))))
        out = self.drop(out)
        out = self.relu(self.norm2(self.conv2(self.pad2(out))))
        out = self.drop(out)
        return self.relu(out + res)


class TCNAutoencoder(nn.Module):
    def __init__(self, n_features, lookback, num_filters, latent_dim, kernel_size, dilations, dropout):
        super().__init__()
        self.n_features = n_features
        self.lookback = lookback
        self.latent_dim = latent_dim
        self.num_filters = num_filters

        enc_layers = []
        in_ch = n_features
        for d in dilations:
            enc_layers.append(CausalConvBlock(in_ch, num_filters, kernel_size, d, dropout))
            in_ch = num_filters
        self.encoder_tcn = nn.Sequential(*enc_layers)
        self.enc_pool = nn.AdaptiveAvgPool1d(1)
        self.enc_fc = nn.Linear(num_filters, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, num_filters * lookback)
        dec_layers = []
        in_ch = num_filters
        for d in reversed(dilations):
            dec_layers.append(CausalConvBlock(in_ch, num_filters, kernel_size, d, dropout))
            in_ch = num_filters
        self.decoder_tcn = nn.Sequential(*dec_layers)
        self.dec_out = nn.Sequential(
            nn.Conv1d(num_filters, n_features, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        z = x.permute(0, 2, 1)
        z = self.encoder_tcn(z)
        z = self.enc_pool(z).squeeze(-1)
        return self.enc_fc(z)

    def decode(self, z):
        out = self.dec_fc(z).view(-1, self.num_filters, self.lookback)
        out = self.decoder_tcn(out)
        return self.dec_out(out).permute(0, 2, 1)

    def forward(self, x):
        return self.decode(self.encode(x))


def load_and_smooth(input_path: Path, smooth_window: int = SMOOTH_WINDOW) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df_raw = pd.read_csv(input_path, parse_dates=["Date"]).set_index("Date").sort_index()
    if df_raw.empty:
        raise ValueError("Input connectedness data is empty")

    df_smooth = df_raw.rolling(window=smooth_window, min_periods=smooth_window).mean().dropna()
    if df_smooth.empty:
        raise ValueError("Data is empty after rolling mean smoothing")

    return df_smooth


def prepare_dataloaders(df_smooth: pd.DataFrame):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_raw = df_smooth.values
    T = data_raw.shape[0]

    split_idx = int(T * 0.80)
    train_raw = data_raw[:split_idx]
    val_raw = data_raw[split_idx:]

    scaler.fit(train_raw)
    train_scaled = scaler.transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    data_scaled = scaler.transform(data_raw)

    train_dataset = ConnectednessDataset(train_scaled, LOOKBACK)
    val_dataset = ConnectednessDataset(val_scaled, LOOKBACK)
    full_dataset = ConnectednessDataset(data_scaled, LOOKBACK)

    if len(train_dataset) <= 0 or len(val_dataset) <= 0 or len(full_dataset) <= 0:
        raise ValueError("Insufficient samples after lookback windowing")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    full_loader = DataLoader(full_dataset, batch_size=128, shuffle=False)

    return scaler, train_loader, val_loader, full_loader, data_raw.shape[1]


def train_tcn_autoencoder(model: TCNAutoencoder, train_loader, val_loader, model_path: Path):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0

    device = next(model.parameters()).device

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n_train = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)
            n_train += x_batch.size(0)

        train_loss = running_loss / max(n_train, 1)

        model.eval()
        val_loss_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                val_loss_sum += criterion(model(x_val), y_val).item() * x_val.size(0)
                n_val += x_val.size(0)

        val_loss = val_loss_sum / max(n_val, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3} | train={train_loss:.6f} | val={val_loss:.6f} | best={best_val_loss:.6f}")

        if patience_counter >= ES_PATIENCE:
            print(f"Early stopping at epoch {epoch} (patience={ES_PATIENCE})")
            break

    model.load_state_dict(torch.load(model_path, map_location=device))
    return model, best_val_loss


def extract_latent_embeddings(model: TCNAutoencoder, full_loader, latent_dates: pd.DatetimeIndex) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    latent_list = []

    with torch.no_grad():
        for x_batch, _ in full_loader:
            z = model.encode(x_batch.to(device))
            latent_list.append(z.cpu().numpy())

    latent_embeddings = np.vstack(latent_list)
    if len(latent_embeddings) != len(latent_dates):
        raise ValueError("Latent embedding length mismatch with latent dates")

    return latent_embeddings


def compute_hmm_bic(embeddings: np.ndarray, n_states: int, n_iter: int = 2000, n_restarts: int = 5):
    T_len, d = embeddings.shape
    n_params = (n_states - 1) + n_states * (n_states - 1) + n_states * d + n_states * d * (d + 1) // 2

    best_loglik = -np.inf
    best_model = None

    for seed in range(n_restarts):
        try:
            hmm = GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                n_iter=n_iter,
                random_state=seed * 7,
                tol=1e-5,
            )
            hmm.fit(embeddings)
            ll = hmm.score(embeddings)
            if ll > best_loglik:
                best_loglik = ll
                best_model = hmm
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError(f"All HMM restarts failed for K={n_states}")

    bic = -2 * best_loglik + n_params * math.log(T_len)
    return bic, best_loglik, best_model


def check_persistence(model: GaussianHMM, embeddings: np.ndarray, min_spell: int = MIN_SPELL) -> bool:
    states = model.predict(embeddings)
    for k in range(model.n_components):
        mask = (states == k).astype(int)
        spells = []
        run = 0
        for v in mask:
            if v == 1:
                run += 1
            elif run > 0:
                spells.append(run)
                run = 0
        if run > 0:
            spells.append(run)
        if not spells or np.mean(spells) <= min_spell:
            return False
    return True


def select_hmm_model(latent_embeddings: np.ndarray):
    bic_scores = {}
    hmm_models = {}

    print(f"{'K':>4}  {'LogLik':>12}  {'BIC':>12}")
    print("-" * 34)
    for k in K_RANGE:
        bic, loglik, hmm_model = compute_hmm_bic(latent_embeddings, k)
        bic_scores[k] = bic
        hmm_models[k] = hmm_model
        print(f"{k:>4}  {loglik:>12.2f}  {bic:>12.2f}")

    sorted_ks = sorted(bic_scores, key=bic_scores.get)
    optimal_k = None

    print(f"\nPersistence guard (min_spell={MIN_SPELL}):")
    for k_candidate in sorted_ks:
        if check_persistence(hmm_models[k_candidate], latent_embeddings, MIN_SPELL):
            optimal_k = k_candidate
            print(f"K={optimal_k} accepted (BIC={bic_scores[optimal_k]:.2f})")
            break
        print(f"K={k_candidate} rejected")

    if optimal_k is None:
        optimal_k = sorted_ks[0]
        print(f"No candidate passed persistence; fallback K={optimal_k}")

    return optimal_k, hmm_models[optimal_k], bic_scores


def compute_spell_stats(states: np.ndarray, n_states: int) -> pd.DataFrame:
    rows = []
    for k in range(n_states):
        mask = (states == k).astype(int)
        spells = []
        run_len = 0
        for val in mask:
            if val == 1:
                run_len += 1
            elif run_len > 0:
                spells.append(run_len)
                run_len = 0
        if run_len > 0:
            spells.append(run_len)
        if spells:
            rows.append(
                {
                    "State": k,
                    "Total Days": int(mask.sum()),
                    "N Spells": len(spells),
                    "Mean Spell": round(np.mean(spells), 1),
                    "Median Spell": round(np.median(spells), 1),
                    "Max Spell": int(np.max(spells)),
                }
            )
    return pd.DataFrame(rows).set_index("State")


def run_regime_detection() -> pd.DataFrame:
    df_smooth = load_and_smooth(INPUT_PATH, SMOOTH_WINDOW)

    scaler, train_loader, val_loader, full_loader, n_features = prepare_dataloaders(df_smooth)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = TCNAutoencoder(
        n_features=n_features,
        lookback=LOOKBACK,
        num_filters=NUM_FILTERS,
        latent_dim=LATENT_DIM,
        kernel_size=KERNEL_SIZE,
        dilations=DILATIONS,
        dropout=DROPOUT,
    ).to(device)

    model, best_val_loss = train_tcn_autoencoder(model, train_loader, val_loader, MODEL_PATH)
    print(f"Best validation loss: {best_val_loss:.6f}")

    latent_dates = df_smooth.index[LOOKBACK - 1:]
    latent_embeddings = extract_latent_embeddings(model, full_loader, latent_dates)

    latent_df = pd.DataFrame(latent_embeddings, index=latent_dates)
    latent_df.index.name = "Date"
    latent_df.columns = [f"z{i+1}" for i in range(latent_embeddings.shape[1])]
    latent_df.to_csv(LATENT_PATH)

    optimal_k, best_hmm, bic_scores = select_hmm_model(latent_embeddings)

    viterbi_states = best_hmm.predict(latent_embeddings)
    regime_df = pd.DataFrame({"hmm_state": viterbi_states}, index=latent_dates)
    regime_df.index.name = "Date"

    spell_stats = compute_spell_stats(viterbi_states, optimal_k)
    spell_stats.to_csv(OUTPUT_DIR / "spell_stats.csv")

    trans_df = pd.DataFrame(
        best_hmm.transmat_,
        index=[f"State {k}" for k in range(optimal_k)],
        columns=[f"State {k}" for k in range(optimal_k)],
    )
    trans_df.to_csv(OUTPUT_DIR / "transition_matrix.csv")

    regime_df.to_csv(OUTPUT_PATH)

    print("\nSaved regime outputs:")
    print(f"- {OUTPUT_PATH}")
    print(f"- {LATENT_PATH}")
    print(f"- {OUTPUT_DIR / 'spell_stats.csv'}")
    print(f"- {OUTPUT_DIR / 'transition_matrix.csv'}")
    print(f"States: {sorted(regime_df['hmm_state'].unique())}")

    return regime_df


def main():
    run_regime_detection()


if __name__ == "__main__":
    main()
