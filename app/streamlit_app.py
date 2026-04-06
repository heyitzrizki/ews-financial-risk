from pathlib import Path
import subprocess
import sys

import altair as alt
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
PRED_DIR = BASE_DIR / "data" / "processed" / "predictive"
TVP_PATH = BASE_DIR / "data" / "processed" / "tvp_var" / "tvp_var_spillover_indices.csv"
INFERENCE_PATH = BASE_DIR / "src" / "inference.py"


def risk_label(prob: float) -> str:
	if prob < 0.30:
		return "Rendah"
	if prob < 0.60:
		return "Sedang"
	return "Tinggi"


@st.cache_data(ttl=120)
def load_csv(path: Path) -> pd.DataFrame:
	if not path.exists():
		return pd.DataFrame()
	return pd.read_csv(path)


def run_pipeline() -> tuple[bool, str]:
	if not INFERENCE_PATH.exists():
		return False, "File inference.py tidak ditemukan."

	cmd = [sys.executable, str(INFERENCE_PATH)]
	result = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True)
	output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
	return result.returncode == 0, output.strip()


def build_transmitter_table(df_tvp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
	if df_tvp.empty:
		return pd.DataFrame(), pd.DataFrame(), "-"

	df_tvp["Date"] = pd.to_datetime(df_tvp["Date"])
	df_tvp = df_tvp.sort_values("Date")
	latest = df_tvp.iloc[-1]

	assets = ["JCI", "SP500", "SSE", "VIX", "WTI", "Gold", "USDIDR"]
	rows = []
	for asset in assets:
		to_col = f"TO_{asset}"
		net_col = f"NET_{asset}"
		to_val = float(latest[to_col]) if to_col in latest.index else 0.0
		net_val = float(latest[net_col]) if net_col in latest.index else 0.0
		role = "Penyebar Tekanan" if net_val > 0 else "Penerima Tekanan"
		rows.append(
			{
				"Pasar": asset,
				"Kontribusi Tekanan": to_val,
				"Arah Bersih": net_val,
				"Peran": role,
			}
		)

	latest_table = pd.DataFrame(rows).sort_values("Kontribusi Tekanan", ascending=False)
	top_three = latest_table.head(3).copy()
	latest_date = latest["Date"].strftime("%Y-%m-%d")
	return latest_table, top_three, latest_date


def main() -> None:
	st.set_page_config(page_title="EWS Financial Risk", layout="wide")
	st.title("Dashboard Peringatan Dini Risiko Pasar")
	st.caption("Bahasa ringkas untuk monitoring risiko harian tanpa istilah teknis.")

	with st.sidebar:
		st.header("Kontrol")
		if st.button("Perbarui Data Sekarang", use_container_width=True):
			with st.spinner("Sedang memperbarui data dan sinyal..."):
				ok, log_text = run_pipeline()
			if ok:
				st.success("Data berhasil diperbarui.")
				st.cache_data.clear()
			else:
				st.error("Proses pembaruan gagal. Cek detail log di bawah.")
			with st.expander("Lihat detail proses"):
				st.text(log_text if log_text else "Tidak ada log.")

		if st.button("Muat Ulang Dashboard", use_container_width=True):
			st.cache_data.clear()
			st.rerun()

		st.info("Tips: tekan 'Perbarui Data Sekarang' sebelum presentasi agar angka terbaru muncul.")

	latest_signal = load_csv(PRED_DIR / "latest_signal.csv")
	signal_history = load_csv(PRED_DIR / "latest_signal_history.csv")
	summary_metrics = load_csv(PRED_DIR / "summary_metrics.csv")
	tvp_df = load_csv(TVP_PATH)

	if latest_signal.empty:
		st.warning("Sinyal belum tersedia. Jalankan pipeline dulu dari sidebar.")
		return

	latest = latest_signal.iloc[-1]
	latest_date = pd.to_datetime(latest["Date"])
	risk_prob = float(latest["y_prob"])
	alert = int(latest["y_pred"])
	horizon = int(latest["horizon"])
	method = str(latest["model"])

	status_text = "Waspada" if alert == 1 else "Stabil"
	status_color = "#ef4444" if alert == 1 else "#22c55e"

	st.markdown(
		f"""
		<div style="padding:14px;border-radius:10px;background:{status_color}22;border:1px solid {status_color};">
			<h3 style="margin:0;color:{status_color};">Status Hari Ini: {status_text}</h3>
			<p style="margin:6px 0 0 0;">Interpretasi singkat: tingkat risiko saat ini berada pada kategori <b>{risk_label(risk_prob)}</b>.</p>
		</div>
		""",
		unsafe_allow_html=True,
	)

	c1, c2, c3, c4 = st.columns(4)
	c1.metric("Peluang Risiko", f"{risk_prob:.1%}")
	c2.metric("Kategori Risiko", risk_label(risk_prob))
	c3.metric("Tanggal Data", latest_date.strftime("%Y-%m-%d"))
	c4.metric("Jangka Waktu", f"{horizon} hari")

	left, right = st.columns([1.35, 1])

	with left:
		st.subheader("Pergerakan Sinyal Risiko")
		if signal_history.empty:
			st.info("Riwayat sinyal belum tersedia.")
		else:
			signal_history["Date"] = pd.to_datetime(signal_history["Date"])
			recent = signal_history.sort_values("Date").tail(250).copy()
			recent["Kategori"] = recent["y_prob"].apply(risk_label)

			chart = (
				alt.Chart(recent)
				.mark_line(point=False)
				.encode(
					x=alt.X("Date:T", title="Tanggal"),
					y=alt.Y("y_prob:Q", title="Peluang Risiko", scale=alt.Scale(domain=[0, 1])),
					tooltip=["Date:T", alt.Tooltip("y_prob:Q", format=".3f"), "Kategori:N"],
				)
				.properties(height=300)
			)

			threshold_line = (
				alt.Chart(pd.DataFrame({"threshold": [float(latest["threshold"])]}))
				.mark_rule(strokeDash=[5, 5], color="#f59e0b")
				.encode(y="threshold:Q")
			)

			st.altair_chart(chart + threshold_line, use_container_width=True)

	with right:
		st.subheader("Ringkasan Metode (Bahasa Sederhana)")
		if summary_metrics.empty:
			st.info("Ringkasan metode belum tersedia.")
		else:
			friendly = summary_metrics.copy()
			friendly["Skor Deteksi Dini"] = (friendly["PR_AUC_mean"] * 100).round(1)
			friendly["Ketepatan Alarm"] = (friendly["Precision_mean"] * 100).round(1)
			friendly["Jangkauan Alarm"] = (friendly["Recall_mean"] * 100).round(1)
			friendly["Metode"] = friendly["model"]
			friendly["Jangka Waktu (hari)"] = friendly["horizon"].astype(int)

			display_cols = [
				"Metode",
				"Jangka Waktu (hari)",
				"Skor Deteksi Dini",
				"Ketepatan Alarm",
				"Jangkauan Alarm",
			]
			st.dataframe(
				friendly[display_cols]
				.sort_values(["Skor Deteksi Dini", "Ketepatan Alarm"], ascending=False)
				.reset_index(drop=True),
				use_container_width=True,
				hide_index=True,
			)

			st.caption(f"Metode aktif saat ini: {method} (untuk horizon {horizon} hari).")

	st.subheader("Pasar Penyebar Tekanan (Transmitter)")
	latest_table, top_three, tx_date = build_transmitter_table(tvp_df)
	if latest_table.empty:
		st.info("Data penyebar tekanan belum tersedia.")
	else:
		st.caption(f"Berdasarkan data terbaru tanggal {tx_date}.")
		t1, t2 = st.columns([1, 1.2])
		with t1:
			st.markdown("**Tiga Penyebar Tekanan Teratas**")
			st.dataframe(top_three, use_container_width=True, hide_index=True)
		with t2:
			chart_df = latest_table.copy()
			bar = (
				alt.Chart(chart_df)
				.mark_bar()
				.encode(
					x=alt.X("Pasar:N", sort="-y"),
					y=alt.Y("Arah Bersih:Q", title="Arah Bersih Tekanan"),
					color=alt.condition(
						alt.datum["Arah Bersih"] > 0,
						alt.value("#16a34a"),
						alt.value("#dc2626"),
					),
					tooltip=["Pasar:N", "Kontribusi Tekanan:Q", "Arah Bersih:Q", "Peran:N"],
				)
				.properties(height=280)
			)
			st.altair_chart(bar, use_container_width=True)

	st.subheader("Riwayat Sinyal Terbaru")
	if signal_history.empty:
		st.info("Belum ada riwayat sinyal.")
	else:
		hist = signal_history.copy()
		hist["Date"] = pd.to_datetime(hist["Date"])
		hist = hist.sort_values("Date", ascending=False).head(30)
		hist["Status"] = hist["y_pred"].map({1: "Waspada", 0: "Stabil"})
		hist["Peluang Risiko"] = (hist["y_prob"] * 100).round(2)
		hist["Tanggal"] = hist["Date"].dt.strftime("%Y-%m-%d")
		hist["Jangka Waktu"] = hist["horizon"].astype(int).astype(str) + " hari"
		hist["Metode"] = hist["model"]

		st.dataframe(
			hist[["Tanggal", "Status", "Peluang Risiko", "Jangka Waktu", "Metode"]],
			use_container_width=True,
			hide_index=True,
		)


if __name__ == "__main__":
	main()

