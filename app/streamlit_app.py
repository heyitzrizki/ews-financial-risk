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


def risk_level(prob: float) -> str:
    if prob < 0.30:
        return "Low"
    if prob < 0.60:
        return "Moderate"
    return "High"


def alert_label(flag: int) -> str:
    return "Alert" if int(flag) == 1 else "Stable"


@st.cache_data(ttl=120)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def run_pipeline() -> tuple[bool, str]:
    if not INFERENCE_PATH.exists():
        return False, "inference.py was not found."

    cmd = [sys.executable, str(INFERENCE_PATH)]
    result = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True)
    output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    return result.returncode == 0, output.strip()


def build_transmitter_table(df_tvp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if df_tvp.empty:
        return pd.DataFrame(), pd.DataFrame(), "-"

    df_tvp = df_tvp.copy()
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
        role = "Pressure Sender" if net_val > 0 else "Pressure Receiver"

        rows.append(
            {
                "Market": asset,
                "Pressure Contribution": to_val,
                "Net Direction": net_val,
                "Role": role,
            }
        )

    latest_table = pd.DataFrame(rows).sort_values("Pressure Contribution", ascending=False)
    top_three = latest_table.head(3).copy()
    latest_date = latest["Date"].strftime("%Y-%m-%d")
    return latest_table, top_three, latest_date


def main() -> None:
    st.set_page_config(page_title="EWS Risk Dashboard", layout="wide")
    st.title("Early Warning Dashboard for Market Stress")
    st.caption("Simple decision language: what is the current risk level, and which markets are sending pressure.")

    with st.sidebar:
        st.header("Actions")
        if st.button("Refresh Full Pipeline", use_container_width=True):
            with st.spinner("Updating data and signals..."):
                ok, log_text = run_pipeline()
            if ok:
                st.success("Pipeline finished successfully.")
                st.cache_data.clear()
            else:
                st.error("Pipeline failed. Check details below.")
            with st.expander("Execution Log"):
                st.text(log_text if log_text else "No log output.")

        if st.button("Reload Dashboard", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.info("Tip: Refresh before meetings so the dashboard shows the latest signal.")

    latest_signal = load_csv(PRED_DIR / "latest_signal.csv")
    signal_history = load_csv(PRED_DIR / "latest_signal_history.csv")
    summary_metrics = load_csv(PRED_DIR / "summary_metrics.csv")
    top_models = load_csv(PRED_DIR / "top_model_selection.csv")
    tvp_df = load_csv(TVP_PATH)

    if latest_signal.empty:
        st.warning("No signal file found yet. Click 'Refresh Full Pipeline' in the sidebar.")
        return

    latest = latest_signal.iloc[-1]
    latest_date = pd.to_datetime(latest["Date"])
    hybrid_prob = float(latest["y_prob"])
    alert = int(latest["y_pred"])
    horizon = int(latest["horizon"])
    ml_model = str(latest.get("ml_model", "-"))
    dl_model = str(latest.get("dl_model", "-"))

    status = alert_label(alert)
    level = risk_level(hybrid_prob)
    status_color = "#dc2626" if alert == 1 else "#16a34a"

    st.markdown(
        f"""
        <div style="padding:14px;border-radius:10px;background:{status_color}18;border:1px solid {status_color};">
            <h3 style="margin:0;color:{status_color};">Current Status: {status}</h3>
            <p style="margin:6px 0 0 0;">Current risk level is <b>{level}</b>, based on the hybrid signal from your top ML and top DL models.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Hybrid Risk Probability", f"{hybrid_prob:.1%}")
    k2.metric("Risk Level", level)
    k3.metric("Data Date", latest_date.strftime("%Y-%m-%d"))
    k4.metric("Prediction Window", f"{horizon} days")

    m1, m2 = st.columns(2)
    with m1:
        st.markdown("**Top Classical Model (ML)**")
        st.write(ml_model)
    with m2:
        st.markdown("**Top Deep Sequence Model (DL)**")
        st.write(dl_model)

    left, right = st.columns([1.4, 1])

    with left:
        st.subheader("Risk Signal Trend")
        if signal_history.empty:
            st.info("No historical signal is available.")
        else:
            hist = signal_history.copy()
            hist["Date"] = pd.to_datetime(hist["Date"])
            hist = hist.sort_values("Date").tail(300)
            hist["Risk Level"] = hist["y_prob"].apply(risk_level)

            line = (
                alt.Chart(hist)
                .mark_line()
                .encode(
                    x=alt.X("Date:T", title="Date"),
                    y=alt.Y("y_prob:Q", title="Risk Probability", scale=alt.Scale(domain=[0, 1])),
                    tooltip=["Date:T", alt.Tooltip("y_prob:Q", format=".3f"), "Risk Level:N"],
                )
                .properties(height=320)
            )

            threshold_line = (
                alt.Chart(pd.DataFrame({"threshold": [float(latest["threshold"])]}))
                .mark_rule(strokeDash=[6, 5], color="#f59e0b")
                .encode(y="threshold:Q")
            )

            st.altair_chart(line + threshold_line, use_container_width=True)

            if {"ml_prob", "dl_prob"}.issubset(hist.columns):
                source = hist[["Date", "ml_prob", "dl_prob"]].melt(
                    id_vars="Date",
                    value_vars=["ml_prob", "dl_prob"],
                    var_name="Signal Component",
                    value_name="Probability",
                )
                component_chart = (
                    alt.Chart(source)
                    .mark_line(strokeDash=[4, 2])
                    .encode(
                        x="Date:T",
                        y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1])),
                        color="Signal Component:N",
                        tooltip=["Date:T", "Signal Component:N", alt.Tooltip("Probability:Q", format=".3f")],
                    )
                    .properties(height=180)
                )
                st.caption("Component view: contribution from ML and DL top performers.")
                st.altair_chart(component_chart, use_container_width=True)

    with right:
        st.subheader("Model Performance Snapshot")
        if summary_metrics.empty:
            st.info("Performance summary is not available.")
        else:
            perf = summary_metrics.copy()
            perf["Early Detection Score"] = (perf["PR_AUC_mean"] * 100).round(1)
            perf["Alarm Precision"] = (perf["Precision_mean"] * 100).round(1)
            perf["Crisis Capture"] = (perf["Recall_mean"] * 100).round(1)
            perf["Model"] = perf["model"]
            perf["Model Family"] = perf["family"]
            perf["Window (days)"] = perf["horizon"].astype(int)

            show_cols = [
                "Model",
                "Model Family",
                "Window (days)",
                "Early Detection Score",
                "Alarm Precision",
                "Crisis Capture",
            ]
            st.dataframe(
                perf[show_cols].sort_values(
                    by=["Early Detection Score", "Alarm Precision"],
                    ascending=[False, False],
                ),
                use_container_width=True,
                hide_index=True,
            )

        if not top_models.empty:
            st.markdown("**Best ML + DL Pair by Horizon**")
            show = top_models.copy()
            show["hybrid_score"] = (show["hybrid_score"] * 100).round(1)
            show = show.rename(
                columns={
                    "horizon": "Window (days)",
                    "ml_model": "Top ML",
                    "dl_model": "Top DL",
                    "hybrid_score": "Combined Score",
                }
            )
            st.dataframe(
                show[["Window (days)", "Top ML", "Top DL", "Combined Score"]],
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("Pressure Sender Markets (Transmitter)")
    latest_table, top_three, tx_date = build_transmitter_table(tvp_df)
    if latest_table.empty:
        st.info("Transmitter data is not available.")
    else:
        st.caption(f"Based on the latest connectedness snapshot ({tx_date}).")
        t1, t2 = st.columns([1, 1.2])
        with t1:
            st.markdown("**Top 3 Pressure Senders**")
            st.dataframe(top_three, use_container_width=True, hide_index=True)
        with t2:
            bar = (
                alt.Chart(latest_table)
                .mark_bar()
                .encode(
                    x=alt.X("Market:N", sort="-y"),
                    y=alt.Y("Net Direction:Q", title="Net Pressure Direction"),
                    color=alt.condition(
                        alt.datum["Net Direction"] > 0,
                        alt.value("#16a34a"),
                        alt.value("#dc2626"),
                    ),
                    tooltip=["Market:N", "Pressure Contribution:Q", "Net Direction:Q", "Role:N"],
                )
                .properties(height=280)
            )
            st.altair_chart(bar, use_container_width=True)

    st.subheader("Latest Signal Records")
    if signal_history.empty:
        st.info("No signal history found.")
    else:
        records = signal_history.copy()
        records["Date"] = pd.to_datetime(records["Date"])
        records = records.sort_values("Date", ascending=False).head(30)
        records["Status"] = records["y_pred"].map({1: "Alert", 0: "Stable"})
        records["Risk Probability (%)"] = (records["y_prob"] * 100).round(2)
        records["Date"] = records["Date"].dt.strftime("%Y-%m-%d")

        view_cols = ["Date", "Status", "Risk Probability (%)", "horizon", "ml_model", "dl_model"]
        rename = {
            "horizon": "Window (days)",
            "ml_model": "Top ML",
            "dl_model": "Top DL",
        }
        st.dataframe(
            records[view_cols].rename(columns=rename, errors="ignore"),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
