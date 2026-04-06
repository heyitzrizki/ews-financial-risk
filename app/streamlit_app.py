from pathlib import Path
import subprocess
import sys

import altair as alt
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
PRED_DIR = BASE_DIR / "data" / "processed" / "predictive"
TVP_PATH = BASE_DIR / "data" / "processed" / "tvp_var" / "tvp_var_spillover_indices.csv"
MERGED_MARKET_PATH = BASE_DIR / "data" / "merged" / "market_close_2001_2026.csv"
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


def resolve_timeframe_range(choice: str, min_date: pd.Timestamp, max_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    presets = {
        "Full History": (min_date, max_date),
        "Global Financial Crisis (2007-2009)": (pd.Timestamp("2007-01-01"), pd.Timestamp("2009-12-31")),
        "Euro Debt Crisis (2010-2012)": (pd.Timestamp("2010-01-01"), pd.Timestamp("2012-12-31")),
        "COVID-19 Shock (2020-2021)": (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
        "Rate Hike Shock (2022-2023)": (pd.Timestamp("2022-01-01"), pd.Timestamp("2023-12-31")),
        "Last 1 Year": (max_date - pd.DateOffset(years=1), max_date),
        "Last 3 Years": (max_date - pd.DateOffset(years=3), max_date),
    }

    start_date, end_date = presets.get(choice, (min_date, max_date))
    start_date = max(start_date, min_date)
    end_date = min(end_date, max_date)
    return start_date, end_date


def render_market_features_page() -> None:
    st.title("Market Data Explorer")
    st.caption("Explore historical market features from Yahoo Finance and inspect periods like COVID-19 or other crises.")

    market_df = load_csv(MERGED_MARKET_PATH)
    if market_df.empty:
        st.warning("Merged Yahoo data was not found. Run the pipeline first.")
        return

    if "Date" not in market_df.columns:
        st.error("Merged market file is missing the 'Date' column.")
        return

    market_df = market_df.copy()
    market_df["Date"] = pd.to_datetime(market_df["Date"])
    market_df = market_df.sort_values("Date")

    feature_cols = [c for c in market_df.columns if c != "Date"]
    if not feature_cols:
        st.warning("No Yahoo feature columns found.")
        return

    min_date = market_df["Date"].min()
    max_date = market_df["Date"].max()

    c1, c2 = st.columns([1.2, 1.8])
    with c1:
        timeframe = st.selectbox(
            "Timeframe",
            [
                "Full History",
                "Global Financial Crisis (2007-2009)",
                "Euro Debt Crisis (2010-2012)",
                "COVID-19 Shock (2020-2021)",
                "Rate Hike Shock (2022-2023)",
                "Last 1 Year",
                "Last 3 Years",
                "Custom Range",
            ],
            index=0,
            key="feature_timeframe",
        )

    if timeframe == "Custom Range":
        with c2:
            c_start, c_end = st.columns(2)
            with c_start:
                custom_start = st.date_input(
                    "Start date",
                    value=min_date.date(),
                    min_value=min_date.date(),
                    max_value=max_date.date(),
                    key="feature_start_date",
                )
            with c_end:
                custom_end = st.date_input(
                    "End date",
                    value=max_date.date(),
                    min_value=min_date.date(),
                    max_value=max_date.date(),
                    key="feature_end_date",
                )

        start_date = pd.Timestamp(custom_start)
        end_date = pd.Timestamp(custom_end)
        if start_date > end_date:
            start_date, end_date = end_date, start_date
    else:
        start_date, end_date = resolve_timeframe_range(timeframe, min_date, max_date)

    filtered = market_df[(market_df["Date"] >= start_date) & (market_df["Date"] <= end_date)].copy()
    if filtered.empty:
        st.warning("No data is available for the selected timeframe.")
        return

    selected_features = st.multiselect(
        "Select features",
        options=feature_cols,
        default=feature_cols,
        key="feature_selector",
    )
    if not selected_features:
        st.info("Select at least one feature to display charts.")
        return

    normalize = st.checkbox("Normalize each feature to 0-1 scale", value=False, key="feature_normalize")

    plot_df = filtered[["Date"] + selected_features].copy()
    if normalize:
        for col in selected_features:
            col_min = plot_df[col].min()
            col_max = plot_df[col].max()
            if pd.notna(col_min) and pd.notna(col_max) and col_max > col_min:
                plot_df[col] = (plot_df[col] - col_min) / (col_max - col_min)

    long_df = plot_df.melt(id_vars="Date", var_name="Feature", value_name="Value")
    long_df = long_df.dropna(subset=["Value"])

    st.caption(
        f"Showing {len(selected_features)} feature(s) from {plot_df['Date'].min().date()} to {plot_df['Date'].max().date()} "
        f"({len(plot_df)} rows)."
    )

    faceted_chart = (
        alt.Chart(long_df)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Value:Q", title="Value"),
            tooltip=["Date:T", "Feature:N", alt.Tooltip("Value:Q", format=".4f")],
        )
        .properties(height=130)
        .facet(row=alt.Row("Feature:N", title=None))
        .resolve_scale(y="independent")
    )
    st.altair_chart(faceted_chart, use_container_width=True)

    stats_df = filtered[selected_features].describe().T.reset_index().rename(columns={"index": "Feature"})
    st.markdown("**Feature Summary Statistics**")
    st.dataframe(stats_df, use_container_width=True, hide_index=True)


def render_ews_dashboard() -> None:
    st.title("Early Warning Dashboard for Market Stress")
    st.caption("Simple decision language: what is the current risk level, and which markets are sending pressure.")


def main() -> None:
    st.set_page_config(page_title="EWS Risk Dashboard", layout="wide")

    with st.sidebar:
        page = st.radio(
            "Page",
            ["EWS Dashboard", "Market Data Explorer"],
            index=0,
        )

        st.markdown("---")
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

    if page == "Market Data Explorer":
        render_market_features_page()
        return

    render_ews_dashboard()

    latest_signal = load_csv(PRED_DIR / "latest_signal.csv")
    signal_history = load_csv(PRED_DIR / "latest_signal_history.csv")
    summary_metrics = load_csv(PRED_DIR / "summary_metrics.csv")
    top_models = load_csv(PRED_DIR / "top_model_selection.csv")
    tvp_df = load_csv(TVP_PATH)

    if latest_signal.empty:
        st.warning("No signal file found yet. Click 'Refresh Full Pipeline' in the sidebar.")
        return

    history_filtered = pd.DataFrame()
    selected_timeframe = "Full History"
    custom_start = None
    custom_end = None
    if not signal_history.empty:
        signal_history = signal_history.copy()
        signal_history["Date"] = pd.to_datetime(signal_history["Date"])
        signal_history = signal_history.sort_values("Date")

        min_hist_date = signal_history["Date"].min()
        max_hist_date = signal_history["Date"].max()

        st.subheader("Historical Signal Explorer")
        tf_col1, tf_col2 = st.columns([1.2, 1])
        with tf_col1:
            selected_timeframe = st.selectbox(
                "Select timeframe",
                [
                    "Full History",
                    "Global Financial Crisis (2007-2009)",
                    "Euro Debt Crisis (2010-2012)",
                    "COVID-19 Shock (2020-2021)",
                    "Rate Hike Shock (2022-2023)",
                    "Last 1 Year",
                    "Last 3 Years",
                    "Custom Range",
                ],
                index=0,
            )

        if selected_timeframe == "Custom Range":
            with tf_col2:
                c_start, c_end = st.columns(2)
                with c_start:
                    custom_start = st.date_input(
                        "Start date",
                        value=min_hist_date.date(),
                        min_value=min_hist_date.date(),
                        max_value=max_hist_date.date(),
                    )
                with c_end:
                    custom_end = st.date_input(
                        "End date",
                        value=max_hist_date.date(),
                        min_value=min_hist_date.date(),
                        max_value=max_hist_date.date(),
                    )

            start_date = pd.Timestamp(custom_start)
            end_date = pd.Timestamp(custom_end)
            if start_date > end_date:
                start_date, end_date = end_date, start_date
        else:
            start_date, end_date = resolve_timeframe_range(selected_timeframe, min_hist_date, max_hist_date)

        history_filtered = signal_history[
            (signal_history["Date"] >= start_date) & (signal_history["Date"] <= end_date)
        ].copy()

        if history_filtered.empty:
            st.warning("No records are available for the selected timeframe.")
        else:
            period_days = (history_filtered["Date"].max() - history_filtered["Date"].min()).days
            alerts_count = int((history_filtered["y_pred"] == 1).sum())
            all_count = int(len(history_filtered))
            st.caption(
                f"Showing {all_count} records from {history_filtered['Date'].min().date()} to {history_filtered['Date'].max().date()} "
                f"({period_days} days), with {alerts_count} alert days."
            )

    latest = latest_signal.iloc[-1]
    latest_date = pd.to_datetime(latest["Date"])
    hybrid_prob = float(latest["y_prob"])
    alert = int(latest["y_pred"])
    horizon = int(latest["horizon"])
    ml_model = str(latest.get("ml_model", "-"))
    dl_model = str(latest.get("dl_model", "-"))
    crisis_state_count = int(latest.get("crisis_state_count", 1))
    crisis_states = str(latest.get("crisis_states", str(latest.get("global_crisis_state", "-"))))

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

    st.caption(
        f"Crisis-event rule: top {crisis_state_count} regime(s) by connectedness level (states: {crisis_states})."
    )

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
        if history_filtered.empty:
            st.info("No historical signal is available.")
        else:
            hist = history_filtered.sort_values("Date").copy()
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
    if history_filtered.empty:
        st.info("No signal history found.")
    else:
        records = history_filtered.copy().sort_values("Date", ascending=False)
        records["Status"] = records["y_pred"].map({1: "Alert", 0: "Stable"})
        records["Risk Probability (%)"] = (records["y_prob"] * 100).round(2)
        records["Date"] = records["Date"].dt.strftime("%Y-%m-%d")

        view_cols = ["Date", "Status", "Risk Probability (%)", "horizon", "ml_model", "dl_model", "crisis_states"]
        rename = {
            "horizon": "Window (days)",
            "ml_model": "Top ML",
            "dl_model": "Top DL",
            "crisis_states": "Crisis States",
        }
        st.dataframe(
            records[view_cols].rename(columns=rename, errors="ignore"),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
