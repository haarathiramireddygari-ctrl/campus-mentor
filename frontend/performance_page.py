import altair as alt
import pandas as pd
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BENCHMARKS = {
    "hybrid_development_100": "Development",
    "independent_holdout_100": "Synthetic holdout",
}
METRICS = {
    "Binary Accuracy": "Accuracy",
    "Fraud Precision": "Precision",
    "Fraud Recall": "Fraud recall",
    "Fraud F1": "Fraud F1",
    "False Positive Rate": "False alerts",
    "Triage Fraud Recall": "Triage recall",
    "Legitimate Triage Rate": "Legitimate triage",
    "Manual Review Rate": "Manual review",
    "Digital Arrest Alert Recall": "Digital arrest recall",
}


def show_performance_page():
    st.title("Model Performance Analytics")
    st.info("Unified V4 results on two synthetic diagnostic benchmarks, not independent real-world validation.")
    rows = []
    for name, label in BENCHMARKS.items():
        path = BASE_DIR / "reports" / "unified_v4" / name / "unified_metrics.csv"
        try:
            df = pd.read_csv(path)
            if len(df) != 1 or "Dataset" not in df.columns:
                raise ValueError("Expected one metrics row")
            row = df.iloc[0].to_dict()
            row["Benchmark"] = label
            rows.append(row)
        except (OSError, ValueError, pd.errors.ParserError):
            st.warning(f"Missing or invalid evaluation report: {name}")
    if rows:
        results = pd.DataFrame(rows).set_index("Benchmark")
        available = [m for m in METRICS if m in results.columns]
        if available:
            numeric = results[available].apply(pd.to_numeric, errors="coerce")
            numeric = numeric.where(numeric.ge(0) & numeric.le(100))
            table = numeric.transpose()
            formatted = table.apply(lambda col: col.map(lambda v: f"{v:.2f}%" if pd.notna(v) else "N/A"))
            st.dataframe(formatted)
            chart_data = numeric.rename(columns=METRICS).reset_index().melt(
                id_vars="Benchmark", var_name="Metric", value_name="Percent"
            ).dropna(subset=["Percent"])
            if not chart_data.empty:
                chart = alt.Chart(chart_data).mark_bar().encode(
                    y=alt.Y("Metric:N", sort=list(METRICS.values()), title=None,
                            axis=alt.Axis(labelLimit=180)),
                    yOffset=alt.YOffset("Benchmark:N", sort=list(BENCHMARKS.values())),
                    x=alt.X("Percent:Q", scale=alt.Scale(domain=[0, 100]), title="Percent (%)"),
                    color=alt.Color("Benchmark:N", legend=alt.Legend(title=None, orient="bottom")),
                    tooltip=["Metric:N", "Benchmark:N", alt.Tooltip("Percent:Q", format=".2f")],
                ).properties(height=max(220, len(available) * 48))
                st.altair_chart(chart, use_container_width=True)
        st.download_button("Download recorded Unified V4 metrics", results.reset_index().to_csv(index=False).encode("utf-8"),
                           file_name="unified_v4_metrics.csv", mime="text/csv")
    st.caption("Binary accuracy counts explicit fraud alerts; manual reviews are non-alerts. "
               "Triage recall counts alerts plus manual reviews. Category accuracy is not established by this evaluation.")
    st.write("142/142 automated tests passed after Unified V4 application integration (24 Nov 2026). "
             "This is a recorded result, not a live test run; rerun the suite after any changes.")
    st.warning("Legitimate messages may trigger unnecessary reviews. Fraud may be missed in real use. "
               "The evaluated datasets are synthetic and have influenced development.")
