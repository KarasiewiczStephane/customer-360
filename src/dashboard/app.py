"""Streamlit dashboard for Customer 360 analytics.

Provides customer search, 360-degree profile view, segment overview,
CLV analysis, cohort retention heatmap, and data export.
"""

import json
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="Customer 360", page_icon="👥", layout="wide")


@st.cache_resource
def _get_connection() -> duckdb.DuckDBPyConnection:
    config = load_config()
    db_path = config["database"]["path"]
    if not Path(db_path).exists():
        st.error(f"Database not found at {db_path}. Run the pipeline first.")
        st.stop()
    return duckdb.connect(db_path, read_only=True)


def _safe_query(
    conn: duckdb.DuckDBPyConnection, sql: str, params: list | None = None
) -> pd.DataFrame:
    """Execute a query and return a DataFrame, returning empty on error."""
    try:
        if params:
            return conn.execute(sql, params).df()
        return conn.execute(sql).df()
    except Exception as exc:
        logger.warning("Query failed: %s", exc)
        return pd.DataFrame()


# ── Customer Search & Profile ────────────────────────────────────────


def render_customer_search(conn: duckdb.DuckDBPyConnection) -> None:
    """Render the customer search and profile view page."""
    st.header("Customer Search")
    query = st.text_input("Search by name, email, or ID", placeholder="Enter search term…")

    if not query:
        return

    pattern = f"%{query}%"
    results = _safe_query(
        conn,
        """
        SELECT
            g.unified_id, g.name, g.email,
            r.rfm_score, r.segment,
            c.predicted_clv, c.clv_tier
        FROM golden_records g
        LEFT JOIN rfm_scores r ON g.unified_id = r.customer_id
        LEFT JOIN clv_predictions c ON g.unified_id = c.customer_id
        WHERE LOWER(g.name) LIKE LOWER(?)
           OR LOWER(g.email) LIKE LOWER(?)
           OR g.unified_id LIKE ?
        LIMIT 50
        """,
        [pattern, pattern, pattern],
    )

    if results.empty:
        st.warning("No customers found.")
        return

    st.dataframe(results, use_container_width=True)

    selected = st.selectbox("Select customer for profile", results["unified_id"].tolist())
    if selected:
        _render_profile(conn, selected)


def _render_profile(conn: duckdb.DuckDBPyConnection, unified_id: str) -> None:
    """Render the 360-degree customer profile."""
    basic = _safe_query(conn, "SELECT * FROM golden_records WHERE unified_id = ?", [unified_id])
    if basic.empty:
        st.error("Customer not found")
        return

    info = basic.iloc[0]
    st.subheader(f"Profile: {info.get('name', 'N/A')}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ID", unified_id)
    col2.metric("Email", str(info.get("email", "N/A")))

    rfm = _safe_query(conn, "SELECT * FROM rfm_scores WHERE customer_id = ?", [unified_id])
    if not rfm.empty:
        col3.metric("Segment", rfm.iloc[0].get("segment", "N/A"))

    clv = _safe_query(conn, "SELECT * FROM clv_predictions WHERE customer_id = ?", [unified_id])
    if not clv.empty:
        col4.metric("CLV", f"${clv.iloc[0].get('predicted_clv', 0):,.2f}")

    # RFM scores
    if not rfm.empty:
        st.markdown("**RFM Scores**")
        rc1, rc2, rc3, rc4 = st.columns(4)
        r = rfm.iloc[0]
        rc1.metric("Recency", r.get("r_score", "-"))
        rc2.metric("Frequency", r.get("f_score", "-"))
        rc3.metric("Monetary", r.get("m_score", "-"))
        rc4.metric("RFM", r.get("rfm_score", "-"))

    # Transaction history
    st.markdown("**Recent Transactions**")
    source_ids = info.get("source_ids", "")
    id_list = [sid.strip() for sid in str(source_ids).split(",") if sid.strip()]
    if id_list:
        placeholders = ",".join(["?" for _ in id_list])
        txn = _safe_query(
            conn,
            f"SELECT date, amount, product_category FROM transactions WHERE customer_id IN ({placeholders}) ORDER BY date DESC LIMIT 20",
            id_list,
        )
        if not txn.empty:
            st.dataframe(txn, use_container_width=True)
        else:
            st.info("No transactions found.")

    # Web & support
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Web Activity**")
        if id_list:
            web = _safe_query(
                conn,
                f"SELECT timestamp, pages_visited, time_on_site, referrer FROM web_sessions WHERE customer_id IN ({placeholders}) ORDER BY timestamp DESC LIMIT 10",
                id_list,
            )
            if not web.empty:
                st.dataframe(web, use_container_width=True)
            else:
                st.info("No web sessions.")
    with c2:
        st.markdown("**Support History**")
        if id_list:
            support = _safe_query(
                conn,
                f"SELECT created_at, category, satisfaction_score, status FROM support_tickets WHERE customer_id IN ({placeholders}) ORDER BY created_at DESC LIMIT 10",
                id_list,
            )
            if not support.empty:
                st.dataframe(support, use_container_width=True)
            else:
                st.info("No support tickets.")


# ── Segment Overview ─────────────────────────────────────────────────


def render_segment_overview(conn: duckdb.DuckDBPyConnection) -> None:
    """Render segment distribution and metrics."""
    st.header("Segment Overview")

    seg = _safe_query(
        conn,
        """
        SELECT segment,
               COUNT(*) AS customer_count,
               AVG(monetary) AS avg_monetary,
               AVG(frequency) AS avg_frequency,
               AVG(recency) AS avg_recency
        FROM rfm_scores GROUP BY segment ORDER BY avg_monetary DESC
        """,
    )
    if seg.empty:
        st.warning("No segment data available.")
        return

    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(
            seg, values="customer_count", names="segment", title="Customer Distribution by Segment"
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        seg["total_revenue"] = seg["customer_count"] * seg["avg_monetary"]
        fig = px.bar(
            seg, x="segment", y="total_revenue", title="Revenue by Segment", color="segment"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment Metrics")
    st.dataframe(seg.round(2), use_container_width=True)


# ── CLV Analysis ─────────────────────────────────────────────────────


def render_clv_analysis(conn: duckdb.DuckDBPyConnection) -> None:
    """Render CLV distribution and tier analysis."""
    st.header("CLV Analysis")

    clv = _safe_query(conn, "SELECT predicted_clv, clv_tier FROM clv_predictions")
    if clv.empty:
        st.warning("No CLV data available.")
        return

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(
            clv, x="predicted_clv", nbins=50, title="CLV Distribution", color="clv_tier"
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        tier = (
            clv.groupby("clv_tier")
            .agg(
                count=("predicted_clv", "count"),
                avg=("predicted_clv", "mean"),
                total=("predicted_clv", "sum"),
            )
            .round(2)
            .reset_index()
        )
        fig = px.bar(tier, x="clv_tier", y="total", title="Total CLV by Tier", color="clv_tier")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tier Metrics")
    st.dataframe(
        clv.groupby("clv_tier")["predicted_clv"].describe().round(2), use_container_width=True
    )


# ── Cohort Analysis ──────────────────────────────────────────────────


def render_cohort_analysis(conn: duckdb.DuckDBPyConnection) -> None:
    """Render cohort retention heatmap."""
    st.header("Cohort Analysis")

    ret = _safe_query(conn, "SELECT * FROM cohort_retention")
    if ret.empty:
        st.warning("No cohort data available.")
        return

    matrix = ret.pivot(index="cohort_month", columns="cohort_age", values="retention_rate")
    fig = px.imshow(
        matrix,
        labels={"x": "Months Since Signup", "y": "Cohort Month", "color": "Retention %"},
        title="Customer Retention by Cohort",
        color_continuous_scale="RdYlGn",
        aspect="auto",
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Retention Curve")
    avg = matrix.mean(axis=0).reset_index()
    avg.columns = ["Month", "Retention Rate"]
    fig = px.line(
        avg, x="Month", y="Retention Rate", title="Average Retention Across Cohorts", markers=True
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Entity Resolution Stats ─────────────────────────────────────────


def render_entity_resolution(conn: duckdb.DuckDBPyConnection) -> None:
    """Render entity resolution quality metrics."""
    st.header("Entity Resolution Statistics")

    total = _safe_query(conn, "SELECT COUNT(*) AS n FROM crm_customers")
    golden = _safe_query(conn, "SELECT COUNT(*) AS n FROM golden_records")

    if total.empty or golden.empty:
        st.warning("Resolution data unavailable.")
        return

    total_n = int(total.iloc[0]["n"])
    golden_n = int(golden.iloc[0]["n"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original Records", f"{total_n:,}")
    c2.metric("Unified Records", f"{golden_n:,}")
    c3.metric("Dedup Rate", f"{(1 - golden_n / total_n) * 100:.1f}%" if total_n else "0%")
    c4.metric("Avg Cluster Size", f"{total_n / golden_n:.2f}" if golden_n else "N/A")


# ── Export ────────────────────────────────────────────────────────────


def render_export(conn: duckdb.DuckDBPyConnection) -> None:
    """Render export UI for segments and profiles."""
    st.header("Export Data")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Export Segment List")
        segments = _safe_query(conn, "SELECT DISTINCT segment FROM rfm_scores ORDER BY segment")
        if not segments.empty:
            seg = st.selectbox("Select Segment", segments["segment"].tolist())
            if st.button("Generate CSV"):
                data = _safe_query(
                    conn,
                    """
                    SELECT g.unified_id, g.name, g.email,
                           r.rfm_score, r.segment, r.monetary,
                           c.predicted_clv, c.clv_tier
                    FROM golden_records g
                    JOIN rfm_scores r ON g.unified_id = r.customer_id
                    LEFT JOIN clv_predictions c ON g.unified_id = c.customer_id
                    WHERE r.segment = ?
                    """,
                    [seg],
                )
                csv_bytes = data.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    csv_bytes,
                    f"segment_{seg.lower().replace(' ', '_')}.csv",
                    "text/csv",
                )

    with c2:
        st.subheader("Export Customer Profile")
        cid = st.text_input("Enter Customer ID for export")
        if cid and st.button("Generate JSON"):
            profile = _safe_query(conn, "SELECT * FROM golden_records WHERE unified_id = ?", [cid])
            if profile.empty:
                st.error("Customer not found.")
            else:
                export = profile.iloc[0].to_dict()
                export = {
                    k: (str(v) if not isinstance(v, int | float | bool | type(None)) else v)
                    for k, v in export.items()
                }
                json_str = json.dumps(export, indent=2, default=str)
                st.download_button(
                    "Download JSON", json_str, f"customer_{cid}.json", "application/json"
                )


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the Streamlit dashboard."""
    conn = _get_connection()

    page = st.sidebar.radio(
        "Navigation",
        [
            "Customer Search",
            "Segment Overview",
            "CLV Analysis",
            "Cohort Analysis",
            "Entity Resolution",
            "Export",
        ],
    )

    if page == "Customer Search":
        render_customer_search(conn)
    elif page == "Segment Overview":
        render_segment_overview(conn)
    elif page == "CLV Analysis":
        render_clv_analysis(conn)
    elif page == "Cohort Analysis":
        render_cohort_analysis(conn)
    elif page == "Entity Resolution":
        render_entity_resolution(conn)
    elif page == "Export":
        render_export(conn)


if __name__ == "__main__":
    main()
