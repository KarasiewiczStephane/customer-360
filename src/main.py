"""Entry point for the Customer 360 analytics pipeline.

Orchestrates data generation, entity resolution, analytics, and
stores results in DuckDB for dashboard consumption.
"""

import argparse
import time
from typing import Any

import pandas as pd

from src.analytics.clv import CLVPredictor
from src.analytics.cohort import CohortAnalyzer
from src.analytics.rfm import RFMAnalyzer
from src.analytics.segmentation import CustomerSegmentation
from src.data.generator import SyntheticDataGenerator
from src.data.loader import DataLoader
from src.data.quality import assess_data_quality
from src.resolution.matcher import EntityMatcher
from src.resolution.merger import RecordMerger
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_pipeline(config_path: str = "configs/config.yaml", sample: bool = False) -> dict[str, Any]:
    """Run the full Customer 360 analytics pipeline.

    Args:
        config_path: Path to the YAML configuration file.
        sample: If True, use reduced data volumes for quick runs.

    Returns:
        Summary dictionary with step timings and record counts.
    """
    config = load_config(config_path)
    db_path = config["database"]["path"]
    summary: dict[str, Any] = {"steps": {}, "timings": {}}

    if sample:
        config["data"]["num_customers"] = config["data"].get("sample_customers", 1000)
        config["data"]["num_transactions"] = config["data"]["num_customers"] * 50
        config["data"]["num_web_sessions"] = config["data"]["num_customers"] * 10
        config["data"]["num_support_tickets"] = config["data"]["num_customers"] * 2
        logger.info("Sample mode: reduced to %d customers", config["data"]["num_customers"])

    pipeline_start = time.time()

    # Step 1: Generate synthetic data
    t0 = time.time()
    logger.info("Step 1/9: Generating synthetic data")
    gen = SyntheticDataGenerator(config, seed=config["data"].get("seed", 42))
    customers = gen.generate_crm_customers()
    customer_ids = customers["customer_id"].tolist()
    transactions = gen.generate_transactions(customer_ids)
    web_sessions = gen.generate_web_sessions(customer_ids)
    support_tickets = gen.generate_support_tickets(customer_ids)
    summary["steps"]["generate"] = {
        "customers": len(customers),
        "transactions": len(transactions),
        "web_sessions": len(web_sessions),
        "support_tickets": len(support_tickets),
    }
    summary["timings"]["generate"] = round(time.time() - t0, 2)
    logger.info("Step 1 complete in %.2fs", summary["timings"]["generate"])

    # Step 2: Load raw data into DuckDB
    t0 = time.time()
    logger.info("Step 2/9: Loading data into DuckDB")
    loader = DataLoader(db_path)
    loader.load_all_data(
        {
            "crm_customers": customers,
            "transactions": transactions,
            "web_sessions": web_sessions,
            "support_tickets": support_tickets,
        }
    )
    summary["timings"]["load"] = round(time.time() - t0, 2)
    logger.info("Step 2 complete in %.2fs", summary["timings"]["load"])

    # Step 3: Data quality assessment
    t0 = time.time()
    logger.info("Step 3/9: Assessing data quality")
    quality_report = assess_data_quality(db_path)
    summary["steps"]["quality"] = quality_report
    summary["timings"]["quality"] = round(time.time() - t0, 2)
    logger.info("Step 3 complete in %.2fs", summary["timings"]["quality"])

    # Step 4: Entity resolution
    t0 = time.time()
    logger.info("Step 4/9: Running entity resolution")
    matcher = EntityMatcher(config.get("resolution", {}))
    matches, uncertain, non_matches, features = matcher.run(customers)
    summary["steps"]["resolution"] = {
        "matches": len(matches),
        "uncertain": len(uncertain),
        "non_matches": len(non_matches),
    }
    summary["timings"]["resolution"] = round(time.time() - t0, 2)
    logger.info("Step 4 complete in %.2fs", summary["timings"]["resolution"])

    # Step 5: Golden record creation
    t0 = time.time()
    logger.info("Step 5/9: Creating golden records")
    merger = RecordMerger(config.get("resolution", {}).get("source_priority"))
    normalized = matcher.normalize_data(customers)
    clusters = merger.build_match_clusters(matches)
    golden_records = merger.merge_all(clusters, normalized)
    loader.load_all_data({"golden_records": golden_records})
    summary["steps"]["golden_records"] = len(golden_records)
    summary["timings"]["golden_records"] = round(time.time() - t0, 2)
    logger.info("Step 5 complete in %.2fs", summary["timings"]["golden_records"])

    # Step 6: RFM analysis
    t0 = time.time()
    logger.info("Step 6/9: Running RFM analysis")
    rfm_analyzer = RFMAnalyzer()
    rfm = rfm_analyzer.calculate_rfm(transactions)
    rfm = rfm_analyzer.assign_scores(rfm, n_segments=config["analytics"].get("rfm_segments", 5))
    rfm = rfm_analyzer.assign_segments(rfm)
    loader.load_all_data({"rfm_scores": rfm})
    summary["steps"]["rfm_segments"] = rfm["segment"].value_counts().to_dict()
    summary["timings"]["rfm"] = round(time.time() - t0, 2)
    logger.info("Step 6 complete in %.2fs", summary["timings"]["rfm"])

    # Step 7: K-Means clustering
    t0 = time.time()
    logger.info("Step 7/9: Running K-Means segmentation")
    segmenter = CustomerSegmentation(
        max_clusters=config["analytics"].get("kmeans_max_clusters", 10),
    )
    seg_features = segmenter.prepare_features(rfm, web_sessions, support_tickets)
    clustered = segmenter.fit_clusters(seg_features)
    profiles = segmenter.get_cluster_profiles(clustered)
    loader.load_all_data({"cluster_profiles": profiles.reset_index()})
    summary["steps"]["clusters"] = int(segmenter.optimal_k or 0)
    summary["timings"]["clustering"] = round(time.time() - t0, 2)
    logger.info("Step 7 complete in %.2fs", summary["timings"]["clustering"])

    # Step 8: CLV prediction
    t0 = time.time()
    logger.info("Step 8/9: Predicting customer lifetime value")
    clv_predictor = CLVPredictor(
        prediction_period=config["analytics"].get("clv_prediction_months", 12),
    )
    clv_summary = clv_predictor.prepare_rfm_summary(transactions)
    clv_predictor.fit_probabilistic_models(clv_summary)
    clv_predictions = clv_predictor.predict_clv_probabilistic(
        clv_summary,
        discount_rate=config["analytics"].get("clv_discount_rate", 0.01),
    )
    clv_predictions = clv_predictor.segment_by_clv(clv_predictions)
    clv_predictions_out = clv_predictions.reset_index().rename(columns={"index": "customer_id"})
    if "customer_id" not in clv_predictions_out.columns:
        clv_predictions_out = clv_predictions_out.rename(
            columns={clv_predictions_out.columns[0]: "customer_id"}
        )
    loader.load_all_data({"clv_predictions": clv_predictions_out})
    summary["steps"]["clv_tiers"] = clv_predictions["clv_tier"].value_counts().to_dict()
    summary["timings"]["clv"] = round(time.time() - t0, 2)
    logger.info("Step 8 complete in %.2fs", summary["timings"]["clv"])

    # Step 9: Cohort analysis
    t0 = time.time()
    logger.info("Step 9/9: Running cohort analysis")
    cohort_analyzer = CohortAnalyzer()
    customers_with_signup = customers[["customer_id", "signup_date"]].copy()
    cohort_data = cohort_analyzer.create_cohorts(customers_with_signup, transactions)
    retention_matrix = cohort_analyzer.calculate_retention(cohort_data)
    cohort_retention = _flatten_retention_matrix(retention_matrix)
    loader.load_all_data({"cohort_retention": cohort_retention})
    summary["steps"]["cohort_months"] = retention_matrix.shape[0]
    summary["timings"]["cohort"] = round(time.time() - t0, 2)
    logger.info("Step 9 complete in %.2fs", summary["timings"]["cohort"])

    total_time = round(time.time() - pipeline_start, 2)
    summary["total_time"] = total_time
    logger.info("Pipeline complete in %.2fs", total_time)

    return summary


def _flatten_retention_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Convert a pivot-table retention matrix into a long-format DataFrame.

    Args:
        matrix: Retention matrix with cohort months as index and
            cohort age as columns.

    Returns:
        Long-format DataFrame with ``cohort_month``, ``cohort_age``,
        and ``retention_rate`` columns.
    """
    records = []
    for cohort_month in matrix.index:
        for cohort_age in matrix.columns:
            val = matrix.loc[cohort_month, cohort_age]
            if pd.notna(val):
                records.append(
                    {
                        "cohort_month": str(cohort_month),
                        "cohort_age": int(cohort_age),
                        "retention_rate": float(val),
                    }
                )
    return pd.DataFrame(records)


def main() -> None:
    """CLI entry point for the pipeline."""
    parser = argparse.ArgumentParser(description="Customer 360 Analytics Pipeline")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use reduced data volumes for a quick test run",
    )
    args = parser.parse_args()
    summary = run_pipeline(config_path=args.config, sample=args.sample)
    logger.info("Pipeline summary: %s", summary)


if __name__ == "__main__":
    main()
