# Customer 360 Analytics Platform

A unified customer analytics platform that consolidates data from CRM, transactions, web sessions, and support tickets into a single customer view with segmentation, lifetime value prediction, and cohort analysis.

## Quick Start

```bash
# Clone and install
git clone git@github.com:KarasiewiczStephane/customer-360.git
cd customer-360
make install
pip install -e .

# Run the full pipeline (generates synthetic data, loads into DuckDB,
# runs entity resolution, RFM, clustering, CLV, and cohort analysis)
make pipeline

# Or run a smaller sample (~1K customers) for a quick test
make pipeline-sample

# Launch the Streamlit dashboard (requires pipeline to have run first)
make dashboard
# Open http://localhost:8501
```

The pipeline generates all data synthetically -- no external datasets are required. Running `make pipeline` executes a 9-step process that populates a DuckDB database at `data/customer360.duckdb`. The dashboard reads from this database, so the pipeline must complete before launching the dashboard.

## Architecture

```
Raw Data Sources       Entity Resolution       Analytics           Presentation
+----------------+    +-------------------+   +----------------+  +-------------+
| CRM            |--->| Blocking          |-->| RFM Analysis   |->| Streamlit   |
| Transactions   |    | Fuzzy Matching    |   | K-Means        |  | Dashboard   |
| Web Sessions   |    | Golden Records    |   | CLV (BG/NBD)   |  |             |
| Support        |    +-------------------+   | Cohort         |  +-------------+
+----------------+            |               +----------------+        |
       |                      v                      |                  v
       v               +------------+                v            +-----------+
 Synthetic Data        |   DuckDB   |<---------------+            |  Export   |
 Generator             +------------+                             |  CSV/JSON |
                                                                  +-----------+
```

**Pipeline Steps (executed by `src/main.py`):**

1. **Data Generation** (`src/data/generator.py`) -- Synthetic CRM, transaction, web, and support records with realistic quality issues (duplicates, missing values, orphans)
2. **DuckDB Loading** (`src/data/loader.py`) -- Bulk-load all sources into an embedded analytical database
3. **Quality Assessment** (`src/data/quality.py`) -- Record counts, orphan rates, missing values, duplicate detection
4. **Entity Resolution** (`src/resolution/matcher.py`) -- First-letter blocking + Jaro-Winkler/exact comparison + threshold classification
5. **Golden Records** (`src/resolution/merger.py`) -- Connected-component clustering + best-value field selection
6. **RFM Analysis** (`src/analytics/rfm.py`) -- Quintile scoring (1-5) with segment mapping
7. **K-Means Clustering** (`src/analytics/segmentation.py`) -- Multi-source feature engineering with elbow/silhouette optimization
8. **CLV Prediction** (`src/analytics/clv.py`) -- BG/NBD + Gamma-Gamma probabilistic models with ML fallback
9. **Cohort Analysis** (`src/analytics/cohort.py`) -- Signup-month cohorts with period-over-period retention matrices

## Dashboard Pages

The Streamlit dashboard (`src/dashboard/app.py`) provides six views via sidebar navigation:

| Page | Description |
|---|---|
| Customer Search | Search by name, email, or ID; view full 360-degree profile with transactions, web activity, and support history |
| Segment Overview | RFM segment distribution pie chart, revenue by segment bar chart, and segment metrics table |
| CLV Analysis | CLV distribution histogram by tier, total CLV by tier, and tier descriptive statistics |
| Cohort Analysis | Retention heatmap by signup cohort and average retention curve |
| Entity Resolution | Deduplication metrics: original vs. unified record counts, dedup rate, average cluster size |
| Export | Download segment lists as CSV or individual customer profiles as JSON |

## Customer Segments

| Segment | R | F | M | Description |
|---|---|---|---|---|
| Champions | 5 | 5 | 5 | Best customers: recent, frequent, high spend |
| Loyal Customers | 4-5 | 4-5 | 4-5 | Consistent high-value customers |
| Potential Loyalists | 3-5 | 3 | 3 | Recent with moderate engagement |
| New Customers | 4-5 | 1 | 1 | Recent first-time buyers |
| At Risk | 1-2 | 1-2 | 2-3 | Declining engagement |
| Can't Lose Them | 1 | 1 | 3-5 | Inactive but previously high value |
| Lost | 1 | 1 | 1 | No recent activity |

## Configuration

All parameters are driven by `configs/config.yaml`:

| Section | Key Parameters |
|---|---|
| `data` | `num_customers` (10K), `num_transactions` (500K), `sample_customers` (1K), `seed` (42) |
| `database` | `path` (data/customer360.duckdb) |
| `resolution` | `match_threshold` (0.85), `uncertain_threshold` (0.6), blocking fields |
| `analytics` | `rfm_segments` (5), `kmeans_max_clusters` (10), `clv_prediction_months` (12) |

## Make Targets

```bash
make install           # pip install -r requirements.txt
make pipeline          # Run full 9-step pipeline (python -m src.main)
make pipeline-sample   # Run pipeline with ~1K customers (--sample flag)
make dashboard         # Launch Streamlit dashboard on port 8501
make test              # pytest tests/ -v --tb=short --cov=src
make lint              # ruff check + ruff format
make clean             # Remove __pycache__, .pyc, and DuckDB file
make run               # Alias for make pipeline
```

## Docker

```bash
# Build and run both pipeline + dashboard
make docker-up

# Or run individually
make docker-build       # Build pipeline and dashboard images
make docker-pipeline    # Run pipeline in container (populates data volume)
make docker-dashboard   # Run dashboard in container on port 8501

# Tear down
make docker-down
```

The dashboard is available at `http://localhost:8501`.

## Project Structure

```
customer-360/
├── src/
│   ├── data/               # Synthetic data generation and loading
│   │   ├── generator.py    # Multi-source data generator with quality issues
│   │   ├── loader.py       # DuckDB bulk loader
│   │   └── quality.py      # Data quality assessment
│   ├── resolution/         # Entity resolution
│   │   ├── matcher.py      # Blocking, comparison, classification
│   │   ├── merger.py       # Golden record creation
│   │   └── quality.py      # Match reporting and review queue
│   ├── analytics/          # Customer analytics
│   │   ├── rfm.py          # RFM scoring and segmentation
│   │   ├── segmentation.py # K-Means with feature engineering
│   │   ├── clv.py          # Probabilistic and ML-based CLV
│   │   └── cohort.py       # Cohort retention analysis
│   ├── dashboard/          # Streamlit dashboard
│   │   └── app.py          # Multi-page dashboard application
│   ├── utils/              # Shared utilities
│   │   ├── config.py       # YAML configuration loader
│   │   ├── database.py     # DuckDB connection manager
│   │   └── logger.py       # Structured logging
│   └── main.py             # Pipeline orchestrator with CLI
├── tests/
├── configs/
│   └── config.yaml         # All configurable parameters
├── .github/workflows/
│   └── ci.yml              # Lint, test, Docker CI pipeline
├── Dockerfile              # Multi-stage: pipeline + dashboard targets
├── docker-compose.yml      # Orchestrated pipeline -> dashboard startup
├── Makefile
├── requirements.txt
└── pyproject.toml
```

## Tech Stack

- **Database:** DuckDB (embedded OLAP)
- **Entity Resolution:** recordlinkage, jellyfish, fuzzywuzzy, networkx
- **ML/Analytics:** scikit-learn, lifetimes (BG/NBD, Gamma-Gamma)
- **Data Generation:** Faker
- **Dashboard:** Streamlit, Plotly
- **Quality:** pytest, ruff, pre-commit
- **Deployment:** Docker, GitHub Actions CI

## License

MIT
