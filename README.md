# Customer 360 Analytics Platform

A unified customer analytics platform that consolidates data from CRM, transactions, web sessions, and support tickets into a single customer view with segmentation, lifetime value prediction, and cohort analysis.

## Architecture

```
Raw Data Sources       Entity Resolution       Analytics           Presentation
┌──────────────┐      ┌─────────────────┐     ┌──────────────┐    ┌───────────┐
│ CRM          │─────>│ Blocking        │────>│ RFM Analysis │───>│ Streamlit │
│ Transactions │      │ Fuzzy Matching  │     │ K-Means      │    │ Dashboard │
│ Web Sessions │      │ Golden Records  │     │ CLV (BG/NBD) │    │           │
│ Support      │      └─────────────────┘     │ Cohort       │    └───────────┘
└──────────────┘               │              └──────────────┘          │
       │                       v                     │                  v
       v               ┌──────────────┐              v           ┌───────────┐
 Synthetic Data        │   DuckDB     │◄─────────────┘           │  Export   │
 Generator             └──────────────┘                          │  CSV/JSON │
                                                                 └───────────┘
```

**Pipeline Steps:**

1. **Data Generation** -- Synthetic CRM, transaction, web, and support records with realistic quality issues (duplicates, missing values, orphans)
2. **DuckDB Loading** -- Bulk-load all sources into an embedded analytical database
3. **Quality Assessment** -- Record counts, orphan rates, missing values, duplicate detection
4. **Entity Resolution** -- First-letter blocking + Jaro-Winkler/exact comparison + threshold classification
5. **Golden Records** -- Connected-component clustering + best-value field selection
6. **RFM Analysis** -- Quintile scoring (1-5) with segment mapping (Champions, Loyal, At Risk, Lost, etc.)
7. **K-Means Clustering** -- Multi-source feature engineering with elbow/silhouette optimization
8. **CLV Prediction** -- BG/NBD + Gamma-Gamma probabilistic models with ML fallback
9. **Cohort Analysis** -- Signup-month cohorts with period-over-period retention matrices

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

## Setup

```bash
git clone git@github.com:KarasiewiczStephane/customer-360.git
cd customer-360

pip install -r requirements.txt
pip install -e .
```

## Usage

```bash
# Run the full pipeline (10K customers, ~500K transactions)
make pipeline

# Quick test run (~1K customers)
make pipeline-sample

# Launch the dashboard
make dashboard

# Run tests
make test

# Lint
make lint
```

### Docker

```bash
# Build and run both pipeline + dashboard
make docker-up

# Or run individually
make docker-build
make docker-pipeline
make docker-dashboard
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
├── tests/                  # 129 tests, 80%+ coverage
├── configs/
│   └── config.yaml         # All configurable parameters
├── .github/workflows/
│   └── ci.yml              # Lint, test, Docker CI pipeline
├── Dockerfile              # Multi-stage: pipeline + dashboard targets
├── docker-compose.yml      # Orchestrated pipeline → dashboard startup
├── Makefile                # Build, test, run, and Docker targets
├── requirements.txt
└── pyproject.toml
```

## Tech Stack

- **Database:** DuckDB (embedded OLAP)
- **Entity Resolution:** recordlinkage, jellyfish, fuzzywuzzy, networkx
- **ML/Analytics:** scikit-learn, lifetimes (BG/NBD, Gamma-Gamma)
- **Dashboard:** Streamlit, Plotly
- **Quality:** pytest, ruff, pre-commit
- **Deployment:** Docker, GitHub Actions CI

## License

MIT
