"""Synthetic data generator for the Customer 360 platform.

Generates realistic multi-source customer data with intentional
data-quality issues such as duplicates, missing values, and
inconsistent formatting.
"""

import random
from typing import Any

import numpy as np
import pandas as pd
from faker import Faker

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SyntheticDataGenerator:
    """Generates synthetic CRM, transaction, web, and support data.

    Args:
        config: Configuration dictionary with ``data`` section.
        seed: Random seed for reproducibility.
    """

    SEGMENTS = ["Enterprise", "SMB", "Startup", "Consumer"]
    PRODUCT_CATEGORIES = ["Electronics", "Clothing", "Home", "Food", "Services", "Software"]

    def __init__(self, config: dict[str, Any], seed: int = 42) -> None:
        self.config = config
        self.seed = seed
        self.fake = Faker()
        Faker.seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    # ------------------------------------------------------------------
    # CRM customers
    # ------------------------------------------------------------------

    def generate_crm_customers(self, n: int | None = None) -> pd.DataFrame:
        """Generate CRM customer records with intentional quality issues.

        Args:
            n: Number of base customer records to create. Defaults to
               the value in ``config['data']['num_customers']``.

        Returns:
            DataFrame with customer records including ~5 % near-duplicates.
        """
        if n is None:
            n = self.config["data"]["num_customers"]
        logger.info("Generating %d CRM customer records …", n)

        customers: list[dict[str, Any]] = []
        for i in range(n):
            customers.append(
                {
                    "customer_id": f"CRM_{i:06d}",
                    "name": self.fake.name(),
                    "email": self._generate_email_with_issues(),
                    "phone": self._generate_phone_with_issues(),
                    "signup_date": self.fake.date_between(start_date="-3y", end_date="today"),
                    "segment": random.choice(self.SEGMENTS),
                    "address": self.fake.address().replace("\n", ", "),
                    "zip_code": self.fake.zipcode(),
                }
            )

        df = pd.DataFrame(customers)

        if df.empty:
            return df

        # Introduce ~5 % near-duplicates
        duplicates = self._create_duplicate_records(df, duplicate_rate=0.05)
        result = pd.concat([df, duplicates], ignore_index=True)
        logger.info(
            "Created %d CRM records (%d base + %d duplicates)",
            len(result),
            len(df),
            len(duplicates),
        )
        return result

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    def generate_transactions(self, customer_ids: list[str], n: int | None = None) -> pd.DataFrame:
        """Generate transaction records with orphan transactions.

        Args:
            customer_ids: Valid customer IDs to sample from.
            n: Number of transactions. Defaults to config value.

        Returns:
            DataFrame with transaction records (~5 % orphans).
        """
        if n is None:
            n = self.config["data"]["num_transactions"]
        logger.info("Generating %d transaction records …", n)

        transactions: list[dict[str, Any]] = []
        for i in range(n):
            # 95 % valid, 5 % orphan
            if random.random() < 0.95:
                cust_id = random.choice(customer_ids)
            else:
                cust_id = f"ORPHAN_{random.randint(100000, 999999)}"

            transactions.append(
                {
                    "transaction_id": f"TXN_{i:08d}",
                    "customer_id": cust_id,
                    "date": self.fake.date_between(start_date="-2y", end_date="today"),
                    "amount": round(random.lognormvariate(3.5, 1.2), 2),
                    "product_category": random.choice(self.PRODUCT_CATEGORIES),
                }
            )

        df = pd.DataFrame(transactions)
        logger.info("Created %d transaction records", len(df))
        return df

    # ------------------------------------------------------------------
    # Web sessions
    # ------------------------------------------------------------------

    REFERRERS = ["google", "facebook", "direct", "email", "linkedin", "twitter", None]
    PAGES = ["home", "products", "pricing", "about", "contact", "checkout", "blog"]

    def generate_web_sessions(self, customer_ids: list[str], n: int | None = None) -> pd.DataFrame:
        """Generate web behaviour session records.

        Args:
            customer_ids: Valid customer IDs to sample from.
            n: Number of sessions. Defaults to config value.

        Returns:
            DataFrame with web session records (~10 % anonymous).
        """
        if n is None:
            n = self.config["data"]["num_web_sessions"]
        logger.info("Generating %d web session records …", n)

        sessions: list[dict[str, Any]] = []
        for i in range(n):
            cust_id = random.choice(customer_ids) if random.random() < 0.9 else None
            sessions.append(
                {
                    "session_id": f"SES_{i:08d}",
                    "customer_id": cust_id,
                    "timestamp": self.fake.date_time_between(start_date="-1y", end_date="now"),
                    "pages_visited": random.randint(1, 20),
                    "time_on_site": random.randint(10, 1800),
                    "referrer": random.choice(self.REFERRERS),
                    "pages_path": ",".join(random.choices(self.PAGES, k=random.randint(1, 8))),
                }
            )

        df = pd.DataFrame(sessions)
        logger.info("Created %d web session records", len(df))
        return df

    # ------------------------------------------------------------------
    # Support tickets
    # ------------------------------------------------------------------

    TICKET_CATEGORIES = ["Billing", "Technical", "Account", "Product", "Returns", "Other"]

    def generate_support_tickets(
        self, customer_ids: list[str], n: int | None = None
    ) -> pd.DataFrame:
        """Generate support ticket records.

        Args:
            customer_ids: Valid customer IDs to sample from.
            n: Number of tickets. Defaults to config value.

        Returns:
            DataFrame with support ticket records.
        """
        if n is None:
            n = self.config["data"]["num_support_tickets"]
        logger.info("Generating %d support ticket records …", n)

        tickets: list[dict[str, Any]] = []
        for i in range(n):
            resolution_hours = random.expovariate(1 / 24)
            tickets.append(
                {
                    "ticket_id": f"TKT_{i:06d}",
                    "customer_id": random.choice(customer_ids),
                    "created_at": self.fake.date_time_between(start_date="-2y", end_date="now"),
                    "category": random.choice(self.TICKET_CATEGORIES),
                    "resolution_time_hours": round(resolution_hours, 2),
                    "satisfaction_score": random.choices(
                        [1, 2, 3, 4, 5], weights=[5, 10, 20, 35, 30]
                    )[0],
                    "status": random.choice(
                        ["resolved", "resolved", "resolved", "open", "pending"]
                    ),
                }
            )

        df = pd.DataFrame(tickets)
        logger.info("Created %d support ticket records", len(df))
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_email_with_issues(self) -> str | None:
        """Return an email with ~10 % missing and ~5 % malformed."""
        if random.random() < 0.10:
            return None
        email = self.fake.email()
        if random.random() < 0.05:
            email = email.upper() if random.random() < 0.5 else email.replace("@", " @ ")
        return email

    def _generate_phone_with_issues(self) -> str | None:
        """Return a phone number with ~8 % missing and varied formats."""
        if random.random() < 0.08:
            return None
        formats = [
            "###-###-####",
            "(###) ###-####",
            "##########",
            "+1-###-###-####",
        ]
        return self.fake.numerify(random.choice(formats))

    def _create_duplicate_records(self, df: pd.DataFrame, duplicate_rate: float) -> pd.DataFrame:
        """Create near-duplicate records with slight name variations.

        Args:
            df: Original customer DataFrame.
            duplicate_rate: Fraction of records to duplicate.

        Returns:
            DataFrame of near-duplicate records.
        """
        n_duplicates = int(len(df) * duplicate_rate)
        duplicates = df.sample(n=n_duplicates, random_state=self.seed).copy()
        duplicates["customer_id"] = [f"CRM_DUP_{i:06d}" for i in range(n_duplicates)]
        duplicates["name"] = duplicates["name"].apply(self._vary_name)
        return duplicates

    @staticmethod
    def _vary_name(name: str) -> str:
        """Introduce variations: typos, nicknames, abbreviations."""
        variations = [
            lambda n: n.replace("Robert", "Bob").replace("William", "Bill").replace("James", "Jim"),
            lambda n: n.lower(),
            lambda n: n + " Jr." if random.random() < 0.3 else n,
            lambda n: "".join(c for c in n if random.random() > 0.08),
        ]
        return random.choice(variations)(name)
