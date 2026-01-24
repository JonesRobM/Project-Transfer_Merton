"""
Octopus Agile API Connector

Provides access to Octopus Energy's Agile tariff pricing data.
Primary use case: Energy price volatility index for systemic latency analysis.

API Documentation: https://developer.octopus.energy/docs/api/

Statistical Rationale:
    The Octopus Agile tariff provides half-hourly electricity prices that
    track wholesale market conditions. High volatility periods correlate
    with energy cost stress for households, potentially leading to food
    insecurity with a characteristic lag time.

    We compute a "volatility index" as the rolling standard deviation of
    daily prices, which captures price uncertainty rather than absolute level.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import urllib.request
import urllib.parse
import json

import polars as pl
import numpy as np


# Default product and tariff codes for Agile Octopus
DEFAULT_PRODUCT_CODE = "AGILE-FLEX-22-11-25"
DEFAULT_TARIFF_CODE = "E-1R-AGILE-FLEX-22-11-25-C"

# South East England region code (includes Merton/SW19)
REGION_CODE = "C"

# Base URL for Octopus API
OCTOPUS_BASE_URL = "https://api.octopus.energy/v1"


@dataclass
class OctopusConfig:
    """Configuration for Octopus API requests.

    Attributes:
        product_code: Octopus product identifier
        tariff_code: Specific tariff code including region
        region: Distribution region code (A-P)
    """
    product_code: str = DEFAULT_PRODUCT_CODE
    tariff_code: str = DEFAULT_TARIFF_CODE
    region: str = REGION_CODE


class OctopusAgileClient:
    """
    Client for fetching energy price data from Octopus Energy API.

    The Octopus Agile tariff offers half-hourly prices that vary based
    on wholesale electricity costs. This provides a high-frequency
    signal of energy market conditions.

    Example:
        >>> client = OctopusAgileClient()
        >>> df = client.fetch_prices(
        ...     start_date="2024-01-01",
        ...     end_date="2024-01-31"
        ... )
        >>> print(df.head())

    Physical Interpretation:
        Energy prices P(t) represent the instantaneous cost burden on
        households. The volatility σ(P) over a rolling window captures
        price uncertainty, which may affect household budgeting and
        food security with a characteristic lag.
    """

    def __init__(self, config: Optional[OctopusConfig] = None):
        """
        Initialize the Octopus client.

        Args:
            config: Optional configuration. Defaults to Agile tariff for SE England.
        """
        self.config = config or OctopusConfig()
        self.base_url = OCTOPUS_BASE_URL

    def _build_url(
        self,
        start_date: str,
        end_date: str,
        page_size: int = 1500,
    ) -> str:
        """
        Construct the API URL for standard unit rates.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            page_size: Number of records per page (max 1500)

        Returns:
            Complete API URL string
        """
        # Convert to ISO format with timezone
        period_from = f"{start_date}T00:00:00Z"
        period_to = f"{end_date}T23:59:59Z"

        params = {
            "period_from": period_from,
            "period_to": period_to,
            "page_size": page_size,
        }

        query_string = urllib.parse.urlencode(params)

        # Build URL path
        url = (
            f"{self.base_url}/products/{self.config.product_code}/"
            f"electricity-tariffs/{self.config.tariff_code}/"
            f"standard-unit-rates/?{query_string}"
        )

        return url

    def fetch_prices(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Fetch half-hourly electricity prices for the configured tariff.

        Returns prices in pence per kWh for each half-hour period.
        The Agile tariff updates prices daily based on wholesale market.

        Args:
            start_date: Start date in YYYY-MM-DD format (default: 30 days ago)
            end_date: End date in YYYY-MM-DD format (default: today)

        Returns:
            polars DataFrame with columns:
                - datetime: datetime, start of half-hour period
                - price_pence_kwh: float, unit rate in pence/kWh
                - date: date, for daily aggregation

        Raises:
            urllib.error.URLError: If the API request fails
            ValueError: If the response cannot be parsed

        Note:
            The Agile API only provides ~6 months of historical data.
            For longer time series, consider using archived data.
        """
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        start_date = start_date or (
            datetime.now() - timedelta(days=30)
        ).strftime("%Y-%m-%d")

        url = self._build_url(start_date, end_date)

        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # Product/tariff not found - return empty DataFrame
                return pl.DataFrame({
                    "datetime": [],
                    "price_pence_kwh": [],
                    "date": [],
                })
            raise

        # Parse response
        results = data.get("results", [])

        if not results:
            return pl.DataFrame({
                "datetime": [],
                "price_pence_kwh": [],
                "date": [],
            })

        # Extract records
        records = []
        for item in results:
            valid_from = item.get("valid_from", "")
            value = item.get("value_inc_vat", 0)

            try:
                dt = datetime.fromisoformat(valid_from.replace("Z", "+00:00"))
            except ValueError:
                continue

            records.append({
                "datetime": dt.replace(tzinfo=None),
                "price_pence_kwh": float(value) if value else 0.0,
            })

        df = pl.DataFrame(records).sort("datetime")

        # Add date column for aggregation
        df = df.with_columns([
            pl.col("datetime").dt.date().alias("date")
        ])

        return df

    def compute_daily_stats(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Aggregate half-hourly prices to daily statistics.

        Computes daily mean, min, max, and standard deviation of prices.
        The standard deviation captures intra-day volatility.

        Args:
            df: DataFrame with 'date' and 'price_pence_kwh' columns

        Returns:
            DataFrame with daily aggregated statistics
        """
        return df.group_by("date").agg([
            pl.col("price_pence_kwh").mean().alias("price_mean"),
            pl.col("price_pence_kwh").min().alias("price_min"),
            pl.col("price_pence_kwh").max().alias("price_max"),
            pl.col("price_pence_kwh").std().alias("price_std"),
            pl.col("price_pence_kwh").count().alias("n_periods"),
        ]).sort("date")

    def compute_volatility_index(
        self,
        df: pl.DataFrame,
        window_days: int = 7,
    ) -> pl.DataFrame:
        """
        Compute the Energy Volatility Index.

        The volatility index is the rolling standard deviation of daily
        mean prices over a specified window. High values indicate price
        uncertainty that may stress household budgets.

        Physical Interpretation:
            σ(P) captures the unpredictability of energy costs. Households
            cannot easily adjust spending when prices fluctuate rapidly,
            leading to potential food budget squeeze.

        Args:
            df: DataFrame with daily price statistics (from compute_daily_stats)
            window_days: Rolling window size in days (default: 7)

        Returns:
            DataFrame with additional 'volatility_index' column
        """
        return df.with_columns([
            pl.col("price_mean")
            .rolling_std(window_size=window_days, min_periods=3)
            .alias("volatility_index")
        ])


def fetch_agile_prices(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    region: str = REGION_CODE,
) -> pl.DataFrame:
    """
    Convenience function to fetch Agile tariff prices.

    This is a simplified interface for fetching half-hourly prices
    for a given date range and region.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        region: Distribution region code (default: C for South East)

    Returns:
        polars DataFrame with price data

    Example:
        >>> df = fetch_agile_prices(start_date="2024-01-01", end_date="2024-01-31")
        >>> print(df.columns)
        ['datetime', 'price_pence_kwh', 'date']
    """
    # Update tariff code with region
    tariff_code = f"E-1R-AGILE-FLEX-22-11-25-{region}"
    config = OctopusConfig(tariff_code=tariff_code, region=region)
    client = OctopusAgileClient(config)
    return client.fetch_prices(start_date, end_date)


def get_monthly_price_summary(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pl.DataFrame:
    """
    Get monthly summary of energy prices for time series alignment.

    Aggregates half-hourly Agile prices to monthly means for comparison
    with other monthly indicators (claimant count, food insecurity).

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        polars DataFrame with monthly price summary:
            - month: date (first of month)
            - price_mean: average price in pence/kWh
            - price_volatility: standard deviation of daily prices
    """
    client = OctopusAgileClient()
    df = client.fetch_prices(start_date, end_date)

    if df.is_empty():
        return pl.DataFrame({
            "month": [],
            "price_mean": [],
            "price_volatility": [],
        })

    # Aggregate to monthly
    monthly = df.with_columns([
        pl.col("datetime").dt.truncate("1mo").alias("month")
    ]).group_by("month").agg([
        pl.col("price_pence_kwh").mean().alias("price_mean"),
        pl.col("price_pence_kwh").std().alias("price_volatility"),
    ]).sort("month")

    return monthly
