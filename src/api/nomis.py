"""
Nomis API Connector

Provides access to ONS labour market statistics via the Nomis API.
Primary use case: Monthly Claimant Count for Merton (SW19).

API Documentation: https://www.nomisweb.co.uk/api/v01/help

Statistical Rationale:
    The Claimant Count is a leading indicator of economic stress at the
    household level. Changes in the derivative (dC/dt) can signal policy
    shocks or economic disruptions that may propagate to food insecurity
    with a characteristic lag time (τ_lag).
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import urllib.request
import urllib.parse
import json

import polars as pl


# Merton Local Authority geography code
MERTON_GEO_CODE = "1946157274"

# Claimant Count dataset ID
CLAIMANT_COUNT_DATASET = "NM_162_1"

# Base URL for Nomis API
NOMIS_BASE_URL = "https://www.nomisweb.co.uk/api/v01"


@dataclass
class NomisConfig:
    """Configuration for Nomis API requests.

    Attributes:
        dataset_id: Nomis dataset identifier (e.g., NM_162_1 for Claimant Count)
        geography: Geographic area code (e.g., 1946157274 for Merton)
        measures: Measure type (20100 = persons, 20201 = rate)
        gender: Gender filter (0 = total, 1 = male, 2 = female)
        age: Age band filter (0 = all ages 16+)
    """
    dataset_id: str = CLAIMANT_COUNT_DATASET
    geography: str = MERTON_GEO_CODE
    measures: str = "20100"  # Persons count
    gender: str = "0"  # Total (male + female)
    age: str = "0"  # All ages 16+


class NomisClient:
    """
    Client for fetching labour market data from the Nomis API.

    The Nomis API provides access to official UK labour market statistics
    from the Office for National Statistics. This client focuses on the
    Claimant Count measure, which tracks unemployment benefit claims.

    Example:
        >>> client = NomisClient()
        >>> df = client.fetch_claimant_count(
        ...     start_date="2020-01-01",
        ...     end_date="2024-12-01"
        ... )
        >>> print(df.head())

    Physical Interpretation:
        The claimant count C(t) represents the instantaneous state of
        economic stress. The "Friction Index" dC/dt captures the rate
        of change, which may correlate with downstream food insecurity
        after a characteristic delay τ.
    """

    def __init__(self, config: Optional[NomisConfig] = None):
        """
        Initialize the Nomis client.

        Args:
            config: Optional configuration. Defaults to Merton Claimant Count.
        """
        self.config = config or NomisConfig()
        self.base_url = NOMIS_BASE_URL

    def _build_url(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> str:
        """
        Construct the API URL with query parameters.

        The Nomis API accepts comma-separated months for time ranges.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Complete API URL string
        """
        # Build time parameter - generate list of months for range
        time_param = "latest"
        if start_date and end_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

            # Generate list of months in range
            months = []
            current = start_dt.replace(day=1)
            end_month = end_dt.replace(day=1)

            while current <= end_month:
                months.append(current.strftime("%Y-%m"))
                # Move to next month
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)

            time_param = ",".join(months)

        params = {
            "geography": self.config.geography,
            "gender": self.config.gender,
            "age": self.config.age,
            "measure": "1",  # Claimant count (not rate/proportion)
            "measures": self.config.measures,
            "time": time_param,
        }

        query_string = urllib.parse.urlencode(params)
        url = f"{self.base_url}/dataset/{self.config.dataset_id}.data.json?{query_string}"

        return url

    def fetch_claimant_count(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Fetch monthly claimant count data for the configured geography.

        The Claimant Count measures the number of people claiming
        unemployment-related benefits. It is a timely indicator of
        labour market conditions, released monthly.

        Args:
            start_date: Start date in YYYY-MM-DD format (default: 2020-01-01)
            end_date: End date in YYYY-MM-DD format (default: latest available)

        Returns:
            polars DataFrame with columns:
                - date: datetime, first of month
                - geography: str, area name
                - claimant_count: int, number of claimants

        Raises:
            urllib.error.URLError: If the API request fails
            ValueError: If the response cannot be parsed
        """
        start_date = start_date or "2020-01-01"
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")

        url = self._build_url(start_date, end_date)

        # Fetch data
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))

        # Parse response
        observations = data.get("obs", [])

        if not observations:
            return pl.DataFrame({
                "date": [],
                "geography": [],
                "claimant_count": [],
            })

        # Extract records
        records = []
        for obs in observations:
            # Time field contains date in "YYYY-MM" format
            time_info = obs.get("time", {})
            date_str = time_info.get("value", "")

            # Geography description contains area name
            geography_info = obs.get("geography", {})
            geography = geography_info.get("description", "")

            # Observation value
            obs_value_info = obs.get("obs_value", {})
            value = obs_value_info.get("value", 0)

            # Parse date (format: "2025-12")
            try:
                date = datetime.strptime(date_str, "%Y-%m")
            except ValueError:
                continue

            records.append({
                "date": date,
                "geography": geography,
                "claimant_count": int(value) if value else 0,
            })

        df = pl.DataFrame(records).sort("date")

        return df

    def compute_friction_index(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute the Friction Index: the rate of change of claimant count.

        The Friction Index dC/dt captures sudden changes in the labour
        market that may signal policy shocks (e.g., benefit changes,
        lockdowns) or economic disruptions.

        Physical Interpretation:
            A positive spike in dC/dt indicates rapid deterioration of
            employment conditions. This may precede increased food bank
            demand with a characteristic lag τ_lag.

        Args:
            df: DataFrame with 'date' and 'claimant_count' columns

        Returns:
            DataFrame with additional 'friction_index' column
        """
        return df.with_columns([
            (pl.col("claimant_count").diff() / pl.col("claimant_count").shift(1))
            .alias("friction_index")
        ])


def fetch_claimant_count(
    geography: str = MERTON_GEO_CODE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pl.DataFrame:
    """
    Convenience function to fetch claimant count data.

    This is a simplified interface for the most common use case:
    fetching monthly claimant count for a given geography.

    Args:
        geography: Nomis geography code (default: Merton)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        polars DataFrame with claimant count data

    Example:
        >>> df = fetch_claimant_count(start_date="2020-01-01")
        >>> print(df.columns)
        ['date', 'geography', 'claimant_count']
    """
    config = NomisConfig(geography=geography)
    client = NomisClient(config)
    return client.fetch_claimant_count(start_date, end_date)
