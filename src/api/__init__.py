"""
API Connectors for External Data Sources

This module provides typed, documented connectors for:
- Nomis API: Labour market statistics (Claimant Count)
- Octopus Agile API: Energy price volatility data
"""

from .nomis import NomisClient, fetch_claimant_count
from .octopus import OctopusAgileClient, fetch_agile_prices

__all__ = [
    "NomisClient",
    "fetch_claimant_count",
    "OctopusAgileClient",
    "fetch_agile_prices",
]
