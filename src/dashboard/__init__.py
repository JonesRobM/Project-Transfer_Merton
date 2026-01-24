"""
Monitoring Dashboard for Systemic Latency

This module provides a lightweight monitoring system that tracks
energy price announcements and triggers alerts when significant
changes are detected.

Designed to run on Raspberry Pi or similar low-power hardware.
"""

from .monitor import (
    EnergyPriceMonitor,
    AlertLevel,
    check_price_cap,
)

__all__ = [
    "EnergyPriceMonitor",
    "AlertLevel",
    "check_price_cap",
]
