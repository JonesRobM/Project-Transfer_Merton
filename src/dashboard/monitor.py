"""
Energy Price Monitoring System

This module provides a monitoring system that tracks the Ofgem energy
price cap and triggers alerts when significant changes are detected.

Alert Logic:
    Based on the systemic latency analysis, we know that energy price
    increases lead to food insecurity with a ~3 month lag. This monitor
    provides early warning when price cap changes exceed thresholds.

    Alert Levels:
    - GREEN: Price cap stable or decreasing
    - AMBER: Price cap increased 5-10%
    - RED: Price cap increased >10% (intervention protocol triggered)

Designed for Raspberry Pi deployment with minimal dependencies.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert levels for the monitoring system."""
    GREEN = "green"    # No action needed
    AMBER = "amber"    # Watch closely
    RED = "red"        # Intervention protocol


@dataclass
class PriceCapRecord:
    """Record of an energy price cap value.

    Attributes:
        date: Effective date of the price cap
        annual_cap: Annual cap in GBP for typical household
        quarterly_change_pct: Percentage change from previous quarter
        source: Source URL or reference
    """
    date: datetime
    annual_cap: float
    quarterly_change_pct: Optional[float] = None
    source: str = "manual"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "date": self.date.isoformat(),
            "annual_cap": self.annual_cap,
            "quarterly_change_pct": self.quarterly_change_pct,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PriceCapRecord":
        """Create from dictionary."""
        return cls(
            date=datetime.fromisoformat(data["date"]),
            annual_cap=data["annual_cap"],
            quarterly_change_pct=data.get("quarterly_change_pct"),
            source=data.get("source", "manual"),
        )


@dataclass
class Alert:
    """An alert triggered by the monitoring system.

    Attributes:
        timestamp: When the alert was generated
        level: Alert level (GREEN, AMBER, RED)
        message: Human-readable description
        price_cap: The price cap that triggered the alert
        intervention_deadline: Deadline for intervention (3 months from trigger)
    """
    timestamp: datetime
    level: AlertLevel
    message: str
    price_cap: PriceCapRecord
    intervention_deadline: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "price_cap": self.price_cap.to_dict(),
            "intervention_deadline": self.intervention_deadline.isoformat(),
        }


class EnergyPriceMonitor:
    """
    Monitoring system for energy price cap changes.

    This class tracks the Ofgem energy price cap and generates alerts
    when significant changes are detected. It is designed to run on
    low-power hardware (Raspberry Pi) with periodic checks.

    Usage:
        monitor = EnergyPriceMonitor(data_dir=Path("./data"))
        monitor.add_price_cap(
            date=datetime(2024, 4, 1),
            annual_cap=1690,
            source="ofgem"
        )
        alert = monitor.check_and_alert()
        if alert.level == AlertLevel.RED:
            print("INTERVENTION PROTOCOL TRIGGERED")

    State Persistence:
        The monitor persists its state to a JSON file, allowing it to
        resume after restarts without losing historical data.
    """

    # Thresholds for alert levels (percentage change)
    AMBER_THRESHOLD = 5.0   # 5% increase
    RED_THRESHOLD = 10.0    # 10% increase

    # Intervention window (months)
    INTERVENTION_WINDOW_MONTHS = 3

    def __init__(
        self,
        data_dir: Path,
        state_file: str = "monitor_state.json",
    ):
        """
        Initialize the monitoring system.

        Args:
            data_dir: Directory for state persistence
            state_file: Name of the state file
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.data_dir / state_file

        self.price_caps: List[PriceCapRecord] = []
        self.alerts: List[Alert] = []

        self._load_state()

    def _load_state(self) -> None:
        """Load persisted state from disk."""
        if self.state_path.exists():
            try:
                with open(self.state_path, "r") as f:
                    data = json.load(f)

                self.price_caps = [
                    PriceCapRecord.from_dict(pc)
                    for pc in data.get("price_caps", [])
                ]
                logger.info(f"Loaded {len(self.price_caps)} price cap records")
            except Exception as e:
                logger.warning(f"Could not load state: {e}")

    def _save_state(self) -> None:
        """Persist state to disk."""
        data = {
            "price_caps": [pc.to_dict() for pc in self.price_caps],
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.state_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_price_cap(
        self,
        date: datetime,
        annual_cap: float,
        source: str = "manual",
    ) -> PriceCapRecord:
        """
        Add a new price cap record.

        Args:
            date: Effective date of the price cap
            annual_cap: Annual cap in GBP for typical household
            source: Source of the data (e.g., "ofgem", "manual")

        Returns:
            The created PriceCapRecord
        """
        # Calculate change from previous
        quarterly_change = None
        if self.price_caps:
            prev = self.price_caps[-1]
            quarterly_change = ((annual_cap - prev.annual_cap) / prev.annual_cap) * 100

        record = PriceCapRecord(
            date=date,
            annual_cap=annual_cap,
            quarterly_change_pct=quarterly_change,
            source=source,
        )

        self.price_caps.append(record)
        self.price_caps.sort(key=lambda x: x.date)
        self._save_state()

        logger.info(
            f"Added price cap: {annual_cap} GBP effective {date.date()}"
            f" (change: {quarterly_change:.1f}%)" if quarterly_change else ""
        )

        return record

    def check_and_alert(self) -> Optional[Alert]:
        """
        Check current state and generate alert if needed.

        Returns:
            Alert object if threshold exceeded, None otherwise
        """
        if not self.price_caps:
            logger.warning("No price cap data available")
            return None

        latest = self.price_caps[-1]
        change = latest.quarterly_change_pct

        if change is None:
            # First record, no comparison possible
            return None

        # Determine alert level
        if change >= self.RED_THRESHOLD:
            level = AlertLevel.RED
            message = (
                f"ALERT: Price cap increased {change:.1f}% to {latest.annual_cap} GBP. "
                f"Intervention protocol should be triggered immediately."
            )
        elif change >= self.AMBER_THRESHOLD:
            level = AlertLevel.AMBER
            message = (
                f"WATCH: Price cap increased {change:.1f}% to {latest.annual_cap} GBP. "
                f"Monitor food bank demand closely."
            )
        elif change < 0:
            level = AlertLevel.GREEN
            message = (
                f"OK: Price cap decreased {abs(change):.1f}% to {latest.annual_cap} GBP. "
                f"Pressure on households should ease."
            )
        else:
            level = AlertLevel.GREEN
            message = (
                f"OK: Price cap change minimal ({change:.1f}%). "
                f"No action required."
            )

        # Calculate intervention deadline
        deadline = latest.date + timedelta(days=self.INTERVENTION_WINDOW_MONTHS * 30)

        alert = Alert(
            timestamp=datetime.now(),
            level=level,
            message=message,
            price_cap=latest,
            intervention_deadline=deadline,
        )

        self.alerts.append(alert)
        logger.info(f"[{level.value.upper()}] {message}")

        return alert

    def get_status_report(self) -> str:
        """
        Generate a human-readable status report.

        Returns:
            Formatted status report string
        """
        if not self.price_caps:
            return "No data available. Add price cap records to begin monitoring."

        latest = self.price_caps[-1]
        alert = self.check_and_alert()

        lines = [
            "=" * 60,
            "ENERGY PRICE CAP MONITOR - STATUS REPORT",
            "=" * 60,
            f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "CURRENT STATUS",
            "-" * 40,
            f"Latest price cap:     {latest.annual_cap:.0f} GBP/year",
            f"Effective date:       {latest.date.strftime('%Y-%m-%d')}",
            f"Quarterly change:     {latest.quarterly_change_pct:+.1f}%"
            if latest.quarterly_change_pct else "Quarterly change:     N/A (first record)",
            "",
        ]

        if alert:
            lines.extend([
                f"ALERT LEVEL:          [{alert.level.value.upper()}]",
                f"Message:              {alert.message}",
                "",
            ])

            if alert.level in (AlertLevel.AMBER, AlertLevel.RED):
                lines.extend([
                    "INTERVENTION TIMELINE",
                    "-" * 40,
                    f"Price cap effective:  {latest.date.strftime('%Y-%m-%d')}",
                    f"Intervention window:  {self.INTERVENTION_WINDOW_MONTHS} months",
                    f"Deadline:             {alert.intervention_deadline.strftime('%Y-%m-%d')}",
                    "",
                ])

        # Historical data
        if len(self.price_caps) > 1:
            lines.extend([
                "HISTORICAL DATA",
                "-" * 40,
            ])
            for pc in self.price_caps[-5:]:  # Last 5 records
                change_str = f"{pc.quarterly_change_pct:+.1f}%" if pc.quarterly_change_pct else "N/A"
                lines.append(f"  {pc.date.strftime('%Y-%m-%d')}: {pc.annual_cap:.0f} GBP ({change_str})")

        lines.append("=" * 60)

        return "\n".join(lines)


def check_price_cap(data_dir: Path = Path("./data/monitor")) -> Alert:
    """
    Quick check of current price cap status.

    This is a convenience function for command-line use.

    Args:
        data_dir: Directory containing monitor state

    Returns:
        Current alert status
    """
    monitor = EnergyPriceMonitor(data_dir=data_dir)
    print(monitor.get_status_report())
    return monitor.check_and_alert()


# Historical price cap data for initialization
HISTORICAL_PRICE_CAPS = [
    # (date, annual_cap_gbp, source)
    (datetime(2020, 10, 1), 1042, "ofgem"),
    (datetime(2021, 4, 1), 1138, "ofgem"),
    (datetime(2021, 10, 1), 1277, "ofgem"),
    (datetime(2022, 4, 1), 1971, "ofgem"),  # +54% - major shock
    (datetime(2022, 10, 1), 2500, "ofgem"),  # EPG intervention
    (datetime(2023, 1, 1), 2500, "ofgem"),
    (datetime(2023, 4, 1), 2500, "ofgem"),
    (datetime(2023, 7, 1), 2074, "ofgem"),
    (datetime(2023, 10, 1), 1834, "ofgem"),
    (datetime(2024, 1, 1), 1928, "ofgem"),
    (datetime(2024, 4, 1), 1690, "ofgem"),
    (datetime(2024, 7, 1), 1568, "ofgem"),
    (datetime(2024, 10, 1), 1717, "ofgem"),
]


def initialize_monitor(data_dir: Path = Path("./data/monitor")) -> EnergyPriceMonitor:
    """
    Initialize monitor with historical price cap data.

    Args:
        data_dir: Directory for state persistence

    Returns:
        Initialized EnergyPriceMonitor
    """
    monitor = EnergyPriceMonitor(data_dir=data_dir)

    if not monitor.price_caps:
        logger.info("Initializing with historical data...")
        for date, cap, source in HISTORICAL_PRICE_CAPS:
            monitor.add_price_cap(date=date, annual_cap=cap, source=source)

    return monitor
