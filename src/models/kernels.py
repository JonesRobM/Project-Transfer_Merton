"""
Vulnerability Kernel Implementations

This module provides parameterized kernel functions that model the
impulse response of the social safety net to economic shocks.

Physical Interpretation:
    When an economic shock occurs (e.g., energy price spike), its
    effects propagate through the system with a characteristic delay
    and shape. The kernel h(τ) captures this transfer function:

    - Delta kernel: Instantaneous effect at fixed lag
    - Exponential kernel: Immediate onset with gradual decay
    - Gamma kernel: Delayed peak followed by decay (most realistic)

Statistical Note:
    The Gamma kernel is preferred because it captures the empirical
    observation that food insecurity peaks ~2-3 months after a shock,
    rather than immediately.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Any

import numpy as np
from numpy.typing import NDArray
from scipy import signal
from scipy.stats import gamma as gamma_dist
from scipy.optimize import minimize_scalar, minimize


@dataclass
class KernelFitResult:
    """Results from fitting a kernel to observed data.

    Attributes:
        kernel_name: Name of the kernel type
        parameters: Dict of fitted parameter values
        r_squared: Coefficient of determination (variance explained)
        rmse: Root mean squared error
        aic: Akaike Information Criterion (for model comparison)
        predicted: Array of predicted values
    """
    kernel_name: str
    parameters: Dict[str, float]
    r_squared: float
    rmse: float
    aic: float
    predicted: NDArray[np.float64]


class VulnerabilityKernel(ABC):
    """
    Abstract base class for vulnerability kernels.

    A vulnerability kernel h(τ) represents the impulse response of the
    social system to an economic shock. It describes how a delta-function
    shock at t=0 propagates to affect food insecurity at later times.

    The kernel must satisfy:
        - h(τ) >= 0 for all τ (non-negative response)
        - ∫h(τ)dτ = 1 (normalized to preserve scale)
        - h(τ) = 0 for τ < 0 (causal - no anticipation of shocks)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this kernel type."""
        pass

    @property
    @abstractmethod
    def n_parameters(self) -> int:
        """Number of free parameters in this kernel."""
        pass

    @abstractmethod
    def evaluate(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate the kernel at given time points.

        Args:
            t: Array of time points (in months)

        Returns:
            Array of kernel values, normalized to sum to 1
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, float]:
        """Return current parameter values as a dictionary."""
        pass

    def convolve(
        self,
        shock_signal: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Convolve the shock signal with this kernel.

        This computes the predicted response D(t) = (S * h)(t).

        Args:
            shock_signal: Array of shock values over time

        Returns:
            Array of predicted response values (same length as input)
        """
        n = len(shock_signal)
        t = np.arange(n, dtype=np.float64)
        kernel = self.evaluate(t)

        # Full convolution, then truncate to original length
        result = signal.convolve(shock_signal, kernel, mode="full")[:n]

        return result

    def fit(
        self,
        shock_signal: NDArray[np.float64],
        response_signal: NDArray[np.float64],
    ) -> KernelFitResult:
        """
        Fit kernel parameters to minimize prediction error.

        Uses scipy.optimize to find parameters that minimize the
        sum of squared errors between predicted and observed response.

        Args:
            shock_signal: Array of shock values (predictor)
            response_signal: Array of response values (target)

        Returns:
            KernelFitResult with fitted parameters and diagnostics
        """
        # Normalize signals for stable optimization
        shock_norm = (shock_signal - shock_signal.mean()) / shock_signal.std()
        response_norm = (response_signal - response_signal.mean()) / response_signal.std()

        # Subclasses must implement _fit_parameters
        self._fit_parameters(shock_norm, response_norm)

        # Compute predictions on normalized data
        predicted_norm = self.convolve(shock_norm)

        # Scale predictions to match response scale
        valid_mask = ~np.isnan(predicted_norm) & ~np.isnan(response_norm)
        if valid_mask.sum() < 2:
            # Not enough data
            return KernelFitResult(
                kernel_name=self.name,
                parameters=self.get_parameters(),
                r_squared=0.0,
                rmse=float("inf"),
                aic=float("inf"),
                predicted=np.full_like(response_signal, np.nan),
            )

        # Linear regression to scale predictions
        pred_valid = predicted_norm[valid_mask]
        resp_valid = response_norm[valid_mask]

        slope, intercept = np.polyfit(pred_valid, resp_valid, 1)
        predicted_scaled = slope * predicted_norm + intercept

        # De-normalize predictions
        predicted = predicted_scaled * response_signal.std() + response_signal.mean()

        # Compute metrics
        ss_res = np.sum((response_signal[valid_mask] - predicted[valid_mask]) ** 2)
        ss_tot = np.sum((response_signal[valid_mask] - response_signal[valid_mask].mean()) ** 2)

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        rmse = np.sqrt(ss_res / valid_mask.sum())

        # AIC: -2*log(likelihood) + 2*k, assuming Gaussian errors
        n = valid_mask.sum()
        k = self.n_parameters + 2  # kernel params + scale + intercept
        aic = n * np.log(ss_res / n) + 2 * k

        return KernelFitResult(
            kernel_name=self.name,
            parameters=self.get_parameters(),
            r_squared=r_squared,
            rmse=rmse,
            aic=aic,
            predicted=predicted,
        )

    @abstractmethod
    def _fit_parameters(
        self,
        shock_norm: NDArray[np.float64],
        response_norm: NDArray[np.float64],
    ) -> None:
        """
        Internal method to fit parameters on normalized data.

        Subclasses must implement this to update their parameters
        based on the observed data.
        """
        pass


class DeltaKernel(VulnerabilityKernel):
    """
    Delta (impulse) kernel with fixed time lag.

    h(τ) = δ(τ - τ_lag)

    This is the simplest kernel, representing an instantaneous
    effect that occurs exactly τ_lag months after the shock.

    Physical Interpretation:
        Assumes all of the shock's impact manifests at a single
        point in time. Useful as a baseline but unrealistic for
        real-world propagation which is typically distributed.

    Parameters:
        lag: Time delay in months (τ_lag)
    """

    def __init__(self, lag: float = 3.0):
        """
        Initialize delta kernel.

        Args:
            lag: Time delay in months (default: 3.0)
        """
        self.lag = lag

    @property
    def name(self) -> str:
        return "Delta"

    @property
    def n_parameters(self) -> int:
        return 1

    def evaluate(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate delta kernel (discrete approximation).

        Returns a spike at the lag point, zero elsewhere.
        """
        kernel = np.zeros_like(t)
        lag_idx = int(round(self.lag))

        if 0 <= lag_idx < len(kernel):
            kernel[lag_idx] = 1.0

        return kernel

    def get_parameters(self) -> Dict[str, float]:
        return {"lag": self.lag}

    def _fit_parameters(
        self,
        shock_norm: NDArray[np.float64],
        response_norm: NDArray[np.float64],
    ) -> None:
        """Find optimal lag via grid search."""
        n = len(shock_norm)
        best_lag = 0
        best_corr = -np.inf

        for lag in range(min(7, n)):
            if lag == 0:
                shifted = shock_norm
            else:
                shifted = np.concatenate([np.zeros(lag), shock_norm[:-lag]])

            valid = ~np.isnan(shifted) & ~np.isnan(response_norm)
            if valid.sum() > 2:
                corr = np.corrcoef(shifted[valid], response_norm[valid])[0, 1]
                if corr > best_corr:
                    best_corr = corr
                    best_lag = lag

        self.lag = float(best_lag)


class ExponentialKernel(VulnerabilityKernel):
    """
    Exponential decay kernel.

    h(τ) = λ * exp(-λτ) for τ >= 0

    The shock's impact begins immediately and decays exponentially.
    The decay rate λ determines how quickly the effect fades.

    Physical Interpretation:
        Represents a "leaky integrator" model where the shock's
        effect accumulates and then gradually dissipates. Common
        in models of memory or stress response.

    Parameters:
        decay_rate: Exponential decay rate λ (larger = faster decay)
    """

    def __init__(self, decay_rate: float = 0.3):
        """
        Initialize exponential kernel.

        Args:
            decay_rate: Decay rate λ (default: 0.3, ~3 month half-life)
        """
        self.decay_rate = decay_rate

    @property
    def name(self) -> str:
        return "Exponential"

    @property
    def n_parameters(self) -> int:
        return 1

    def evaluate(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate exponential decay kernel."""
        kernel = np.exp(-self.decay_rate * t)
        kernel[t < 0] = 0

        # Normalize
        if kernel.sum() > 0:
            kernel = kernel / kernel.sum()

        return kernel

    def get_parameters(self) -> Dict[str, float]:
        return {"decay_rate": self.decay_rate}

    def _fit_parameters(
        self,
        shock_norm: NDArray[np.float64],
        response_norm: NDArray[np.float64],
    ) -> None:
        """Find optimal decay rate via optimization."""

        def objective(rate: float) -> float:
            if rate <= 0.01 or rate > 2.0:
                return 1e10

            self.decay_rate = rate
            pred = self.convolve(shock_norm)

            valid = ~np.isnan(pred) & ~np.isnan(response_norm)
            if valid.sum() < 2:
                return 1e10

            # Use correlation as objective (maximize -> minimize negative)
            corr = np.corrcoef(pred[valid], response_norm[valid])[0, 1]
            return -corr if not np.isnan(corr) else 1e10

        result = minimize_scalar(objective, bounds=(0.05, 1.5), method="bounded")
        self.decay_rate = result.x


class GammaKernel(VulnerabilityKernel):
    """
    Gamma distribution kernel.

    h(τ) = Gamma(τ; shape, scale)

    The Gamma distribution produces a delayed peak, making it ideal
    for modeling processes with inherent delays (like household
    budget adjustment after a price shock).

    Physical Interpretation:
        The shape parameter controls whether the response is:
        - Monotonic decay (shape < 1)
        - Immediate peak then decay (shape = 1, reduces to exponential)
        - Delayed peak then decay (shape > 1, most realistic)

        The scale parameter controls the timescale of the response.
        Peak occurs at (shape - 1) * scale months for shape > 1.

    Parameters:
        shape: Shape parameter α (controls peak timing)
        scale: Scale parameter β (controls spread/duration)
    """

    def __init__(self, shape: float = 2.0, scale: float = 1.5):
        """
        Initialize gamma kernel.

        Args:
            shape: Shape parameter α (default: 2.0)
            scale: Scale parameter β (default: 1.5)
        """
        self.shape = shape
        self.scale = scale

    @property
    def name(self) -> str:
        return "Gamma"

    @property
    def n_parameters(self) -> int:
        return 2

    @property
    def peak_time(self) -> float:
        """Time of peak response in months."""
        if self.shape >= 1:
            return (self.shape - 1) * self.scale
        return 0.0

    def evaluate(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate gamma distribution kernel."""
        kernel = gamma_dist.pdf(t, a=self.shape, scale=self.scale)
        kernel[t < 0] = 0

        # Normalize
        if kernel.sum() > 0:
            kernel = kernel / kernel.sum()

        return kernel

    def get_parameters(self) -> Dict[str, float]:
        return {
            "shape": self.shape,
            "scale": self.scale,
            "peak_time": self.peak_time,
        }

    def _fit_parameters(
        self,
        shock_norm: NDArray[np.float64],
        response_norm: NDArray[np.float64],
    ) -> None:
        """Find optimal shape and scale via optimization."""

        def objective(params: NDArray[np.float64]) -> float:
            shape, scale = params

            if shape < 0.5 or shape > 10 or scale < 0.5 or scale > 5:
                return 1e10

            self.shape = shape
            self.scale = scale
            pred = self.convolve(shock_norm)

            valid = ~np.isnan(pred) & ~np.isnan(response_norm)
            if valid.sum() < 2:
                return 1e10

            corr = np.corrcoef(pred[valid], response_norm[valid])[0, 1]
            return -corr if not np.isnan(corr) else 1e10

        # Grid search for starting point
        best_params = np.array([2.0, 1.5])
        best_obj = objective(best_params)

        for shape in [1.0, 2.0, 3.0, 4.0]:
            for scale in [0.5, 1.0, 1.5, 2.0]:
                params = np.array([shape, scale])
                obj = objective(params)
                if obj < best_obj:
                    best_obj = obj
                    best_params = params

        # Refine with optimization
        result = minimize(
            objective,
            best_params,
            method="L-BFGS-B",
            bounds=[(0.5, 10.0), (0.5, 5.0)],
        )

        self.shape = result.x[0]
        self.scale = result.x[1]


def fit_kernels(
    shock_signal: NDArray[np.float64],
    response_signal: NDArray[np.float64],
) -> List[KernelFitResult]:
    """
    Fit all kernel types to the observed data.

    This is a convenience function that fits Delta, Exponential,
    and Gamma kernels to the same data and returns results for
    model comparison.

    Args:
        shock_signal: Array of shock values (e.g., energy prices)
        response_signal: Array of response values (e.g., food insecurity)

    Returns:
        List of KernelFitResult objects, one per kernel type
    """
    kernels: List[VulnerabilityKernel] = [
        DeltaKernel(),
        ExponentialKernel(),
        GammaKernel(),
    ]

    results = []
    for kernel in kernels:
        result = kernel.fit(shock_signal, response_signal)
        results.append(result)

    return results


def select_best_kernel(
    results: List[KernelFitResult],
    criterion: str = "r_squared",
) -> KernelFitResult:
    """
    Select the best-fitting kernel from a list of results.

    Args:
        results: List of KernelFitResult objects
        criterion: Selection criterion ("r_squared", "aic", or "rmse")

    Returns:
        The KernelFitResult with the best score

    Raises:
        ValueError: If criterion is not recognized or results is empty
    """
    if not results:
        raise ValueError("No results to compare")

    if criterion == "r_squared":
        # Higher is better
        return max(results, key=lambda r: r.r_squared)
    elif criterion == "aic":
        # Lower is better
        return min(results, key=lambda r: r.aic)
    elif criterion == "rmse":
        # Lower is better
        return min(results, key=lambda r: r.rmse)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
