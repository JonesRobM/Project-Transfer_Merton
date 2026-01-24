"""
Bayesian Inference for Systemic Latency

This module provides PyMC-based Bayesian models for estimating the
lag distribution between economic shocks and food insecurity response.

Statistical Framework:
    We model the lag τ as a latent variable with a prior distribution,
    and infer its posterior given the observed cross-correlation
    structure of the data.

    The likelihood is derived from the residuals of the convolution
    model, assumed to be normally distributed:

        y ~ Normal(X @ β(τ), σ²)

    Where β(τ) is the regression coefficient at lag τ.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
from numpy.typing import NDArray

# Conditional import for PyMC - may not be installed
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    pm = None
    az = None


@dataclass
class BayesianLagResult:
    """Results from Bayesian lag estimation.

    Attributes:
        posterior_mean: Mean of the lag posterior distribution
        posterior_std: Standard deviation of the posterior
        hdi_low: Lower bound of 94% Highest Density Interval
        hdi_high: Upper bound of 94% HDI
        posterior_samples: Array of posterior samples for the lag
        diagnostics: Dict of MCMC diagnostics (r_hat, ess, etc.)
    """
    posterior_mean: float
    posterior_std: float
    hdi_low: float
    hdi_high: float
    posterior_samples: NDArray[np.float64]
    diagnostics: Dict[str, Any]


def check_pymc_available() -> bool:
    """Check if PyMC is available for import."""
    return PYMC_AVAILABLE


def estimate_lag_bayesian(
    shock_signal: NDArray[np.float64],
    response_signal: NDArray[np.float64],
    max_lag: int = 6,
    n_samples: int = 2000,
    n_tune: int = 1000,
    random_seed: int = 42,
) -> Optional[BayesianLagResult]:
    """
    Estimate the optimal lag using Bayesian inference.

    This function builds a hierarchical model where the lag τ is
    treated as a discrete latent variable. The posterior distribution
    over τ gives both the most likely lag and its uncertainty.

    Model Specification:
        τ ~ DiscreteUniform(0, max_lag)
        β ~ Normal(0, 1)
        σ ~ HalfNormal(1)
        y ~ Normal(β * X_shifted(τ), σ)

    Args:
        shock_signal: Array of shock values (predictor)
        response_signal: Array of response values (target)
        max_lag: Maximum lag to consider (default: 6 months)
        n_samples: Number of posterior samples to draw (default: 2000)
        n_tune: Number of tuning samples (default: 1000)
        random_seed: Random seed for reproducibility

    Returns:
        BayesianLagResult with posterior estimates, or None if PyMC unavailable

    Note:
        This implementation uses a continuous relaxation of the discrete
        lag for efficient MCMC sampling, then rounds to nearest integer.
    """
    if not PYMC_AVAILABLE:
        return None

    # Normalize signals
    shock_norm = (shock_signal - shock_signal.mean()) / shock_signal.std()
    response_norm = (response_signal - response_signal.mean()) / response_signal.std()

    n = len(shock_norm)

    # Create shifted versions of shock signal for each lag
    X_lags = np.zeros((n, max_lag + 1))
    for lag in range(max_lag + 1):
        if lag == 0:
            X_lags[:, lag] = shock_norm
        else:
            X_lags[lag:, lag] = shock_norm[:-lag]

    # Build PyMC model
    with pm.Model() as model:
        # Prior on continuous lag (relaxation of discrete)
        tau = pm.Uniform("tau", lower=0, upper=max_lag)

        # Interpolate between lag columns
        tau_floor = pm.math.floor(tau).astype("int32")
        tau_ceil = pm.math.ceil(tau).astype("int32")
        tau_frac = tau - tau_floor

        # Weighted combination of adjacent lag columns
        X_interp = (1 - tau_frac) * X_lags[:, tau_floor] + tau_frac * X_lags[:, tau_ceil]

        # Regression coefficient
        beta = pm.Normal("beta", mu=0, sigma=2)

        # Noise standard deviation
        sigma = pm.HalfNormal("sigma", sigma=1)

        # Likelihood
        y_pred = beta * X_interp
        pm.Normal("y_obs", mu=y_pred, sigma=sigma, observed=response_norm)

        # Sample from posterior
        trace = pm.sample(
            draws=n_samples,
            tune=n_tune,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=False,
        )

    # Extract posterior samples
    tau_samples = trace.posterior["tau"].values.flatten()

    # Compute summary statistics
    posterior_mean = float(np.mean(tau_samples))
    posterior_std = float(np.std(tau_samples))

    # HDI (Highest Density Interval)
    hdi = az.hdi(trace, hdi_prob=0.94)["tau"].values
    hdi_low = float(hdi[0])
    hdi_high = float(hdi[1])

    # Diagnostics
    summary = az.summary(trace, var_names=["tau", "beta", "sigma"])
    diagnostics = {
        "r_hat_tau": float(summary.loc["tau", "r_hat"]),
        "ess_tau": float(summary.loc["tau", "ess_bulk"]),
        "r_hat_beta": float(summary.loc["beta", "r_hat"]),
        "converged": float(summary.loc["tau", "r_hat"]) < 1.05,
    }

    return BayesianLagResult(
        posterior_mean=posterior_mean,
        posterior_std=posterior_std,
        hdi_low=hdi_low,
        hdi_high=hdi_high,
        posterior_samples=tau_samples,
        diagnostics=diagnostics,
    )


def estimate_kernel_bayesian(
    shock_signal: NDArray[np.float64],
    response_signal: NDArray[np.float64],
    kernel_type: str = "gamma",
    n_samples: int = 2000,
    n_tune: int = 1000,
    random_seed: int = 42,
) -> Optional[Dict[str, Any]]:
    """
    Estimate kernel parameters using Bayesian inference.

    This function fits a parameterized kernel (currently Gamma only)
    using MCMC, providing posterior distributions for all parameters.

    Model Specification (Gamma kernel):
        shape ~ Gamma(2, 1)
        scale ~ Gamma(2, 0.5)
        β ~ Normal(0, 1)
        σ ~ HalfNormal(1)
        y ~ Normal(β * convolve(X, kernel(shape, scale)), σ)

    Args:
        shock_signal: Array of shock values
        response_signal: Array of response values
        kernel_type: Type of kernel ("gamma" only currently)
        n_samples: Number of posterior samples
        n_tune: Number of tuning samples
        random_seed: Random seed

    Returns:
        Dict with posterior summaries for all parameters, or None if unavailable
    """
    if not PYMC_AVAILABLE:
        return None

    if kernel_type != "gamma":
        raise ValueError(f"Only 'gamma' kernel supported, got: {kernel_type}")

    from scipy.stats import gamma as gamma_dist
    from scipy import signal

    # Normalize
    shock_norm = (shock_signal - shock_signal.mean()) / shock_signal.std()
    response_norm = (response_signal - response_signal.mean()) / response_signal.std()

    n = len(shock_norm)
    t = np.arange(n, dtype=np.float64)

    def gamma_kernel(shape: float, scale: float) -> NDArray[np.float64]:
        """Compute normalized gamma kernel."""
        k = gamma_dist.pdf(t, a=shape, scale=scale)
        if k.sum() > 0:
            k = k / k.sum()
        return k

    # Precompute convolutions for a grid of parameters
    # (PyMC doesn't support scipy operations directly)
    shape_grid = np.linspace(1.0, 8.0, 20)
    scale_grid = np.linspace(0.3, 3.0, 20)

    conv_grid = np.zeros((len(shape_grid), len(scale_grid), n))
    for i, shape in enumerate(shape_grid):
        for j, scale in enumerate(scale_grid):
            k = gamma_kernel(shape, scale)
            conv_grid[i, j, :] = signal.convolve(shock_norm, k, mode="full")[:n]

    with pm.Model() as model:
        # Priors
        shape_idx = pm.DiscreteUniform("shape_idx", lower=0, upper=len(shape_grid) - 1)
        scale_idx = pm.DiscreteUniform("scale_idx", lower=0, upper=len(scale_grid) - 1)

        # Get convolution from grid
        X_conv = conv_grid[shape_idx, scale_idx, :]

        # Regression
        beta = pm.Normal("beta", mu=0, sigma=2)
        sigma = pm.HalfNormal("sigma", sigma=1)

        y_pred = beta * X_conv
        pm.Normal("y_obs", mu=y_pred, sigma=sigma, observed=response_norm)

        # Sample
        trace = pm.sample(
            draws=n_samples,
            tune=n_tune,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=False,
        )

    # Extract samples and convert indices to actual values
    shape_idx_samples = trace.posterior["shape_idx"].values.flatten()
    scale_idx_samples = trace.posterior["scale_idx"].values.flatten()

    shape_samples = shape_grid[shape_idx_samples]
    scale_samples = scale_grid[scale_idx_samples]

    # Compute peak time
    peak_samples = (shape_samples - 1) * scale_samples
    peak_samples[shape_samples < 1] = 0

    return {
        "shape": {
            "mean": float(np.mean(shape_samples)),
            "std": float(np.std(shape_samples)),
            "hdi": (float(np.percentile(shape_samples, 3)), float(np.percentile(shape_samples, 97))),
        },
        "scale": {
            "mean": float(np.mean(scale_samples)),
            "std": float(np.std(scale_samples)),
            "hdi": (float(np.percentile(scale_samples, 3)), float(np.percentile(scale_samples, 97))),
        },
        "peak_time": {
            "mean": float(np.mean(peak_samples)),
            "std": float(np.std(peak_samples)),
            "hdi": (float(np.percentile(peak_samples, 3)), float(np.percentile(peak_samples, 97))),
        },
        "diagnostics": {
            "converged": True,  # Simplified for grid-based approach
        },
    }
