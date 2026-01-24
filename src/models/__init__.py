"""
Systemic Vulnerability Models

This module provides implementations of the vulnerability kernel h(τ)
that characterizes the transfer function between economic shocks and
food insecurity response.

Mathematical Framework:
    The food insecurity response D(t) is modeled as a convolution of
    the economic shock signal S(t) with the vulnerability kernel h(τ):

        D(t) = (S * h)(t) + η(t)

    Where:
        - S(t): Economic shock signal (e.g., energy price rise)
        - h(τ): Vulnerability kernel (impulse response function)
        - η(t): Noise/unexplained variance
        - τ: Lag time between shock and response
"""

from .kernels import (
    VulnerabilityKernel,
    DeltaKernel,
    ExponentialKernel,
    GammaKernel,
    fit_kernels,
    select_best_kernel,
)

from .bayesian import (
    check_pymc_available,
    estimate_lag_bayesian,
    estimate_kernel_bayesian,
    BayesianLagResult,
)

__all__ = [
    # Kernels
    "VulnerabilityKernel",
    "DeltaKernel",
    "ExponentialKernel",
    "GammaKernel",
    "fit_kernels",
    "select_best_kernel",
    # Bayesian
    "check_pymc_available",
    "estimate_lag_bayesian",
    "estimate_kernel_bayesian",
    "BayesianLagResult",
]
