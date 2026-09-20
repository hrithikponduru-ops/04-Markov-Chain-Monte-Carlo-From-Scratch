"""Two applications of the sampler and their figures.

1. The coin problem, where the posterior is known exactly (Beta(9, 5)).
2. Hubble's 1929 data: Bayesian linear regression of recession velocity on
   distance, giving a posterior for the Hubble constant with uncertainty.

Run from this folder:  python run.py
"""

import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mcmc.diagnostics import autocorrelation, effective_sample_size, gelman_rubin_rhat, integrated_autocorrelation_time
from mcmc.metropolis import metropolis_hastings
from mcmc.models import RegressionPriors, log_beta_binomial_posterior, log_linear_regression_posterior

SEED = 0
COIN_K, COIN_N, COIN_A, COIN_B = 7, 10, 2.0, 2.0
COIN_STEPS, COIN_STEP_SIZE, COIN_BURN = 100_000, 0.25, 2_000
N_CHAINS, CHAIN_STEPS, BURN_IN = 4, 20_000, 5_000
STEP_SIZES = np.array([90.0, 110.0, 0.2])  # slope, intercept, log sigma
PRIORS = RegressionPriors(slope_sd=2_000.0, intercept_sd=2_000.0)
PARAM_NAMES = ("slope (km/s per Mpc)", "intercept (km/s)", "log sigma")
BLUE, GREEN, RED, GREY, INK = "#2563eb", "#059669", "#dc2626", "#9ca3af", "#1f2937"
CHAIN_COLOURS = (BLUE, GREEN, RED, "#d97706")
FIG = Path("figures")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    coin_problem()
    hubble_regression()


# -----------------------------------------------------------------------------
# 1. Coin problem
# -----------------------------------------------------------------------------
def beta_pdf(theta: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    normaliser = math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)
    return theta ** (alpha - 1) * (1 - theta) ** (beta - 1) / normaliser


def beta_quantile(p: float, alpha: float, beta: float) -> float:
    """Invert the CDF numerically: integrate the pdf on a fine grid (trapezoid rule)."""
    grid = np.linspace(0, 1, 100_001)
    pdf = beta_pdf(grid, alpha, beta)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    return float(np.interp(p, cdf / cdf[-1], grid))


def coin_problem() -> None:
    print("-" * 64)
    print(f"COIN: {COIN_K} heads in {COIN_N} flips, prior Beta({COIN_A:.0f}, {COIN_B:.0f})")
    print("-" * 64)
    
    alpha, beta = COIN_A + COIN_K, COIN_B + COIN_N - COIN_K
    rng = np.random.default_rng(SEED)
    chain = metropolis_hastings(
        lambda t: log_beta_binomial_posterior(t[0], COIN_K, COIN_N, COIN_A, COIN_B),
        np.array([0.5]), COIN_STEP_SIZE, COIN_STEPS, rng,
    )
    
    x = chain.samples[COIN_BURN:, 0]
    n_eff = effective_sample_size(x)
    true_mean = alpha / (alpha + beta)
    true_var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
    
    print(f"acceptance rate {chain.acceptance_rate:.3f}, kept {len(x)} samples, n_eff = {n_eff:.0f}")
    print(f"{'quantity':<14}{'MCMC':>10}{'exact Beta(9,5)':>18}{'MC std error':>14}")
    print(f"{'mean':<14}{x.mean():>10.4f}{true_mean:>18.4f}{np.sqrt(true_var / n_eff):>14.4f}")
    print(f"{'variance':<14}{x.var():>10.5f}{true_var:>18.5f}")
    for p in (0.05, 0.5, 0.95):
        print(f"{f'p={p:.0%} quantile':<14}{np.quantile(x, p):>10.4f}{beta_quantile(p, alpha, beta):>18.4f}")
    
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(x, bins=60, density=True, color=BLUE, alpha=0.5, label="MCMC samples")
    grid = np.linspace(0.01, 0.99, 400)
    ax.plot(grid, beta_pdf(grid, alpha, beta), color=INK, lw=2, label="exact posterior Beta(9, 5)")
    ax.axvline(true_mean, color=RED, ls="--", lw=1.2, label=f"exact mean {true_mean:.4f}")
    ax.axvline(x.mean(), color=GREEN, ls=":", lw=1.5, label=f"MCMC mean {x.mean():.4f}")
    ax.set_xlabel(r"$\theta$ = probability of heads")
    ax.set_ylabel("posterior density")
    ax.set_title(f"{COIN_K} heads in {COIN_N} flips: the sampler versus the closed form")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "coin_posterior.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# 2. Hubble 1929
# -----------------------------------------------------------------------------
def hubble_regression() -> None:
    print("\n" + "=" * 64)
    print("HUBBLE 1929: velocity = intercept + slope * distance")
    print("=" * 64)
    
    data = np.loadtxt("data/hubble_1929.csv", delimiter=",", skiprows=1)
    x, y = data[:, 0], data[:, 1]
    
    design = np.column_stack([x, np.ones_like(x)])
    ols_slope, ols_intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    ols_sigma = np.sqrt(np.sum((y - design @ [ols_slope, ols_intercept]) ** 2) / (len(y) - 2))
    print(f"OLS: slope {ols_slope:.1f}, intercept {ols_intercept:.1f}, residual sd {ols_sigma:.1f}")

    log_target = lambda p: log_linear_regression_posterior(p, x, y, PRIORS)  # noqa: E731
    rng = np.random.default_rng(SEED)
    
    starts = np.array([ols_slope, ols_intercept, np.log(ols_sigma)]) + rng.normal(size=(N_CHAINS, 3)) * [150, 150, 0.5]
    t0 = time.perf_counter()
    
    chains = [metropolis_hastings(log_target, s, STEP_SIZES, BURN_IN + CHAIN_STEPS, rng) for s in starts]
    elapsed = time.perf_counter() - t0
    
    kept = np.stack([c.samples[BURN_IN:] for c in chains])  # (m, n, 3)
    acceptance = np.mean([c.acceptance_rate for c in chains])
    
    print(f"{N_CHAINS} chains x {CHAIN_STEPS} kept steps (+{BURN_IN} burn-in each) in {elapsed:.1f}s, "
          f"mean acceptance {acceptance:.3f}")

    pooled = kept.reshape(-1, 3)
    
    print(f"\n{'parameter':<24}{'post. mean':>11}{'post. sd':>10}{'95% interval':>22}{'R-hat':>8}{'n_eff':>8}")
    for j, name in enumerate(PARAM_NAMES):
        col = pooled[:, j]
        lo, hi = np.quantile(col, [0.025, 0.975])
        rhat = gelman_rubin_rhat(kept[:, :, j])
        n_eff = sum(effective_sample_size(kept[m, :, j]) for m in range(N_CHAINS))
        print(f"{name:<24}{col.mean():>11.2f}{col.std():>10.2f}{f'[{lo:.1f}, {hi:.1f}]':>22}{rhat:>8.4f}{n_eff:>8.0f}")
        
    sigma = np.exp(pooled[:, 2])
    print(f"{'sigma (km/s)':<24}{sigma.mean():>11.1f}{sigma.std():>10.1f}"
          f"{f'[{np.quantile(sigma, 0.025):.0f}, {np.quantile(sigma, 0.975):.0f}]':>22}")

    fig_trace(kept)
    fig_autocorrelation(kept)
    fig_posterior(pooled, ols_slope, ols_intercept)
    fig_fit(x, y, pooled, ols_slope, ols_intercept, rng)


def fig_trace(kept: np.ndarray) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)
    
    for j, ax in enumerate(axes):
        for m in range(N_CHAINS):
            ax.plot(kept[m, :3_000, j], lw=0.5, color=CHAIN_COLOURS[m], alpha=0.8)
        ax.set_ylabel(PARAM_NAMES[j], fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        
    axes[-1].set_xlabel("step after burn-in")
    axes[0].set_title(f"Four chains from dispersed starts, first 3 000 kept steps. "
                      f"R-hat: {', '.join(f'{gelman_rubin_rhat(kept[:, :, j]):.3f}' for j in range(3))}", fontsize=10)
                      
    fig.tight_layout()
    fig.savefig(FIG / "trace.png", dpi=150)
    plt.close(fig)


def fig_autocorrelation(kept: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    
    for j, colour in enumerate((BLUE, GREEN, RED)):
        chain = kept[0, :, j]
        rho = autocorrelation(chain, max_lag=150)
        tau = integrated_autocorrelation_time(chain)
        ax.plot(rho, color=colour, lw=2, label=f"{PARAM_NAMES[j]}: $\\tau$ = {tau:.0f}")
        
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xlabel(r"lag $k$")
    ax.set_ylabel(r"autocorrelation $\rho_k$")
    ax.set_title(r"Successive samples are correlated: $n_{\mathrm{eff}} = n / \tau$")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    
    fig.tight_layout()
    fig.savefig(FIG / "autocorrelation.png", dpi=150)
    plt.close(fig)


def fig_posterior(pooled: np.ndarray, ols_slope: float, ols_intercept: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    
    for ax, j, ref, colour in ((axes[0], 0, ols_slope, BLUE), (axes[1], 1, ols_intercept, GREEN)):
        col = pooled[:, j]
        ax.hist(col, bins=60, density=True, color=colour, alpha=0.55)
        ax.axvline(ref, color=INK, ls="--", lw=1.2, label=f"OLS {ref:.0f}")
        lo, hi = np.quantile(col, [0.025, 0.975])
        ax.axvspan(lo, hi, color=colour, alpha=0.12, label=f"95 % interval [{lo:.0f}, {hi:.0f}]")
        ax.set_xlabel(PARAM_NAMES[j])
        ax.set_ylim(0, ax.get_ylim()[1] * 1.35)  # headroom so the legend clears the peak
        ax.legend(fontsize=8, loc="upper center")
        
    axes[0].set_title("Posterior of the Hubble constant", fontsize=10)
    axes[1].set_title("Posterior of the intercept", fontsize=10)
    
    ax = axes[2]
    idx = np.random.default_rng(1).choice(len(pooled), 4_000, replace=False)
    ax.scatter(pooled[idx, 0], pooled[idx, 1], s=3, alpha=0.3, color=BLUE)
    ax.scatter([ols_slope], [ols_intercept], s=60, color=RED, zorder=3, label="OLS")
    
    corr = np.corrcoef(pooled[:, 0], pooled[:, 1])[0, 1]
    ax.set_xlabel(PARAM_NAMES[0])
    ax.set_ylabel(PARAM_NAMES[1])
    ax.set_title(f"Joint posterior: correlation {corr:.2f}", fontsize=10)
    ax.legend(fontsize=8)
    
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        
    fig.tight_layout()
    fig.savefig(FIG / "posterior_hubble.png", dpi=150)
    plt.close(fig)


def fig_fit(x, y, pooled, ols_slope, ols_intercept, rng) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    grid = np.linspace(0, 2.2, 50)
    
    for i in rng.choice(len(pooled), 200, replace=False):
        ax.plot(grid, pooled[i, 1] + pooled[i, 0] * grid, color=BLUE, alpha=0.05, lw=1)
        
    ax.plot(grid, ols_intercept + ols_slope * grid, color=RED, lw=2, label=f"OLS: {ols_slope:.0f} km/s per Mpc")
    mean_slope, mean_intercept = pooled[:, 0].mean(), pooled[:, 1].mean()
    ax.plot(grid, mean_intercept + mean_slope * grid, color=INK, lw=1.5, ls="--",
            label=f"posterior mean: {mean_slope:.0f} km/s per Mpc")
            
    ax.plot([], [], color=BLUE, alpha=0.5, label="200 lines drawn from the posterior")
    ax.scatter(x, y, color=INK, s=28, zorder=3, label="Hubble's 24 nebulae (1929)")
    
    ax.set_xlabel("distance (Mpc)")
    ax.set_ylabel("recession velocity (km/s)")
    ax.set_title("Hubble's law with the uncertainty the data actually supports")
    ax.spines[["top", "right"]].set_visible(False)
    
    fig.tight_layout()
    fig.savefig(FIG / "hubble_fit.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()