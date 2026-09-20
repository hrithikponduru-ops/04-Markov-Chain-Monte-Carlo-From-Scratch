"""Conceptual diagrams for README.md (data-independent). Writes to figures/.

Run from this folder: python make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

from mcmc.metropolis import metropolis_hastings
from mcmc.models import log_gaussian

INK, BLUE, GREEN, RED, GREY, AMBER = "#1f2937", "#2563eb", "#059669", "#dc2626", "#9ca3af", "#d97706"
FIG = Path("figures")


def fig_metropolis_step():
    """Uphill proposals are always accepted; downhill ones with probability p(theta')/p(theta)."""
    density = lambda t: 0.6 * np.exp(-0.5 * ((t - 1) / 0.7) ** 2) + 0.4 * np.exp(-0.5 * ((t + 1.5) / 0.5) ** 2)  # noqa: E731
    grid = np.linspace(-3.5, 3.5, 400)
    theta = 0.2
    uphill, downhill = 0.9, 2.3

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(grid, density(grid), color=INK, lw=2)
    ax.fill_between(grid, density(grid), color=GREY, alpha=0.15)
    ax.plot([theta], [density(theta)], "o", color=INK, ms=9, zorder=4)
    ax.text(theta, density(theta) + 0.045, r"current $\theta$", ha="center", fontsize=10)
    for prop, colour, label, offset in ((uphill, GREEN, "uphill", (0.0, 0.09)), (downhill, RED, "downhill", (0.0, -0.075))):
        ratio = density(prop) / density(theta)
        ax.annotate("", xy=(prop, density(prop)), xytext=(theta, density(theta)),
                    arrowprops=dict(arrowstyle="->", color=colour, lw=2, connectionstyle="arc3,rad=-0.3"))
        ax.plot([prop], [density(prop)], "o", color=colour, ms=9, zorder=4)
        alpha_text = "1 (always)" if ratio >= 1 else f"{ratio:.2f}"
        ax.text(prop + offset[0], density(prop) + offset[1], f"{label} proposal\n" + r"$\alpha$ = " + alpha_text,
                ha="center", va="bottom" if offset[1] > 0 else "top", color=colour, fontsize=9)
    ax.set_ylim(-0.12, 0.85)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"target density $p(\theta)$ (normalising constant unknown)")
    ax.set_title(r"Accept with probability $\alpha = \min\left(1, p(\theta')/p(\theta)\right)$")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "metropolis_step.png", dpi=150)
    plt.close(fig)


def fig_step_size():
    """Too small a step accepts everything but crawls; too large rejects almost everything."""
    target = lambda t: log_gaussian(t, np.zeros(1), np.eye(1))  # noqa: E731
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True, sharey=True)
    for ax, step, colour, verdict in zip(axes, (0.1, 2.4, 40.0), (AMBER, GREEN, RED),
                                         ("too small: high acceptance, but the chain crawls",
                                          "about right: explores the whole distribution",
                                          "too large: almost every proposal is rejected")):
        chain = metropolis_hastings(target, np.array([-2.5]), step, 1_000, np.random.default_rng(0))
        ax.plot(chain.samples[:, 0], lw=0.8, color=colour)
        ax.set_title(f"step size {step}: acceptance rate {chain.acceptance_rate:.2f}. {verdict}", fontsize=10, loc="left")
        ax.axhline(0, color=GREY, lw=0.8, ls="--")
        ax.set_ylabel(r"$\theta$")
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("step")
    fig.suptitle("The same target N(0, 1), three proposal step sizes, 1 000 steps each", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "step_size.png", dpi=150)
    plt.close(fig)


def fig_detailed_balance():
    """Probability flow between the three states of the worked example is balanced pair by pair."""
    pi = np.array([0.2, 0.5, 0.3])
    centres = [(0.0, 0.0), (2.2, 0.0), (4.4, 0.0)]
    flows = {(0, 1): 0.1, (1, 0): 0.1, (1, 2): 0.15, (2, 1): 0.15}

    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, ((cx, cy), p) in enumerate(zip(centres, pi)):
        ax.add_patch(Circle((cx, cy), 0.28 + 0.6 * p, color=BLUE, alpha=0.25))
        ax.text(cx, cy, f"state {i}\n" + r"$\pi_i$ = " + f"{p}", ha="center", va="center", fontsize=10)
    for (i, j), flow in flows.items():
        (x0, _), (x1, _) = centres[i], centres[j]
        direction = np.sign(x1 - x0)
        r0, r1 = 0.28 + 0.6 * pi[i], 0.28 + 0.6 * pi[j]
        y = 0.25 * direction
        ax.add_patch(FancyArrowPatch((x0 + direction * r0, y), (x1 - direction * r1, y),
                                     arrowstyle="-|>", mutation_scale=16, color=GREEN if direction > 0 else RED, lw=2))
        ax.text((x0 + x1) / 2, y + 0.18 * direction, r"$\pi_{%d} P_{%d%d}$ = %.2f" % (i, i, j, flow),
                ha="center", va="center", fontsize=9, color=GREEN if direction > 0 else RED)
    ax.text(2.2, -1.25, "Detailed balance: the flow each way across every link is equal, so nothing accumulates "
            "and $\pi_i$ never changes.", ha="center", fontsize=10)
    ax.set_xlim(-1.0, 5.6)
    ax.set_ylim(-1.5, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "detailed_balance.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig_metropolis_step()
    fig_step_size()
    fig_detailed_balance()
    print("wrote figures/")