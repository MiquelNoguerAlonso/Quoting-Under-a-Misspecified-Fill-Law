"""Generate the ten reproducible figures used in Papers II and III.

All panels are synthetic illustrations or deterministic evaluations of the
models verified by this package.  No market data are used.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.stats import norm, poisson


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
P2_FIGURES = ROOT / "paper_2" / "figures"
P3_FIGURES = ROOT / "paper_3" / "figures"
RESULTS_PATH = HERE / "verification_results.json"

sys.path.insert(0, str(HERE))
from order_book_foundations import event_objects  # noqa: E402
from partial_fills_latency import (  # noqa: E402
    compound_fill_pmf,
    delayed_value,
    queue_cell,
)


BLUE = "#1f4e79"
ORANGE = "#d97706"
GREEN = "#177245"
PURPLE = "#6b5b95"
RED = "#b03a2e"
GREY = "#68737d"
LIGHT_BLUE = "#d9e7f4"
LIGHT_ORANGE = "#f7dfc2"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.color": "#d7dce0",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.65,
            "lines.linewidth": 1.8,
            "savefig.dpi": 240,
        }
    )


def load_results() -> dict[str, dict]:
    payload = json.loads(RESULTS_PATH.read_text())
    return {row["check"]: row for row in payload["results"]}


def finish(fig: plt.Figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(f"({letter}) {title}", loc="left", fontweight="semibold")


def p2_queue_priority() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.05), constrained_layout=True)

    ranks = np.arange(0, 7)
    mu = 1.0
    for hazard, color in zip([0.25, 0.5, 1.0], [GREEN, ORANGE, BLUE]):
        value = (mu / (mu + hazard)) ** (ranks + 1)
        axes[0].plot(ranks, value, marker="o", color=color,
                     label=fr"$\beta+\xi={hazard:g}$")
    axes[0].set_xlabel("Lots ahead, $n$")
    axes[0].set_ylabel("Discounted execution probability")
    axes[0].set_xticks(ranks)
    axes[0].set_ylim(0, 0.84)
    axes[0].grid(axis="y")
    axes[0].legend(frameon=False)
    panel_title(axes[0], "a", "Priority value from the product formula")

    horizon = np.linspace(0, 6, 300)
    for n, color in zip(range(4), [BLUE, GREEN, ORANGE, PURPLE]):
        fill = poisson.sf(n, mu * horizon)
        axes[1].plot(horizon, fill, color=color, label=fr"$n={n}$")
    axes[1].set_xlabel("Deadline, $H$")
    axes[1].set_ylabel("Fill probability by $H$")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(axis="y")
    axes[1].legend(frameon=False, ncol=2)
    panel_title(axes[1], "b", "Erlang fill-time distributions")

    finish(fig, P2_FIGURES / "01_queue_priority.png")


def p2_shutdown_threshold() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.05), constrained_layout=True)
    lambda_u, kappa, k = 2.0, 1.2, 1.0
    critical = k * lambda_u / (math.e * (kappa - k))
    offsets = np.linspace(0, 9, 500)
    for multiplier, color in zip([0.5, 1.0, 1.5], [GREEN, BLUE, RED]):
        lambda_i = multiplier * critical
        reward = lambda_u * offsets * np.exp(-kappa * offsets)
        reward -= lambda_i / k * np.exp(-k * offsets)
        axes[0].plot(offsets, reward, color=color,
                     label=fr"$\lambda_i={multiplier:g}\lambda_i^\ast$")
    axes[0].axhline(0, color="#333333", linewidth=0.9)
    axes[0].axvline(1 / (kappa - k), color=GREY, linestyle=":",
                    label=r"binding offset $1/(\kappa-k)$")
    axes[0].set_xlabel(r"Ask offset, $\delta$")
    axes[0].set_ylabel(r"Gross markout rate, $g(\delta)$")
    axes[0].grid(axis="y")
    axes[0].legend(frameon=False, loc="lower right")
    panel_title(axes[0], "a", "Profit vanishes at the critical intensity")

    gap = np.linspace(0.08, 1.0, 400)
    boundary = k * lambda_u / (math.e * gap)
    axes[1].plot(gap, boundary, color=BLUE,
                 label=r"$\lambda_i^\ast=k\lambda_u/[e(\kappa-k)]$")
    axes[1].fill_between(gap, 0, boundary, color=LIGHT_BLUE,
                         label="some profitable finite offset")
    axes[1].fill_between(gap, boundary, 10, color=LIGHT_ORANGE,
                         label="gross-profit shutdown")
    axes[1].scatter([kappa - k], [critical], color=RED, zorder=4)
    axes[1].annotate(f"{critical:.4f}", (kappa - k, critical),
                     xytext=(8, 7), textcoords="offset points", color=RED)
    axes[1].set_xlabel(r"Decay gap, $\kappa-k$")
    axes[1].set_ylabel(r"Informed intensity, $\lambda_i$")
    axes[1].set_ylim(0, 10)
    axes[1].grid(axis="y")
    axes[1].legend(frameon=False, loc="upper right")
    panel_title(axes[1], "b", "Exact shutdown boundary")

    finish(fig, P2_FIGURES / "02_shutdown_threshold.png")


def p2_pending_cancel() -> None:
    states, index, q_fraction, r_fraction, w_fraction, _ = queue_cell()
    q = np.asarray(q_fraction, dtype=float)
    r = np.asarray(r_fraction, dtype=float)
    w = np.asarray(w_fraction, dtype=float)
    start = index[(3, 2)]
    durations = np.linspace(0, 1.0, 161)
    fill_probability, filled_volume, marked_value = [], [], []
    for duration in durations:
        pmf, _ = compound_fill_pmf(3, 2, float(duration), 1.4,
                                   [(1, 0.7), (3, 0.3)], cutoff=60)
        fill_probability.append(float(pmf[1] + pmf[2]))
        filled_volume.append(float(pmf[1] + 2 * pmf[2]))
        marked_value.append(float(delayed_value(q, r, w, 0.8, duration)[start]))

    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.05), constrained_layout=True)
    axes[0].plot(durations, fill_probability, color=BLUE,
                 label="probability of any fill")
    axes[0].plot(durations, filled_volume, color=ORANGE,
                 label="expected filled volume")
    axes[0].set_xlabel("Cancellation-acknowledgement delay")
    axes[0].set_ylabel("Fill quantity")
    axes[0].set_ylim(0, max(filled_volume) * 1.08)
    axes[0].grid(axis="y")
    axes[0].legend(frameon=False)
    panel_title(axes[0], "a", "Fills during pending cancellation")

    axes[1].plot(durations, marked_value, color=RED)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    check_delays = np.array([0.01, 0.12, 0.35, 0.8])
    check_values = [marked_value[int(round(x * 160))] for x in check_delays]
    axes[1].scatter(check_delays, check_values, color=RED, edgecolor="white", zorder=4)
    axes[1].set_xlabel("Cancellation-acknowledgement delay")
    axes[1].set_ylabel("Pending-period marked value")
    axes[1].grid(axis="y")
    panel_title(axes[1], "b", "Value with partial-fill revaluation")

    finish(fig, P2_FIGURES / "03_pending_cancel.png")


def p2_erlang_approximation(results: dict[str, dict]) -> None:
    row = results["erlang_cancel_approximation"]
    stages = np.array([x["stages"] for x in row["approximations"]], dtype=float)
    bias = np.array([x["initial_state_bias"] for x in row["approximations"]])
    weak = np.array([x["uniform_weak_bound"] for x in row["approximations"]])
    leading = abs(row["initial_state_leading_bias"])

    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.05), constrained_layout=True)
    axes[0].loglog(stages, np.abs(bias), marker="o", color=BLUE,
                   label="initial-state absolute bias")
    axes[0].loglog(stages, leading / stages, linestyle="--", color=ORANGE,
                   label=r"leading $k^{-1}$ term")
    axes[0].loglog(stages, weak, linestyle=":", color=GREY,
                   label="uniform weak bound")
    axes[0].set_xlabel("Erlang stages, $k$")
    axes[0].set_ylabel("Absolute value error")
    axes[0].grid(which="both")
    axes[0].legend(frameon=False)
    panel_title(axes[0], "a", "Mean-matched stage approximation")

    scaled = -stages * bias
    axes[1].semilogx(stages, scaled, marker="o", color=PURPLE,
                     label=r"$-k(V^{D_k}-V^d)$")
    axes[1].axhline(leading, color=ORANGE, linestyle="--",
                    label="analytic limit")
    axes[1].set_xlabel("Erlang stages, $k$")
    axes[1].set_ylabel("Scaled initial-state bias")
    axes[1].set_ylim(0, leading * 1.18)
    axes[1].grid(axis="y")
    axes[1].legend(frameon=False)
    panel_title(axes[1], "b", "$k^{-1}$ rate is attained")

    finish(fig, P2_FIGURES / "04_erlang_approximation.png")


def p2_certificates_learning(results: dict[str, dict]) -> None:
    dynamic = results["dynamic_market_making_performance_identity"]
    computed = results["computed_model_and_solver_certificate"]
    queue = results["interval_queue_inventory_control"]
    actual = np.array(
        [
            dynamic["direct_regret"],
            computed["direct_regret"],
            max(queue["rank_blind_policy_losses"]),
        ]
    )
    bounds = np.array(
        [
            dynamic["integrated_gradient_certificate"],
            computed["integrated_upper_bound"],
            float(Fraction(queue["certified_loss_upper"])),
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.05), constrained_layout=True)
    labels = ["marked\ninventory", "computed\nHJB", "74-state\nqueue"]
    x = np.arange(len(labels))
    width = 0.36
    axes[0].bar(x - width / 2, actual, width, color=BLUE, label="evaluated loss")
    axes[0].bar(x + width / 2, bounds, width, color=ORANGE, label="certificate")
    axes[0].set_xticks(x, labels)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Loss in the stated synthetic cell")
    axes[0].grid(axis="y", which="both")
    axes[0].legend(frameon=False)
    panel_title(axes[0], "a", "Certified versus evaluated loss")

    learning = results["two_quote_coverage_and_expected_loss"]
    for case, color in zip(learning["cases"], [BLUE, GREEN]):
        tau = np.array([z["tau"] for z in case["poisson_probability_sums"]])
        scaled = np.array([z["tau_times_loss"] for z in case["poisson_probability_sums"]])
        axes[1].semilogx(tau, scaled, marker="o", color=color,
                         label=fr"$\kappa={case['kappa']:g}$")
        axes[1].axhline(case["asymptotic_loss_constant"], color=color,
                        linestyle="--", alpha=0.75)
    axes[1].set_xlabel(r"Logging duration per quote, $\tau$")
    axes[1].set_ylabel(r"Scaled expected loss, $\tau\,E[L]$")
    axes[1].grid(axis="y")
    axes[1].legend(frameon=False)
    panel_title(axes[1], "b", "Two-quote $\tau^{-1}$ learning rate")

    finish(fig, P2_FIGURES / "05_certificates_learning.png")


def p3_decision_geometry() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.05), constrained_layout=True)
    signal = np.linspace(-1.6, 1.6, 500)
    for fee, color in zip([0.0, 0.2, 0.5], [GREY, BLUE, ORANGE]):
        position = np.sign(signal) * np.maximum(np.abs(signal) - fee, 0)
        position = np.clip(position, -1, 1)
        axes[0].plot(signal, position, color=color, label=fr"$c={fee:g}$")
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].axvline(0, color="#333333", linewidth=0.8)
    axes[0].set_xlabel(r"Forecast, $\mu$")
    axes[0].set_ylabel(r"Optimal position, $w(\mu)$")
    axes[0].set_ylim(-1.08, 1.08)
    axes[0].grid(axis="y")
    axes[0].legend(frameon=False)
    panel_title(axes[0], "a", "Thresholding and position clipping")

    error = np.linspace(0, 2.2, 500)
    envelope = np.where(error <= 1, error**2 / 2, error - 0.5)
    axes[1].plot(error, envelope, color=BLUE, label="sharp combined envelope")
    axes[1].plot(error, error**2 / 2, color=ORANGE, linestyle="--",
                 label="curvature-only bound")
    axes[1].plot(error, error, color=GREEN, linestyle=":",
                 label="diameter-only bound")
    axes[1].axvline(1, color=GREY, linewidth=0.8)
    axes[1].annotate(r"$E=aW$", (1, 0.5), xytext=(7, -20),
                     textcoords="offset points", color=GREY)
    axes[1].set_xlabel(r"Forecast error norm, $E$")
    axes[1].set_ylabel("Uniform decision-loss bound")
    axes[1].set_ylim(0, 2.0)
    axes[1].grid(axis="y")
    axes[1].legend(frameon=False, loc="upper left")
    panel_title(axes[1], "b", "Curvature--diameter crossover ($a=W=1$)")

    finish(fig, P3_FIGURES / "01_decision_geometry.png")


def p3_threshold_rates(results: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.05), constrained_layout=True)
    z = np.linspace(0.02, 4.5, 500)
    exact = 2 * (norm.pdf(z) - z * norm.sf(z))
    asymptotic = 2 * norm.pdf(z) / z**2
    axes[0].semilogy(z, exact, color=BLUE, label="exact Gaussian value $V/s$")
    axes[0].semilogy(z[z >= 1], asymptotic[z >= 1], color=ORANGE,
                     linestyle="--", label="Mills-ratio leading term")
    axes[0].scatter([3], [2 * (norm.pdf(3) - 3 * norm.sf(3))],
                    color=RED, zorder=4)
    axes[0].annotate(r"$c/s=3$", (3, exact[np.argmin(np.abs(z - 3))]),
                     xytext=(-42, 10), textcoords="offset points", color=RED)
    axes[0].set_xlabel(r"Cost-to-signal ratio, $c/s$")
    axes[0].set_ylabel(r"Oracle value divided by $s$")
    axes[0].grid(which="both")
    axes[0].legend(frameon=False)
    panel_title(axes[0], "a", "Gaussian threshold value")

    alpha = np.linspace(0.05, 4.0, 400)
    exponent = (alpha + 1) / (alpha + 2)
    axes[1].plot(alpha, exponent, color=PURPLE,
                 label=r"sharp exponent $(\alpha+1)/(\alpha+2)$")
    axes[1].axhline(1, color=GREY, linestyle="--",
                    label="independent smooth-law exponent")
    checks = results["dependent_error_MSE_margin_sharpness"]["fixed_signal_laws"]
    axes[1].scatter([x["margin_exponent"] for x in checks],
                    [x["observed_MSE_exponent"] for x in checks],
                    color=RED, edgecolor="white", zorder=4, label="verified cases")
    axes[1].set_xlabel(r"Margin exponent, $\alpha$")
    axes[1].set_ylabel("Exponent on MSE")
    axes[1].set_ylim(0.48, 1.03)
    axes[1].grid(axis="y")
    axes[1].legend(frameon=False, loc="lower right")
    panel_title(axes[1], "b", "Dependence slows the loss rate")

    finish(fig, P3_FIGURES / "02_threshold_rates.png")


def posterior_path(prior: np.ndarray, k: np.ndarray, matrices: list[np.ndarray],
                   intervals: list[float], events: list[int]) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = [0.0]
    probabilities: list[float] = [float(prior[0])]
    row = prior.copy()
    elapsed = 0.0
    for i, duration in enumerate(intervals):
        base = row.copy()
        for s in np.linspace(0, duration, 36)[1:]:
            current = base @ expm(s * k)
            current /= current.sum()
            times.append(elapsed + float(s))
            probabilities.append(float(current[0]))
        row = base @ expm(duration * k)
        row /= row.sum()
        elapsed += duration
        if i < len(events):
            row = row @ matrices[events[i]]
            row /= row.sum()
            times.append(elapsed)
            probabilities.append(float(row[0]))
    return np.array(times), np.array(probabilities)


def p3_filter_certificate(results: dict[str, dict]) -> None:
    u = np.array([[-0.2, 0.2], [0.1, -0.1]])
    matrices = [np.array([[1.2, 0.1], [0.04, 0.5]]),
                np.array([[0.4, 0.02], [0.06, 0.9]])]
    uhat = np.array([[-0.205, 0.205], [0.098, -0.098]])
    fitted_matrices = [m * 1.003 for m in matrices]
    k, _ = event_objects(u, matrices)
    khat, _ = event_objects(uhat, fitted_matrices)
    intervals = [0.12, 0.07, 0.16, 0.1]
    events = [0, 1, 0]
    t, true_p = posterior_path(np.array([0.6, 0.4]), k, matrices, intervals, events)
    th, fitted_p = posterior_path(np.array([0.602, 0.398]), khat,
                                  fitted_matrices, intervals, events)

    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.05), constrained_layout=True)
    axes[0].plot(t, true_p, color=BLUE, label="true filter")
    axes[0].plot(th, fitted_p, color=ORANGE, linestyle="--", label="fitted filter")
    boundaries = np.cumsum(intervals)[:-1]
    for j, boundary in enumerate(boundaries):
        axes[0].axvline(boundary, color=GREY, linewidth=0.8, linestyle=":")
        axes[0].text(boundary, 0.98, f"event {events[j]}", rotation=90,
                     va="top", ha="right", color=GREY, fontsize=7)
    axes[0].set_xlabel("Observed-history time")
    axes[0].set_ylabel("Posterior probability of state 0")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y")
    axes[0].legend(frameon=False)
    panel_title(axes[0], "a", "Event and silence updates")

    filter_row = results["event_history_filter_and_silence"]
    complete = results["events_to_coarse_forecast_to_position"]
    actual = np.array(
        [
            filter_row["actual_posterior_error"],
            abs(complete["true_forecast"] - complete["deployed_forecast"]),
            complete["actual_decision_loss"],
        ]
    )
    bound = np.array(
        [
            filter_row["posterior_error_bound"],
            complete["forecast_error_bound"],
            complete["certified_decision_loss"],
        ]
    )
    labels = ["posterior\n$L^1$", "forecast", "decision\nloss"]
    x = np.arange(3)
    width = 0.36
    axes[1].bar(x - width / 2, actual, width, color=BLUE, label="evaluated")
    axes[1].bar(x + width / 2, bound, width, color=ORANGE, label="certified")
    axes[1].set_xticks(x, labels)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Stage-specific error or loss")
    axes[1].grid(axis="y", which="both")
    axes[1].legend(frameon=False)
    panel_title(axes[1], "b", "Composed certificate")

    finish(fig, P3_FIGURES / "03_filter_certificate.png")


def p3_latency_frontier() -> None:
    kappa, drift, horizon, curvature, fee = 20.0, 2.0, 0.05, 0.7, 0.012
    value = drift * (-math.expm1(-2 * kappa * horizon)) / (2 * kappa)
    threshold = math.log(value / fee) / (2 * kappa)
    delay = np.linspace(0, 0.05, 400)
    decay = np.exp(-2 * kappa * delay)
    mean = value * decay
    quadratic = value**2 / (2 * curvature) * (1 - np.exp(-4 * kappa * delay))
    naive = value**2 / (2 * curvature) * (1 - decay) ** 2
    mixed = ((max(0, value - fee) ** 2)
             - np.maximum(0, mean - fee) ** 2) / (2 * curvature)
    position = np.maximum(0, mean - fee) / curvature

    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.05), constrained_layout=True)
    axes[0].plot(1000 * delay, quadratic, color=BLUE,
                 label="quadratic information loss")
    axes[0].plot(1000 * delay, mixed, color=ORANGE,
                 label="mixed-cost information loss")
    axes[0].plot(1000 * delay, naive, color=GREEN, linestyle="--",
                 label="naive stale-mean term")
    axes[0].set_xlabel("Observation delay (ms)")
    axes[0].set_ylabel("Decision loss")
    axes[0].grid(axis="y")
    axes[0].legend(frameon=False)
    panel_title(axes[0], "a", "Information versus stale-mean loss")

    axes[1].plot(1000 * delay, position, color=PURPLE)
    axes[1].axvline(1000 * threshold, color=RED, linestyle="--",
                    label=f"no-trade boundary: {1000 * threshold:.1f} ms")
    axes[1].fill_between(1000 * delay, 0, position, color="#e8e2f2")
    axes[1].set_xlabel("Observation delay (ms)")
    axes[1].set_ylabel("Absolute optimal stale position")
    axes[1].set_ylim(0, position.max() * 1.08)
    axes[1].grid(axis="y")
    axes[1].legend(frameon=False)
    panel_title(axes[1], "b", "Latency induces no-trade")

    finish(fig, P3_FIGURES / "04_latency_frontier.png")


def p3_causal_tracking() -> None:
    rho, kappa, variance = 1.0, 1.0, 1.0
    theta = kappa / (kappa + rho)
    dt = 0.005
    total = 16.0
    rng = np.random.default_rng(20260915)
    times = np.arange(0, total + dt, dt)
    target = np.empty_like(times)
    position = np.empty_like(times)
    target[0] = rng.normal(scale=math.sqrt(variance))
    position[0] = 0.0
    decay = math.exp(-rho * dt)
    innovation = math.sqrt(variance * (1 - math.exp(-2 * rho * dt)))
    for j in range(1, len(times)):
        target[j] = decay * target[j - 1] + innovation * rng.normal()
        position[j] = position[j - 1] + dt * kappa * (
            theta * target[j - 1] - position[j - 1]
        )
    keep = times >= 8.0

    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.05), constrained_layout=True)
    axes[0].plot(times[keep] - 8, target[keep], color="#a8b0b8", linewidth=1,
                 label="OU target $x_t$")
    axes[0].plot(times[keep] - 8, theta * target[keep], color=ORANGE, alpha=0.85,
                 label=r"causal aim $\theta x_t$")
    axes[0].plot(times[keep] - 8, position[keep], color=BLUE,
                 label="optimal causal position $w_t$")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Target and position")
    axes[0].grid(axis="y")
    axes[0].legend(frameon=False, ncol=1)
    panel_title(axes[0], "a", "A fixed-seed stationary tracking path")

    ratio = np.logspace(-1, 1.2, 400)
    theta_grid = 1 / (1 + ratio)
    axes[1].semilogx(ratio, theta_grid**2, color=BLUE,
                     label=r"covariance/$v=\theta^2$")
    axes[1].semilogx(ratio, np.sqrt(theta_grid), color=GREEN,
                     label=r"correlation $=\sqrt{\theta}$")
    axes[1].semilogx(ratio, 1 - theta_grid**2, color=ORANGE,
                     label=r"causal cost $=1-\theta^2$")
    axes[1].semilogx(ratio, theta_grid / (1 + theta_grid), color=PURPLE,
                     linestyle="--", label="causality share")
    axes[1].set_xlabel(r"Target decay relative to trading, $\rho/\kappa$")
    axes[1].set_ylabel("Normalized quantity")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(axis="y")
    axes[1].legend(frameon=False, loc="center right")
    panel_title(axes[1], "b", "Four distinct transfer quantities")

    finish(fig, P3_FIGURES / "05_causal_tracking.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the analytic and synthetic PNG figures for Papers II and III."
    )
    parser.add_argument(
        "--paper", choices=("2", "3", "all"), default="all",
        help="figure set to generate (default: all)",
    )
    parser.add_argument(
        "--output", type=Path,
        help="output directory when generating exactly one paper",
    )
    args = parser.parse_args()
    if args.output is not None and args.paper == "all":
        parser.error("--output requires --paper 2 or --paper 3")
    return args


def main() -> None:
    global P2_FIGURES, P3_FIGURES
    args = parse_args()
    if args.output is not None:
        if args.paper == "2":
            P2_FIGURES = args.output.resolve()
        else:
            P3_FIGURES = args.output.resolve()
    configure_style()
    results = load_results()
    if args.paper in ("2", "all"):
        p2_queue_priority()
        p2_shutdown_threshold()
        p2_pending_cancel()
        p2_erlang_approximation(results)
        p2_certificates_learning(results)
    if args.paper in ("3", "all"):
        p3_decision_geometry()
        p3_threshold_rates(results)
        p3_filter_certificate(results)
        p3_latency_frontier()
        p3_causal_tracking()


if __name__ == "__main__":
    main()
