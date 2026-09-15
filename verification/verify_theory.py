"""Synthetic mathematical checks for Papers 2 and 3. No market data are used."""
import json
import math
from pathlib import Path

import numpy as np
import scipy
from fractions import Fraction
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm, solve_continuous_are, solve_continuous_lyapunov
from scipy.optimize import minimize_scalar
from scipy.stats import norm

RESULTS = []
RNG = np.random.default_rng(20260914)


def record(name, **metrics):
    RESULTS.append({"check": name, "status": "passed", **metrics})


def check_local_constrained_certificate():
    worst = 0.0
    for m in [0.2, 1.0, 3.0]:
        for true_target in [-1.0, 0.3, 2.0]:
            for error in [-0.7, 0.15, 1.2]:
                f = lambda x: -m * (x - true_target) ** 2 / 2
                astar = np.clip(true_target, 0, 1)
                ahat = np.clip(true_target + error / m, 0, 1)
                gap = f(astar) - f(ahat)
                bound = error**2 / (2 * m)
                assert gap <= bound + 1e-12
                assert abs(astar - ahat) <= abs(error) / m + 1e-12
                worst = max(worst, gap / bound)
    record("constrained_gradient_certificate", configurations=27,
           largest_gap_to_bound=float(worst))


def check_shutdown():
    lu, kappa, k = 10.0, 2.0, 1.0
    critical = k * lu / (math.e * (kappa - k))
    g = lambda d, li: lu*d*math.exp(-kappa*d)-li/k*math.exp(-k*d)
    assert abs(g(1.0, critical)) < 1e-14
    assert g(1.0, 0.9 * critical) > 0
    for d in np.linspace(0, 20, 2001):
        assert g(d, 1.1 * critical) < 0
    numeric = minimize_scalar(lambda d: -d*math.exp(-(kappa-k)*d),
                              bounds=(0, 5), method="bounded")
    assert abs(numeric.x - 1) < 1e-5
    record("shutdown_threshold", critical_intensity=critical,
           numerical_binding_offset=float(numeric.x))


def check_dynamic_market_making():
    qs = np.arange(-2, 3)
    T, lower, upper = 1.0, 0.15, 1.0
    true = {"kappa": 1.2, "lambdas": [2.2, 2.0], "mark": 0.04}
    estimated = {"kappa": 1.15, "lambdas": [2.35, 1.85], "mark": 0.055}
    terminal = -0.05 * qs.astype(float)**2
    penalty = 0.02 * qs.astype(float)**2

    def enabled(i):
        return [(s, i + dq) for s, dq in enumerate([-1, 1])
                if 0 <= i + dq < len(qs)]

    def policy(value, model):
        result = {}
        for i in range(len(qs)):
            for side, j in enabled(i):
                shadow = value[i] - value[j] + model["mark"]
                result[i, side] = float(np.clip(
                    shadow + 1 / model["kappa"], lower, upper))
        return result

    def generator_reward(actions, model):
        K = np.zeros((len(qs), len(qs)))
        reward = -penalty.copy()
        for i in range(len(qs)):
            for side, j in enabled(i):
                d = actions[i, side]
                rate = model["lambdas"][side] * math.exp(-model["kappa"] * d)
                K[i, j] += rate
                K[i, i] -= rate
                reward[i] += rate * (d - model["mark"])
        return K, reward

    def solve_value(model):
        def rhs(t, value):
            K, r = generator_reward(policy(value, model), model)
            return -(r + K @ value)
        out = solve_ivp(rhs, (T, 0), terminal, rtol=2e-10, atol=2e-12,
                        dense_output=True, max_step=0.02)
        assert out.success
        return out

    h = solve_value(true)
    hhat = solve_value(estimated)
    def eval_rhs(t, value):
        K, r = generator_reward(policy(hhat.sol(t), estimated), true)
        return -(r + K @ value)
    evaluation = solve_ivp(eval_rhs, (T, 0), terminal, rtol=2e-10,
                           atol=2e-12, dense_output=True, max_step=0.02)
    assert evaluation.success
    minimum_m = [math.inf]

    def forward(t, state):
        value, fit = h.sol(t), hhat.sol(t)
        ahat = policy(fit, estimated)
        K, r = generator_reward(ahat, true)
        Kstar, rstar = generator_reward(policy(value, true), true)
        gap = rstar + Kstar @ value - (r + K @ value)
        local_bound = np.zeros(len(qs))
        for i in range(len(qs)):
            grad_sq, curvature = 0.0, math.inf
            for side, j in enabled(i):
                d = ahat[i, side]
                p = value[i] - value[j] + true["mark"]
                phat = fit[i] - fit[j] + estimated["mark"]
                lam = true["lambdas"][side] * math.exp(-true["kappa"] * d)
                lamhat = estimated["lambdas"][side] * math.exp(-estimated["kappa"] * d)
                residual = (lam * (1 - true["kappa"] * (d-p))
                            - lamhat * (1 - estimated["kappa"] * (d-phat)))
                grad_sq += residual**2
                m = (true["lambdas"][side] * math.exp(-true["kappa"]*upper)
                     * true["kappa"] * (2 - true["kappa"]*(upper-p)))
                curvature = min(curvature, m)
            assert curvature > 0
            minimum_m[0] = min(minimum_m[0], curvature)
            local_bound[i] = grad_sq / (2 * curvature)
            assert gap[i] <= local_bound[i] + 2e-9
        prob = state[:len(qs)]
        return np.r_[prob @ K, prob @ gap, prob @ local_bound]

    start = np.r_[np.eye(len(qs))[2], 0.0, 0.0]
    occupancy = solve_ivp(forward, (0, T), start, rtol=2e-9, atol=2e-11,
                          max_step=0.01)
    assert occupancy.success
    integrated_gap, certificate = occupancy.y[-2:, -1]
    direct_regret = float(h.sol(0)[2] - evaluation.sol(0)[2])
    assert abs(direct_regret - integrated_gap) < 2e-7
    assert direct_regret <= certificate + 2e-7
    assert abs(sum(occupancy.y[:len(qs), -1]) - 1) < 1e-9
    record("dynamic_market_making_performance_identity",
           direct_regret=direct_regret, integrated_hamiltonian_gap=float(integrated_gap),
           integrated_gradient_certificate=float(certificate),
           minimum_concavity_modulus=minimum_m[0],
           identity_absolute_error=abs(direct_regret-float(integrated_gap)))


def check_quadratic_and_mixed_costs():
    errors = []
    for _ in range(50):
        A = RNG.normal(size=(4, 4))
        H = A.T @ A + np.eye(4)
        Hhat = H + 0.2 * np.eye(4)
        mu, muhat = RNG.normal(size=(2, 4))
        w, what = np.linalg.solve(H, mu), np.linalg.solve(Hhat, muhat)
        utility = lambda v: mu @ v - 0.5 * v @ H @ v
        residual = H @ what - mu
        exact = 0.5 * residual @ np.linalg.solve(H, residual)
        errors.append(abs(utility(w)-utility(what)-exact))
    assert max(errors) < 1e-11
    largest_violation = 0.0
    for a in [0.5, 1.0, 3.0]:
        for c in [0.1, 0.8]:
            for mu in np.linspace(-2, 2, 9):
                cost = lambda w: a*w*w/2+c*abs(w)
                w = float(np.clip(np.sign(mu)*max(abs(mu)-c, 0)/a, -1, 1))
                opt = minimize_scalar(lambda x: cost(x)-mu*x,
                                      bounds=(-1, 1), method="bounded",
                                      options={"xatol": 1e-12})
                assert mu*w-cost(w) >= -opt.fun - 1e-9
                for e in [-0.7, 0.2]:
                    mhat = mu + e
                    what = float(np.clip(np.sign(mhat)*max(abs(mhat)-c, 0)/a, -1, 1))
                    gap = mu*w-cost(w)-(mu*what-cost(what))
                    bound = e*e/(2*a)
                    assert gap <= bound + 1e-12
                    largest_violation = max(largest_violation, gap-bound)
    record("quadratic_matrix_and_mixed_cost_loss",
           matrix_identity_max_error=max(errors),
           mixed_cost_max_bound_violation=largest_violation)


def check_threshold_and_smooth_law():
    c = 3.0
    exact = 2*(norm.pdf(c)-c*norm.sf(c))
    numeric = 2*quad(lambda x: (x-c)*norm.pdf(x), c, np.inf,
                     epsabs=1e-13)[0]
    assert abs(exact-numeric) < 1e-12
    ratios = []
    for s in [0.1, 0.05, 0.025]:
        def conditional_loss(e):
            points = sorted(set([-12., -1., 1., -1.-e, 1.-e, 12.]))
            def integrand(mu):
                w = np.sign(mu) * (abs(mu) > 1)
                what = np.sign(mu+e) * (abs(mu+e) > 1)
                loss = mu*w-abs(w)-(mu*what-abs(what))
                return loss * norm.pdf(mu)
            return sum(quad(integrand, a, b, epsabs=1e-13)[0]
                       for a, b in zip(points[:-1], points[1:]))
        loss = (conditional_loss(s)+conditional_loss(-s))/2
        ratios.append(loss/s**2)
    expected = norm.pdf(1)
    assert abs(ratios[-1]-expected) < 1e-5
    record("gaussian_value_and_smooth_threshold_expansion",
           gaussian_value_at_cost_3=exact,
           retained_fraction=exact/math.sqrt(2/math.pi),
           small_error_coefficients=ratios,
           limiting_coefficient=expected)


def check_order_book_generator_transfer():
    largest_ratio = 0.0
    for _ in range(30):
        off = RNG.uniform(0, 1, (4, 4))
        np.fill_diagonal(off, 0)
        Q = off - np.diag(off.sum(axis=1))
        offhat = off * RNG.uniform(0.8, 1.2, (4, 4))
        Qhat = offhat - np.diag(offhat.sum(axis=1))
        p = np.array([100., 100., 101., 101.])
        pi = np.r_[RNG.dirichlet([2, 2]), 0., 0.]
        pihat = np.r_[RNG.dirichlet([2, 2]), 0., 0.]
        horizon = 0.4
        error = abs(pihat @ expm(horizon*Qhat) @ p - pi @ expm(horizon*Q) @ p)
        bound = 0.5 * (np.abs(pihat-pi).sum()
                       + horizon*np.abs(Qhat-Q).sum(axis=1).max())
        assert error <= bound + 1e-11
        largest_ratio = max(largest_ratio, error/bound)
    mu = (1-math.exp(-0.8))/2
    muhat = (1-math.exp(-1.2))/2
    loss = (mu-muhat)**2/4
    record("order_book_generator_and_posterior_transfer",
           random_valid_generators=30, largest_error_to_bound=largest_ratio,
           two_state_true_mean=mu, two_state_estimated_mean=muhat,
           two_state_quadratic_loss=loss)


def check_causal_hjb_and_covariances():
    exact_checks = 0
    for eta_q in [Fraction(1), Fraction(3, 2)]:
        for kap_q in [Fraction(1, 2), Fraction(1), Fraction(7, 3)]:
            for rho_q in [Fraction(1, 3), Fraction(1), Fraction(2), Fraction(10)]:
                v_q = Fraction(5, 4)
                th_q = kap_q/(kap_q+rho_q)
                A = eta_q*kap_q
                B = -2*eta_q*kap_q*th_q
                C = eta_q*kap_q**2*(1-th_q**2)/(2*rho_q)
                lhs = {"u2": eta_q, "w2": eta_q*kap_q**2,
                       "x2": eta_q*kap_q**2-2*rho_q*C,
                       "wx": -2*eta_q*kap_q**2-rho_q*B,
                       "uw": 2*A, "ux": B, "constant": 2*rho_q*v_q*C}
                rhs = {"u2": eta_q, "w2": eta_q*kap_q**2,
                       "x2": eta_q*kap_q**2*th_q**2,
                       "wx": -2*eta_q*kap_q**2*th_q,
                       "uw": 2*eta_q*kap_q, "ux": -2*eta_q*kap_q*th_q,
                       "constant": eta_q*kap_q**2*v_q*(1-th_q**2)}
                assert lhs == rhs
                exact_checks += 1
    rows = []
    for rho0 in [0.5, 1.0, 2.0, 10.0]:
        eta0, kap0, v0 = 1., 1., 1.
        th = kap0/(kap0+rho0)
        A = np.array([[0., 0.], [0., -rho0]])
        B = np.array([[1.], [0.]])
        Q = eta0*kap0**2*np.array([[1., -1.], [-1., 1.]])
        P = solve_continuous_are(A, B, Q, np.array([[eta0]]))
        K = B.T @ P/eta0
        assert np.max(abs(K-np.array([[kap0, -kap0*th]]))) < 1e-10
        noise = np.array([[0., 0.], [0., 2*rho0*v0]])
        C = solve_continuous_lyapunov(A-B@K, -noise)
        assert abs(C[0, 1]/v0-th**2) < 1e-11
        assert abs(C[0, 0]/v0-th**3) < 1e-11
        J = float(eta0*(K@C@K.T)[0, 0]+eta0*kap0**2*(C[0, 0]-2*C[0, 1]+v0))
        assert abs(J-eta0*kap0**2*v0*(1-th**2)) < 1e-11
        alpha, beta = 1.1*kap0, 0.9*kap0*th
        Cwrong = solve_continuous_lyapunov(
            np.array([[-alpha, beta], [0., -rho0]]), -noise)
        ku = np.array([-alpha, beta])
        Jwrong = eta0*(ku@Cwrong@ku)+eta0*kap0**2*(Cwrong[0, 0]-2*Cwrong[0, 1]+v0)
        kres = np.array([kap0-alpha, beta-kap0*th])
        assert abs(Jwrong-J-eta0*(kres@Cwrong@kres)) < 1e-11
        rows.append({"rho": rho0, "normalized_covariance": float(C[0, 1]),
                     "correlation": float(C[0, 1]/math.sqrt(C[0, 0]*v0)),
                     "causal_cost": J, "wrong_rule_excess": float(Jwrong-J)})
    record("causal_hjb_riccati_and_wrong_rule_identity",
           exact_rational_polynomial_checks=exact_checks, independent_riccati_checks=rows)


def check_information_bounds():
    prior = np.array([0.5, 0.5])
    posterior = [np.array([0.8, 0.2]), np.array([0.2, 0.8])]
    returns = np.array([[1.2, 0.9], [0.85, 1.15]])
    def value(prob):
        def objective(b):
            return prob @ np.log(b*returns[:, 0]+(1-b)*returns[:, 1])
        sol = minimize_scalar(lambda b: -objective(b), bounds=(0, 1), method="bounded")
        return max(objective(0), objective(1), -sol.fun)
    growth_gain = sum(value(p) for p in posterior)/2-value(prior)
    information = sum(float(np.sum(p*np.log(p/prior))) for p in posterior)/2
    assert 0 <= growth_gain <= information + 1e-10
    classification_value = 0.3
    assert classification_value <= math.sqrt(information/2)+1e-12
    record("bounded_reward_and_log_growth_information_bounds",
           log_growth_gain=growth_gain, mutual_information=information,
           bounded_reward_value=classification_value)


def main():
    for check in [
        check_local_constrained_certificate, check_shutdown,
        check_dynamic_market_making, check_quadratic_and_mixed_costs,
        check_threshold_and_smooth_law, check_order_book_generator_transfer,
        check_causal_hjb_and_covariances, check_information_bounds,
    ]:
        check()
    from trilogy_extensions import run_checks
    run_checks(record)
    from submitted_extensions import run_checks as run_submitted_checks
    run_submitted_checks(record)
    from order_book_foundations import run_checks as run_book_checks
    run_book_checks(record)
    from queue_certificates import run_checks as run_queue_checks
    run_queue_checks(record)
    from forecast_certificates import run_checks as run_forecast_checks
    run_forecast_checks(record)
    from partial_fills_latency import run_checks as run_partial_fill_checks
    run_partial_fill_checks(record)
    from observation_value import run_checks as run_observation_checks
    run_observation_checks(record)
    report = {"status": "all_passed", "check_groups": len(RESULTS),
              "data": "Synthetic mathematical checks; no market data.",
              "versions": {"numpy": np.__version__, "scipy": scipy.__version__}, "results": RESULTS}
    output = Path(__file__).with_name("verification_results.json")
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
