"""Independent synthetic checks for the explicit order-book constructions.

Queue resolvents are compared with holding-time products; controlled values
with forward occupation integrals; filtering with a normalized ODE; and
aggregation with matrix exponentials. These are numerical checks, except for
the polynomial coefficient enclosures, which use exact fractions throughout.
"""
from fractions import Fraction as F
import math

import numpy as np
from scipy.integrate import quad, quad_vec, solve_ivp
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from scipy.stats import gamma, poisson


def infnorm(matrix):
    return float(np.linalg.norm(matrix, ord=np.inf))


def integrated_semigroup(q, vector, horizon):
    """Augmented exponential, valid also for a singular generator."""
    n = len(vector)
    block = np.zeros((n + 1, n + 1))
    block[:n, :n] = q
    block[:n, n] = vector
    return expm(horizon * block)[:n, n]


def mix_constants(omega, horizon):
    if omega == 0:
        return horizon, horizon**2 / 2
    a = -math.expm1(-omega * horizon) / omega
    return a, (horizon - a) / omega


def envelope(a, diameter, error):
    return error**2 / (2 * a) if error <= a * diameter else (
        diameter * error - a * diameter**2 / 2)


def queue_matrix(n, mu, cancellation, invalidation):
    s = np.zeros((n + 1, n + 1))
    fill = np.zeros(n + 1)
    fill[0] = mu
    for j in range(n + 1):
        progress = mu + (cancellation * j if j else 0)
        s[j, j] = -progress - invalidation
        if j:
            s[j, j-1] = progress
    return s, fill


def check_tagged_queue(record):
    largest_discrepancy = 0.0
    for n in [0, 1, 3, 6]:
        for mu in [0.6, 1.7]:
            beta, xi, cancellation = 0.2, 0.3, 0.1
            s, fill = queue_matrix(n, mu, cancellation, xi)
            resolvent = np.linalg.solve(beta * np.eye(n+1) - s, fill)[n]
            product = mu / (mu + beta + xi)
            for j in range(1, n + 1):
                product *= (mu+cancellation*j)/(mu+cancellation*j+beta+xi)
            largest_discrepancy = max(largest_discrepancy, abs(product-resolvent))
            assert abs(product-resolvent) < 2e-14
            s0, f0 = queue_matrix(n, mu, 0.0, 0.0)
            for horizon in [0.1, 1.0, 3.0]:
                matrix_cdf = integrated_semigroup(s0, f0, horizon)[n]
                erlang_cdf = gamma.cdf(horizon, a=n+1, scale=1/mu)
                assert abs(matrix_cdf-erlang_cdf) < 2e-13
            shat, fhat = queue_matrix(n, mu+0.08, cancellation, xi+0.05)
            g, ghat = 0.4*fill, 0.37*fhat
            horizon = 1.8
            value = integrated_semigroup(s, g, horizon)
            fitted = integrated_semigroup(shat, ghat, horizon)
            bound = horizon*infnorm(g-ghat) + horizon**2/2*infnorm(g)*infnorm(s-shat)
            assert infnorm(value-fitted) <= bound + 1e-13
    # Exact priority and one-replacement comparisons in the text.
    front, behind = F(1, 2), F(1, 16)
    old_front, old_back = F(1, 2)*front, F(1, 2)*F(1, 8)
    replacement = F(7, 10)*F(1, 4)-F(1, 100)
    assert old_front == F(1, 4) and old_back == F(1, 16)
    assert old_front > replacement > old_back and replacement == F(33, 200)
    record('tagged_queue_absorption_and_priority', queue_configurations=8,
           erlang_comparisons=24, largest_resolvent_discrepancy=largest_discrepancy,
           replacement_value=float(replacement))


def check_replacement_clock(record):
    # Four active states: (price level, number ahead), plus fill and invalidation.
    n, horizon, nu, fee = 6, 2.0, 2.5, 0.02
    q0, r0 = np.zeros((n, n)), np.zeros(n)
    target = np.arange(n)
    for level, (mu, reward) in enumerate([(1.4, 0.45), (0.7, 0.8)]):
        for rank in [0, 1]:
            i = 2*level+rank
            target[i] = 2*(1-level)+1  # A replacement joins the other queue's back.
            if rank:
                q0[i, i-1] = mu+0.2
            else:
                q0[i, 4] = mu
                r0[i] = mu*reward
            q0[i, 5] = 0.25
            q0[i, i] = -q0[i].sum()
            r0[i] -= 0.02
    active = np.arange(n) < 4

    def decision_gains(h):
        return np.where(active, h[target]-h-fee, 0.0)

    def rhs(tau, h):
        return r0+q0@h+nu*np.maximum(decision_gains(h), 0)

    value = solve_ivp(rhs, [0, horizon], np.zeros(n), dense_output=True,
                      rtol=2e-11, atol=2e-13, max_step=0.01)
    assert value.success
    # Constant replacement policy: independent linear value and occupation law.
    qpi, rpi = q0.copy(), r0.copy()
    for i in range(4):
        qpi[i, target[i]] += nu
        qpi[i, i] -= nu
        rpi[i] -= nu*fee
    jpi = integrated_semigroup(qpi, rpi, horizon)
    initial = np.eye(n)[0]

    def loss_integrand(t):
        h = value.sol(horizon-t)
        optimal_hamiltonian = rhs(horizon-t, h)
        policy_hamiltonian = rpi+qpi@h
        return initial@expm(t*qpi)@(optimal_hamiltonian-policy_hamiltonian)

    occupation_loss = quad(loss_integrand, 0, horizon, epsabs=1e-10,
                           points=np.linspace(0, horizon, 101), limit=200)[0]
    gap = value.y[0, -1]-jpi[0]
    assert gap >= -1e-10 and abs(gap-occupation_loss) < 2e-8
    # Evaluate the maximizing feedback by a forward probability/reward equation.
    def forward(t, y):
        h = value.sol(horizon-t)
        replace = decision_gains(h) > 0
        q, r = q0.copy(), r0.copy()
        for i in range(4):
            if replace[i]:
                q[i, target[i]] += nu
                q[i, i] -= nu
                r[i] -= nu*fee
        return np.r_[y[:n]@q, y[:n]@r]
    forward_value = solve_ivp(forward, [0, horizon], np.r_[initial, 0.0],
                              rtol=2e-10, atol=2e-12, max_step=0.005)
    assert forward_value.success
    assert abs(forward_value.y[-1, -1]-value.y[0, -1]) < 3e-7
    assert abs(forward_value.y[:n, -1].sum()-1) < 1e-10
    record('queue_replacement_clock_hjb', states=n,
           optimal_initial_value=float(value.y[0, -1]),
           constant_replacement_regret=float(gap),
           occupation_identity_error=float(abs(gap-occupation_loss)),
           feedback_value_error=float(abs(forward_value.y[-1, -1]-value.y[0, -1])))


def padd(a, b):
    out = [F(0)]*max(len(a), len(b))
    for i, x in enumerate(a):
        out[i] += x
    for i, x in enumerate(b):
        out[i] += x
    return out


def psub(a, b):
    return padd(a, [-x for x in b])


def pmul(a, b):
    out = [F(0)]*(len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i+j] += x*y
    return out


def derivative_bound(coefficients, order):
    p = coefficients[:]
    for _ in range(order):
        p = [i*p[i] for i in range(1, len(p))] or [F(0)]
    # On [0,1], the coefficient absolute sum is a global enclosure.
    return sum(map(abs, p))


def check_event_derivatives(record):
    rates = [[F(1), F(1,3), F(1,5)], [F(4,5), F(-1,10), F(1,5)],
             [F(1,2), F(1,4)]]
    rate_errors = [[F(1,100), F(-1,50)], [F(1,80), F(1,70)], [F(-1,90)]]
    marks = [[F(1,2), F(-1,5)], [F(-1,3), F(1,4), F(1,8)], [F(-1,10)]]
    mark_errors = [[F(1,60)], [F(-1,50), F(1,90)], [F(1,70)]]
    fitted_rates = [psub(x, e) for x, e in zip(rates, rate_errors)]
    fitted_marks = [psub(x, e) for x, e in zip(marks, mark_errors)]
    c, chat = [F(-1,5), F(0), F(-1,8)], [F(-1,5), F(1,100), F(-1,9)]
    r, rhat = c[:], chat[:]
    for lam, lamhat, g, ghat in zip(rates, fitted_rates, marks, fitted_marks):
        assert lam[0] > sum(abs(x) for x in lam[1:] if x < 0)
        assert lamhat[0] > sum(abs(x) for x in lamhat[1:] if x < 0)
        r, rhat = padd(r, pmul(lam, g)), padd(rhat, pmul(lamhat, ghat))
    eps_l = [sum(derivative_bound(p, j) for p in rate_errors) for j in range(3)]
    eps_g = [max(derivative_bound(p, j) for p in mark_errors) for j in range(3)]
    true_g = [max(derivative_bound(p, j) for p in marks) for j in range(3)]
    fitted_l = [sum(derivative_bound(p, j) for p in fitted_rates) for j in range(3)]
    budgets = []
    for j in range(3):
        bound = derivative_bound(psub(c, chat), j)
        for ell in range(j+1):
            bound += math.comb(j, ell)*(true_g[j-ell]*eps_l[ell]
                                       +fitted_l[ell]*eps_g[j-ell])
        # The complete reward-error polynomial is independently expanded first.
        assert derivative_bound(psub(r, rhat), j) <= bound
        # Channels 0 and 1 share an off-diagonal destination; channel 2 is a self-event.
        grouped_error = padd(rate_errors[0], rate_errors[1])
        assert derivative_bound(grouped_error, j) <= eps_l[j]
        budgets.append(str(bound))
    record('event_rate_and_mark_derivative_enclosures', arithmetic='exact fractions',
           action_interval=[0, 1], derivative_orders=[0, 1, 2],
           reward_error_budgets=budgets)


def check_book_truncation(record):
    configurations, worst_ratio = 0, 0.0
    for cap in [1, 3, 6]:
        for rate in [0.5, 2.0]:
            horizon = 1.3
            exit_probability = poisson.sf(cap, rate*horizon)
            post_exit_time = quad(lambda t: poisson.sf(cap, rate*t), 0, horizon,
                                  epsabs=1e-13)[0]
            # Full reward is 1 after exit and full terminal payoff is 1 on exit;
            # capped rewards are zero. Direct birth-count sums give the difference.
            actual_gap = exit_probability+post_exit_time
            # Independent first-exit-time density is Gamma(cap+1, rate).
            coupling_bound = 2*quad(
                lambda t: (1+horizon-t)*gamma.pdf(t, a=cap+1, scale=1/rate),
                0, horizon, epsabs=1e-13)[0]
            uniform_bound = 2*(1+horizon)*exit_probability
            assert abs(2*actual_gap-coupling_bound) < 2e-12
            assert actual_gap <= coupling_bound+1e-12
            assert coupling_bound <= uniform_bound+1e-12
            worst_ratio = max(worst_ratio, actual_gap/uniform_bound)
            configurations += 1
    record('finite_book_exit_coupling', configurations=configurations,
           largest_gap_to_uniform_bound=float(worst_ratio))


def check_additive_forecast(record):
    q = np.array([[-0.4, 0.4], [0.7, -0.7]])
    qhat = np.array([[-0.43, 0.43], [0.68, -0.68]])
    d, dhat = np.array([0.12, -0.08]), np.array([0.126, -0.085])
    pi, pihat = np.array([0.8, 0.2]), np.array([0.77, 0.23])
    ratios = []
    for horizon in [0.2, 1.0, 3.0, 10.0]:
        truth = pi@integrated_semigroup(q, d, horizon)
        fitted = pihat@integrated_semigroup(qhat, dhat, horizon)
        # Independently integrate state probabilities and expected price drift.
        flow = solve_ivp(lambda t, y: np.r_[y[:2]@q, y[:2]@d],
                         [0, horizon], np.r_[pi, 0.0], rtol=1e-11, atol=1e-13)
        assert flow.success and abs(flow.y[-1, -1]-truth) < 1e-10
        for omega in [0.0, 1.1]:
            a, c = mix_constants(omega, horizon)
            bound = horizon*infnorm(d-dhat)+np.ptp(d)/2*(
                a*np.linalg.norm(pi-pihat, 1)+c*infnorm(q-qhat))
            assert abs(truth-fitted) <= bound+1e-12
            ratios.append(abs(truth-fitted)/bound)
    # Negative control: the one-state relative generator is identical, price is not.
    one_state_q = np.zeros((1, 1))
    means = [integrated_semigroup(one_state_q, np.array([lam*0.5]), 1)[0]
             for lam in [1.0, 4.0]]
    assert np.allclose(means, [0.5, 2.0])
    record('marked_price_clock_and_forecast_transfer', comparisons=8,
           largest_error_to_bound=float(max(ratios)),
           same_generator_different_clock_means=means)


def event_objects(u, matrices):
    nu = sum(m@np.ones(len(u)) for m in matrices)
    k = u-np.diag(nu)
    q = k+sum(matrices)
    assert np.allclose(q.sum(axis=1), 0)
    assert np.all(q-np.diag(np.diag(q)) >= 0)
    return k, q


def history_row(prior, k, matrices, intervals, events):
    row = prior.copy()
    for i, event in enumerate(events):
        row = row@expm(intervals[i]*k)@matrices[event]
    return row@expm(intervals[-1]*k)


def filter_comparison(prior, fitted_prior, k, khat, matrices, fitted_matrices,
                      intervals, events):
    constants = [max(infnorm(m), infnorm(mhat))
                 for m, mhat in zip(matrices, fitted_matrices)]
    b = [m/a for m, a in zip(matrices, constants)]
    bhat = [m/a for m, a in zip(fitted_matrices, constants)]
    row = history_row(prior, k, b, intervals, events)
    fitted_row = history_row(fitted_prior, khat, bhat, intervals, events)
    e = np.linalg.norm(prior-fitted_prior, 1)+sum(intervals)*infnorm(k-khat)
    e += sum(infnorm(b[j]-bhat[j]) for j in events)
    assert np.linalg.norm(row-fitted_row, 1) <= e+1e-13
    pi, pihat = row/row.sum(), fitted_row/fitted_row.sum()
    bound = min(2.0, 2*e/fitted_row.sum())
    assert np.linalg.norm(pi-pihat, 1) <= bound+1e-12
    return pi, pihat, bound, float(fitted_row.sum())


def check_event_filter(record):
    u = np.array([[-0.2, 0.2], [0.1, -0.1]])
    matrices = [np.array([[1.2, 0.1], [0.04, 0.5]]),
                np.array([[0.4, 0.02], [0.06, 0.9]])]
    uhat = np.array([[-0.205, 0.205], [0.098, -0.098]])
    fitted_matrices = [m*1.003 for m in matrices]
    k, _ = event_objects(u, matrices)
    khat, _ = event_objects(uhat, fitted_matrices)
    prior, fitted_prior = np.array([0.6, 0.4]), np.array([0.602, 0.398])
    intervals, events = [0.12, 0.07, 0.16, 0.1], [0, 1, 0]
    pi, pihat, bound, zhat = filter_comparison(
        prior, fitted_prior, k, khat, matrices, fitted_matrices, intervals, events)
    # A normalized nonlinear filtering ODE provides an independent comparison.
    normalized = prior.copy()
    for i, duration in enumerate(intervals):
        ode = solve_ivp(lambda t, p: p@k-p*(p@k@np.ones(2)),
                        [0, duration], normalized, rtol=1e-12, atol=1e-14)
        assert ode.success
        normalized = ode.y[:, -1]
        if i < len(events):
            normalized = normalized@matrices[events[i]]
            normalized /= normalized.sum()
    assert np.linalg.norm(normalized-pi, 1) < 1e-11
    quiet = np.array([0.5, 0.5])@expm(-np.diag([3.0, 1.0]))
    quiet /= quiet.sum()
    assert abs(quiet[0]-1/(1+math.exp(2))) < 1e-14
    assert abs(quiet@np.array([3.0, 1.0])-1.238405844044235) < 1e-13
    record('event_history_filter_and_silence',
           posterior_ode_discrepancy=float(np.linalg.norm(normalized-pi, 1)),
           actual_posterior_error=float(np.linalg.norm(pi-pihat, 1)),
           posterior_error_bound=float(bound), rescaled_fitted_likelihood=zhat,
           fast_regime_probability_after_silence=float(quiet[0]))


def full_book_generator():
    q = np.array([[0, 1, 0.4, 0.2], [0.3, 0, 0.1, 0.5],
                  [0.2, 0.2, 0, 0.7], [0.3, 0.1, 0.8, 0]], dtype=float)
    q -= np.diag(q.sum(axis=1))
    return q


def check_aggregation(record):
    q = full_book_generator()
    lift = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    coarse = np.array([[-0.6, 0.6], [0.4, -0.4]])
    assert infnorm(q@lift-lift@coarse) < 1e-15
    exact_error = 0.0
    for t in [0.1, 1.0, 4.0]:
        exact_error = max(exact_error, infnorm(expm(t*q)@lift-lift@expm(t*coarse)))
    assert exact_error < 3e-14
    # Exact forecast aggregation does not alone imply filter aggregation.
    matrices = [np.diag([1.3, 1.3, 0.6, 0.6]), np.diag([0.4, 0.4, 0.9, 0.9])]
    coarse_matrices = [np.diag([1.3, 0.6]), np.diag([0.4, 0.9])]
    k, _ = event_objects(q, matrices)
    coarse_k, _ = event_objects(coarse, coarse_matrices)
    assert infnorm(k@lift-lift@coarse_k) < 1e-15
    for m, mc in zip(matrices, coarse_matrices):
        assert infnorm(m@lift-lift@mc) < 1e-15
    prior = np.array([0.4, 0.1, 0.3, 0.2])
    row = history_row(prior, k, matrices, [0.1, 0.2, 0.15], [0, 1])
    coarse_row = history_row(prior@lift, coarse_k, coarse_matrices,
                             [0.1, 0.2, 0.15], [0, 1])
    filter_error = infnorm(row/row.sum()@lift-coarse_row/coarse_row.sum())
    assert filter_error < 1e-13
    unequal_marks = matrices[0].copy()
    unequal_marks[0, 0] += 0.01
    assert infnorm(unequal_marks@lift-lift@coarse_matrices[0]) > 0.009
    q[1, 3] += 0.15
    q[1, 1] -= 0.15
    residual = q@lift-lift@coarse
    horizon = 1.7
    direct = expm(horizon*q)@lift-lift@expm(horizon*coarse)
    integral = quad_vec(lambda s: expm((horizon-s)*q)@residual@expm(s*coarse),
                        0, horizon, epsabs=1e-12)[0]
    assert infnorm(direct-integral) < 2e-13
    pi, pihat = np.array([0.4, 0.1, 0.3, 0.2]), np.array([0.38, 0.12, 0.3, 0.2])
    coarse_d = np.array([0.2, -0.1])
    d = lift@coarse_d+np.array([0, 0.04, 0, 0])
    truth = pi@integrated_semigroup(q, d, horizon)
    fitted = (pihat@lift)@integrated_semigroup(coarse, coarse_d, horizon)
    a, c = mix_constants(1.0, horizon)
    bound = horizon*infnorm(d-lift@coarse_d)+np.ptp(coarse_d)/2*(
        a*np.linalg.norm(pi-pihat, 1)+c*infnorm(residual))
    assert abs(truth-fitted) <= bound+1e-12
    assert infnorm(residual) > 0.29
    record('book_state_aggregation_and_intertwining',
           exact_aggregation_error=exact_error,
           exact_coarse_filter_error=filter_error,
           perturbed_rate_residual=infnorm(residual),
           duhamel_identity_error=infnorm(direct-integral),
           forecast_error=float(abs(truth-fitted)), forecast_bound=float(bound))


def check_complete_decision(record):
    u = full_book_generator()
    u[1, 3] += 0.02
    u[1, 1] -= 0.02
    uhat = u.copy()
    uhat[1, 3] += 0.003
    uhat[1, 1] -= 0.003
    matrices = [np.diag([1.5, 1.56, 0.5, 0.52]),
                np.diag([0.65, 0.66, 1.2, 1.22])]
    fitted_matrices = [matrices[0]*1.001, matrices[1]*0.999]
    k, q = event_objects(u, matrices)
    khat, qhat = event_objects(uhat, fitted_matrices)
    prior, fitted_prior = np.array([0.4, 0.1, 0.3, 0.2]), np.array([0.401, 0.099, 0.3, 0.2])
    pi, pihat, epi, zhat = filter_comparison(
        prior, fitted_prior, k, khat, matrices, fitted_matrices,
        [0.08, 0.11, 0.04, 0.09], [0, 1, 0])
    lift = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    average = lift.T/2
    coarse = average@qhat@lift
    d = (matrices[0]-matrices[1])@np.ones(4)
    dhat = (fitted_matrices[0]-fitted_matrices[1])@np.ones(4)
    coarse_d = average@dhat
    eq, ed = infnorm(q-qhat), infnorm(d-dhat)
    dr = infnorm(qhat@lift-lift@coarse)+eq
    dd = infnorm(dhat-lift@coarse_d)+ed
    assert infnorm(q@lift-lift@coarse) <= dr+1e-12
    assert infnorm(d-lift@coarse_d) <= dd+1e-12
    horizon, curvature, fee, w0, limit = 0.7, 0.7, 0.02, 0.04, 0.6
    mu = pi@integrated_semigroup(q, d, horizon)
    muhat = (pihat@lift)@integrated_semigroup(coarse, coarse_d, horizon)
    a, c = mix_constants(coarse[0, 1]+coarse[1, 0], horizon)
    error_bound = horizon*dd+np.ptp(coarse_d)/2*(a*epi+c*dr)
    assert abs(mu-muhat) <= error_bound+1e-12

    def position(signal):
        residual = signal-curvature*w0
        return float(np.clip(w0+np.sign(residual)*max(abs(residual)-fee, 0)/curvature,
                             -limit, limit))

    def cost(w):
        return curvature*w*w/2+fee*abs(w-w0)

    wstar, fitted = position(mu), position(muhat)
    deployed = float(np.clip(fitted+0.007, -limit, limit))
    g = curvature*deployed+fee*np.sign(deployed-w0)
    if abs(deployed-w0) < 1e-13:
        g = float(np.clip(muhat, curvature*w0-fee, curvature*w0+fee))
    if deployed == limit:
        g = max(g, muhat)
    if deployed == -limit:
        g = min(g, muhat)
    chi = abs(g-muhat)
    loss = (mu*wstar-cost(wstar))-(mu*deployed-cost(deployed))
    bound = envelope(curvature, 2*limit, error_bound+chi)
    numeric = minimize_scalar(lambda w: cost(w)-mu*w, bounds=(-limit, limit),
                              method='bounded', options={'xatol': 1e-13})
    assert abs(numeric.fun-(cost(wstar)-mu*wstar)) < 1e-10
    assert loss >= -1e-12 and loss <= bound+1e-12
    record('events_to_coarse_forecast_to_position', full_states=4, coarse_states=2,
           rescaled_history_likelihood=zhat, posterior_error_bound=float(epi),
           rate_reduction_budget=dr, price_drift_budget=dd,
           true_forecast=float(mu), deployed_forecast=float(muhat),
           forecast_error_bound=float(error_bound), optimizer_residual=float(chi),
           actual_decision_loss=float(loss), certified_decision_loss=float(bound))


def run_checks(record):
    check_tagged_queue(record)
    check_replacement_clock(record)
    check_event_derivatives(record)
    check_book_truncation(record)
    check_additive_forecast(record)
    check_event_filter(record)
    check_aggregation(record)
    check_complete_decision(record)
