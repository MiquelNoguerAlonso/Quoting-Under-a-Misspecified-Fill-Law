"""Queue-control enclosures and statistical lower-bound checks.

The interval Bellman iterations use exact fractions with outward rounding.
Independent policy iteration and linear evaluation use floating-point arrays.
"""
from fractions import Fraction as F
import math

import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize_scalar
from scipy.stats import norm, poisson


ROUND = 10**12


def down(x):
    return F((x.numerator*ROUND)//x.denominator, ROUND)


def up(x):
    return -down(-x)


def book_cell():
    states = []
    for z in [0, 1]:
        for q in range(-2, 3):
            states.append((z, q, 0, 0, 0))
            for side in [-1, 1]:
                if -2 <= q+side <= 2:
                    for level in [0, 1]:
                        for rank in [0, 1]:
                            states.append((z, q, side, level, rank))
    index = {s: i for i, s in enumerate(states)}
    events, actions, base = [], [], []
    rate_radius, mark_radius = F(1, 5000), F(1, 100000)
    for i, (z, q, side, level, rank) in enumerate(states):
        raw = [(index[(1-z, q, side, level, rank)], [F(35, 100), F(55, 100)][z], F(0))]
        if side:
            mu = [[F(6, 5), F(3, 5)], [F(11, 5), F(1)]][z][level]
            mu *= F(26, 25) if side == -1 else F(24, 25)
            gain = [[F(16, 100), F(29, 100)], [F(3, 100), F(1, 5)]][z][level]
            if rank:
                raw.append((index[(z, q, side, level, 0)], mu+F(1, 4), F(0)))
            else:
                raw.append((index[(z, q+side, 0, 0, 0)], mu, gain))
            raw.append((index[(z, q, 0, 0, 0)], F(1, 5), F(0)))
        rows = []
        for j, lam, gain in raw:
            radius = mark_radius if gain else F(0)
            rows.append((j, lam-rate_radius, lam+rate_radius,
                         gain-radius, gain+radius))
        aa = [('keep', i, F(0))]
        if side:
            aa.append(('cancel', index[(z, q, 0, 0, 0)], F(1, 500)))
        for new_side in [-1, 1]:
            if -2 <= q+new_side <= 2:
                for new_level in [0, 1]:
                    aa.append((str((new_side, new_level)),
                               index[(z, q, new_side, new_level, 1)],
                               F(1, 250) if side else F(1, 1000)))
        events.append(rows)
        actions.append(aa)
        base.append(-F(3, 100)*q*q)
    return states, index, events, actions, base


def check_interval_book_control(record):
    states, index, events, actions, base = book_cell()
    n, beta, nu, rate = len(states), F(3, 5), F(3), F(13, 2)
    assert n == 74
    assert all(sum(e[2] for e in row)+nu <= rate for row in events)
    assert all(e[1] > 0 for row in events for e in row)

    def matrices(policy, fraction=F(0)):
        q, r = np.zeros((n, n)), np.array(list(map(float, base)))
        for i in range(n):
            for j, lo, hi, gl, gu in events[i]:
                lam = float((lo+hi)/2+fraction*(hi-lo)/2)
                gain = float((gl+gu)/2+fraction*(gu-gl)/2)
                q[i, j] += lam
                q[i, i] -= lam
                r[i] += lam*gain
            _, j, fee = actions[i][policy[i]]
            q[i, j] += float(nu)
            q[i, i] -= float(nu)
            r[i] -= float(nu*fee)
        assert np.max(np.abs(q.sum(axis=1))) < 2e-15
        return q, r

    def optimum(fraction=F(0)):
        policy = np.zeros(n, dtype=int)
        for _ in range(40):
            q, r = matrices(policy, fraction)
            value = solve(float(beta)*np.eye(n)-q, r)
            new = np.array([np.argmax([value[j]-float(c) for _, j, c in row])
                            for row in actions])
            if np.all(new == policy):
                return policy, value
            policy = new
        raise AssertionError('Policy iteration failed to stabilize.')

    center_policy, center_value = optimum()
    deployed = center_policy.copy()
    differences = []
    for i, (z, q, side, level, rank) in enumerate(states):
        if side and rank:
            front = index[(z, q, side, level, 0)]
            name = actions[front][center_policy[front]][0]
            deployed[i] = next(j for j, a in enumerate(actions[i]) if a[0] == name)
            if deployed[i] != center_policy[i]:
                differences.append(i)

    def bellman(vector, upper):
        result = []
        choose = max if upper else min
        rounding = up if upper else down
        for i in range(n):
            external = base[i]
            for j, lo, hi, gl, gu in events[i]:
                external += choose(lam*(g+vector[j]-vector[i])
                                   for lam in [lo, hi] for g in [gl, gu])
            choice = max(vector[j]-vector[i]-fee for _, j, fee in actions[i])
            result.append(rounding((rate*vector[i]+external+nu*choice)/(beta+rate)))
        return result

    lower, upper = [-1/beta]*n, [1/beta]*n
    # Reward magnitudes are bounded by 1 throughout the declared uncertainty box.
    for i, row in enumerate(events):
        assert abs(base[i])+sum(e[2]*max(abs(e[3]), abs(e[4])) for e in row) + nu*F(1,250) < 1
    for _ in range(320):
        lower, upper = bellman(lower, False), bellman(upper, True)
    assert all(lo <= hi for lo, hi in zip(lower, upper))

    gaps = []
    for i, row in enumerate(actions):
        _, chosen, chosen_fee = row[deployed[i]]
        candidates = []
        for _, target, fee in row:
            value_diff = upper[target]-lower[chosen] if target != chosen else F(0)
            candidates.append(value_diff-fee+chosen_fee)
        gaps.append(nu*max(F(0), *candidates))
    loss_upper = [max(gaps)/beta]*n
    for _ in range(320):
        new = []
        for i, row in enumerate(events):
            external = sum(max(lo*(loss_upper[j]-loss_upper[i]),
                               hi*(loss_upper[j]-loss_upper[i]))
                           for j, lo, hi, gl, gu in row)
            _, target, _ = actions[i][deployed[i]]
            numerator = gaps[i]+rate*loss_upper[i]+external
            numerator += nu*(loss_upper[target]-loss_upper[i])
            new.append(up(numerator/(beta+rate)))
        loss_upper = new
    start = index[(0, 0, 0, 0, 0)]
    comparisons = []
    for fraction in [F(0), F(1,2), F(-1,2)]:
        true_policy, true_value = optimum(fraction)
        q, r = matrices(deployed, fraction)
        direct = solve(float(beta)*np.eye(n)-q, r)
        actual_gaps = np.zeros(n)
        for i, row in enumerate(actions):
            _, chosen, fee = row[deployed[i]]
            actual_gaps[i] = float(nu)*(max(true_value[j]-float(c) for _, j, c in row)
                                       -true_value[chosen]+float(fee))
        occupation = solve(float(beta)*np.eye(n)-q, actual_gaps)
        assert np.max(np.abs(occupation-(true_value-direct))) < 2e-13
        assert np.all(true_value >= np.array(list(map(float, lower)))-1e-13)
        assert np.all(true_value <= np.array(list(map(float, upper)))+1e-13)
        assert np.all(actual_gaps <= np.array(list(map(float, gaps)))+1e-13)
        assert np.all(true_value-direct <= np.array(list(map(float, loss_upper)))+1e-12)
        comparisons.append(float(true_value[start]-direct[start]))
    assert comparisons[0] > 0
    assert loss_upper[start] < max(gaps)/beta
    record('interval_queue_inventory_control', states=n, outward_decimal_places=12,
           exact_rational_iterations=320, model_comparisons=3,
           rank_sensitive_states=len(differences), center_optimal_value=float(center_value[start]),
           rank_blind_policy_losses=comparisons,
           certified_loss_upper=str(up(loss_upper[start])),
           uniform_gap_upper=float(max(gaps)/beta),
           certified_zero_gap_states=sum(g == 0 for g in gaps),
           largest_value_interval_width=float(max(hi-lo for lo, hi in zip(lower, upper))))


def check_rank_information_lower_bound(record):
    front, back, replacement = F(1,4), F(1,16), F(33,200)
    a, b = front-replacement, replacement-back
    keep_probability, minimax = a/(a+b), a*b/(a+b)
    assert minimax == F(697,15000)
    assert (1-keep_probability)*a == keep_probability*b == minimax
    numeric = minimize_scalar(lambda p: max((1-p)*float(a), p*float(b)),
                              bounds=(0,1), method='bounded')
    assert abs(numeric.fun-float(minimax)) < 1e-6
    record('queue_rank_information_lower_bound', arithmetic='exact fractions',
           optimal_keep_probability=str(keep_probability), minimax_loss=str(minimax))


def check_queue_statistical_boundary(record):
    n, reward, theta, mu0, mulow, muhigh = 2, 0.5, 1.0, 1.0, 0.8, 1.2
    value = lambda mu: reward*(mu/(mu+theta))**(n+1)
    derivative = lambda mu: reward*(n+1)*theta*mu**n/(mu+theta)**(n+2)
    alternative = value(mu0)
    lipschitz = derivative(1.0)
    derivative_lower = reward*(n+1)*theta*mulow**n/(muhigh+theta)**(n+2)
    d0 = math.sqrt(mulow)/8
    max_ratios, limits = [], []
    z = 0.75
    for duration in [100, 1000, 10000, 100000]:
        displacement = d0/math.sqrt(duration)
        mm, mp = mu0-displacement, mu0+displacement
        kl = duration*(mm*math.log(mm/mp)-mm+mp)
        assert kl <= 1/16+1e-11
        lower = 3*derivative_lower*d0/(8*math.sqrt(duration))
        actual_gaps = [alternative-value(mm), value(mp)-alternative]
        testing_lower = min(actual_gaps)*(1-math.sqrt(kl/2))/2
        assert testing_lower >= lower
        for mu in np.linspace(mulow,muhigh,81):
            threshold = math.ceil(duration*mu0)
            # Ties choose keeping; the observation is market-order count, not fills.
            wrong = poisson.cdf(threshold-1,duration*mu) if mu>=mu0 else poisson.sf(threshold-1,duration*mu)
            risk = abs(value(mu)-alternative)*wrong
            upper = lipschitz*math.sqrt(muhigh/duration)
            assert risk <= upper+1e-14
            max_ratios.append(risk/upper)
        local_mu = mu0+z*math.sqrt(mu0/duration)
        probability = poisson.cdf(math.ceil(duration*mu0)-1,duration*local_mu)
        limits.append(math.sqrt(duration)*(value(local_mu)-alternative)*probability)
    limit = derivative(mu0)*math.sqrt(mu0)*z*norm.cdf(-z)
    assert abs(limits[-1]-limit) < 1e-4
    record('queue_replacement_statistical_rate', rate='duration^(-1/2)',
           poisson_risk_comparisons=324, largest_risk_to_upper=float(max(max_ratios)),
           scaled_local_risks=limits, analytical_local_limit=float(limit),
           lower_bound_constant=float(3*derivative_lower*d0/8))


def run_checks(record):
    check_interval_book_control(record)
    check_rank_information_lower_bound(record)
    check_queue_statistical_boundary(record)
