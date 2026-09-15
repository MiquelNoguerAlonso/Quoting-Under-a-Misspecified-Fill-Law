"""Sharp forecast constants and exact arithmetic interval-to-position bounds.

The complete interval calculation encloses exponentials by rational Taylor
sums and tails. Floating-point matrix exponentials serve only as comparisons.
"""
from fractions import Fraction as F
from itertools import product
import math

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

from order_book_foundations import (integrated_semigroup, mix_constants,
                                    full_book_generator, event_objects, history_row)


def floor_fraction(x, places=14):
    scale = 10**places
    return F(x.numerator*scale//x.denominator, scale)


def ceil_fraction(x, places=14):
    return -floor_fraction(-x, places)


def eye(n):
    return [[F(i == j) for j in range(n)] for i in range(n)]


def matmul(a, b):
    return [[sum(x*y for x, y in zip(row, column)) for column in zip(*b)] for row in a]


def rowmul(row, matrix):
    return [sum(x*y for x,y in zip(row,column)) for column in zip(*matrix)]


def exp_minus_enclosure(x, order=44):
    # The Lagrange remainder has the sign of the next Taylor term for x >= 0.
    assert x >= 0 and order % 2 == 0
    term, total = F(1), F(1)
    for k in range(1, order+1):
        term *= -x/k
        total += term
    upper = total
    lower = total+term*(-x)/(order+1)
    return max(F(0),lower), upper


def metzler_exp_enclosure(k, duration, order=30):
    n = len(k)
    shift = max(-k[i][i] for i in range(n))
    if shift == 0:
        shift = F(1)
    p = [[F(i == j)+k[i][j]/shift for j in range(n)] for i in range(n)]
    assert all(x >= 0 for row in p for x in row)
    x = shift*duration
    e_lower, e_upper = exp_minus_enclosure(x)
    term, total = eye(n), eye(n)
    for j in range(1, order+1):
        term = [[v*x/j for v in row] for row in matmul(term,p)]
        total = [[a+b for a,b in zip(ra,rb)] for ra,rb in zip(total,term)]
    y = x*max(sum(row) for row in p)
    assert y < order+2
    tail = e_upper*y**(order+1)/math.factorial(order+1)/(1-y/F(order+2))
    lower = [[floor_fraction(e_lower*v,16) for v in row] for row in total]
    upper = [[ceil_fraction(e_upper*v+tail,16) for v in row] for row in total]
    return lower, upper


def interval_history(prior_lower, prior_upper, k_lower, k_upper,
                     event_lower, event_upper, intervals, events):
    lower, upper = prior_lower[:], prior_upper[:]
    for j, duration in enumerate(intervals):
        pl, _ = metzler_exp_enclosure(k_lower, duration)
        _, pu = metzler_exp_enclosure(k_upper, duration)
        lower, upper = rowmul(lower,pl), rowmul(upper,pu)
        if j < len(events):
            lower = rowmul(lower,event_lower[events[j]])
            upper = rowmul(upper,event_upper[events[j]])
        lower = [floor_fraction(v,16) for v in lower]
        upper = [ceil_fraction(v,16) for v in upper]
    assert sum(lower) > 0
    return lower, upper


def forecast_enclosure(q_lower, q_upper, d_lower, d_upper, horizon,
                       rate=F(2), order=24):
    n = len(d_lower)
    assert all(sum(q_upper[i][j] for j in range(n) if j != i) <= rate for i in range(n))
    el, eu = exp_minus_enclosure(rate*horizon)
    power_l, power_u = d_lower[:], d_upper[:]
    out_l, out_u = [F(0)]*n, [F(0)]*n
    poisson_term, poisson_prefix, weight_sum_l = F(1), F(1), F(0)
    for k in range(order+1):
        if k:
            poisson_term *= rate*horizon/k
            poisson_prefix += poisson_term
        wl = max(F(0),(1-eu*poisson_prefix)/rate)
        wu = max(F(0),(1-el*poisson_prefix)/rate)
        weight_sum_l += wl
        for i in range(n):
            out_l[i] += min(wl*power_l[i],wu*power_l[i])
            out_u[i] += max(wl*power_u[i],wu*power_u[i])
        if k < order:
            nl, nu = [], []
            for i in range(n):
                low_sum, high_sum = F(0), F(0)
                for j in range(n):
                    if i != j:
                        dl, du = power_l[j]-power_l[i], power_u[j]-power_u[i]
                        low_sum += min(q_lower[i][j]*dl,q_upper[i][j]*dl)
                        high_sum += max(q_lower[i][j]*du,q_upper[i][j]*du)
                nl.append(power_l[i]+low_sum/rate)
                nu.append(power_u[i]+high_sum/rate)
            power_l,power_u = nl,nu
    tail = max(F(0),horizon-weight_sum_l)
    bound = max(abs(v) for v in d_lower+d_upper)
    return ([floor_fraction(v-bound*tail) for v in out_l],
            [ceil_fraction(v+bound*tail) for v in out_u],tail)


def ratio_extreme(lower, upper, scores, maximize):
    order = sorted(range(len(scores)),key=lambda i:scores[i])
    candidates = []
    for cut in range(len(scores)+1):
        below = set(order[:cut])
        weights = [((lower[i] if i in below else upper[i]) if maximize else
                    (upper[i] if i in below else lower[i])) for i in range(len(scores))]
        if sum(weights) > 0:
            candidates.append(sum(a*b for a,b in zip(weights,scores))/sum(weights))
    return (max if maximize else min)(candidates)


def position(mu, a, fee, w0, limit):
    residual = mu-a*w0
    soft = max(F(0),abs(residual)-fee)*(1 if residual >= 0 else -1)
    return min(limit,max(-limit,w0+soft/a))


def cost(w, a, fee, w0):
    return a*w*w/2+fee*abs(w-w0)


def conjugate(mu, a, fee, w0, limit):
    w = position(mu,a,fee,w0,limit)
    return mu*w-cost(w,a,fee,w0)


def check_sharp_forecast_constants(record):
    ratios = []
    for omega in [0.,0.7,2.0]:
        q = np.array([[0.,0.],[omega,-omega]])
        d = np.array([0.,0.8])
        for horizon in [0.2,1.,4.]:
            a,c = mix_constants(omega,horizon)
            for epsilon in [1e-2,1e-3,1e-4]:
                fitted_q = q+np.array([[-epsilon,epsilon],[0.,0.]])
                actual = integrated_semigroup(fitted_q,d,horizon)[0]
                formula = d[1]*epsilon*mix_constants(omega+epsilon,horizon)[1]
                bound = d[1]*epsilon*c
                assert abs(actual-formula) < 2e-12
                assert 0 <= actual <= bound+2e-12
                if epsilon == 1e-4:
                    ratios.append(actual/bound)
            posterior = np.array([0.98,0.02])
            p_error = posterior@integrated_semigroup(q,d,horizon)
            assert abs(p_error-0.02*d[1]*a) < 1e-13
            shifted = integrated_semigroup(q,d+0.003,horizon)[0]
            assert abs(shifted-0.003*horizon) < 1e-13
    assert min(ratios) > 0.9998
    record('sharp_additive_forecast_coefficients', generator_cases=27,
           exact_posterior_and_drift_cases=18,
           smallest_near_sharp_ratio=float(min(ratios)),
           largest_near_sharp_ratio=float(max(ratios)))


def check_interval_minimax(record):
    cases = 0
    for a in [F(1,2),F(2)]:
        for fee in [F(0),F(1,5)]:
            for limit in [F(1,2),F(2)]:
                w0 = F(1,10)
                for left,right in [(F(-2),F(-1)),(F(-1),F(1)),(F(0),F(1)),
                                   (F(1,20),F(1,10)),(F(1),F(3))]:
                    fl,fr = [conjugate(x,a,fee,w0,limit) for x in [left,right]]
                    center = (fr-fl)/(right-left)
                    value = max(fl-left*center,fr-right*center)+cost(center,a,fee,w0)
                    assert -limit <= center <= limit
                    def objective(w):
                        return max(float(fl)-float(left)*w,float(fr)-float(right)*w)+float(a)*w*w/2+float(fee)*abs(w-float(w0))
                    solution = minimize_scalar(objective,bounds=(-float(limit),float(limit)),
                                               method='bounded',options={'xatol':1e-12})
                    numerical_value = min(solution.fun,objective(-float(limit)),objective(float(limit)))
                    assert abs(numerical_value-float(value)) < 2e-8
                    cases += 1
    # Exact quadratic equality and a complete no-trade interval.
    a,left,right = F(2),F(-1),F(3)
    center = (left+right)/(2*a)
    assert max((left-a*center)**2,(right-a*center)**2)/(2*a)==(right-left)**2/(8*a)
    a,fee,w0,limit = F(1),F(1,5),F(1,10),F(1)
    left,right = F(-1,20),F(1,5)
    fl,fr = [conjugate(x,a,fee,w0,limit) for x in [left,right]]
    assert (fr-fl)/(right-left)==w0
    assert fl-left*w0+cost(w0,a,fee,w0)==0
    assert fr-right*w0+cost(w0,a,fee,w0)==0
    # A proportional-cost threshold gives an exact linear minimax radius.
    fee,error = F(1,5),F(1,20)
    left,right = fee-error,fee+error
    fl,fr = max(F(0),abs(left)-fee),max(F(0),abs(right)-fee)
    pure_center = (fr-fl)/(right-left)
    assert pure_center == F(1,2)
    assert max(fl-left*pure_center,fr-right*pure_center)+fee*abs(pure_center) == error/2
    # A nonquadratic friction checks the same secant rule independently.
    for left,right in [(-1.3,0.8),(0.2,1.5),(-0.4,0.1)]:
        f = lambda w: 0.4*w*w+0.2*w**4
        star = lambda mu: -minimize_scalar(lambda w:f(w)-mu*w,bounds=(-3,3),
                                            method='bounded',options={'xatol':1e-13}).fun
        fl,fr = star(left),star(right)
        center = (fr-fl)/(right-left)
        direct = minimize_scalar(lambda w:f(w)+max(fl-left*w,fr-right*w),
                                  bounds=(-3,3),method='bounded',options={'xatol':1e-12})
        assert abs(direct.fun-(f(center)+max(fl-left*center,fr-right*center)))<2e-8
    record('exact_interval_minimax_position', mixed_cost_cases=cases,
           nonquadratic_cases=3, exact_quadratic_radius='(upper-lower)^2/(8a)',
           no_trade_interval_loss='0', pure_threshold_center='1/2',
           pure_threshold_radius='error/2')


def check_interval_book_forecast(record):
    q0 = [[F(-16,10),F(1),F(4,10),F(2,10)],
          [F(3,10),F(-9,10),F(1,10),F(5,10)],
          [F(2,10),F(2,10),F(-11,10),F(7,10)],
          [F(3,10),F(1,10),F(8,10),F(-12,10)]]
    qhat = [row[:] for row in q0]
    qhat[1][3] += F(23,1000)
    qhat[1][1] -= F(23,1000)
    ql,qu = [row[:] for row in qhat],[row[:] for row in qhat]
    ql[1][3] -= F(3,1000)
    qu[1][3] += F(3,1000)
    true_plus = list(map(F,['1.5','1.56','0.5','0.52']))
    true_minus = list(map(F,['0.65','0.66','1.2','1.22']))
    plus = [v*F(1001,1000) for v in true_plus]
    minus = [v*F(999,1000) for v in true_minus]
    plus_l,plus_u = [v-F(2,1000) for v in plus],[v+F(2,1000) for v in plus]
    minus_l,minus_u = [v-F(15,10000) for v in minus],[v+F(15,10000) for v in minus]
    kl,ku = [row[:] for row in ql],[row[:] for row in qu]
    for i in range(4):
        kl[i][i] = -sum(qu[i][j] for j in range(4) if i != j)-plus_u[i]-minus_u[i]
        ku[i][i] = -sum(ql[i][j] for j in range(4) if i != j)-plus_l[i]-minus_l[i]
    diagonal = lambda v:[[v[i] if i==j else F(0) for j in range(4)] for i in range(4)]
    prior = list(map(F,['0.401','0.099','0.3','0.2']))
    prior_l = [v-(F(1,1000) if i<2 else F(0)) for i,v in enumerate(prior)]
    prior_u = [v+(F(1,1000) if i<2 else F(0)) for i,v in enumerate(prior)]
    intervals = list(map(F,['0.08','0.11','0.04','0.09']))
    alpha_l,alpha_u = interval_history(prior_l,prior_u,kl,ku,
        [diagonal(plus_l),diagonal(minus_l)],[diagonal(plus_u),diagonal(minus_u)],
        intervals,[0,1,0])
    dl,du = [x-y for x,y in zip(plus_l,minus_u)],[x-y for x,y in zip(plus_u,minus_l)]
    future_l,future_u,tail = forecast_enclosure(ql,qu,dl,du,F(7,10))
    lo = floor_fraction(ratio_extreme(alpha_l,alpha_u,future_l,False),12)
    hi = ceil_fraction(ratio_extreme(alpha_l,alpha_u,future_u,True),12)
    for scores,maximize in [(future_l,False),(future_u,True)]:
        vertices = []
        for bits in product([0,1],repeat=4):
            weights = [alpha_u[i] if bits[i] else alpha_l[i] for i in range(4)]
            vertices.append(sum(x*y for x,y in zip(weights,scores))/sum(weights))
        assert ratio_extreme(alpha_l,alpha_u,scores,maximize)==(max if maximize else min)(vertices)
    # Reference model from the previous worked example, evaluated independently.
    true_q = full_book_generator()
    true_q[1,3] += 0.02
    true_q[1,1] -= 0.02
    true_ms = [np.diag(list(map(float,true_plus))),np.diag(list(map(float,true_minus)))]
    true_k,_ = event_objects(true_q,true_ms)
    alpha = history_row(np.array([.4,.1,.3,.2]),true_k,true_ms,list(map(float,intervals)),[0,1,0])
    future = integrated_semigroup(true_q,np.array(list(map(float,[x-y for x,y in zip(true_plus,true_minus)]))),.7)
    actual = float(alpha@future/alpha.sum())
    assert all(float(l)-1e-14 <= x <= float(u)+1e-14 for l,u,x in zip(alpha_l,alpha_u,alpha))
    assert all(float(l)-1e-13 <= x <= float(u)+1e-13 for l,u,x in zip(future_l,future_u,future))
    assert float(lo) <= actual <= float(hi)
    a,fee,w0,limit = F(7,10),F(1,50),F(1,25),F(3,5)
    fl,fr = [conjugate(x,a,fee,w0,limit) for x in [lo,hi]]
    deployed = F(287417666770,10**12)
    loss_bound = max(fl-lo*deployed,fr-hi*deployed)+cost(deployed,a,fee,w0)
    minimax = (fr-fl)/(hi-lo)
    minimax_bound = max(fl-lo*minimax,fr-hi*minimax)+cost(minimax,a,fee,w0)
    assert minimax_bound <= loss_bound
    implementable_minimax = floor_fraction(minimax,12)
    rounded_minimax_bound = max(fl-lo*implementable_minimax,fr-hi*implementable_minimax)+cost(implementable_minimax,a,fee,w0)
    assert loss_bound < F(3088774,10**9)
    assert tail < F(1,10**20)
    # Recompute the earlier norm certificate for exactly this primitive box.
    khat = [row[:] for row in qhat]
    for i in range(4):
        khat[i][i] -= plus[i]+minus[i]
    fitted_alpha_l,fitted_alpha_u = interval_history(prior,prior,khat,khat,
        [diagonal(plus),diagonal(minus)],[diagonal(plus),diagonal(minus)],intervals,[0,1,0])
    scale_plus,scale_minus = max(plus_u),max(minus_u)
    z_lower = sum(fitted_alpha_l)/(scale_plus**2*scale_minus)
    ehist = F(2,1000)+sum(intervals)*F(95,10000)
    ehist += 2*F(2,1000)/scale_plus+F(15,10000)/scale_minus
    posterior_error = min(F(2),2*ehist/z_lower)
    dhat = [x-y for x,y in zip(plus,minus)]
    coarse_d = [(dhat[0]+dhat[1])/2,(dhat[2]+dhat[3])/2]
    coarse_q = [[-F(1223,2000),F(1223,2000)],[F(2,5),-F(2,5)]]
    omega = F(2023,2000)
    el,eu = exp_minus_enclosure(omega*F(7,10))
    a_upper = (1-el)/omega
    c_upper = F(7,10)/omega-(1-eu)/omega**2
    delta_d = max(abs(dhat[i]-coarse_d[i//2]) for i in range(4))+F(35,10000)
    global_error = F(7,10)*delta_d+(max(coarse_d)-min(coarse_d))/2*(
        a_upper*posterior_error+c_upper*F(29,1000))
    coarse_l,coarse_u,_ = forecast_enclosure(coarse_q,coarse_q,coarse_d,coarse_d,F(7,10))
    lifted_l,lifted_u = [coarse_l[i//2] for i in range(4)],[coarse_u[i//2] for i in range(4)]
    mhat_l = ratio_extreme(fitted_alpha_l,fitted_alpha_u,lifted_l,False)
    mhat_u = ratio_extreme(fitted_alpha_l,fitted_alpha_u,lifted_u,True)
    chi = max(abs(a*deployed+fee-mhat_l),abs(a*deployed+fee-mhat_u))
    same_box_global = (global_error+chi)**2/(2*a)
    record('rational_event_to_position_interval', arithmetic='exact rational enclosures',
           metzler_series_order=30, exponential_taylor_order=44,
           uniformization_order=24, certified_poisson_tail_upper=str(ceil_fraction(tail,24)),
           forecast_lower=str(lo), forecast_upper=str(hi),
           reference_forecast=actual, deployed_position=str(deployed),
           deployed_loss_upper=str(ceil_fraction(loss_bound,12)),
           minimax_position=str(implementable_minimax),
           minimax_loss_upper=str(ceil_fraction(rounded_minimax_bound,12)),
           earlier_example_uniform_loss_bound=0.0030887738336221075,
           same_box_uniform_loss_upper=str(ceil_fraction(same_box_global,12)),
           uniform_to_interval_improvement=float(same_box_global/loss_bound))


def run_checks(record):
    check_sharp_forecast_constants(record)
    check_interval_minimax(record)
    check_interval_book_forecast(record)
