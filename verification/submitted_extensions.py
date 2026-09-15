"""Independent synthetic checks for the six submitted paper extensions.

Finite-sample Poisson expectations use truncated probability sums in floating point,
not Monte Carlo. Continuous formulas are compared with optimization,
matrix exponentials, quadrature, singular values, and Lyapunov equations.
"""
import math
import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm, solve_continuous_lyapunov, svdvals
from scipy.optimize import minimize_scalar, brentq
from scipy.stats import poisson


def check_discounted_certificates(record):
    true = (np.array([.4, .6]), np.array([.3, .4]), np.array([.2, .1]))
    fitted = (np.array([.42, .58]), np.array([.31, .41]), np.array([.22, .12]))

    def policy(h, model):
        b, _, d = model
        return np.clip(b+d*(h[::-1]-h), 0, 1)

    def coefficients(a, model):
        b, lam, d = model
        rates = lam+d*a
        return np.array([[-rates[0], rates[0]],
                         [rates[1], -rates[1]]]), b*a-a*a/2

    def solve(beta, model):
        a = np.array([.5, .5])
        for _ in range(50):
            K, r = coefficients(a, model)
            h = np.linalg.solve(beta*np.eye(2)-K, r)
            anew = policy(h, model)
            if np.max(abs(anew-a)) < 1e-13:
                return h
            a = anew
        raise AssertionError('Policy iteration did not converge')

    rows = []
    for beta in [.2, 1., 3.]:
        h, hh = solve(beta, true), solve(beta, fitted)
        ah = policy(hh, fitted)
        K, r = coefficients(ah, true)
        j = np.linalg.solve(beta*np.eye(2)-K, r)
        gap = beta*h-r-K@h
        integral = np.array([quad(lambda t: float(
            (expm((K-beta*np.eye(2))*t)@gap)[i]), 0, np.inf,
            epsabs=1e-12)[0] for i in range(2)])
        identity_error = float(np.max(abs(h-j-integral)))
        assert identity_error < 2e-12
        B = .5/beta
        d = (.02+2*B*.03)/beta
        b = .02+2*B*.02+2*d*.22
        assert np.max(abs(h-hh)) <= d
        assert np.min(h-j) >= -1e-12
        assert np.max(h-j) <= b*b/(2*beta)
        v = hh+np.array([.01, -.007])
        amax = policy(v, fitted)
        Kh, rh = coefficients(amax, fitted)
        rho = np.max(abs(beta*v-rh-Kh@v))
        span = float(np.ptp(v))
        D = (rho+.02+span*.03)/beta
        assert np.max(abs(h-v)) <= D
        aimp = np.clip(amax+.01, 0, 1)
        gradient = fitted[0]-aimp+fitted[2]*(v[::-1]-v)
        chi = np.where(aimp == 0, np.maximum(gradient, 0),
                       np.where(aimp == 1, np.maximum(-gradient, 0),
                                abs(gradient)))
        Kimp, rimp = coefficients(aimp, true)
        jimp = np.linalg.solve(beta*np.eye(2)-Kimp, rimp)
        b1 = .02+span*.02+2*D*.24
        computed_bound = np.max((b1+chi)**2)/(2*beta)
        assert np.max(h-jimp) <= computed_bound+1e-12
        rows.append({'discount_rate': beta, 'maximum_regret': float(max(h-j)),
                     'performance_identity_error': identity_error,
                     'computed_bound': float(computed_bound),
                     'implemented_regret': float(max(h-jimp))})
    # The transition-level term yields beta^-2 in b and beta^-5 in its bound.
    powers = []
    for beta in [1e-3, 1e-4, 1e-5]:
        b = .01+(.02+.02)/beta+.04/beta**2
        scaled = b*b/(2*beta)*beta**5
        powers.append(float(scaled))
    assert abs(powers[-1]/(.04**2/2)-1) < 3e-5
    record('discounted_hjb_and_computed_certificate', cases=rows,
           beta_fifth_power_rescaled_bounds=powers,
           limiting_rescaled_bound=.04**2/2)


def poisson_expected_loss(kappa, tau, delta0=.4, Delta=.2, D=.6):
    means = [tau, tau*math.exp(-kappa*Delta)]
    counts, weights = [], []
    omitted_probability = 0.
    for mean in means:
        lo, hi = int(poisson.ppf(1e-13, mean)), int(poisson.ppf(1-1e-13, mean))
        n = np.arange(lo, hi+1)
        omitted_probability += float(poisson.cdf(lo-1, mean)+poisson.sf(hi, mean))
        counts.append(n)
        weights.append(poisson.pmf(n, mean))
    Fstar = math.exp(kappa*delta0-1)/kappa
    expectation = 0.
    for start in range(0, len(counts[0]), 256):
        n0 = counts[0][start:start+256, None]
        n1 = counts[1][None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            kh = np.log(n0/n1)/Delta
            dh = np.where((n0 > 0) & (n1 > 0) & (kh > 0),
                          np.clip(1/kh, 0, D), 0.)
        losses = Fstar-dh*np.exp(-kappa*(dh-delta0))
        assert np.min(losses) > -1e-13
        expectation += float(weights[0][start:start+256] @ losses @ weights[1])
    return expectation, omitted_probability*Fstar


def check_two_quote_experiment(record):
    rows = []
    for kappa in [2., 3.]:
        Delta, delta0, tau, epsilon = .2, .4, 20000., .05
        m0, m1 = tau, tau*math.exp(-kappa*Delta)
        t = math.sqrt(3*math.log(4/epsilon)/m1)
        assert t <= min(.5, kappa*Delta/8)
        prob_event = 1.
        for mean in [m0, m1]:
            lo, hi = math.ceil((1-t)*mean), math.floor((1+t)*mean)
            prob_event *= float(poisson.cdf(hi, mean)-poisson.cdf(lo-1, mean))
        assert prob_event >= 1-epsilon
        M = 2*kappa*math.exp(kappa*delta0)
        bound1 = 8*M*t*t/(Delta**2*kappa**2*(kappa-4*t/Delta)**2)
        bound2 = 64*math.exp(kappa*(delta0+Delta))/(Delta**2*kappa**3)*3*math.log(4/epsilon)/tau
        assert bound1 <= bound2
        # All count ratios on the concentration event lie in this interval.
        for e0 in [-t, 0., t]:
            for e1 in [-t, 0., t]:
                kh = math.log(m0*(1+e0)/(m1*(1+e1)))/Delta
                dh = np.clip(1/kh, 0, .6)
                loss = math.exp(kappa*delta0-1)/kappa-dh*math.exp(-kappa*(dh-delta0))
                assert 0 <= loss+1e-13 <= bound1+1e-12
        leading = math.exp(kappa*delta0-1)*(1+math.exp(kappa*Delta))/(2*Delta**2*kappa**3)
        expected_rows = []
        for duration in [1000., 5000., 20000.]:
            expected, tail = poisson_expected_loss(kappa, duration)
            expected_rows.append({'tau': duration, 'expected_loss': expected,
                                  'tau_times_loss': duration*expected,
                                  'omitted_tail_loss_bound': tail})
        assert abs(expected_rows[-1]['tau_times_loss']/leading-1) < .015
        rows.append({'kappa': kappa, 'concentration_event_probability': prob_event,
                     'asymptotic_loss_constant': leading, 'poisson_probability_sums': expected_rows})
    root = brentq(lambda x: x-2*(1+math.exp(-x)), 2, 3)
    assert root > 2
    for x in np.linspace(.001, 2, 100):
        assert math.exp(x)*(x-2)-2 < 0
    record('two_quote_coverage_and_expected_loss', cases=rows,
           unconstrained_spacing_root=root, largest_feasible_scaled_spacing=2.)


def check_unbounded_signal_ceiling(record):
    largest_mgf_error = 0.
    for k in [.6, 2., 5.]:
        for delta in [0., .2, 1.5]:
            p = math.exp(-k*delta)
            for s in [.01, 1., 10., 100.]:
                exact = 1-p*s/(k+s)
                numeric = (quad(lambda j: k*math.exp(-k*j), 0, delta)[0]
                           +quad(lambda j: math.exp(s*(delta-j))*k*math.exp(-k*j), delta, np.inf)[0])
                largest_mgf_error = max(largest_mgf_error, abs(exact-numeric))
                centered = math.log(exact)+p*s/k
                assert centered <= p*s*s/(k*(k+s))+1e-12
                assert centered <= s/k-math.log1p(s/k)+1e-12
                assert centered <= s*s/(2*k**2)+1e-12
    assert largest_mgf_error < 1e-10
    k, cut = 2., .5
    mass = np.array([1-math.exp(-k*cut), math.exp(-k*cut)])
    likelihood = np.array([[.8, .2], [.2, .8]])  # rows: observed signal; columns: mark bin
    py = likelihood@mass
    kl = np.sum((likelihood*mass[None, :]/py[:, None])
                *np.log(likelihood/py[:, None]), axis=1)
    information = float(py@kl)
    actions = [0., .15, .3, .5, .8]
    conditional_rewards = np.zeros((2, len(actions)))
    for y in range(2):
        for i, delta in enumerate(actions):
            points = sorted(set([0., delta, cut]))+[np.inf]
            value = 0.
            for lo, hi in zip(points[:-1], points[1:]):
                def integrand(j):
                    qy = likelihood[y, int(j > cut)]
                    return -max(j-delta, 0)*qy*k*math.exp(-k*j)
                value += quad(integrand, lo, hi)[0]
            conditional_rewards[y, i] = 2*delta*math.exp(-3*delta)+value/py[y]
    gain = float(py@conditional_rewards.max(axis=1)-(py@conditional_rewards).max())
    conditional_bound = float(py@(np.minimum(np.sqrt(2*kl), 1.)/k))
    bound = min(math.sqrt(2*information), 1.)/k
    assert 0 < gain <= conditional_bound <= bound+1e-12
    for t in [.001, .1, .5, 2., 8.]:
        sigma, b = .7, .5
        expected = (math.sqrt(2*sigma*sigma*t) if t <= sigma*sigma/(2*b*b)
                    else b*t+sigma*sigma/(2*b))
        f = lambda s: (t+sigma*sigma*s*s/2)/s
        optimized = minimize_scalar(f, bounds=(1e-10, 1/b), method='bounded',
                                    options={'xatol': 1e-12})
        assert abs(optimized.fun-expected) < 1e-6
    record('unbounded_mgf_and_signal_value', maximum_mgf_quadrature_error=largest_mgf_error,
           signal_value_per_opportunity=gain, mutual_information_nats=information,
           conditional_entropy_bound=conditional_bound, information_bound=bound)


def check_curvature_diameter(record):
    largest_ratio = 0.
    count = 0
    for a in [.4, 1., 3.]:
        for W in [.2, 2.]:
            for c in [0., .4]:
                F = lambda w: a*w*w/2+c*abs(w)
                def solve(mu):
                    out = minimize_scalar(lambda w: F(w)-mu*w,
                                          bounds=(-W/2, W/2), method='bounded',
                                          options={'xatol': 1e-13})
                    candidates = [out.x, -W/2, W/2, 0.]
                    return max(candidates, key=lambda w: mu*w-F(w))
                for mu in [-3*a*W, -.3*a*W, .5*a*W, 2*a*W]:
                    for muh in [-2*a*W, 0., .7*a*W, 3*a*W]:
                        w, wh = solve(mu), solve(muh)
                        loss = mu*(w-wh)-F(w)+F(wh)
                        E = abs(mu-muh)
                        envelope = E*E/(2*a) if E <= a*W else W*E-a*W*W/2
                        assert loss >= -1e-11
                        assert loss <= envelope+2e-10
                        assert loss <= (mu-muh)*(w-wh)+2e-10
                        if envelope > 0:
                            largest_ratio = max(largest_ratio, loss/envelope)
                        count += 1
    # Exact rational witness of attainment on both branches, without fitting.
    from fractions import Fraction as F
    witnesses = []
    a, W, muh = F(2), F(1), F(-1)
    for E in [F(1, 4), F(1), F(2), F(4)]:
        mu = muh+E
        wh = -W/2
        w = max(-W/2, min(W/2, mu/a))
        loss = mu*(w-wh)-a*(w*w-wh*wh)/2
        envelope = E*E/(2*a) if E <= a*W else W*E-a*W*W/2
        assert loss == envelope
        witnesses.append({'error': str(E), 'loss': str(loss)})
    record('sharp_curvature_diameter_envelope', optimization_cases=count,
           maximum_loss_to_envelope=float(largest_ratio), exact_rational_witnesses=witnesses)


def check_filtered_forecast(record):
    T, m = 2., 1.3
    omega = math.pi/(2*T)
    z = lambda t: math.sin(omega*t)
    # Independently integrate the future signal rather than substituting its adjoint formula.
    e = lambda s: quad(z, s, T, epsabs=1e-12)[0]
    z2 = quad(lambda t: z(t)**2, 0, T)[0]
    e2 = quad(lambda s: e(s)**2, 0, T)[0]
    sharp_ratio = 4*T*T/math.pi**2
    assert abs(e2/z2-sharp_ratio) < 1e-12
    bias2 = quad(lambda s: (T-s)**2, 0, T)[0]
    assert abs(bias2/T-T*T/3) < 1e-12
    assert e2/z2 > bias2/T
    # Direct true objective evaluation at v*=B*z/m and fitted v=0.
    vstar = lambda t: e(t)/m
    wstar = lambda t: quad(vstar, 0, t, epsabs=1e-12)[0]
    objective = quad(lambda t: m*vstar(t)**2/2-z(t)*wstar(t), 0, T)[0]
    loss = -objective
    bound = 2*T*T/(m*math.pi**2)*z2
    assert abs(loss-bound) < 1e-12
    # A finite-dimensional approximation supplies an independent singular-value comparison.
    n = 320
    B = T/n*(np.tril(np.ones((n, n)), -1)+.5*np.eye(n))
    norm_B = float(svdvals(B)[0])
    assert abs(norm_B/(2*T/math.pi)-1) < 1e-5
    # On the one-feature feasible class of constant rates, projection keeps
    # only integral(t*z(t))dt. A nonzero signal with that moment zero is invisible.
    znull = lambda t: t-2*T/3
    coefficient = quad(lambda t: t*znull(t), 0, T)[0]
    assert abs(coefficient) < 1e-12
    assert quad(lambda t: znull(t)**2, 0, T)[0] > 0
    record('causal_forecast_norm_and_sharpness', sharp_operator_norm=2*T/math.pi,
           numerical_operator_norm=norm_B, exact_loss=loss, sharp_bound=bound,
           persistent_bias_squared_norm_ratio=bias2/T,
           worst_squared_norm_ratio=e2/z2,
           restricted_class_invisible_signal_moment=coefficient)


def check_wrong_tracking_primitives(record):
    errors = []
    count = 0
    for eta in [.2, 1.]:
        for kap in [.4, 1., 3.]:
            for rho in [.2, 2.]:
                v = .7
                th = kap/(kap+rho)
                optimum = eta*kap*kap*v*(1-th*th)
                def evaluate(alpha, beta):
                    system = np.array([[-alpha, beta], [0., -rho]])
                    noise = np.array([[0., 0.], [0., 2*rho*v]])
                    covariance = solve_continuous_lyapunov(system, -noise)
                    u = np.array([-alpha, beta])
                    return (eta*(u@covariance@u)
                            +eta*kap*kap*(covariance[0, 0]-2*covariance[0, 1]+v))
                for factor in [.5, .8, 1.2, 2.]:
                    rh = factor*rho
                    thh = kap/(kap+rh)
                    actual = evaluate(kap, kap*thh)-optimum
                    formula = eta*kap*kap*v*(thh-th)**2
                    errors.append(abs(actual-formula))
                    count += 1
                    kh = factor*kap
                    thh = kh/(kh+rho)
                    actual = evaluate(kh, kh*thh)-optimum
                    a, b = kap-kh, kh*thh-kap*th
                    formula = eta*v*(a*a*thh**3+2*a*b*thh**2+b*b)
                    errors.append(abs(actual-formula))
                    count += 1
    assert max(errors) < 2e-13
    eta, kap, rho, v, d = 1., 1., 2., 1., .5
    th = kap/(kap+rho)
    under = eta*kap*kap*v*(kap/(kap+rho-d)-th)**2
    over = eta*kap*kap*v*(kap/(kap+rho+d)-th)**2
    ratio = ((kap+rho+d)/(kap+rho-d))**2
    assert abs(under/over-ratio) < 1e-12
    # Biased two-point estimates: the leading term must use MSE, not variance alone.
    offsets = 1e-5*np.array([-1., 2.])
    probabilities = np.array([.25, .75])
    actual = float(probabilities@(eta*kap*kap*v*(kap/(kap+rho+offsets)-th)**2))
    mse = float(probabilities@(offsets*offsets))
    variance = float(probabilities@((offsets-probabilities@offsets)**2))
    leading = eta*v*th**4*mse
    assert abs(actual/leading-1) < 2e-5
    assert mse > 1.5*variance
    record('wrong_persistence_and_friction', independent_lyapunov_cases=count,
           maximum_formula_error=float(max(errors)), equal_error_cost_ratio=ratio,
           biased_estimator_mse=mse, biased_estimator_variance=variance,
           exact_mixture_loss=actual, local_mse_approximation=leading)


def run_checks(record):
    for check in [check_discounted_certificates, check_two_quote_experiment,
                  check_unbounded_signal_ceiling, check_curvature_diameter,
                  check_filtered_forecast, check_wrong_tracking_primitives]:
        check(record)
