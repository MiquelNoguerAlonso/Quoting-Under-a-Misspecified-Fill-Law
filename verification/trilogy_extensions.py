"""Independent checks for the added model, margin and information certificates.

All inputs are synthetic. Exact rational bounds are distinguished from
floating-point ODE, optimization and quadrature comparisons.
"""
from fractions import Fraction as F
import math
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.linalg import null_space
from scipy.optimize import minimize_scalar


def pval(c, x):
    return sum(a*x**i for i, a in enumerate(c))


def pmul(a, b):
    out = [F(0)]*(len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i+j] += x*y
    return out


def quadratic_abs_max(c):
    points = [F(0), F(1)]
    if len(c) > 2 and c[2]:
        vertex = -c[1]/(2*c[2])
        if 0 < vertex < 1:
            points.append(vertex)
    return max(abs(pval(c, x)) for x in points)


def check_computed_hjb_certificate(record):
    # Time-to-go u in [0,1]. v=(.09u+.005,.17u-.005).
    # All coefficients and global supremum bounds below are exact rationals.
    r_b = [F(2, 5), F(3, 5)]
    h_b = [F(21, 50), F(29, 50)]
    r_k = [F(3, 10), F(2, 5)]
    h_k = [F(31, 100), F(41, 100)]
    r_d = [F(1, 5), F(1, 10)]
    h_d = [F(11, 50), F(3, 25)]
    value_slope = [F(9, 100), F(17, 100)]
    value_diff = [[F(-1, 100), F(2, 25)],
                  [F(1, 100), F(-2, 25)]]
    residual_polynomials = []
    for x in range(2):
        c = [h_b[x]+h_d[x]*value_diff[x][0],
             h_d[x]*value_diff[x][1]]
        # The estimated action optimum is c, strictly inside [0,1].
        assert all(0 < pval(c, u) < 1 for u in [F(0), F(1)])
        poly = [z/2 for z in pmul(c, c)]
        poly[0] += h_k[x]*value_diff[x][0]-value_slope[x]
        poly[1] += h_k[x]*value_diff[x][1]
        residual_polynomials.append(poly)
    rho = max(quadratic_abs_max(p) for p in residual_polynomials)
    # span(v) <= .01+.08u; error r0=.02, k0=.03, r1=.02, k1=.02.
    # D=.005+(.02+.0003+rho)u+.0012u^2; K1+ = .24; m=1.
    D = [F(1, 200), F(203, 10000)+rho, F(3, 2500)]
    b = [F(1, 50)+F(1, 5000)+F(12, 25)*D[0],
         F(1, 625)+F(12, 25)*D[1], F(12, 25)*D[2]]
    optimizer_residual = F(1, 100)
    b[0] += optimizer_residual
    squared = pmul(b, b)
    exact_bound = sum(c/F(2*(i+1)) for i, c in enumerate(squared))

    true_b, true_k, true_d = map(
        lambda a: np.array([float(x) for x in a]), [r_b, r_k, r_d])
    est_b, est_d = map(
        lambda a: np.array([float(x) for x in a]), [h_b, h_d])

    def policy(t):
        diff = np.array([.08*(1-t)-.01, -.08*(1-t)+.01])
        return est_b+est_d*diff+.01

    def hamiltonian(h, a):
        return true_b*a-.5*a*a+(true_k+true_d*a)*(h[::-1]-h)

    def true_hjb(t, h):
        a = np.clip(true_b+true_d*(h[::-1]-h), 0, 1)
        return -hamiltonian(h, a)

    opt = solve_ivp(true_hjb, (1., 0.), np.zeros(2),
                    rtol=2e-12, atol=2e-14, dense_output=True)
    pol = solve_ivp(lambda t, h: -hamiltonian(h, policy(t)),
                    (1., 0.), np.zeros(2), rtol=2e-12, atol=2e-14)
    assert opt.success and pol.success
    regret = float(opt.y[0, -1]-pol.y[0, -1])
    assert 0 <= regret <= float(exact_bound)
    # Independent forward occupation and accumulated reward.
    def forward(t, y):
        a = policy(t)
        k = true_k+true_d*a
        flow = y[0]*k[0]-y[1]*k[1]
        rew = true_b*a-.5*a*a
        return [-flow, flow, y[:2]@rew]
    fwd = solve_ivp(forward, (0., 1.), [1., 0., 0.],
                    rtol=2e-12, atol=2e-14)
    assert abs(fwd.y[2, -1]-pol.y[0, -1]) < 2e-11
    # This grid is only an additional diagnostic. The supremum rho and
    # integral bound were enclosed analytically with rational arithmetic.
    max_value_error = 0.
    for t in np.linspace(0, 1, 301):
        v = np.array([.09*(1-t)+.005, .17*(1-t)-.005])
        err = float(np.max(abs(opt.sol(t)-v)))
        assert err <= float(pval(D, F(str(1-t))))+1e-11
        max_value_error = max(max_value_error, err)
    record("computed_model_and_solver_certificate",
           exact_global_hjb_residual=str(rho),
           global_hjb_residual=float(rho),
           exact_integrated_upper_bound=str(exact_bound),
           integrated_upper_bound=float(exact_bound),
           direct_regret=regret,
           forward_backward_discrepancy=abs(fwd.y[2, -1]-pol.y[0, -1]),
           maximum_evaluated_value_error=max_value_error,
           quote_stationarity_residual=float(optimizer_residual))


def check_hybrid_margin(record):
    rows = []
    for alpha in [.5, 1., 2.]:
        values = []
        for zeta in [.02, .01, .005]:
            numeric = quad(lambda g: alpha*g**alpha, 0, 2*zeta,
                           epsabs=1e-13)[0]
            formula = alpha/(alpha+1)*(2*zeta)**(alpha+1)
            assert abs(numeric-formula) < 1e-12
            assert formula <= 2*zeta*(2*zeta)**alpha
            values.append(numeric)
        slope = math.log(values[0]/values[-1])/math.log(4)
        assert abs(slope-alpha-1) < 1e-9
        rows.append({"margin_exponent": alpha, "observed_loss_exponent": slope})
    record("hybrid_switching_margin_sharpness", rates=rows)


def check_quote_coverage(record):
    def reward(a, k):
        return a*math.exp(-k*(a-.4))
    minima = [math.exp(-k/5)*k*(2-3*k/5) for k in [2, 3]]
    m = min(minima)
    lower = m/8*(.5-1/3)**2
    def regret(a, k):
        return reward(1/k, k)-reward(a, k)
    sol = minimize_scalar(lambda a: max(regret(a, 2), regret(a, 3)),
                          bounds=(0., .6), method="bounded",
                          options={"xatol": 1e-14})
    assert sol.success and sol.fun >= lower
    for k in [2, 3]:
        assert math.exp(-k*(.4-.4)) == 1
        s = minimize_scalar(lambda a: -reward(a, k),
                            bounds=(0., .6), method="bounded")
        assert abs(s.x-1/k) < 2e-6
    # Randomization cannot lower the optimum: each regret is convex,
    # so replacing a random quote by its mean weakly improves both.
    record("fixed_quote_nonidentification",
           common_logged_intensity=1., theorem_lower_bound=lower,
           computed_minimax_regret=float(sol.fun),
           minimax_quote=float(sol.x))


def check_mse_margin_sharpness(record):
    rows = []
    for alpha in [.5, 1., 2.]:
        pairs = []
        d0 = .4
        C = d0**(-alpha)
        for t in [.08, .04, .02, .01]:
            loss = quad(lambda d: alpha*d**alpha/d0**alpha,
                        0, t, epsabs=1e-14)[0]
            mse = quad(lambda d: 4*alpha*d**(alpha+1)/d0**alpha,
                       0, t, epsabs=1e-14)[0]
            assert abs(loss-alpha/(alpha+1)*t**(alpha+1)/d0**alpha) < 1e-11
            assert abs(mse-4*alpha/(alpha+2)*t**(alpha+2)/d0**alpha) < 1e-11
            z = (mse/(C*(alpha+1)))**(1/(alpha+2))
            assert loss <= 2*C*z**(alpha+1)+2*mse/z+1e-12
            pairs.append((mse, loss))
        slope = math.log(pairs[0][1]/pairs[-1][1])/math.log(pairs[0][0]/pairs[-1][0])
        assert abs(slope-(alpha+1)/(alpha+2)) < 1e-9
        rows.append({"margin_exponent": alpha, "observed_MSE_exponent": slope})
    record("dependent_error_MSE_margin_sharpness", fixed_signal_laws=rows)


def coordinate_minimum(A, f, charge, lo=-.45, hi=.65):
    x = np.zeros(len(f))
    for _ in range(20000):
        old = x.copy()
        for i in range(len(x)):
            z = f[i]-A[i]@x+A[i, i]*x[i]
            x[i] = np.clip(np.sign(z)*max(abs(z)-charge, 0)/A[i, i], lo, hi)
        if np.max(abs(x-old)) < 1e-14:
            return x
    raise AssertionError("Coordinate optimization did not converge")


def check_joint_residual(record):
    rng = np.random.default_rng(20260915)
    errors = []
    ratios = []
    for _ in range(24):
        n = 6
        raw = rng.normal(size=(n, n))
        A = raw.T@raw+.7*np.eye(n)
        perturbation = rng.normal(size=(n, n))*.012
        Ah = A+(perturbation+perturbation.T)/2
        f = rng.normal(size=n)
        fh = f+rng.normal(size=n)*.06
        m = float(np.linalg.eigvalsh(A)[0])
        assert np.linalg.eigvalsh(Ah)[0] > 0
        # Common affine class with two independent completion equalities.
        C = np.vstack([np.ones(n), np.arange(n)])
        target = np.array([1., .7])
        v0 = C.T@np.linalg.solve(C@C.T, target)
        V = null_space(C)
        v = v0+V@np.linalg.solve(V.T@A@V, V.T@(f-A@v0))
        vh = v0+V@np.linalg.solve(V.T@Ah@V, V.T@(fh-Ah@v0))
        residual = V.T@((A-Ah)@vh-(f-fh))
        def objective(z, M=A, b=f):
            return .5*z@M@z-b@z
        gap = objective(vh)-objective(v)
        identity = .5*residual@np.linalg.solve(V.T@A@V, residual)
        errors.append(abs(gap-identity))
        assert errors[-1] < 3e-12
        assert gap <= (np.linalg.norm((A-Ah)@vh-(f-fh)))**2/(2*m)+1e-12
        # Independent nonsmooth coordinate minimization with binding boxes.
        charge = .2
        u = coordinate_minimum(A, f, charge)
        uh = coordinate_minimum(Ah, fh, charge)
        xi = charge*np.sign(uh)
        g = Ah@uh-fh
        for i in range(n):
            if abs(uh[i]) < 1e-12:
                xi[i] = np.clip(-g[i], -charge, charge)
            elif abs(uh[i]-.65) < 1e-12:
                xi[i] += max(0., -g[i]-xi[i])
            elif abs(uh[i]+.45) < 1e-12:
                xi[i] += min(0., -g[i]-xi[i])
        sh = Ah@uh-fh+xi
        assert np.linalg.norm(sh) < 1e-10
        r = (A-Ah)@uh-(f-fh)
        loss = objective(uh)+charge*abs(uh).sum()-objective(u)-charge*abs(u).sum()
        bound = np.linalg.norm(r+sh)**2/(2*m)
        assert loss >= -1e-11 and loss <= bound+1e-11
        ratios.append(float(loss/bound) if bound else 0.)
    # Exact scalar cancellation with both primitive inputs wrong.
    assert F(2)*F(1)-F(2) == 0
    assert (F(2)-F(3))*F(1)-(F(2)-F(3)) == 0
    record("joint_forecast_impact_residual",
           affine_and_mixed_constraint_instances=len(errors),
           largest_affine_identity_error=max(errors),
           largest_mixed_loss_to_bound=max(ratios),
           exact_compensating_error_example=True)


def check_information_and_adjoint(record):
    # Two dates, two equally likely states. The first rate is chosen
    # before Z is revealed; each scenario must execute one unit in total.
    vF = [[F(1), F(0)], [F(0), F(1)]]
    vG = [[F(1, 2), F(1, 2)]]*2
    vh = [[F(3, 5), F(2, 5)]]*2
    forecasts = [F(1), F(-1)]
    def cost(v):
        return sum(sum(z*z for z in row)/2-forecasts[i]*row[0]
                   for i, row in enumerate(v))/2
    def half_energy_diff(a, b):
        return sum(sum((x-y)**2 for x, y in zip(row_a, row_b))
                   for row_a, row_b in zip(a, b))/4
    info = cost(vG)-cost(vF)
    model = cost(vh)-cost(vG)
    assert info == F(1, 4) == half_energy_diff(vG, vF)
    assert model == F(1, 100) == half_energy_diff(vh, vG)
    assert cost(vh)-cost(vF) == info+model == F(13, 50)
    # A separate four-leaf, three-date tree checks the causal adjoint
    # by a matrix adjoint and by explicit conditional future averaging.
    L = np.zeros((12, 7))
    for leaf in range(4):
        L[3*leaf, 0] = 1
        L[3*leaf+1, 1+leaf//2] = 1
        L[3*leaf+2, 3+leaf] = 1
    W = np.eye(12)/4
    Gram = L.T@W@L
    B = np.kron(np.eye(4), np.tril(np.ones((3, 3))))
    rng = np.random.default_rng(20260916)
    max_error = 0.
    for _ in range(20):
        z = L@rng.normal(size=7)
        matrix_adj = L@np.linalg.solve(Gram, L.T@B.T@W@z)
        future = (B.T@z).reshape(4, 3)
        conditional = np.zeros((4, 3))
        conditional[:, 0] = future[:, 0].mean()
        for group in [slice(0, 2), slice(2, 4)]:
            conditional[group, 1] = future[group, 1].mean()
        conditional[:, 2] = future[:, 2]
        max_error = max(max_error, float(np.max(abs(matrix_adj-conditional.ravel()))))
        v = L@rng.normal(size=7)
        assert abs((B@v)@W@z-v@W@matrix_adj) < 1e-11
    assert max_error < 1e-12
    record("information_energy_decomposition_and_causal_adjoint",
           exact_information_loss=str(info), exact_model_loss=str(model),
           exact_total_loss=str(info+model),
           tree_adjoint_instances=20, largest_adjoint_error=max_error)


def run_checks(record):
    for check in [
        check_computed_hjb_certificate, check_hybrid_margin,
        check_quote_coverage, check_mse_margin_sharpness,
        check_joint_residual, check_information_and_adjoint,
    ]:
        check(record)
