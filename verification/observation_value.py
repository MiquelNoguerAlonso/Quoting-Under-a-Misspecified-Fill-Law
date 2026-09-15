"""Decision-sufficient book features and the value of delayed observations."""
from fractions import Fraction as F
import math

import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import expm

from order_book_foundations import integrated_semigroup


def scalar_position(mu,a,fee,limit):
    if a==0:
        return 0. if abs(mu)<=fee else math.copysign(limit,mu)
    return np.clip(math.copysign(max(0.,abs(mu)-fee)/a,mu),-limit,limit)


def check_information_identity(record):
    rng=np.random.default_rng(742713)
    cases=0; max_difference=0.
    for a in [0.,.4,1.3]:
        for fee in [0.,.15]:
            for _ in range(8):
                prior=rng.dirichlet(np.ones(5))
                observation=rng.dirichlet(np.ones(3),size=5)
                signal=rng.normal(size=5)
                joint=prior[:,None]*observation
                probabilities=joint.sum(axis=0)
                means=joint.T@signal/probabilities
                cost=lambda w:a*w*w/2+fee*abs(w)
                star=lambda mu:mu*scalar_position(mu,a,fee,.8)-cost(scalar_position(mu,a,fee,.8))
                oracle=sum(prior[i]*star(signal[i]) for i in range(5))
                value=sum(probabilities[o]*star(means[o]) for o in range(3))
                deployed=np.clip(np.array([scalar_position(mu,a,fee,.8) for mu in means])+.03,-.8,.8)
                actual=sum(joint[i,o]*(signal[i]*deployed[o]-cost(deployed[o])) for i in range(5) for o in range(3))
                implementation=sum(probabilities[o]*(star(means[o])-means[o]*deployed[o]+cost(deployed[o])) for o in range(3))
                difference=abs(oracle-actual-(oracle-value)-implementation)
                assert difference<1e-13
                assert oracle>=value-1e-13 and implementation>=-1e-13
                if a>0:
                    mse=sum(joint[i,o]*(signal[i]-means[o])**2 for i in range(5) for o in range(3))
                    assert oracle-value<=mse/(2*a)+2e-13
                max_difference=max(max_difference,difference); cases+=1
    # Same nontrivial hidden variation, zero decision value inside no-trade.
    signals=[F(-1,10),F(1,10)]; fee=F(1,5)
    assert all(abs(mu)<fee for mu in signals)
    record('nonlinear_observation_value', finite_experiment_cases=cases,
           maximum_decomposition_difference=max_difference,
           hidden_variation_without_trading_value='exact no-trade example')


def check_price_specific_reduction(record):
    qf=[[F(-3),F(1),F(1),F(1)],
        [F(1),F(-5),F(2),F(2)],
        [F(1),F(1),F(-2),F(0)],
        [F(1),F(1),F(0),F(-2)]]
    df=[F(0),F(0),F(1),F(-1)]
    mv=lambda mat,vec:[sum(x*y for x,y in zip(row,vec)) for row in mat]
    power=df
    for k in range(4):
        assert power[0]==power[1]
        assert power==[(-2)**k*v for v in df]
        power=mv(qf,power)
    q,d=np.array(qf,dtype=float),np.array(df,dtype=float)
    lift=np.array([[1.,0,0],[1.,0,0],[0,1.,0],[0,0,1.]])
    average=np.diag([.5,1.,1.])@lift.T
    coarse_q=average@q@lift
    coarse_d=average@d
    residual=q@lift-lift@coarse_q
    assert np.max(np.sum(np.abs(residual),axis=1))==2.
    forecast_errors=[]
    for horizon in [.01,.1,.5,1.,3.]:
        true=integrated_semigroup(q,d,horizon)
        reduced=lift@integrated_semigroup(coarse_q,coarse_d,horizon)
        forecast_errors.append(float(np.max(np.abs(true-reduced))))
        assert np.max(np.abs(true-reduced))<4e-13
        assert np.max(np.abs(residual@expm(horizon*coarse_q)@coarse_d))<3e-13
    assert max(forecast_errors)<4e-13
    # Perturb one rate; compare the exact intertwining identity and target bound.
    qp=q.copy(); qp[0,2]+=.03; qp[0,0]-=.03
    rp=qp@lift-lift@coarse_q
    rows=[]
    for horizon in [.1,.5,1.]:
        diff=integrated_semigroup(qp,d,horizon)-lift@integrated_semigroup(coarse_q,coarse_d,horizon)
        allowance,_=quad_vec(lambda t:(horizon-t)*np.max(np.abs(rp@expm(t*coarse_q)@coarse_d)),
                              0,horizon,epsabs=1e-13,epsrel=1e-13)
        assert np.max(np.abs(diff))<=float(allowance)+2e-13
        rows.append({'horizon':horizon,'actual_forecast_error':float(np.max(np.abs(diff))),
                     'price_specific_bound':float(allowance)})
    record('price_specific_state_reduction', exact_krylov_orders=4,
           full_rate_residual_norm=2.,maximum_exact_forecast_difference=max(forecast_errors),
           perturbed_comparisons=rows)


def check_feed_latency_identity(record):
    rng=np.random.default_rng(438372)
    comparisons=0; maximum=0.; timestamp_cases=0
    for n in [2,4,6]:
        for _ in range(4):
            q=rng.uniform(.05,.6,(n,n)); np.fill_diagonal(q,0.)
            q-=np.diag(q.sum(axis=1))
            drift=rng.normal(size=n)
            prior=rng.dirichlet(np.ones(n))
            forecast=integrated_semigroup(q,drift,.6)
            for delay in [.001,.05,.4]:
                p=expm(delay*q); mean=p@forecast
                conditional_variance=p@(forecast**2)-mean**2
                gamma_q=lambda f:np.sum(q*(f[None,:]-f[:,None])**2,axis=1)
                integral,_=quad_vec(lambda s:expm(s*q)@gamma_q(expm((delay-s)*q)@forecast),
                                     0,delay,epsabs=1e-13,epsrel=1e-13)
                maximum=max(maximum,float(np.max(np.abs(integral-conditional_variance))))
                assert np.max(np.abs(integral-conditional_variance))<5e-13
                a=.7
                information=float(prior@conditional_variance)/(2*a)
                stale_extra=float(prior@((mean-forecast)**2))/(2*a)
                direct=sum(prior[i]*p[i,j]*(forecast[j]-forecast[i])**2 for i in range(n) for j in range(n))/(2*a)
                assert abs(direct-information-stale_extra)<5e-13
                assert information>=-1e-13
                # At stationarity data processing makes latency loss increase.
                lhs=q.T.copy(); rhs=np.zeros(n); lhs[-1,:]=1.; rhs[-1]=1.
                stationary=np.linalg.solve(lhs,rhs)
                times=[0.,.001,.01,.1,1.]
                losses=[]
                for t in times:
                    pt=expm(t*q)
                    losses.append(float(stationary@(pt@(forecast**2)-(pt@forecast)**2)))
                assert min(np.diff(losses))>=-1e-12
                comparisons+=1
            delays=[.002,.08,.3]; weights=np.array([.2,.5,.3])
            transitions=[expm(t*q) for t in delays]
            means=np.array([pt@forecast for pt in transitions])
            mixture=weights@means
            second=weights@np.array([pt@(forecast**2) for pt in transitions])
            observed_delay_loss=float(prior@(second-weights@(means**2)))/(2*a)
            timestamp_value=float(prior@(weights@(means**2)-mixture**2))/(2*a)
            hidden_delay_loss=float(prior@(second-mixture**2))/(2*a)
            assert timestamp_value>=-1e-13
            assert abs(hidden_delay_loss-observed_delay_loss-timestamp_value)<3e-13
            mean_delay=float(weights@delays)
            mean_delay_forecast=expm(mean_delay*q)@forecast
            approximation_loss=float(prior@((mixture-mean_delay_forecast)**2))/(2*a)
            direct=sum(weights[k]*prior[i]*transitions[k][i,j]*(forecast[j]-mean_delay_forecast[i])**2
                       for k in range(3) for i in range(n) for j in range(n))/(2*a)
            assert abs(direct-hidden_delay_loss-approximation_loss)<4e-13
            timestamp_cases+=1
    record('feed_latency_carre_du_champ', nonreversible_comparisons=comparisons,
           maximum_variance_integral_difference=maximum,timestamp_experiments=timestamp_cases)


def check_two_state_latency(record):
    kappa,drift,horizon,a,fee=20.,2.,.05,.7,.012
    q=np.array([[-kappa,kappa],[kappa,-kappa]])
    v=drift*(-math.expm1(-2*kappa*horizon))/(2*kappa)
    threshold=math.log(v/fee)/(2*kappa)
    rows=[]
    for delay in [0.,.001,.005,.01,.02,.04]:
        decay=math.exp(-2*kappa*delay)
        mean=v*decay
        quadratic_info=v*v/(2*a)*(-math.expm1(-4*kappa*delay))
        naive_extra=v*v/(2*a)*(1-decay)**2
        mixed_info=(max(0,v-fee)**2-max(0,mean-fee)**2)/(2*a)
        signal=np.array([v,-v]); p=expm(delay*q); stale=p@signal
        oracle=np.mean([max(0,abs(mu)-fee)**2/(2*a) for mu in signal])
        fitted=np.mean([max(0,abs(mu)-fee)**2/(2*a) for mu in stale])
        assert abs(oracle-fitted-mixed_info)<1e-14
        direct=np.mean(p@(signal**2)-stale**2)/(2*a)
        assert abs(direct-quadratic_info)<1e-14
        rows.append({'delay_seconds':delay,'quadratic_information_loss':quadratic_info,
                     'naive_additional_loss':naive_extra,
                     'mixed_cost_information_loss':mixed_info,
                     'absolute_optimal_stale_position':max(0,mean-fee)/a})
    record('two_state_latency_frontier', instantaneous_forecast_magnitude=v,
           no_trade_delay_seconds=threshold,
           initial_quadratic_loss_slope=2*kappa*v*v/a,delay_examples=rows)


def run_checks(record):
    check_information_identity(record)
    check_price_specific_reduction(record)
    check_feed_latency_identity(record)
    check_two_state_latency(record)
