"""Partial-fill matching and cancellation-delay comparisons.

Finite-state matrix calculations are compared with compound-Poisson sums,
quadrature, and exact moment bounds. No exchange data are used.
"""
from fractions import Fraction as F
import math

import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import expm, solve
from scipy.stats import poisson, gamma


def match(ahead, remaining, batch):
    fill = min(remaining, max(0, batch-ahead))
    return max(0, ahead-batch), remaining-fill, fill


def queue_cell(rate=F(7,5), batches=((1,F(7,10)),(3,F(3,10)))):
    states = [(ahead,remaining) for ahead in range(4) for remaining in [1,2]]+[(0,0)]
    index = {state:i for i,state in enumerate(states)}
    events = []
    q = [[F(0) for _ in states] for _ in states]
    reward = []
    salvage = []
    for i,(ahead,remaining) in enumerate(states):
        inventory = 2-remaining
        reward.append(-F(3,100)*inventory**2)
        salvage.append(-F(1,20)*inventory**2)
        row = []
        if remaining:
            for batch,probability in batches:
                new_ahead,new_remaining,fill = match(ahead,remaining,batch)
                target = index[(new_ahead,new_remaining)] if new_remaining else index[(0,0)]
                lam = rate*probability
                reference = {2:F(0),1:-F(1,100),0:-F(4,25)}
                old_reference,new_reference=reference[remaining],reference[new_remaining]
                offset=old_reference+F(1,50)  # Fixed bid price is -0.02.
                mark=new_reference-old_reference
                net=fill*offset+(inventory+fill)*mark
                old_wealth=inventory*(old_reference+F(1,50))
                new_wealth=(inventory+fill)*(new_reference+F(1,50))
                assert net==new_wealth-old_wealth
                q[i][target] += lam
                q[i][i] -= lam
                reward[i] += lam*net
                row.append((target,lam,fill,net))
        events.append(row)
    return states,index,q,reward,salvage,events


def delayed_value(q,r,w,beta,duration):
    n = len(r)
    augmented = np.zeros((n+1,n+1))
    augmented[:n,:n] = q-beta*np.eye(n)
    augmented[:n,n] = r
    return (expm(duration*augmented)@np.r_[w,1.])[:n]


def compound_fill_pmf(ahead,remaining,duration,rate,batches,cutoff=50):
    out = np.zeros(remaining+1)
    distribution = {0:1.}
    for count in range(cutoff+1):
        mass = poisson.pmf(count,rate*duration)
        for consumed,probability in distribution.items():
            filled = min(remaining,max(0,consumed-ahead))
            out[filled] += mass*probability
        new = {}
        for consumed,probability in distribution.items():
            for batch,p in batches:
                new[consumed+batch] = new.get(consumed+batch,0.)+probability*p
        distribution = new
    return out,float(poisson.sf(cutoff,rate*duration))


def check_partial_fill_matching(record):
    states,index,q,_,_,events = queue_cell()
    q = np.array(q,dtype=float)
    comparisons,max_error = 0,0.
    for ahead in range(4):
        for remaining in [1,2]:
            start = index[(ahead,remaining)]
            for duration in [0.01,0.12,0.35,1.2]:
                probabilities = expm(duration*q)[start]
                by_state = np.zeros(remaining+1)
                for (_,left),probability in zip(states,probabilities):
                    if probability>0 and left<=remaining:
                        by_state[remaining-left] += probability
                by_count,tail = compound_fill_pmf(ahead,remaining,duration,1.4,[(1,.7),(3,.3)])
                error = float(np.max(np.abs(by_state-by_count)))
                assert error < 2e-13+tail
                for z in [0.,0.35,1.,1.6]:
                    tilted = np.zeros_like(q)
                    for i,row in enumerate(events):
                        for target,lam,fill,_ in row:
                            tilted[i,target] += float(lam)*z**fill
                            tilted[i,i] -= float(lam)
                    matrix_pgf = (expm(duration*tilted)@np.ones(len(states)))[start]
                    direct_pgf = sum(p*z**j for j,p in enumerate(by_count))
                    assert abs(matrix_pgf-direct_pgf)<5e-13+tail*z**remaining
                max_error=max(max_error,error)
                comparisons+=1
    # Check pathwise inventory reservations for both sides and all batch sizes.
    reserve_cases=0
    for side in [-1,1]:
        for inventory in range(-4,5):
            for remaining in range(1,5):
                if abs(inventory+side*remaining)>4:
                    continue
                for ahead in range(5):
                    for batch in range(1,9):
                        _,left,fill=match(ahead,remaining,batch)
                        new_inventory=inventory+side*fill
                        assert -4<=new_inventory<=4
                        assert new_inventory+side*left==inventory+side*remaining
                        reserve_cases+=1
    small=[]
    for duration in [1e-2,1e-3,1e-4]:
        pmf,_=compound_fill_pmf(3,2,duration,1.4,[(1,.7),(3,.3)])
        small.append(float((pmf[1]+pmf[2])/duration**2))
    # Two arrivals first permit a fill: Pr(B1+B2>=4)=.51.
    exact_coefficient=F(7,5)**2*F(51,100)/2
    assert abs(small[-1]-float(exact_coefficient))<1e-4
    record('compound_partial_fill_kernel', distribution_comparisons=comparisons,
           pgf_comparisons=4*comparisons,reservation_cases=reserve_cases,
           maximum_distribution_difference=max_error,
           first_fill_small_time_coefficient=str(exact_coefficient),
           scaled_first_fill_probabilities=small)


def check_cancellation_value(record):
    states,index,qf,rf,wf,_=queue_cell()
    q,r,w=np.array(qf,dtype=float),np.array(rf,dtype=float),np.array(wf,dtype=float)
    beta=.8
    a=q-beta*np.eye(len(states))
    residual=r+a@w
    rows=[]
    maximum=0.
    start=index[(3,2)]
    for duration in [.01,.12,.35,.8]:
        direct=delayed_value(q,r,w,beta,duration)
        integral,_=quad_vec(lambda t:expm(t*a)@residual,0,duration,epsabs=1e-13,epsrel=1e-13)
        error=np.max(np.abs(direct-(w+integral)))
        maximum=max(maximum,float(error))
        assert error<3e-13
        bound=np.max(np.abs(residual))*(-math.expm1(-beta*duration))/beta
        assert np.max(np.abs(direct-w))<=bound+1e-13
        pmf,_=compound_fill_pmf(3,2,duration,1.4,[(1,.7),(3,.3)])
        rows.append({'delay':duration,'first_fill_probability':float(pmf[1]+pmf[2]),
                     'expected_filled_volume':float(pmf[1]+2*pmf[2]),
                     'marked_value':float(direct[start])})
    # Match two delay distributions by their increasing quantile coupling.
    delays=[.1,.3,.8]; perturbed=[.15,.25,.95]; weights=[.2,.5,.3]
    v1=sum(p*delayed_value(q,r,w,beta,t) for p,t in zip(weights,delays))
    v2=sum(p*delayed_value(q,r,w,beta,t) for p,t in zip(weights,perturbed))
    distance=sum(p*abs(t-s) for p,t,s in zip(weights,delays,perturbed))
    assert np.max(np.abs(v1-v2))<=np.max(np.abs(residual))*distance+1e-13
    record('pending_cancel_value_identity', states=len(states),
           maximum_dynkin_difference=maximum, coupled_delay_distance=distance,
           delay_examples=rows)


def check_erlang_approximation(record):
    states,index,qf,rf,wf,_=queue_cell()
    beta_f,duration_f=F(4,5),F(7,20)
    af=[[v-(beta_f if i==j else 0) for j,v in enumerate(row)] for i,row in enumerate(qf)]
    mv=lambda mat,vec:[sum(x*y for x,y in zip(row,vec)) for row in mat]
    g=[x+y for x,y in zip(rf,mv(af,wf))]
    ag=mv(af,g); aag=mv(af,ag)
    g_bound=max(map(abs,g)); b_bound=max(map(abs,ag)); c_bound=max(map(abs,aag))
    q,r,w=np.array(qf,dtype=float),np.array(rf,dtype=float),np.array(wf,dtype=float)
    beta,duration=float(beta_f),float(duration_f)
    a=np.array(af,dtype=float)
    direct=delayed_value(q,r,w,beta,duration)
    leading=duration**2/2*expm(duration*a)@np.array(ag,dtype=float)
    constant=-solve(a,r)
    rows=[]
    start=index[(3,2)]
    for stages in [1,2,4,8,16,32,64,128]:
        rate=stages/duration
        stage=w.copy()
        b=rate*np.eye(len(states))-a
        for _ in range(stages):
            stage=solve(b,r+rate*stage)
        resolvent=solve(b,rate*np.eye(len(states)))
        independent=constant+np.linalg.matrix_power(resolvent,stages)@(w-constant)
        assert np.max(np.abs(stage-independent))<4e-13
        if stages in [1,4,16]:
            integrated,_=quad_vec(lambda t:delayed_value(q,r,w,beta,t)*gamma.pdf(t,stages,scale=duration/stages),
                                   0,np.inf,epsabs=2e-12,epsrel=2e-12)
            assert np.max(np.abs(stage-integrated))<3e-11
        bias=stage-direct
        path_bound=float(g_bound)*duration/math.sqrt(stages)
        weak_bound=float(b_bound)*duration**2/(2*stages)
        remainder=float(c_bound)*duration**3*math.sqrt(3+6/stages)/(6*stages**1.5)
        assert np.max(np.abs(bias))<=min(path_bound,weak_bound)+2e-13
        assert np.max(np.abs(bias-leading/stages))<=remainder+3e-13
        rows.append({'stages':stages,'initial_state_bias':float(bias[start]),
                     'uniform_weak_bound':weak_bound,'scaled_initial_bias':float(stages*bias[start])})
    # The general 1/k order is attained by a one-unit queue.
    mu,reward,disc,delay=1.3,.2,.7,.25
    speed=mu+disc
    target=mu*reward/speed*(-math.expm1(-speed*delay))
    limit=-mu*reward*speed*delay**2*math.exp(-speed*delay)/2
    witnesses=[]
    for k in [10,100,1000,10000]:
        approx=mu*reward/speed*(-math.expm1(-k*math.log1p(speed*delay/k)))
        witnesses.append(k*(approx-target))
    assert abs(witnesses[-1]-limit)<1e-6
    record('erlang_cancel_approximation', arithmetic_constants='exact fractions',
           residual_sup=str(g_bound),second_derivative_sup=str(b_bound),
           third_derivative_sup=str(c_bound),initial_state_leading_bias=float(leading[start]),
           approximations=rows,one_unit_scaled_biases=witnesses,
           one_unit_nonzero_limit=limit)


def run_checks(record):
    check_partial_fill_matching(record)
    check_cancellation_value(record)
    check_erlang_approximation(record)
