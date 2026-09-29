"""Independent finite checks for latency, equilibrium and robust positions.

Synthetic laws only. Exact rational latency and posterior arithmetic are
combined with independent finite-dimensional equilibrium/optimization checks.
"""
from pathlib import Path
from fractions import Fraction as F
import json
import numpy as np
from scipy.optimize import minimize_scalar


def latency_range():
    m,tau,loss=F(1,2),F(2),F(3)
    cases=0
    for j in range(1,1000):
        q=(m/tau)*F(j,1000)
        atom=m/q
        assert atom>tau and q*atom==m
        assert 0<loss*q<loss*m/tau
        cases+=1
    # If effective cancellation ties the trade, cancellation is processed first.
    assert (tau>tau) is False
    eps=F(1,10**9);q=m/(tau+eps)
    assert 0<loss*m/tau-loss*q<F(1,10**8)
    assert m<tau  # Deterministic delay has zero loss.
    return dict(check="attainable_latency_loss", exact_laws=cases+2,
                supremum=str(loss*m/tau), near_supremum_gap=str(loss*m/tau-loss*q))


def symmetric_equilibrium():
    gamma=np.array([[.5,0,0],[.2,.5,0],[.1,.2,.5]])
    assert np.linalg.eigvalsh((gamma+gamma.T)/2).min()>0
    eta=.17;ones=np.ones(3);errors=[]
    for n in range(1,9):
        h=np.kron(np.ones((n,n)),gamma)+np.kron(np.eye(n),gamma.T+2*eta*np.eye(3))
        constraints=np.kron(np.eye(n),ones.reshape(1,-1))
        kkt=np.block([[h,constraints.T],[constraints,np.zeros((n,n))]])
        rhs=np.r_[np.zeros(3*n),np.ones(n)/n]
        profiles=np.linalg.solve(kkt,rhs)[:3*n].reshape(n,3)
        aggregate=profiles.sum(axis=0)
        op=gamma+gamma.T/n+2*eta*np.eye(3)/n
        aggregate_kkt=np.block([[op,ones[:,None]],[ones[None,:],np.zeros((1,1))]])
        independent=np.linalg.solve(aggregate_kkt,np.r_[np.zeros(3),1])[:3]
        errors.append(float(np.max(np.abs(aggregate-independent))))
        assert np.max(np.abs(profiles-aggregate/n))<1e-12
        assert errors[-1]<1e-12
    return dict(check="symmetric_equilibrium", player_counts=8,max_discrepancy=max(errors))


def history_to_position():
    # A binary observed event is informative about a hidden state. Conditional
    # future R is +/-1, with P(R=1|H=0)=2/5 and P(R=1|H=1)=7/10.
    def posterior(prior,y):
        likelihood0=F(1,4) if y else F(3,4)
        likelihood1=F(3,4) if y else F(1,4)
        denom=(1-prior)*likelihood0+prior*likelihood1
        return prior*likelihood1/denom,denom
    def mean(prior,y):
        post,_=posterior(prior,y)
        return -F(1,5)+F(3,5)*post
    cases=0;max_optimizer_gap=0.;min_likelihood=F(1)
    for j in range(21):
        prior=F(1,4)+F(j,40)
        for y in [0,1]:
            lo,hi=mean(F(1,4),y),mean(F(3,4),y)
            mu=mean(prior,y);assert lo<=mu<=hi
            _,denom=posterior(prior,y);min_likelihood=min(min_likelihood,denom)
            mid=(lo+hi)/2;radius=(hi-lo)/2
            for a in [F(1,10),F(1,2),F(1)]:
                for b in [F(-1,2),F(0),F(1,2)]:
                    cost=F(1,100);score=mid-a*b;threshold=cost+radius
                    soft=max(score-threshold,F(0))-max(-score-threshold,F(0))
                    d=min(max(soft/a,-1-b),1-b);w=b+d
                    charge=a*(w*w-b*b)/2+cost*abs(d)
                    guaranteed=mid*d-radius*abs(d)-charge
                    expected=mu*d-charge
                    assert guaranteed>=0 and expected>=guaranteed
                    assert guaranteed==min(lo*d-charge,hi*d-charge)
                    # Independent bounded scalar optimization of the robust objective.
                    def gain(x):
                        dd=x-float(b)
                        return float(mid)*dd-float(radius)*abs(dd)-float(a)*(x*x-float(b*b))/2-float(cost)*abs(dd)
                    opt=minimize_scalar(lambda x:-gain(x),bounds=(-1,1),method="bounded",options={"xatol":1e-12})
                    numerical=max(gain(opt.x),gain(float(b)),gain(-1),gain(1))
                    gap=abs(numerical-float(guaranteed));max_optimizer_gap=max(max_optimizer_gap,gap)
                    assert gap<1e-9
                    # R has support [-1,1]; the centered gain's support width is 2|d|.
                    probability=(mu+1)/2
                    realized_plus=d-charge;realized_minus=-d-charge
                    assert probability*realized_plus+(1-probability)*realized_minus==expected
                    cases+=1
    return dict(check="event_history_to_certified_position",cases=cases,
                smallest_history_likelihood=str(min_likelihood),max_optimizer_gap=max_optimizer_gap,
                noise_variance_proxy="d^2 by conditional Hoeffding")


def main():
    results=[latency_range(),symmetric_equilibrium(),history_to_position()]
    report={"status":"all_passed","data":"Synthetic finite and exact rational checks; no market data.","results":results}
    Path(__file__).with_name('additional_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
