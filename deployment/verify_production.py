"""Reproduce the v4 additions; synthetic data only, no network access.
Run from any working directory. Outputs are written relative to this script.
"""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import heapq,json,math,sys
import numpy as np
from scipy.optimize import linprog, minimize_scalar
from reference_engine import Ledger, Queue, cancel_race

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
RNG=np.random.default_rng(20260928)
REPORT={'data':'Synthetic mathematical and normalized event-model checks; no market data or hardware measurements.',
        'seed':20260928,'checks':[]}

def record(name, **data):
    REPORT['checks'].append({'name':name,'status':'passed',**data})

def raises(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('expected rejection')

def ledger_checks():
    trace=cancel_race()
    (OUT/'event_trace.json').write_text(json.dumps(trace,indent=2)+'\n')
    l=Ledger(parent=10)
    l.submit('old',6); l.cancel_request('old')
    raises(lambda:l.submit('replacement',10))
    l.submit('other',4)
    l.report('old',1,3)
    before=(l.filled,l.inventory,l.outstanding())
    assert not l.report('old',1,3)
    assert not l.report('old',0,0)
    assert before == (l.filled,l.inventory,l.outstanding())
    l.report('old',2,3,True)
    l.submit('replacement',3)
    l.report('replacement',0,0,True) # rejected submission
    assert l.outstanding() == 4
    l.recovery=True
    raises(lambda:l.submit('during_gap',1))
    l.invariant()
    d=Ledger(inventory=2,limit=5)
    raises(lambda:d.submit('too_many',4))
    d.submit('buy',3);d.submit('sell',7,-1)
    d.report('buy',0,2);d.report('sell',0,4);d.invariant()
    q=Queue();q.add('a',3);q.add('tag',4);q.add('b',2)
    q.cancel('b',1);assert q.ahead('tag')==3
    q.cancel('a',2);assert q.ahead('tag')==1
    assert q.consume(3)==[('a',1),('tag',2)]
    assert q.ahead('tag')==0
    q.cancel('tag');q.add('tag_new',2);assert q.ahead('tag_new')==1
    record('lifecycle_and_queue',checks=14,**{k:v for k,v in trace.items() if k!='trace'})
    return trace

def greedy(costs,Q):
    x=[0]*len(costs);h=[]
    for i,c in enumerate(costs):
        if len(c)>1:heapq.heappush(h,(c[1]-c[0],i))
    for _ in range(Q):
        if not h:raise ValueError('insufficient capacity')
        _,i=heapq.heappop(h);x[i]+=1
        if x[i]+1<len(costs[i]):
            heapq.heappush(h,(costs[i][x[i]+1]-costs[i][x[i]],i))
    return tuple(x)

def execution_checks():
    count=0
    for a,b in product(range(1,5),repeat=2):
        for cap1,cap2 in product(range(1,6),repeat=2):
            costs=[[F(a*n*n,2) for n in range(cap1+1)],
                   [F(b*n*n,2) for n in range(cap2+1)]]
            for Q in range(cap1+cap2+1):
                x=greedy(costs,Q)
                val=sum(costs[i][x[i]] for i in range(2))
                exact=min(costs[0][n]+costs[1][Q-n] for n in range(cap1+1) if 0<=Q-n<=cap2)
                assert val == exact
                count+=1
    assert greedy([[F(n*n,2) for n in range(6)],[F(n*n) for n in range(6)]],5)==(3,2)
    feasible_cases=0
    for r in range(11):
        for C in range(11):
            for fmin in range(r+1):
                for fmax in range(fmin,r+1):
                    assert all(r-f<=C for f in range(fmin,fmax+1)) == (r-fmin<=C)
                    feasible_cases+=1
    # Independent viability recursion for 2 stages: deterministic or binary fill supports.
    V={0}; masks={}
    for stage in (1,0):
        new=set()
        for r in range(7):
            supports={'wait':{r},'passive':{r,max(0,r-2)},'take':{max(0,r-3)}}
            acts=[a for a,s in supports.items() if s <= V]
            masks[stage,r]=acts
            if acts:new.add(r)
        V=new
    assert V==set(range(7)) and 'passive' not in masks[0,5]
    record('integer_allocation_and_viability',allocation_cases=count,capacity_cases=feasible_cases,
           integer_cost=8.5,continuous_cost=float(F(25,3)),gap=str(F(1,6)))

def evaluate(P,r,g,policy=None):
    N,S,A,_=P.shape
    V=np.zeros((N+1,S));V[N]=g
    Qs=np.zeros((N,S,A));pi=np.zeros((N,S),int)
    for k in reversed(range(N)):
        Qs[k]=r[k]+np.einsum('say,y->sa',P[k],V[k+1])
        pi[k]=np.argmax(Qs[k],axis=1) if policy is None else policy[k]
        V[k]=Qs[k,np.arange(S),pi[k]]
    return V,Qs,pi

def transfer_checks():
    ratios=[];maxviol=-math.inf
    for _ in range(200):
        N,S,A=4,4,3
        P=RNG.dirichlet(np.ones(S),size=(N,S,A))
        Ph=.96*P+.04*RNG.dirichlet(np.ones(S),size=(N,S,A))
        r=RNG.uniform(-1,1,(N,S,A));rh=r+RNG.uniform(-.01,.01,r.shape)
        g=RNG.uniform(-.5,.5,S);gh=g+RNG.uniform(-.01,.01,S)
        V,Q,pi=evaluate(P,r,g);Vh,Qh,pih=evaluate(Ph,rh,gh)
        # Perturb emitted policy scores, then evaluate that actual policy independently.
        score=Qh+RNG.uniform(-.08,.08,Qh.shape)
        emitted=np.argmax(score,axis=2)
        Vlive,_,_=evaluate(P,r,g,emitted)
        Vfit,_,_=evaluate(Ph,rh,gh,emitted)
        span=np.zeros(N+1);span[N]=np.ptp(gh)
        for k in reversed(range(N)):span[k]=span[k+1]+np.ptp(rh[k])
        eps=.5*np.abs(P-Ph).sum(-1).max(axis=(1,2))
        eta=np.abs(r-rh).max(axis=(1,2))
        B=float(np.max(abs(g-gh))+sum(eta+eps*span[1:]))
        assert np.max(abs(Vlive[0]-Vfit[0])) <= B+1e-12
        assert np.max(abs(V[0]-Vh[0])) <= B+1e-12
        zeta=Vh[:-1]-np.take_along_axis(Qh,emitted[...,None],axis=2)[...,0]
        occ=np.eye(S)[0];occupation=0.
        for k in range(N):
            occupation+=occ@zeta[k]
            occ=occ@Ph[k,np.arange(S),emitted[k]]
        assert abs(occupation-(Vh[0,0]-Vfit[0,0]))<1e-12
        loss=V[0,0]-Vlive[0,0];bound=2*B+occupation
        assert loss <= bound+1e-12
        ratios.append(loss/bound);maxviol=max(maxviol,loss-bound)
    record('finite_horizon_transfer',random_models=200,maximum_loss_bound_ratio=max(ratios),largest_violation=maxviol)
    # Exact binomial sums verify coverage for the Bernoulli specialization.
    worst=0.
    for n in (5,10,20,50):
        delta=.05;u=min(1,math.sqrt(((2+2)*math.log(2)+math.log(1/delta))/(2*n)))
        for p in np.linspace(.05,.95,19):
            fail=sum(math.comb(n,k)*p**k*(1-p)**(n-k) for k in range(n+1) if abs(k/n-p)>u)
            assert fail <= delta/2+1e-12
            worst=max(worst,fail)
    record('finite_sample_coverage',exact_binomial_cases=76,largest_failure_probability=worst)

def quoting_checks():
    largest=-1.
    for _ in range(1000):
        score=RNG.normal(size=5);err=RNG.uniform(.001,.2,5)
        true=score+RNG.uniform(-1,1,5)*err
        emitted=int(RNG.integers(5))
        bound=max(score+err)-(score[emitted]-err[emitted])
        assert max(true)-true[emitted] <= bound+1e-12
        a0=0
        lower=score[emitted]-score[a0]-err[emitted]-err[a0]
        assert true[emitted]-true[a0]>=lower-1e-12
        largest=max(largest,max(true)-true[emitted]-bound)
    d0=F(1,2);d=[F(1,10),F(41,10)];p=[F(9,10),F(1,10)]
    assert sum(x*y for x,y in zip(d,p))==d0
    loss=sum(-3*y for x,y in zip(d,p) if x>2)
    assert loss==F(-3,10)
    record('compiled_scores_and_latency_tail',message_cases=1000,largest_violation=largest,
           constant_delay_payoff=0,mixture_delay_payoff=float(loss),mean_delay=float(d0))

def payoff(phi,R,u,b=0.,a=.1,c=.01,kappa=0.):
    return u*phi*R-a*b*u*phi-c*phi*abs(u)-.5*(a*phi**2+kappa)*u*u

def forecast_checks():
    scenarios=np.array([[0,.14],[0,-.06],[1,.14],[1,-.06]])
    independent=np.full(4,.25);adverse=np.array([.5,0,0,.5])
    edge=.8*independent+.2*adverse
    # w = independent + theta*(adverse-independent), theta in [0,.2].
    D=adverse-independent
    Aeq=np.c_[np.eye(4),-D];beq=independent
    maxerr=0.
    grid=[F(2*n,25) for n in range(13)]
    values=[]
    for uq in grid:
        u=float(uq);cost=payoff(scenarios[:,0],scenarios[:,1],u)
        result=linprog(np.r_[cost,0],A_eq=Aeq,b_eq=beq,bounds=[(0,1)]*4+[(0,.2)],method='highs')
        assert result.success
        exact=F(1,200)*uq-F(1,40)*uq*uq
        maxerr=max(maxerr,abs(result.fun-float(exact)))
        assert abs(result.fun-float(exact))<1e-12
        values.append(exact)
    k=max(range(len(grid)),key=values.__getitem__)
    assert grid[k]==F(2,25) and values[k]==F(3,12500)
    # General three-moment polytopes: compare a continuous scalar optimizer to dense search.
    max_grid_gap=0.
    for _ in range(120):
        phi=RNG.uniform(0,1,6);ret=RNG.uniform(-.4,.4,6)
        weights=RNG.dirichlet(np.ones(6),size=4)
        p=weights@phi;s=weights@(phi**2);m=weights@(phi*ret)
        assert np.all(p*p<=s+1e-12) and np.all(s<=p+1e-12)
        a=.5;c=.01;b=RNG.uniform(-.2,.2);kap=.1
        def objective(u):return float(min(u*m-a*b*p*u-c*p*abs(u)-.5*(a*s+kap)*u*u))
        opt=minimize_scalar(lambda u:-objective(u),bounds=(0,1),method='bounded',options={'xatol':1e-13})
        u=max([0.,1.,float(opt.x)],key=objective)
        exact=max(objective(v) for v in np.linspace(0,1,2001))
        assert objective(u)>=exact-1e-9
        assert objective(u)>=.5*min(a*s+kap)*u*u-1e-8
        max_grid_gap=max(max_grid_gap,objective(u)-exact)
    for _ in range(100):
        h,b,u=RNG.normal(size=(3,5))
        support=h@b+sum(np.maximum(h*u,0))
        brute=max(h@(b+np.array(phi)*u) for phi in product((0,1),repeat=5))
        assert abs(support-brute)<1e-12
    record('robust_joint_moments',rational_grid_actions=len(grid),max_lp_error=maxerr,
           random_moment_families=120,fill_box_checks=100,largest_dense_grid_gap=max_grid_gap,
           robust_continuous_order=.1,robust_continuous_gain=.00025,
           robust_grid_order=float(grid[k]),robust_grid_gain=float(values[k]))

def sequential_checks():
    worst_mgf=-math.inf
    for _ in range(150):
        phi=RNG.uniform(0,1,5);r=RNG.uniform(-.3,.3,5);p=RNG.dirichlet(np.ones(5))
        u=RNG.uniform(-1,1);b=RNG.uniform(-.5,.5)
        Y=payoff(phi,r,u,b=b,a=.4,c=.02,kappa=.1)
        centered=Y-p@Y;W=np.ptp(Y)
        for lam in np.linspace(-10,10,31):
            lhs=np.log(p@np.exp(lam*centered));rhs=lam*lam*W*W/8
            assert lhs<=rhs+1e-12
            worst_mgf=max(worst_mgf,lhs-rhs)
    rho=1.;delta=.1;cross=0
    for signs in product((-1,1),repeat=12):
        S=0;hit=False
        for n,z in enumerate(signs,1):
            S+=z
            boundary=math.sqrt((n+rho)*math.log((n+rho)/(rho*delta*delta)))
            hit|=abs(S)>=boundary
        cross+=hit
    prob=cross/2**12
    assert prob<=delta
    # Importance weighting identity under supported fixed contextual laws.
    beta=np.array([.8,.2]);pi=np.array([.25,.75]);means=np.array([.1,.4])
    assert abs(sum(beta*pi/beta*means)-pi@means)<1e-15
    record('random_fill_noise_and_off_policy_identity',joint_laws=150,mgf_values=4650,
           max_mgf_violation=worst_mgf,enumerated_paths=4096,boundary_crossing_probability=prob,
           importance_weighted_value=float(pi@means))

def figures_and_tables():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.dpi':180,'figure.facecolor':'white'})
    number=1 if (ROOT/'main.tex').exists() else 2 if (ROOT/'p2_quoting.tex').exists() else 3
    if number==1:
        rows=[('Parent size (shares)','10'),('Reservation retained on cancel request','6'),
              ('Fill before cancellation is effective','3'),('Integer allocation cost', '8.500000'),
              ('Continuous allocation cost','8.333333'),('Integrality gap','0.166667')]
        x=np.linspace(0,5,501);y=.5*x*x+(5-x)**2
        fig,axs=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
        axs[0].plot(x,y,color='#123d6a',label='Continuous cost');k=np.arange(6)
        axs[0].scatter(k,.5*k*k+(5-k)**2,color='#b55035',label='Integer choices')
        axs[0].set(xlabel='Shares at destination 1',ylabel='Synthetic execution cost');axs[0].legend()
        r=np.arange(7);axs[1].plot(r,np.maximum(r-3,0),'o-',color='#123d6a')
        axs[1].set(xlabel='Remaining shares',ylabel='Minimum guaranteed current fill',title='Next-stage capacity = 3 shares')
    elif number==2:
        rows=[('Mean cancellation delay, both laws','0.500000'),('Constant-delay payoff (ticks)','0.000000'),
              ('Mixture-delay payoff (ticks)','-0.300000'),('Compiled-score comparisons','1,000'),
              ('Random finite-model comparisons','200')]
        fig,axs=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
        t=np.linspace(0,5,501)
        axs[0].step(t,(t<.5).astype(float),where='post',color='#123d6a',label='Constant delay')
        axs[0].step(t,.9*(t<.1)+.1*(t<4.1),where='post',color='#b55035',label='Same-mean mixture')
        axs[0].axvline(2,color='gray',ls='--',label='Adverse trade')
        axs[0].set(xlabel='Synthetic clock time',ylabel='Cancellation survival probability');axs[0].legend(fontsize=8)
        margin=np.linspace(0,.3,101)
        axs[1].plot(margin,margin-.1,color='#123d6a');axs[1].axhline(0,color='gray',lw=1)
        axs[1].set(xlabel='Computed advantage (ticks)',ylabel='Certified advantage lower bound',title='Two score errors of 0.05 ticks')
    else:
        rows=[('Robust continuous order','0.100000'),('Robust continuous gain','0.000250'),
              ('Robust lot-grid order','0.080000'),('Robust lot-grid gain','0.000240'),
              ('Joint-moment families checked','120'),('Sequential paths enumerated','4,096')]
        fig,axs=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
        u=np.linspace(0,.4,401)
        for theta,col in [(0,'#9dabad'),(.1,'#557e90'),(.2,'#123d6a')]:
            axs[0].plot(u,(.015-.05*theta)*u-.025*u*u,color=col,label=f'Adverse weight {theta:.1f}')
        grid=np.arange(6)*.08
        axs[0].scatter(grid,.005*grid-.025*grid*grid,color='#b55035',s=18)
        axs[0].axhline(0,color='gray',lw=.7);axs[0].set(xlabel='Attempted order',ylabel='Synthetic risk-adjusted gain');axs[0].legend(fontsize=8)
        e=np.linspace(0,.4,201);axs[1].plot(e,np.maximum(.3-e,0),color='#123d6a')
        axs[1].set(xlabel='Upper adverse-coupling weight',ylabel='Robust attempted buy')
    fig.savefig(ROOT/'figures/07_production_bridge.png');plt.close(fig)
    (ROOT/'tables/production_rows.tex').write_text(''.join(f'{k} & {v}\\\\\n' for k,v in rows))


def main():
    ledger_checks();execution_checks();transfer_checks();quoting_checks();forecast_checks();sequential_checks()
    figures_and_tables()
    REPORT['status']='all_passed'
    REPORT['numpy_version']=np.__version__
    (OUT/'production_results.json').write_text(json.dumps(REPORT,indent=2)+'\n')
    print(json.dumps(REPORT,indent=2))

if __name__=='__main__':main()
