"""Baseline quoting improvement with exact outward-rounded queue enclosures."""
from pathlib import Path
from fractions import Fraction as F
import json
import numpy as np
from scipy.linalg import solve
from queue_certificates import book_cell, down, up
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    states,index,original,actions,base=book_cell()
    n=len(states); beta=F(3,5); nu=F(3); clock=F(13,2)
    start=index[(0,0,0,0,0)]
    def matrices(policy,events,fractions):
        L=np.zeros((n,n)); r=np.array(base,dtype=float)
        cursor=0
        for i,row in enumerate(events):
            for j,lo,hi,gl,gu in row:
                f,h=fractions[cursor]; cursor+=1
                lam=float((lo+hi)/2)+f*float((hi-lo)/2)
                mark=float((gl+gu)/2)+h*float((gu-gl)/2)
                L[i,j]+=lam;L[i,i]-=lam;r[i]+=lam*mark
            _,target,fee=actions[i][policy[i]]
            L[i,target]+=float(nu);L[i,i]-=float(nu);r[i]-=float(nu*fee)
        assert np.max(np.abs(L.sum(axis=1)))<2e-15
        return L,r
    count=sum(map(len,original));zero=np.zeros((count,2))
    proposal=np.zeros(n,dtype=int)
    for _ in range(50):
        L,r=matrices(proposal,original,zero)
        value=solve(float(beta)*np.eye(n)-L,r)
        candidate=np.array([np.argmax([value[j]-float(c) for _,j,c in row]) for row in actions])
        if np.array_equal(proposal,candidate):break
        proposal=candidate
    else:raise AssertionError('Nominal policy iteration did not converge')
    baseline=proposal.copy()
    for i,(z,q,side,level,rank) in enumerate(states):
        if side and rank:
            front=index[(z,q,side,level,0)]
            name=actions[front][proposal[front]][0]
            baseline[i]=next(k for k,a in enumerate(actions[i]) if a[0]==name)
    rows=[]; max_identity_error=0.; total_models=0
    rng=np.random.default_rng(5258)
    for multiplier in [0,1,4,16,64]:
        events=[[(j,(lo+hi)/2-multiplier*(hi-lo)/2,(lo+hi)/2+multiplier*(hi-lo)/2,
                     (gl+gu)/2-multiplier*(gu-gl)/2,(gl+gu)/2+multiplier*(gu-gl)/2)
                  for j,lo,hi,gl,gu in row] for row in original]
        assert all(lo>0 for row in events for j,lo,hi,gl,gu in row)
        assert all(sum(e[2] for e in row)+nu<=clock for row in events)
        for i,row in enumerate(events):
            assert abs(base[i])+sum(e[2]*max(abs(e[3]),abs(e[4])) for e in row)+nu*F(1,250)<1
        def fixed_update(v,upper):
            out=[]; choose=max if upper else min; rounding=up if upper else down
            for i,row in enumerate(events):
                h=base[i]+sum(choose(lam*(mark+v[j]-v[i]) for lam in [lo,hi] for mark in [gl,gu]) for j,lo,hi,gl,gu in row)
                _,dest,fee=actions[i][baseline[i]]
                out.append(rounding((clock*v[i]+h+nu*(v[dest]-v[i]-fee))/(beta+clock)))
            return out
        lower=[-1/beta]*n;upper=[1/beta]*n
        for _ in range(320):lower,upper=fixed_update(lower,False),fixed_update(upper,True)
        accepted=baseline.copy();g=[]
        for i,row in enumerate(actions):
            _,z,cb=row[baseline[i]]
            aa=[nu*((lower[y]-upper[z] if y!=z else F(0))-fee+cb) for _,y,fee in row]
            assert aa[baseline[i]]==0
            best=max(range(len(row)),key=lambda k:aa[k])
            if aa[best]>F(1,10**10):accepted[i]=best
            g.append(aa[accepted[i]])
        gain=[F(0)]*n
        for _ in range(320):
            new=[]
            for i,row in enumerate(events):
                external=sum(min(lo*(gain[j]-gain[i]),hi*(gain[j]-gain[i])) for j,lo,hi,gl,gu in row)
                _,dest,_=actions[i][accepted[i]]
                new.append(max(F(0),down((g[i]+clock*gain[i]+external+nu*(gain[dest]-gain[i]))/(beta+clock))))
            gain=new
        comparisons=[]
        draws=[zero,np.full((count,2),-1.),np.full((count,2),1.)]+[rng.uniform(-1,1,(count,2)) for _ in range(20)]
        for fractions in draws:
            Lb,rb=matrices(baseline,events,fractions);Lp,rp=matrices(accepted,events,fractions)
            vb=solve(float(beta)*np.eye(n)-Lb,rb);vp=solve(float(beta)*np.eye(n)-Lp,rp)
            advantage=rp-rb+(Lp-Lb)@vb
            resolvent=solve(float(beta)*np.eye(n)-Lp,advantage)
            err=float(np.max(np.abs(vp-vb-resolvent)));max_identity_error=max(max_identity_error,err)
            assert err<2e-13
            assert np.all(vb>=np.array(lower,dtype=float)-1e-12) and np.all(vb<=np.array(upper,dtype=float)+1e-12)
            assert np.all(advantage>=np.array(g,dtype=float)-1e-12)
            assert np.all(vp-vb>=np.array(gain,dtype=float)-1e-12)
            comparisons.append(float(vp[start]-vb[start]));total_models+=1
        rows.append({'uncertainty_multiplier':multiplier,'changed_states':int(np.sum(accepted!=baseline)),
                     'certified_gain':float(gain[start]),'certified_gain_exact':str(gain[start]),
                     'nominal_gain':comparisons[0],'minimum_sampled_gain':min(comparisons),
                     'max_baseline_interval_width':float(max(u-l for l,u in zip(lower,upper))),
                     'policy':accepted.tolist(),'baseline':baseline.tolist()})
        print('Completed uncertainty multiplier',multiplier,flush=True)
    assert rows[0]['changed_states']>0 and rows[0]['certified_gain']>0
    report={'status':'all_passed','data':'Synthetic queue model; no market data.',
            'states':n,'seed':5258,'exact_rational_iterations':320,'outward_decimal_places':12,
            'independently_evaluated_models':total_models,'maximum_identity_error':max_identity_error,'rows':rows}
    (ROOT/'verification/certified_improvement_results.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'tables').mkdir(exist_ok=True)
    (ROOT/'tables/certified_improvement_rows.tex').write_text(''.join(
        f"{r['uncertainty_multiplier']} & {r['changed_states']} & {r['certified_gain']:.8f} & {r['nominal_gain']:.8f} \\\\\n" for r in rows))
    fig,ax=plt.subplots(1,2,figsize=(10.8,3.8),layout='constrained')
    xx=np.arange(len(rows));labels=[str(r['uncertainty_multiplier']) for r in rows]
    ax[0].bar(xx,[r['changed_states'] for r in rows],color='#0057a6')
    ax[0].set(xticks=xx,xticklabels=labels,xlabel='Uncertainty multiplier',ylabel='Accepted message changes')
    ax[1].plot(xx,[r['nominal_gain'] for r in rows],'o-',label='Nominal gain',color='#0057a6')
    ax[1].plot(xx,[r['certified_gain'] for r in rows],'s--',label='Certified lower gain',color='#b05e00')
    ax[1].set(xticks=xx,xticklabels=labels,xlabel='Uncertainty multiplier',ylabel='Discounted objective gain')
    ax[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax[1].legend(frameon=False)
    for a in ax:a.spines[['top','right']].set_visible(False)
    fig.savefig(ROOT/'figures/06_certified_improvement.png',dpi=220,facecolor='white');plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
