"""Sharp one-opportunity fill/jump ambiguity; no fitted market law."""
from pathlib import Path
from fractions import Fraction as F
import json
import numpy as np
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parents[1]

def main():
 # Cell order: (X,Y)=(0,0),(0,1),(1,0),(1,1).
 A=np.array([[1,1,1,1],[0,0,1,1],[0,1,0,1]],float)
 gain=np.array([0,0,1,-1],float);error=0.;count=0
 for pi in range(11):
  for qi in range(11):
   p,q=F(pi,10),F(qi,10)
   lower=p-2*min(p,q);upper=p-2*max(F(0),p+q-1)
   for z in [max(F(0),p+q-1),min(p,q)]:
    masses=[1-p-q+z,q-z,p-z,z]
    assert min(masses)>=0 and sum(masses)==1
    assert masses[2]+masses[3]==p and masses[1]+masses[3]==q
   for sign,expected in [(1,lower),(-1,upper)]:
    sol=linprog(sign*gain,A_eq=A,b_eq=[1,float(p),float(q)],bounds=(0,None),method='highs')
    assert sol.success,sol.message
    error=max(error,abs(gain@sol.x-float(expected)));count+=1
 assert error<1e-12
 p,q=F(2,5),F(1,4);independent=p-2*p*q;worst=p-2*min(p,q);best=p-2*max(F(0),p+q-1)
 assert (independent,worst,best)==(F(1,5),F(-1,10),F(2,5))
 for i in range(101):
  theta=F(i,100);z=(1-theta)*p*q+theta*min(p,q)
  assert p-2*z==F(1,5)-F(3,10)*theta
 rows=[('Independent',float(independent),'Quote'),('Adverse coupling',float(worst),'Inactive'),('Marginal-only worst case',float(worst),'Inactive')]
 (ROOT/'tables/realism_rows.tex').write_text(''.join(f'{name} & {val:.2f} & {choice} \\\\\n' for name,val,choice in rows))
 result={'synthetic_only':True,'independent_gain':float(independent),'worst_gain':float(worst),'best_gain':float(best),'switch_theta':2/3,'lp_checks':count,'exact_mixture_cases':101,'max_lp_error':error}
 (ROOT/'verification/realism_stress.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
