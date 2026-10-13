"""95% confidence intervals by inverting the 126-labelling permutation test.
For a candidate value v of gamma (or beta), subtract v * (true LCC x M) (or v * true LCC) from y, re-estimate under
every labelling of 5 of 9 carriers as LCC, and keep v if the true labelling's estimate is not in the extreme 5% (two-sided).
Demeaning is linear, so shifted outcomes are handled exactly with precomputed cross-products. Usage: python3 ci.py NAME"""
import pandas as pd, numpy as np, itertools, json, sys, time
NAME=sys.argv[1]; t0=time.time()
LCCset={'WN','B6','NK','F9','G4'}
wx=['dw_sknt','dw_gust','dw_vsby','dw_p01i','dw_ts','dw_sn','dw_fg','dest_cong','Distance']
d=pd.read_pickle('/home/claude/bts/work/slim13.pkl')
d=d.dropna(subset=['dw_sknt','dw_vsby','dw_gust','dest_cong'])
if NAME=='A1_months_perday': d=d[d.month.isin([6,7,8,1,2,12])]; M=d.month.isin([6,7,8]).astype(float).values
elif NAME=='A2_PI_dow': M=d.PI_dow.values.astype(float)
elif NAME=='C7_hour': M=d.HCI.values.astype(float)
elif NAME=='C7_wx': d=d[d.severe.notna()]; M=d.severe.values.astype(float)
elif NAME=='B6_mech': M=d.PI_dow.values.astype(float); d['ground_f']=d.ground.fillna(0); wx=wx+['ground_f','first','legs']
keep=(d.groupby('cell').LCC.transform('nunique')==2).values; d=d[keep]; M=M[keep]
car=d.carrier.astype(str).values; carr=sorted(set(car)); isL=np.array([c in LCCset for c in carr])
f1=pd.factorize(d.cell.values)[0]; f2=pd.factorize(d.Dest.astype(str).values)[0]; n1=np.bincount(f1); n2=np.bincount(f2)
def dm(v):
    v=np.array(v,dtype=float)
    for _ in range(2000):
        old=v.copy(); v-=(np.bincount(f1,v)/n1)[f1]; v-=(np.bincount(f2,v)/n2)[f2]
        if np.max(np.abs(v-old))<1e-8: break
    return v
cols=[dm((car==c).astype(float)) for c in carr]+[dm((car==c)*M) for c in carr]+[dm(d[w].values) for w in wx]
Z=np.column_stack(cols); del cols
Yd=dm(d.y.values)
ZZ=Z.T@Z; ZY=Z.T@Yd; K=Z.shape[1]; kc=len(wx)
print('demeaned',round(time.time()-t0),flush=True)
del Z
labs=[]
for combo in itertools.combinations(range(9),5):
    m=np.zeros(9,bool); m[list(combo)]=True; labs.append(m)
ti=[i for i,m in enumerate(labs) if (m==isL).all()][0]
def S(m):
    s=np.zeros((K,2+kc)); s[:9,0]=m; s[9:18,1]=m; s[18:,2:]=np.eye(kc); return s
Ss=[S(m) for m in labs]; A=[np.linalg.inv(s.T@ZZ@s) for s in Ss]
Lt=np.r_[isL.astype(float),np.zeros(9+kc)]; Pt=np.r_[np.zeros(9),isL.astype(float),np.zeros(kc)]
ZL=ZZ@Lt; ZP=ZZ@Pt                      # Z'(true LCC), Z'(true LCC x M)
def ests(k,v):                           # k=0 beta, 1 gamma; outcome shifted by v
    zy=ZY-v*(ZL if k==0 else ZP)
    return np.array([ (A[i]@(Ss[i].T@zy))[k] for i in range(len(labs)) ])
def pval(k,v):
    e=ests(k,v); return float((np.abs(e)>=abs(e[ti])-1e-12).mean())
est={k:float(ests(k,0)[ti]) for k in (0,1)}
def bound(k,side):
    c=est[k]; step=max(abs(c),1.0)*0.5
    lo=c; hi=c+side*step
    while pval(k,hi)>0.05: lo=hi; hi=hi+side*step; step*=1.6
    for _ in range(40):
        mid=(lo+hi)/2
        if pval(k,mid)>0.05: lo=mid
        else: hi=mid
    return (lo+hi)/2
out={'name':NAME,'beta':est[0],'gamma':est[1],
     'beta_ci':[bound(0,-1),bound(0,1)],'gamma_ci':[bound(1,-1),bound(1,1)],
     'p_gamma_two_check':pval(1,0.0),'p_beta_two_check':pval(0,0.0)}
json.dump(out,open(f'/home/claude/bts/work/ci_{NAME}.json','w'),indent=1)
print(json.dumps(out),round(time.time()-t0),flush=True)
