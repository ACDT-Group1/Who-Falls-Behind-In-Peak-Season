import numpy as np, pandas as pd, itertools, json
z=np.load('/home/claude/bts/work/demeaned.npz'); Yd,Cd,Dd,DPd=z['Yd'],z['Cd'],z['Dd'],z['DPd']
meta=pd.read_pickle('/home/claude/bts/work/regmeta.pkl')
carr=sorted(meta.Reporting_Airline.unique()); LCCset={'WN','B6','NK','F9','G4'}
def bg(mask):
    X=np.column_stack([Dd[:,mask].sum(1),DPd[:,mask].sum(1),Cd]); b,*_=np.linalg.lstsq(X,Yd,rcond=None); return b[0],b[1]
res=[]
for combo in itertools.combinations(range(9),5):
    m=np.zeros(9,bool); m[list(combo)]=True
    b,g=bg(m); res.append((tuple(carr[i] for i in combo),b,g))
obs=[x for x in res if set(x[0])==LCCset][0]
B=np.array([x[1] for x in res]); G=np.array([x[2] for x in res])
out={'obs_beta':obs[1],'obs_gamma':obs[2],
 'p_beta_onesided':float((B>=obs[1]).mean()),          # H1: LCC worse off-peak
 'p_gamma_onesided':float((G>=obs[2]).mean()),         # H1 (pre-registered): gamma > 0
 'p_gamma_twosided':float((np.abs(G)>=abs(obs[2])).mean()),
 'rank_beta':int((B>obs[1]).sum())+1,'rank_gamma_from_top':int((G>obs[2]).sum())+1}
# carrier-level effects (relative to the FSC average), off-peak and peak
X=np.column_stack([Dd[:,1:],DPd,Cd]); b,*_=np.linalg.lstsq(X,Yd,rcond=None)
off=np.r_[0,b[:8]]; pk=off+b[8:17]
isF=np.array([c not in LCCset for c in carr])
out['carrier_off']=dict(zip(carr,(off-off[isF].mean()).round(3).tolist()))
out['carrier_peak']=dict(zip(carr,(pk-pk[isF].mean()).round(3).tolist()))
json.dump(out,open('/home/claude/bts/work/res_perm.json','w'),indent=1)
print(json.dumps(out,indent=1))
