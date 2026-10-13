import pandas as pd, numpy as np, json
LCCset={'WN','B6','NK','F9','G4'}
d=pd.read_pickle('/home/claude/bts/work/slim12.pkl')
d['y']=d.y.astype(float); car=d.carrier.astype(str).values
out={'n_flights':int(len(d))}
def rawgap(m): L=d.LCC.values[m]; y=d.y.values[m]; return float(y[L==1].mean()-y[L==0].mean())
def cellgap(m):
    c=pd.factorize(d.cell.values[m])[0]; L=d.LCC.values[m].astype(float); y=d.y.values[m]
    nL=np.bincount(c,L); nF=np.bincount(c,1-L); sL=np.bincount(c,y*L); sF=np.bincount(c,y*(1-L))
    ok=(nL>0)&(nF>0); w=(nL+nF)[ok]; return float(((sL[ok]/nL[ok]-sF[ok]/nF[ok])*w).sum()/w.sum()), float(w.sum()/m.sum())
allm=np.ones(len(d),bool)
out['raw']=rawgap(allm); out['cell'],out['coverage']=cellgap(allm)
TOP,BOT=[6,7,8],[1,2,12]
mo=d.month.values
for k,ms in [('peak',TOP),('off',BOT)]:
    m=np.isin(mo,ms); out[f'raw_{k}']=rawgap(m); out[f'cell_{k}']=cellgap(m)[0]
pi=d.PI_dow.values
bins=[('<-10%',pi<-0.10),('-10–0%',(pi>=-0.10)&(pi<0)),('0–+10%',(pi>=0)&(pi<0.10)),('>+10%',pi>=0.10)]
out['bins']={b:{'raw':rawgap(m),'cell':cellgap(m)[0],'share':float(m.mean())} for b,m in bins}
out['bdrop']={}
for c in ['none']+sorted(set(car)):
    keep=allm if c=='none' else car!=c
    out['bdrop'][c]={b:cellgap(m&keep)[0] for b,m in bins}
# carrier effects in the PI_dow spec (same FE + controls as runspec)
wx=['dw_sknt','dw_gust','dw_vsby','dw_p01i','dw_ts','dw_sn','dw_fg','dest_cong','Distance']
r=d.dropna(subset=['dw_sknt','dw_vsby','dw_gust','dest_cong']); r=r[r.groupby('cell').LCC.transform('nunique')==2]
out['pi_pct']=r.PI_dow.quantile([.1,.25,.5,.75,.9]).round(3).tolist()
del d
f1=pd.factorize(r.cell.values)[0]; f2=pd.factorize(r.Dest.astype(str).values)[0]; n1=np.bincount(f1); n2=np.bincount(f2)
def dm(v):
    v=np.array(v,dtype=float)
    for _ in range(2000):
        old=v.copy(); v-=(np.bincount(f1,v)/n1)[f1]; v-=(np.bincount(f2,v)/n2)[f2]
        if np.max(np.abs(v-old))<1e-8: break
    return v
rc=r.carrier.astype(str).values; carr=sorted(set(rc)); P=r.PI_dow.values.astype(float)
Yd=dm(r.y.values); Cd=np.column_stack([dm(r[w].values) for w in wx])
Dd=np.column_stack([dm((rc==c).astype(float)) for c in carr]); DPd=np.column_stack([dm((rc==c)*P) for c in carr])
b,*_=np.linalg.lstsq(np.column_stack([Dd[:,1:],DPd,Cd]),Yd,rcond=None); lv=np.r_[0,b[:8]]; sl=b[8:17]
isF=np.array([c not in LCCset for c in carr])
for name,p in [('pi0',0.0),('pi15',0.15)]:
    v=lv+sl*p; out.setdefault('carrier',{})[name]=dict(zip(carr,np.round(v-v[isF].mean(),3).tolist()))
c=pd.read_pickle('/home/claude/bts/work/origin_day_all.pkl')
m=c.groupby('MONTH').agg(t=('n','sum'),dd=('date','nunique')); out['mo_perday']=(m.t/m.dd).round(0).tolist()
json.dump(out,open('/home/claude/bts/work/sitefig.json','w'),indent=1)
print(json.dumps({k:v for k,v in out.items() if k!='bdrop'},indent=1))
