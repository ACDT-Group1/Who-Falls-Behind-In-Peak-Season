import pandas as pd, numpy as np, itertools, json, time
t0=time.time()
LCCset={'WN','B6','NK','F9','G4'}
d=pd.read_pickle('/home/claude/bts/work/slim12.pkl')
d['y']=d.y.astype('float64')
bins=[-10,-0.10,0,0.10,10]; labs=['<-10%','-10–0%','0–+10%','>+10%']
d['bin']=pd.cut(d.PI,bins,labels=labs)
out={'n_flights':len(d)}
def rawgap(x): r=x.groupby('LCC').y.mean(); return float(r[1]-r[0])
def cellgap(x):
    c=pd.factorize(x.cell.values)[0]; L=x.LCC.values.astype(float); y=x.y.values
    nL=np.bincount(c,L); nF=np.bincount(c,1-L); sL=np.bincount(c,y*L); sF=np.bincount(c,y*(1-L))
    ok=(nL>0)&(nF>0); w=(nL+nF)[ok]
    return float(((sL[ok]/nL[ok]-sF[ok]/nF[ok])*w).sum()/w.sum()), float(w.sum()/len(x))
out['raw']=rawgap(d); out['cell'],out['coverage']=cellgap(d)
out['bins']={b:{'raw':rawgap(d[d.bin==b]),'cell':cellgap(d[d.bin==b])[0],'share':float((d.bin==b).mean())} for b in labs}
# same-cell gap by bin, dropping one carrier at a time (for the interactive explorer)
out['bins_drop']={}
for c in ['none']+sorted(d.carrier.unique()):
    s=d if c=='none' else d[d.carrier!=c]
    out['bins_drop'][c]={'all':cellgap(s)[0],**{b:cellgap(s[s.bin==b])[0] for b in labs}}
print('descriptives',round(time.time()-t0),flush=True)
# ---------- regression ----------
wx=['dw_sknt','dw_gust','dw_vsby','dw_p01i','dw_ts','dw_sn','dw_fg','dest_cong','Distance']
r=d.dropna(subset=['dw_sknt','dw_vsby','dw_gust','dest_cong']).copy()
r=r[r.groupby('cell').LCC.transform('nunique')==2].reset_index(drop=True)
out['reg_n']=len(r); out['reg_PI_pct']=r.PI.quantile([.01,.1,.5,.9,.99]).round(3).tolist()
f1=pd.factorize(r.cell.values)[0]; f2=pd.factorize(r.Dest.astype(str).values)[0]; n1=np.bincount(f1); n2=np.bincount(f2)
def demean1(v):
    v=np.array(v,dtype=float)
    for _ in range(2000):
        old=v.copy()
        v-=(np.bincount(f1,v)/n1)[f1]; v-=(np.bincount(f2,v)/n2)[f2]
        if np.max(np.abs(v-old))<1e-8: break
    return v
def demean(M):
    M=np.asarray(M)
    if M.ndim==1: M=M[:,None]
    return np.column_stack([demean1(M[:,j]) for j in range(M.shape[1])])
carr=sorted(r.carrier.astype(str).unique())
Dm=np.column_stack([(r.carrier.astype(str)==c).values for c in carr]).astype(float)
r_y=r.y.values; r_wx=r[wx].values.astype(float); r_org=r.Origin.astype(str).values; r_car=r.carrier.astype(str).values; PI=r.PI.values.astype(float)
del d
Yd=demean(r_y)[:,0]; Cd=demean(r_wx); Dd=demean(Dm); DPd=demean(Dm*PI[:,None])
print('demeaned',round(time.time()-t0),flush=True)
isL=np.array([c in LCCset for c in carr])
def ols(X): b,*_=np.linalg.lstsq(X,Yd,rcond=None); return b
def clse(X,b,cl):
    e=Yd-X@b; XtXi=np.linalg.pinv(X.T@X); g=pd.factorize(cl)[0]; G=g.max()+1
    sc=np.zeros((G,X.shape[1])); np.add.at(sc,g,X*e[:,None]); n,k=X.shape
    V=G/(G-1)*(n-1)/(n-k)*XtXi@(sc.T@sc)@XtXi; return np.sqrt(np.diag(V))
X=np.column_stack([Dd[:,isL].sum(1),DPd[:,isL].sum(1),Cd]); b=ols(X)
out['reg']={'beta':float(b[0]),'gamma':float(b[1]),'gamma_per10':float(b[1]*0.1),
  'se_origin':clse(X,b,r_org)[:2].tolist(),'se_carrier':clse(X,b,r_car)[:2].tolist(),
  'controls':dict(zip(wx,np.round(b[2:],4).tolist()))}
# permutation over 126 labelings
B=[];G=[]
for combo in itertools.combinations(range(9),5):
    m=np.zeros(9,bool); m[list(combo)]=True
    bb=ols(np.column_stack([Dd[:,m].sum(1),DPd[:,m].sum(1),Cd])); B.append(bb[0]); G.append(bb[1])
B=np.array(B);G=np.array(G)
out['perm']={'p_beta_one':float((B>=b[0]).mean()),'p_gamma_one':float((G>=b[1]).mean()),'p_gamma_two':float((np.abs(G)>=abs(b[1])).mean()),
  'rank_beta':int((B>b[0]).sum())+1,'rank_gamma':int((G>b[1]).sum())+1}
# carrier effects relative to FSC mean: at PI=0 and PI=+0.15
bc=ols(np.column_stack([Dd[:,1:],DPd,Cd])); lv=np.r_[0,bc[:8]]; sl=bc[8:17]
isF=~isL
for name,p in [('pi0',0.0),('pi15',0.15)]:
    v=lv+sl*p; out.setdefault('carrier',{})[name]=dict(zip(carr,np.round(v-v[isF].mean(),3).tolist()))
json.dump(out,open('/home/claude/bts/work/res12.json','w'),indent=1)
print(json.dumps({k:v for k,v in out.items() if k!='bins_drop'},indent=1),flush=True)
print('done',round(time.time()-t0))
