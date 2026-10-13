import pandas as pd, numpy as np, itertools, json
fl=pd.read_pickle('/home/claude/bts/work/flights4.pkl')
d=fl[fl.ArrDel15.notna()].copy()
d['y']=d.ArrDel15*100   # percentage points
out={}
# ---------- Step 1: raw gap ----------
def raw(x): r=x.groupby('LCC').y.mean(); return r[1]-r[0]
out['raw']={'all':raw(d),'off':raw(d[d.Peak==0]),'peak':raw(d[d.Peak==1])}
out['rates']={f'peak{k[0]}_lcc{k[1]}':v for k,v in d.groupby(['Peak','LCC']).y.mean().round(3).to_dict().items()}
# ---------- Step 2: same-cell gap (weighted by flights in cell) ----------
def cellgap(x):
    g=x.groupby(['cell','LCC']).y.agg(['mean','size']).unstack()
    g=g.dropna()
    w=g['size'].sum(axis=1)
    return ((g['mean'][1]-g['mean'][0])*w).sum()/w.sum(), w.sum()/len(x), len(g)
for k,sub in [('all',d),('off',d[d.Peak==0]),('peak',d[d.Peak==1])]:
    gp,cov,n=cellgap(sub); out.setdefault('cell',{})[k]={'gap':gp,'coverage':cov,'cells':n}
# ---------- Step 3: FE regression ----------
wx=['dw_sknt','dw_gust','dw_vsby','dw_p01i','dw_ts','dw_sn','dw_fg','dest_cong','Distance']
r=d.copy(); r['dw_p01i']=r.dw_p01i.fillna(0)
r=r.dropna(subset=['dw_sknt','dw_vsby','dw_gust','dest_cong']).copy()
for c in ['dw_ts','dw_sn','dw_fg']: r[c]=r[c].astype(float)
# keep only cells with both types (singletons/one-type cells carry no information on LCC within cell)
mix=r.groupby('cell').LCC.transform('nunique')==2
r=r[mix].reset_index(drop=True)
out['reg_n']=len(r)
fe1=pd.factorize(r.cell)[0]; fe2=pd.factorize(r.Dest)[0]
def demean(M,tol=1e-9,it=500):
    M=np.array(M,dtype=float).copy()
    if M.ndim==1: M=M[:,None]
    n1=np.bincount(fe1); n2=np.bincount(fe2)
    for k in range(it):
        old=M.copy()
        for fe,nn in ((fe1,n1),(fe2,n2)):
            for j in range(M.shape[1]):
                M[:,j]-=(np.bincount(fe,M[:,j])/nn)[fe]
        if np.max(np.abs(M-old))<tol: break
    return M
carr=sorted(r.Reporting_Airline.unique())
Dm=np.column_stack([(r.Reporting_Airline==c).values for c in carr]).astype(float)
DmP=Dm*r.Peak.values[:,None]
Yd=demean(r.y.values)[:,0]
Cd=demean(r[wx].values)
Dd=demean(Dm); DPd=demean(DmP)
np.savez('/home/claude/bts/work/demeaned.npz',Yd=Yd,Cd=Cd,Dd=Dd,DPd=DPd)
r[['Reporting_Airline','Origin','Peak','LCC']].to_pickle('/home/claude/bts/work/regmeta.pkl')
def fit(sel, Xd, cl=None):
    X=np.column_stack([Xd,Cd]); b,*_=np.linalg.lstsq(X,Yd,rcond=None)
    if cl is None: return b
    e=Yd-X@b; XtXi=np.linalg.pinv(X.T@X)
    g=pd.factorize(cl)[0]; S=np.zeros((X.shape[1],X.shape[1]))
    sc=np.zeros((g.max()+1,X.shape[1])); np.add.at(sc,g,X*e[:,None])
    S=sc.T@sc; G=g.max()+1; n,k=X.shape; adj=G/(G-1)*(n-1)/(n-k)
    V=adj*XtXi@S@XtXi; return b,np.sqrt(np.diag(V))
isL=np.array([c in {'WN','B6','NK','F9','G4'} for c in carr])
L=Dd[:,isL].sum(1); LP=DPd[:,isL].sum(1)
b,se=fit(None,np.column_stack([L,LP]),r.Origin.values)
_,se_c=fit(None,np.column_stack([L,LP]),r.Reporting_Airline.values)
out['reg']={'beta':b[0],'gamma':b[1],'se_origin':se[:2].tolist(),'se_carrier':se_c[:2].tolist(),'controls':dict(zip(wx,b[2:].round(4).tolist()))}
json.dump(out,open('/home/claude/bts/work/res_123.json','w'),indent=1,default=float)
print(json.dumps(out,indent=1,default=float))
