import pandas as pd, numpy as np, json
d=pd.read_pickle('/home/claude/bts/work/slim12.pkl')
wx=['dw_sknt','dw_gust','dw_vsby','dw_p01i','dw_ts','dw_sn','dw_fg','dest_cong','Distance']
d=d.dropna(subset=['dw_sknt','dw_vsby','dw_gust','dest_cong'])
car=d.carrier.astype(str).values
out={}
for c in sorted(set(car)):
    r=d[car!=c]; r=r[r.groupby('cell').LCC.transform('nunique')==2]
    f1=pd.factorize(r.cell.values)[0]; f2=pd.factorize(r.Dest.astype(str).values)[0]; n1=np.bincount(f1); n2=np.bincount(f2)
    L=r.LCC.values.astype(float); cols=[r.y.values.astype(float),L,L*r.PI.values.astype(float)]+[r[w].values.astype(float) for w in wx]
    M=[]
    for v in cols:
        v=v.copy()
        for _ in range(2000):
            old=v.copy(); v-=(np.bincount(f1,v)/n1)[f1]; v-=(np.bincount(f2,v)/n2)[f2]
            if np.max(np.abs(v-old))<1e-8: break
        M.append(v)
    M=np.column_stack(M); b,*_=np.linalg.lstsq(M[:,1:],M[:,0],rcond=None)
    out[c]={'beta':round(float(b[0]),3),'gamma':round(float(b[1]),3),'n':int(len(r))}; print(c,out[c],flush=True)
json.dump(out,open('/home/claude/bts/work/res12_loo.json','w'),indent=1)
