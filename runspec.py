"""One estimation: y = b*LCC + g*(LCC x M) [+ extra LCC interactions] + cell FE + dest FE + controls.
Permutation over 126 labelings and leave-one-carrier-out. Usage: python3 runspec.py NAME"""
import pandas as pd, numpy as np, itertools, json, sys, time
NAME=sys.argv[1]; t0=time.time()
LCCset={'WN','B6','NK','F9','G4'}
wx=['dw_sknt','dw_gust','dw_vsby','dw_p01i','dw_ts','dw_sn','dw_fg','dest_cong','Distance']
d=pd.read_pickle('/home/claude/bts/work/slim12.pkl')
d=d.dropna(subset=['dw_sknt','dw_vsby','dw_gust','dest_cong'])
TOP_PD,BOT_PD=[6,7,8],[1,2,12]          # daily-average ranking, all carriers
TOP_TOT,BOT_TOT=[7,8,10],[1,2,11]       # total-flights ranking, all carriers
extra=None
if NAME=='A1_months_perday':
    d=d[d.month.isin(TOP_PD+BOT_PD)]; M=d.month.isin(TOP_PD).astype(float).values
elif NAME=='A1_months_total':
    d=d[d.month.isin(TOP_TOT+BOT_TOT)]; M=d.month.isin(TOP_TOT).astype(float).values
elif NAME=='A2_PI_dow':  M=d.PI_dow.values.astype(float)
elif NAME=='A2_PI_x_weekday':
    M=d.PI_all.values.astype(float); extra=np.column_stack([(d.dow.values==k).astype(float) for k in range(1,7)])
elif NAME=='A3_PI_all':  M=d.PI_all.values.astype(float)
keep=(d.groupby('cell').LCC.transform('nunique')==2).values
d=d[keep]; M=M[keep]; extra=None if extra is None else extra[keep]
car=d.carrier.astype(str).values; carr=sorted(set(car)); isL=np.array([c in LCCset for c in carr])
def fe_codes(r): 
    f1=pd.factorize(r.cell.values)[0]; f2=pd.factorize(r.Dest.astype(str).values)[0]; return f1,f2,np.bincount(f1),np.bincount(f2)
def demean1(v,f1,f2,n1,n2):
    v=np.array(v,dtype=float)
    for _ in range(2000):
        old=v.copy(); v-=(np.bincount(f1,v)/n1)[f1]; v-=(np.bincount(f2,v)/n2)[f2]
        if np.max(np.abs(v-old))<1e-8: break
    return v
f1,f2,n1,n2=fe_codes(d)
dm=lambda v: demean1(v,f1,f2,n1,n2)
Yd=dm(d.y.values); Cd=np.column_stack([dm(d[w].values) for w in wx])
Dm=np.column_stack([(car==c).astype(float) for c in carr])
Dd=np.column_stack([dm(Dm[:,j]) for j in range(9)]); DMd=np.column_stack([dm(Dm[:,j]*M) for j in range(9)])
if extra is not None:
    DEd=[np.column_stack([dm(Dm[:,j]*extra[:,k]) for j in range(9)]) for k in range(extra.shape[1])]
def est(mask):
    cols=[Dd[:,mask].sum(1),DMd[:,mask].sum(1)]
    if extra is not None: cols+= [E[:,mask].sum(1) for E in DEd]
    X=np.column_stack(cols+[Cd]); b,*_=np.linalg.lstsq(X,Yd,rcond=None); return b
b=est(isL)
B=[];G=[]
for combo in itertools.combinations(range(9),5):
    m=np.zeros(9,bool); m[list(combo)]=True; bb=est(m); B.append(bb[0]); G.append(bb[1])
B=np.array(B);G=np.array(G)
out={'name':NAME,'n':int(len(d)),'beta':float(b[0]),'gamma':float(b[1]),
 'p_beta_one':float((B>=b[0]).mean()),'p_gamma_one':float((G>=b[1]).mean()),'p_gamma_two':float((np.abs(G)>=abs(b[1])).mean()),
 'rank_gamma':int((G>b[1]).sum())+1,'M_sd':float(np.std(M))}
print('main+perm',round(time.time()-t0),json.dumps(out),flush=True)
# leave one carrier out (re-demean on each reduced sample)
loo={}
for c in carr:
    r=d[car!=c]; Mr=M[car!=c]; Lr=r.LCC.values.astype(float)
    k2=(r.groupby('cell').LCC.transform('nunique')==2).values; r=r[k2]; Mr=Mr[k2]; Lr=Lr[k2]
    g1,g2,m1,m2=fe_codes(r); dd=lambda v: demean1(v,g1,g2,m1,m2)
    cols=[dd(r.y.values),dd(Lr),dd(Lr*Mr)]
    if extra is not None:
        er=extra[car!=c][k2]; cols+=[dd(Lr*er[:,k]) for k in range(er.shape[1])]
    cols+=[dd(r[w].values) for w in wx]
    X=np.column_stack(cols); bb,*_=np.linalg.lstsq(X[:,1:],X[:,0],rcond=None)
    loo[c]={'beta':round(float(bb[0]),3),'gamma':round(float(bb[1]),3)}
out['loo']=loo
json.dump(out,open(f'/home/claude/bts/work/spec_{NAME}.json','w'),indent=1)
print('done',round(time.time()-t0),json.dumps(loo),flush=True)
