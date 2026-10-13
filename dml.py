"""Step 5: partially linear DML, D = LCC, Y = late 15+ (pp). Nuisance models: sklearn HistGradientBoosting.
5-fold cross-fitting grouped by date (same-day weather never in both train and test). Run per subgroup and compare theta."""
import pandas as pd, numpy as np, json, time
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
t0=time.time()
d=pd.read_pickle('/home/claude/bts/work/slim13.pkl')
d=d.dropna(subset=['dw_sknt','dw_vsby','dw_gust','dest_cong','ow_vsby'])
d['org']=d.Origin.astype('category').cat.codes; d['dst']=d.Dest.astype('category').cat.codes
W=['org','dst','dep_hr','dow','month','doy','Distance','dest_cong','dw_sknt','dw_gust','dw_vsby','dw_p01i','dw_ts','dw_sn','dw_fg',
   'ow_vsby','ow_gust','ow_p01i','severe','HCI','PI_dow']
q1,q2=d.PI_dow.quantile([1/3,2/3])
groups={'off_months':d.month.isin([1,2,12]),'peak_months':d.month.isin([6,7,8]),
        'quiet_days':d.PI_dow<=q1,'busy_days':d.PI_dow>=q2}
N=300000; rng=np.random.default_rng(42); out={'pi_dow_terciles':[float(q1),float(q2)]}
def fit(sub):
    X=sub[W].astype(float).values; y=sub.y.values.astype(float); D=sub.LCC.values.astype(float); g=sub.doy.values
    cat=[W.index('org'),W.index('dst')]
    yr=np.zeros(len(y)); dr=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(X,y,g):
        my=HistGradientBoostingRegressor(max_iter=300,learning_rate=.08,max_leaf_nodes=63,categorical_features=cat,random_state=0).fit(X[tr],y[tr])
        md=HistGradientBoostingClassifier(max_iter=300,learning_rate=.08,max_leaf_nodes=63,categorical_features=cat,random_state=0).fit(X[tr],D[tr])
        yr[te]=y[te]-my.predict(X[te]); dr[te]=D[te]-md.predict_proba(X[te])[:,1]
    th=(yr*dr).sum()/(dr*dr).sum(); psi=(yr-th*dr)*dr
    # standard error clustered by date
    cl=pd.Series(psi).groupby(g).sum().values; se=np.sqrt((cl**2).sum())/(dr*dr).sum()
    r2y=1-np.var(yr)/np.var(y); r2d=1-np.var(dr)/np.var(D)
    return {'theta':float(th),'se':float(se),'n':int(len(y)),'r2_g':float(r2y),'r2_m':float(r2d)}
for k,m in groups.items():
    sub=d[m.values]; sub=sub.iloc[rng.choice(len(sub),min(N,len(sub)),replace=False)]
    out[k]=fit(sub); print(k,out[k],round(time.time()-t0),flush=True)
for a,b,name in [('peak_months','off_months','months'),('busy_days','quiet_days','days')]:
    diff=out[a]['theta']-out[b]['theta']; se=np.hypot(out[a]['se'],out[b]['se'])
    out['diff_'+name]={'diff':float(diff),'se':float(se),'z':float(diff/se)}
json.dump(out,open('/home/claude/bts/work/res_dml.json','w'),indent=1)
print(json.dumps({k:v for k,v in out.items() if k.startswith('diff')}),round(time.time()-t0))
