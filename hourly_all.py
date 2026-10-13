import pandas as pd, glob, zipfile
U='/root/.claude/uploads/8ce57bb5-0be2-5dd5-973f-423d4702ddb8'
dep=[];arr=[]
for z in sorted(glob.glob(f'{U}/*_.zip')):
    with zipfile.ZipFile(z) as zf, zf.open('T_ONTIME_REPORTING.csv') as f:
        d=pd.read_csv(f,usecols=['MONTH','DAY_OF_MONTH','ORIGIN','DEST','CRS_DEP_TIME','CRS_ARR_TIME'])
    d['date']=pd.to_datetime(dict(year=2024,month=d.MONTH,day=d.DAY_OF_MONTH))
    d['dh']=(d.CRS_DEP_TIME//100)%24; d['ah']=(d.CRS_ARR_TIME//100)%24
    d['ad']=d.date+pd.to_timedelta((d.CRS_ARR_TIME<d.CRS_DEP_TIME).astype(int),unit='D')
    dep.append(d.groupby(['ORIGIN','date','dh']).size().rename('n').reset_index().rename(columns={'ORIGIN':'ap','dh':'hr'}))
    arr.append(d.groupby(['DEST','ad','ah']).size().rename('n').reset_index().rename(columns={'DEST':'ap','ad':'date','ah':'hr'}))
c=pd.concat(dep+arr).groupby(['ap','date','hr']).n.sum().rename('cong_all').reset_index()
c.to_pickle('/home/claude/bts/work/hourly_all.pkl'); print(len(c))
