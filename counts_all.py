import pandas as pd, glob, zipfile, io
U='/root/.claude/uploads/8ce57bb5-0be2-5dd5-973f-423d4702ddb8'
parts=[]
for z in sorted(glob.glob(f'{U}/*_.zip')):
    with zipfile.ZipFile(z) as zf:
        with zf.open('T_ONTIME_REPORTING.csv') as f:
            d=pd.read_csv(f,usecols=['YEAR','MONTH','DAY_OF_MONTH','ORIGIN','OP_UNIQUE_CARRIER'])
    parts.append(d.groupby(['MONTH','DAY_OF_MONTH','ORIGIN']).size().rename('n').reset_index())
c=pd.concat(parts); c['date']=pd.to_datetime(dict(year=2024,month=c.MONTH,day=c.DAY_OF_MONTH))
c.to_pickle('/home/claude/bts/work/origin_day_all.pkl')
m=c.groupby('MONTH').agg(total=('n','sum'),days=('date','nunique')); m['per_day']=m.total/m.days
m['rank_total']=m.total.rank(ascending=False).astype(int); m['rank_perday']=m.per_day.rank(ascending=False).astype(int)
print(m.round(0)); print('months in raw:',sorted(c.MONTH.unique()))
