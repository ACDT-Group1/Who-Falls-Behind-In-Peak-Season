import pandas as pd, numpy as np, glob, os
B='/home/claude/bts/proj/acdt_data'
MONTHS=[1,4,7,12]; LCC={'WN','B6','NK','F9','G4'}
fl=pd.concat([pd.read_csv(f'{B}/flights/flights_2024_{m:02d}.csv') for m in MONTHS],ignore_index=True)
fl['date']=pd.to_datetime(fl.FlightDate); fl['month']=fl.date.dt.month
fl['dep_hr']=(fl.CRSDepTime//100)%24; fl['arr_hr']=(fl.CRSArrTime//100)%24
# arrival date: next day if scheduled arrival clock time earlier than departure (overnight); approximation, cross-timezone noted
fl['arr_date']=fl.date+pd.to_timedelta((fl.CRSArrTime<fl.CRSDepTime).astype(int),unit='D')
# congestion from schedule (all 9 carriers, incl. later-cancelled flights): departures+arrivals at the airport in that hour
dep=fl.groupby(['Origin','date','dep_hr']).size().rename('n').reset_index().rename(columns={'Origin':'ap','dep_hr':'hr'})
arr=fl.groupby(['Dest','arr_date','arr_hr']).size().rename('n').reset_index().rename(columns={'Dest':'ap','arr_date':'date','arr_hr':'hr'})
cong=pd.concat([dep,arr]).groupby(['ap','date','hr']).n.sum().rename('cong').reset_index()
# weather: hourly aggregates per station (valid times are local)
ws=[]
for f in glob.glob(f'{B}/asos/*.csv'):
    w=pd.read_csv(f,na_values=['M'],usecols=['station','valid','sknt','gust','vsby','p01i','wxcodes'])
    w['valid']=pd.to_datetime(w.valid); w=w[w.valid.dt.month.isin(MONTHS)]
    w['date']=w.valid.dt.normalize(); w['hr']=w.valid.dt.hour
    wx=w.wxcodes.fillna('')
    w['ts']=wx.str.contains('TS'); w['sn']=wx.str.contains('SN'); w['fg']=wx.str.contains('FG')
    g=w.groupby(['station','date','hr']).agg(sknt=('sknt','max'),gust=('gust','max'),vsby=('vsby','min'),p01i=('p01i','max'),ts=('ts','max'),sn=('sn','max'),fg=('fg','max')).reset_index()
    ws.append(g)
W=pd.concat(ws).rename(columns={'station':'ap'})
W['gust']=W.gust.fillna(W.sknt)
fl=fl.merge(cong.rename(columns={'ap':'Dest','date':'arr_date','hr':'arr_hr','cong':'dest_cong'}),on=['Dest','arr_date','arr_hr'],how='left')
fl=fl.merge(W.add_prefix('dw_').rename(columns={'dw_ap':'Dest','dw_date':'arr_date','dw_hr':'arr_hr'}),on=['Dest','arr_date','arr_hr'],how='left')
fl['LCC']=fl.Reporting_Airline.isin(LCC).astype(int); fl['Peak']=fl.month.isin([7,12]).astype(int)
fl['cell']=fl.Origin+'_'+fl.FlightDate+'_'+fl.dep_hr.astype(str)
fl.to_pickle('/home/claude/bts/work/flights4.pkl')
print('rows',len(fl),'cancelled/diverted',(fl.ArrDel15.isna()).sum())
print('dest weather matched',fl.dw_sknt.notna().mean().round(3),'stations',W.ap.nunique())
print(fl.groupby('month').size())
