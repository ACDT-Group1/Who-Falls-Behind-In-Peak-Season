import pandas as pd, numpy as np, glob, os
B='/home/claude/bts/proj/acdt_data'
MONTHS=list(range(1,13)); LCC={'WN','B6','NK','F9','G4'}
USE=['FlightDate','Reporting_Airline','Tail_Number','Origin','Dest','CRSDepTime','CRSArrTime','ArrDel15','Distance']
fl=pd.concat([pd.read_csv(f'{B}/flights/flights_2024_{m:02d}.csv',usecols=USE,dtype={'Reporting_Airline':'category','Tail_Number':'category','Origin':'category','Dest':'category','Distance':'float32','ArrDel15':'float32'}) for m in MONTHS],ignore_index=True)
for c in ['Reporting_Airline','Tail_Number','Origin','Dest']: fl[c]=fl[c].astype(str).replace('nan',np.nan)
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
# origin weather at the scheduled departure hour
fl=fl.merge(W.add_prefix('ow_').rename(columns={'ow_ap':'Origin','ow_date':'date','ow_hr':'dep_hr'}),on=['Origin','date','dep_hr'],how='left')
fl['severe']=((fl.ow_ts==True)|(fl.ow_sn==True)|(fl.ow_fg==True)|(fl.ow_vsby<3)|(fl.ow_gust>=30)).astype(float)
fl.loc[fl.ow_sknt.isna(),'severe']=np.nan
# hourly crowding at origin, all carriers reporting to BTS, vs that airport's usual for that hour of day
ha=pd.read_pickle('/home/claude/bts/work/hourly_all.pkl')
fl=fl.merge(ha.rename(columns={'ap':'Origin','hr':'dep_hr','cong_all':'org_cong_all'}),on=['Origin','date','dep_hr'],how='left')
fl['HCI']=fl.org_cong_all/fl.groupby(['Origin','dep_hr']).org_cong_all.transform('mean')-1
# aircraft rotation from tail numbers (schedule, same local day)
hm=lambda t:(t//100)*60+t%100
fl=fl.sort_values(['Tail_Number','date','CRSDepTime'])
g=fl.groupby(['Tail_Number','date'],sort=False)
fl['prev_arr']=g.CRSArrTime.shift(); fl['prev_dest']=g.Dest.shift(); fl['prev_dep']=g.CRSDepTime.shift()
fl['legs']=g.CRSDepTime.transform('size')
fl['first']=np.where(fl.Tail_Number.isna(),np.nan,fl.prev_arr.isna().astype(float))
gt=hm(fl.CRSDepTime)-hm(fl.prev_arr)
ok=(fl.prev_dest==fl.Origin)&(fl.prev_arr>=fl.prev_dep)&(gt>=0)
fl['ground']=gt.where(ok)
fl['LCC']=fl.Reporting_Airline.isin(LCC).astype(int); fl['Peak']=fl.month.isin([7,12]).astype(int)
# Peak Index: origin airport's scheduled departures that day vs its 2024 daily mean
od=fl.groupby(['Origin','date']).size().rename('od')
fl=fl.merge(od.reset_index(),on=['Origin','date'],how='left')
fl['PI']=fl.od/fl.groupby('Origin').od.transform(lambda s: s.groupby(fl.loc[s.index,'date']).first().mean())-1
fl['cell']=fl.groupby(['Origin','date','dep_hr'],sort=False).ngroup()
fl=fl[fl.ArrDel15.notna()]
slim=pd.DataFrame({'carrier':fl.Reporting_Airline.astype('category'),'Origin':fl.Origin.astype('category'),'Dest':fl.Dest.astype('category'),
  'cell':fl.cell.astype('int32'),'LCC':fl.LCC.astype('int8'),'PI':fl.PI.astype('float32'),'month':fl.month.astype('int8'),'doy':fl.date.dt.dayofyear.astype('int16'),'dow':fl.date.dt.dayofweek.astype('int8'),
  'y':(fl.ArrDel15*100).astype('float32'),'Distance':fl.Distance.astype('float32'),'dest_cong':fl.dest_cong.astype('float32'),
  'dw_sknt':fl.dw_sknt.astype('float32'),'dw_gust':fl.dw_gust.astype('float32'),'dw_vsby':fl.dw_vsby.astype('float32'),
  'dw_p01i':fl.dw_p01i.fillna(0).astype('float32'),'dw_ts':fl.dw_ts.astype('float32'),'dw_sn':fl.dw_sn.astype('float32'),'dw_fg':fl.dw_fg.astype('float32'),
  'severe':fl.severe.astype('float32'),'HCI':fl.HCI.astype('float32'),'ground':fl.ground.astype('float32'),'first':fl['first'].astype('float32'),'legs':fl.legs.astype('float32'),
  'dep_hr':fl.dep_hr.astype('int8'),'ow_vsby':fl.ow_vsby.astype('float32'),'ow_gust':fl.ow_gust.astype('float32'),'ow_p01i':fl.ow_p01i.fillna(0).astype('float32')})
slim=slim.reset_index(drop=True)
c=pd.read_pickle('/home/claude/bts/work/origin_day_all.pkl'); c['doy']=c.date.dt.dayofyear; c['dow']=c.date.dt.dayofweek
c['PI_all']=c.n/c.groupby('ORIGIN').n.transform('mean')-1; c['PI_dow']=c.n/c.groupby(['ORIGIN','dow']).n.transform('mean')-1
slim=slim.merge(c[['ORIGIN','doy','PI_all','PI_dow']].rename(columns={'ORIGIN':'Origin'}).astype({'Origin':'category'}),on=['Origin','doy'],how='left')
slim['PI_all']=slim.PI_all.astype('float32'); slim['PI_dow']=slim.PI_dow.astype('float32')
slim.to_pickle('/home/claude/bts/work/slim13.pkl')
print('ground',slim.ground.describe().round(1).to_dict(),'HCI na',slim.HCI.isna().mean().round(3),'severe',slim.severe.mean().round(3),'PI_dow na',slim.PI_dow.isna().sum())
print('rows',len(fl),'cancelled/diverted',(fl.ArrDel15.isna()).sum())
print('dest weather matched',fl.dw_sknt.notna().mean().round(3),'stations',W.ap.nunique())
print(fl.groupby('month').size())
