##############################################################################
# Q-Factor Hou, Xue, Zhang (2015)
#
#
# September 2020
#
# Adapted from Qingyi Song code
# https://www.fredasongdrechsler.com/full-python-code/fama-french
##############################################################################
##
## Contents
##
## Load data from WRDS ##
# Set working directory
# Save pandas locally 
# Load pandas locally
## Annual COMPUSTAT Block
## CRSP Block
## CCM Block
# select NYSE stocks for bucket breakdown
## Form Factors from 2x3 sort on ME and I/A
## Monthly portfolio on ROE 
## assign ROE portfolio (monthly)
## Form Factors from 2x3x3 sort on ME, I/A and ROE
##
##############################################################################


import pandas as pd
import numpy as np
import datetime as dt
import wrds
import psycopg2 
import matplotlib.pyplot as plt
import os
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
from scipy import stats

## Load data from WRDS #######################################################
# conn=wrds.Connection()

# # set sample date range
# begdate = '01/01/1966'
# enddate = '12/31/2019'

# comp = conn.raw_sql(f"""
#                     select gvkey, datadate, at, pstkl, txditc,
#                     pstkrv, seq, pstk, ib, sich
#                     from comp.funda
#                     where indfmt='INDL' 
#                     and datafmt='STD'
#                     and popsrc='D'
#                     and consol='C'
#                     and datadate between '{begdate}' and '{enddate}'
#                     """)
# # and datadate between '{begdate}' and '{enddate}'
# # and datadate >= '01/01/1966'
             
# compq = conn.raw_sql(f"""
#                       select gvkey, datadate, ibq, txditcq, seqq,
#                           ceqq, pstkq, pstkrq, atq, ltq, rdq, fqtr, fyearq
#                       from comp.fundq
#                       where indfmt='INDL'
#                           and datafmt='STD'
#                           and popsrc='D'
#                           and consol='C'
#                           and datadate between '{begdate}' and '{enddate}'
#                       """)
# # and datadate between '{begdate}' and '{enddate}'

# crsp_m = conn.raw_sql(f"""
#                       select a.permno, a.permco, a.date,
#                           b.shrcd, b.exchcd, b.siccd,
#                           a.ret, a.retx, a.shrout, a.prc 
#                       from crsp.msf as a
#                       left join crsp.msenames as b
#                           on a.permno=b.permno
#                           and b.namedt<=a.date
#                           and a.date<=b.nameendt
#                       where a.date between '{begdate}' and '{enddate}' 
#                           and b.exchcd between 1 and 3
#                       """) 
# # where a.date between '01/01/1966' and '12/31/2019' 

# dlret = conn.raw_sql("""
#                      select permno, dlret, dlstdt 
#                      from crsp.msedelist
#                      """)

# ccm=conn.raw_sql("""
#                  select gvkey, lpermno as permno, linktype, linkprim,
#                      linkdt, linkenddt
#                  from crsp.ccmxpf_linktable
#                  where substr(linktype,1,1)='L'
#                      and (linkprim ='C' or linkprim='P')
#                  """)

# del conn

## Set working directory #####################################################

os.chdir('C:/Data/2020_HMXZ') # My data path in my windows PC
# os.chdir('../../../Data/2020_HMXZ') # My data path in my windows PC
# os.chdir('Dropbox/Code/Python/hmxz') # My data path in my linux PC
# os.chdir('C:/Dropbox/Code/Python/hmxz') # My dropbox path in my windows PC

## Save pandas locally #######################################################

# print('Saving data to: ' + os.getcwd())

# ## Save as Native Pickle
# comp.to_pickle('comp.pkl')
# compq.to_pickle('compq.pkl')
# crsp_m.to_pickle('crsp_m.pkl')
# dlret.to_pickle('dlret.pkl')
# ccm.to_pickle('ccm.pkl')

# ## Save as excel spreadsheet
# comp.to_excel('comp.xlsx')
# compq.to_excel('compq.xlsx')
# crsp_m.to_excel('crsp_m.xlsx')
# dlret.to_excel('dlret.xlsx')
# ccm.to_excel('ccm.xlsx')

# ## Save as csv files
# comp.to_csv('comp.csv')
# compq.to_csv('compq.csv')
# crsp_m.to_csv('crsp_m.csv')
# dlret.to_csv('dlret.csv')
# ccm.to_csv('ccm.csv')

## Load pandas locally #######################################################

comp = pd.read_pickle('comp.pkl')
compq = pd.read_pickle('compq.pkl')
crsp_m = pd.read_pickle('crsp_m.pkl')
dlret = pd.read_pickle('dlret.pkl')
ccm = pd.read_pickle('ccm.pkl')
print('Data loaded from: ' + os.getcwd())

# comp = pd.read_csv('comp.csv')
# compq = pd.read_csv('compq.csv')
# crsp_m = pd.read_csv('crsp_m.csv')
# dlret = pd.read_csv('dlret.csv')
# ccm = pd.read_csv('ccm.csv')

# comp = pd.read_sas('comp.sas7bdat', format = 'sas7bdat', encoding='latin-1')
# crsp_m = pd.read_sas('crsp_m.sas7bdat', format = 'sas7bdat', encoding='latin-1')
# dlret = pd.read_sas('dlret.sas7bdat', format = 'sas7bdat', encoding='latin-1')
# ccm = pd.read_sas('ccm.sas7bdat', format = 'sas7bdat', encoding='latin-1')
# crsp_m.columns = map(str.lower, crsp_m.columns)
# dlret.columns = map(str.lower, dlret.columns)

# ## Load pandas locally (my Stata files)
# os.chdir('../ccmData')
# comp = pd.read_stata('COMP.dta')
# compq = pd.read_stata('COMPQ.dta')
# crsp_m = pd.read_stata('CRSP_M.dta')
# dlret = pd.read_stata('DLRET.dta')
# ccm = pd.read_stata('CCM.dta')

## Retirar observacoes duplicadas ------------------------------------------
comp = comp.drop_duplicates(['gvkey', 'datadate'],keep= 'last')
compq = compq.drop_duplicates(['gvkey', 'datadate'],keep= 'last')

## exclude financial firms (SIC between 6000 and 6999) ----------------------
comp = comp[~comp["sich"].between(6000, 6999, inclusive = True)]
crsp_m = crsp_m[~crsp_m["siccd"].between(6000, 6999, inclusive = True)]


## Annual COMPUSTAT Block ###################################################

comp['datadate']=pd.to_datetime(comp['datadate']) #convert datadate to date fmt
comp['year']=comp['datadate'].dt.year

# create preferrerd stock
comp['ps']=np.where(comp['pstkrv'].isnull(), comp['pstkl'], comp['pstkrv'])
comp['ps']=np.where(comp['ps'].isnull(),comp['pstk'], comp['ps'])
comp['ps']=np.where(comp['ps'].isnull(),0,comp['ps'])

comp['txditc']=comp['txditc'].fillna(0)

# create book equity
comp['be']=comp['seq']+comp['txditc']-comp['ps']
comp['be']=np.where(comp['be']>0, comp['be'], np.nan)

# number of years in Compustat
comp=comp.sort_values(by=['gvkey','datadate'])
comp['count']=comp.groupby(['gvkey']).cumcount()

# Sort by gvkey and year and create lag_at to calculate I/A
comp = comp.sort_values(['gvkey', 'year'])
comp['lag_at']=comp.groupby(['gvkey'])['at'].shift(1)

# Calculate change in AT
comp['atchange']= comp['at'] - comp['lag_at']

# Calculate AT
comp['ia']= comp['atchange'] / comp['lag_at']
comp[['gvkey', 'datadate', 'at', 'lag_at', 'atchange', 'ia']]

## Create Lagged Book Equity
comp = comp.sort_values(['gvkey', 'datadate'])
comp['lag_be']=comp.groupby(['gvkey'])['be'].shift(1)

# Calculate Annual ROE
comp['roe'] = comp['ib'] / comp['lag_be']

comp=comp[['gvkey','datadate','year','be', 'ia','count']][comp['ia'].notnull()]

## CRSP Block ################################################################

# change variable format to int
crsp_m[['permco','permno',
        'shrcd','exchcd', 'siccd']] = crsp_m[['permco','permno',
                                              'shrcd','exchcd',
                                              'siccd']].astype(int)

# Line up date to be end of month
crsp_m['date']=pd.to_datetime(crsp_m['date'])
crsp_m['jdate']=crsp_m['date']+MonthEnd(0)

# add delisting return
dlret.permno=dlret.permno.astype(int)
dlret['dlstdt']=pd.to_datetime(dlret['dlstdt'])
dlret['jdate']=dlret['dlstdt']+MonthEnd(0)

crsp = pd.merge(crsp_m, dlret, how='left',on=['permno','jdate'])
crsp['dlret']=crsp['dlret'].fillna(0)
crsp['ret']=crsp['ret'].fillna(0)
crsp['retadj']=(1+crsp['ret'])*(1+crsp['dlret'])-1
crsp['me']=crsp['prc'].abs()*crsp['shrout'] # calculate market equity
crsp=crsp.drop(['dlret','dlstdt','prc','shrout'], axis=1)
crsp=crsp.sort_values(by=['jdate','permco','me'])

### Aggregate Market Cap ###
# sum of me across different permno belonging to same permco a given date
crsp_summe = crsp.groupby(['jdate','permco'])['me'].sum().reset_index()
# largest mktcap within a permco/date
crsp_maxme = crsp.groupby(['jdate','permco'])['me'].max().reset_index()
# join by jdate/maxme to find the permno
crsp1=pd.merge(crsp, crsp_maxme, how='inner', on=['jdate','permco','me'])
# drop me column and replace with the sum me
crsp1=crsp1.drop(['me'], axis=1)
# join with sum of me to get the correct market cap info
crsp2=pd.merge(crsp1, crsp_summe, how='inner', on=['jdate','permco'])
# sort by permno and date and also drop duplicates
crsp2=crsp2.sort_values(by=['permno','jdate']).drop_duplicates()

# keep December market cap
crsp2['year']=crsp2['jdate'].dt.year
crsp2['month']=crsp2['jdate'].dt.month
decme=crsp2[crsp2['month']==12]
decme=decme[['permno','date','jdate','me','year']].rename(columns={'me':'dec_me'})

### July to June dates
crsp2['ffdate']=crsp2['jdate']+MonthEnd(-6)
crsp2['ffyear']=crsp2['ffdate'].dt.year
crsp2['ffmonth']=crsp2['ffdate'].dt.month
crsp2['1+retx']=1+crsp2['retx']
crsp2=crsp2.sort_values(by=['permno','date'])

# cumret by stock
crsp2['cumretx']=crsp2.groupby(['permno','ffyear'])['1+retx'].cumprod()
# lag cumret
crsp2['lcumretx']=crsp2.groupby(['permno'])['cumretx'].shift(1)

# lag market cap
crsp2['lme']=crsp2.groupby(['permno'])['me'].shift(1)

# if first permno then use me/(1+retx) to replace the missing value
crsp2['count']=crsp2.groupby(['permno']).cumcount()
crsp2['lme']=np.where(crsp2['count']==0, crsp2['me']/crsp2['1+retx'], crsp2['lme'])

# baseline me
mebase=crsp2[crsp2['ffmonth']==1][['permno','ffyear', 'lme']].rename(columns={'lme':'mebase'})

# merge result back together
crsp3=pd.merge(crsp2, mebase, how='left', on=['permno','ffyear'])
crsp3['wt']=np.where(crsp3['ffmonth']==1, crsp3['lme'], crsp3['mebase']*crsp3['lcumretx'])

decme['year']=decme['year']+1
decme=decme[['permno','year','dec_me']]

# Info as of June
crsp3_jun = crsp3[crsp3['month']==6]

crsp_jun = pd.merge(crsp3_jun, decme, how='inner', on=['permno','year'])
crsp_jun=crsp_jun[['permno','date', 'jdate', 'shrcd','exchcd','retadj','me','wt','cumretx','mebase','lme','dec_me', 'siccd']]
crsp_jun=crsp_jun.sort_values(by=['permno','jdate']).drop_duplicates()


## CCM Block #################################################################

ccm['linkdt']=pd.to_datetime(ccm['linkdt'])
ccm['linkenddt']=pd.to_datetime(ccm['linkenddt'])
# if linkenddt is missing then set to today date
ccm['linkenddt']=ccm['linkenddt'].fillna(pd.to_datetime('today'))

ccm1=pd.merge(comp[['gvkey','datadate','be', 'ia','count']],ccm,how='left',on=['gvkey'])
ccm1['yearend']=ccm1['datadate']+YearEnd(0)
ccm1['jdate']=ccm1['yearend']+MonthEnd(6)

# set link date bounds
ccm2=ccm1[(ccm1['jdate']>=ccm1['linkdt'])&(ccm1['jdate']<=ccm1['linkenddt'])]
ccm2=ccm2[['gvkey','permno','datadate','yearend', 'jdate','be', 'ia','count']]

# link comp and crsp
ccm_jun=pd.merge(crsp_jun, ccm2, how='inner', on=['permno', 'jdate'])
ccm_jun['beme']=ccm_jun['be']*1000/ccm_jun['dec_me']


## POR DESENVOLVER: gerar empresas com ROE disponivel
# mesclar ROE valido com empresas que tem ia valido e me>0

# select NYSE stocks for bucket breakdown
# exchcd = 1 (Nyse exchange code)
# and positive beme and positive me
# and shrcd in (10,11) / share code of 10 or 11 (CRSP share code of 10 or 11)
# and at least 2 years in comp
#
# Falta fazer: exclude financial firms (SIC between 6000 and 6999) 
nyse=ccm_jun[(ccm_jun['exchcd']==1) & 
             (ccm_jun['me']>0) & (ccm_jun['ia'].notnull()) & (ccm_jun['count']>1) &
             ((ccm_jun['shrcd']==10) | (ccm_jun['shrcd']==11))]

# nyse=ccm_jun[(ccm_jun['exchcd']==1) & (ccm_jun['beme']>0) &
#              (ccm_jun['me']>0) & (ccm_jun['ia'].notnull()) & (ccm_jun['count']>1) &
#              ((ccm_jun['shrcd']==10) | (ccm_jun['shrcd']==11))]

# size breakdown
nyse_sz=nyse.groupby(['jdate'])['me'].median().to_frame().reset_index().rename(columns={'me':'sizemedn'})

# # beme breakdown
# nyse_bm=nyse.groupby(['jdate'])['beme'].describe(percentiles=[0.3, 0.7]).reset_index()
# nyse_bm=nyse_bm[['jdate','30%','70%']].rename(columns={'30%':'bm30', '70%':'bm70'})

# investment-to-asset (ia) breakdown
nyse_ia=nyse.groupby(['jdate'])['ia'].describe(percentiles=[0.3, 0.7]).reset_index()
nyse_ia=nyse_ia[['jdate','30%','70%']].rename(columns={'30%':'ia30', '70%':'ia70'})

# nyse_breaks = pd.merge(nyse_sz, nyse_bm, how='inner', on=['jdate'])
nyse_breaks = pd.merge(nyse_sz, nyse_ia, how='inner', on=['jdate'])
# join back size and beme breakdown
ccm1_jun = pd.merge(ccm_jun, nyse_breaks, how='left', on=['jdate'])


# function to assign sz and bm bucket
def sz_bucket(row):
    if row['me']==np.nan:
        value=''
    elif row['me']<=row['sizemedn']:
        value='S'
    else:
        value='B'
    return value

def ia_bucket(row):
    if row['ia']<=row['ia30']:
        value = 'L'
    elif row['ia']<=row['ia70']:
        value='M'
    elif row['ia']>row['ia70']:
        value='H'
    else:
        value=''
    return value


## assign size portfolio
ccm1_jun['szport']=np.where((ccm1_jun['ia'].notnull())&
                            (ccm1_jun['beme']>0)&(ccm1_jun['me']>0)&
                            (ccm1_jun['count']>=1),
                            ccm1_jun.apply(sz_bucket, axis=1), '')
## assign I/A portfolio
ccm1_jun['iaport']=np.where((ccm1_jun['ia'].notnull())&
                            (ccm1_jun['beme']>0)&(ccm1_jun['me']>0)&
                            (ccm1_jun['count']>=1),
                            ccm1_jun.apply(ia_bucket, axis=1), '')

## create positivebmeme and nonmissport variable
ccm1_jun['posbm']=np.where((ccm1_jun['beme']>0)&(ccm1_jun['me']>0)&(ccm1_jun['count']>=1), 1, 0)
ccm1_jun['nonmissport']=np.where((ccm1_jun['iaport']!=''), 1, 0)
ccm1_jun['ianotnull']=np.where((ccm1_jun['ia'].notnull())&(ccm1_jun['count']>=1), 1, 0)

# store portfolio assignment as of June
june=ccm1_jun[['permno','date', 'jdate', 'iaport', 'szport','posbm','nonmissport', 'ianotnull']]
june['ffyear']=june['jdate'].dt.year

# merge back with monthly records
crsp3 = crsp3[['date','permno','shrcd','exchcd','retadj','me','wt','cumretx','ffyear','jdate']]

ccm3=pd.merge(crsp3, 
        june[['permno','ffyear','szport','posbm','iaport','ianotnull', 'nonmissport']],
        how='left', on=['permno','ffyear'])

# keeping only records that meet the criteria
ccm4=ccm3[(ccm3['wt']>0)& (ccm3['ianotnull']==1)& (ccm3['posbm']==1) & (ccm3['nonmissport']==1) & 
          ((ccm3['shrcd']==10) | (ccm3['shrcd']==11))]
ccm4.groupby(by='szport').count()

## Form Factors from 2x3 sort on ME and I/A ##################################

# function to calculate value weighted return
def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan

# value-weigthed return
vwret=ccm4.groupby(['jdate','szport','iaport']).apply(wavg, 'retadj','wt').to_frame().reset_index().rename(columns={0: 'vwret'})
vwret['sbport']=vwret['szport']+vwret['iaport']

# firm count
vwret_n=ccm4.groupby(['jdate','szport','iaport'])['retadj'].count().reset_index().rename(columns={'retadj':'n_firms'})
vwret_n['sbport']=vwret_n['szport']+vwret_n['iaport']

# tranpose
ff_factors=vwret.pivot(index='jdate', columns='sbport', values='vwret').reset_index()
ff_nfirms=vwret_n.pivot(index='jdate', columns='sbport', values='n_firms').reset_index()

# create SMB and HML factors
ff_factors['ia_3']=(ff_factors['BH']+ff_factors['SH'])/2
ff_factors['ia_1']=(ff_factors['BL']+ff_factors['SL'])/2
# ff_factors['WI_A'] = ff_factors['WIAH']-ff_factors['WIAL']
ff_factors['r_ia'] = ff_factors['ia_1']-ff_factors['ia_3']

ff_factors['me_2']=(ff_factors['BL']+ff_factors['BM']+ff_factors['BH'])/3
ff_factors['me_1']=(ff_factors['SL']+ff_factors['SM']+ff_factors['SH'])/3
ff_factors['r_me'] = ff_factors['me_1']-ff_factors['me_2']
ff_factors=ff_factors.rename(columns={'jdate':'date'})

# n firm count
ff_nfirms['H']=ff_nfirms['SH']+ff_nfirms['BH']
ff_nfirms['L']=ff_nfirms['SL']+ff_nfirms['BL']
ff_nfirms['HML']=ff_nfirms['H']+ff_nfirms['L']

ff_nfirms['B']=ff_nfirms['BL']+ff_nfirms['BM']+ff_nfirms['BH']
ff_nfirms['S']=ff_nfirms['SL']+ff_nfirms['SM']+ff_nfirms['SH']
ff_nfirms['SMB']=ff_nfirms['B']+ff_nfirms['S']
ff_nfirms['TOTAL']=ff_nfirms['SMB']
ff_nfirms=ff_nfirms.rename(columns={'jdate':'date'})

ff_nfirms[(ff_nfirms['date']=='2018-12-31')]

os.chdir('C:\Dropbox\Code\Python')
q5_factors = pd.read_csv('hmxz/q5_factors_monthly_2019a.csv')
# print(q5_factors)
# print(ff_factors['r_ia'])


ff_factors['year'] = ff_factors['date'].dt.year
ff_factors['month'] = ff_factors['date'].dt.month
ff_factors[['r_me','r_ia']] = ff_factors[['r_me','r_ia']] * 100
qfcomp = pd.merge(q5_factors,
                   ff_factors[['year','month','r_me','r_ia']][ff_factors['year']>=1972],
                   how='inner', on=['year', 'month'])

print(stats.pearsonr(qfcomp['R_ME'], qfcomp['r_me']))
# (0.97723366230532, 0.0)

print(stats.pearsonr(qfcomp['R_IA'], qfcomp['r_ia']))
# (0.9438473340470327, 2.90160265201413e-278)

## Monthly ROE ###############################################################
# Calculate ROE
# ROE = income_before_extraordinary_items / lag_be
# be = equity + txditcq - book_value_of_preferred_stock
#       where
#            equity = (seqq ou ceqq) + pstkq
#               or
#            equity = (atq - ltq)
#
#            book_value_of_preferred_stock = seq
#            or = CEQ + PSTKRQ (PSTKQ if not availeble)
#            or = ATQ - LTQ

compq['datadate']=pd.to_datetime(compq['datadate']) #convert datadate to date fmt
compq['year']=compq['datadate'].dt.year
compq = compq.sort_values(['gvkey', 'datadate'])

## Create book value of preferred stock
#
# We use redemption value (item PSTKRQ) if available, or carrying value for
# the book value of preferred stock (item PSTKQ).
# http://global-q.org/uploads/1/2/2/6/122679606/factorstd_2020july.pdf
#
# ps = (redemption value (item PSTKRQ))
#    = or carrying value for the book value of preferred stock (item PSTKQ)
#

compq['ps'] = np.where(compq['pstkq'].isnull(), compq['pstkrq'], compq['pstkq'])

## Create stockholdersâ equity
#
# Depending on availability, we use stockholdersâ equity (item SEQQ), or
# common equity (item CEQQ) plus the carrying value of preferred stock (item
# PSTKQ), or total assets (item ATQ) minus total liabilities (item LTQ) in
# that order as shareholdersâ equity.
#
compq['pstkq']=compq['pstkq'].fillna(0)
#    shareholdersâ equity = stockholdersâ equity (item SEQQ)
# or shareholdersâ equity = or common equity (item CEQQ) + the carrying value of preferred stock (item PSTKQ)
compq['se']=np.where(compq['seqq'].isnull(), compq['ceqq']+compq['pstkq'], compq['seqq'])
compq.loc[compq['se'].isnull() & compq['atq'].notnull(),]
# or shareholdersâ equity = or total assets (item ATQ) - total liabilities (item LTQ) in that order as 
compq['se']=np.where(compq['se'].isnull(), compq['atq']-compq['ltq'], compq['se'])


## Create Quartely book equity
#
# In particular, book equity is shareholdersâ equity, plus balance sheet
# deferred taxes and investment tax credit (item TXDITCQ) if available, minus
# the book value of preferred stock.
#
# be (book equity) = se (shareholdersâ equity) 
#                       + tax (balance sheet deferred taxes and investment tax credit (item TXDITCQ))
#                       - ps (book value of preferred stock)
compq['txditcq']=compq['txditcq'].fillna(0)
compq['be']=compq['se']+compq['txditcq']-compq['ps']
compq['be']=np.where(compq['be']>0, compq['be'], np.nan)

## Create Lagged Book Equity
compq = compq.sort_values(['gvkey', 'datadate'])
compq['lag_be']=compq.groupby(['gvkey'])['be'].shift(1)

## Create ROE
# Roe is income before extraordinary items (Compustat quarterly item IBQ)
# scaled by the 1-quarter-lagged book equity
compq['roe'] = compq['ibq'] / compq['lag_be']
compq

### -------------------------------------------------------------------------
### Create a monthly ROE
# Earnings data in Compustat quarterly files are used in the months
# immediately after the most recent public quarterly earnings announcement
# dates (Compustat quarterly item RDQ). For example, if the earnings for the
# fourth fiscal quarter of year t â 1 are publicly announced on March 5
# (or March 25) of year t, we use the announced earnings (divided by the
# book equity from the third quarter of year t â 1) to form portfolios at
# the beginning of April of year t.
# http://global-q.org/uploads/1/2/2/6/122679606/factorstd_2020july.pdf


# roe_q = compq[['gvkey', 'datadate', 'ibq', 'lag_be', 'roe', 'rdq']].dropna()
roe_q = compq[['gvkey', 'rdq', 'datadate', 'roe']].dropna()

roe_q['rdq'] = pd.to_datetime(roe_q['rdq']) # Transformar rdq em data
roe_q['datadate'] = pd.to_datetime(roe_q['datadate']) # Transformar rdq em data

# Caso uma empresa tenha duas divulgacao no mesmo dia, ficar apenas com o
# ROE do trimestre mais recente.
roe_q = roe_q.sort_values(['gvkey', 'rdq', 'datadate'])
roe_q = roe_q.drop_duplicates(['gvkey', 'rdq'], keep='last')

del roe_q['datadate']

# # Retirar empresas que so tem um balanco na amostra (nao Ã© mais necessario)
# retirar = x[['gvkey','rdq']].groupby('gvkey').count()
# retirar = retirar[retirar['rdq'] == 1].index
# x = x[~x['gvkey'].isin(retirar)]

# Repetir ultimo ROE disponivel nos meses que nao ha divulgacao.
x = roe_q
roe_m = (x.set_index('rdq')
      .groupby('gvkey')['roe']
      .apply(lambda x: x.reindex(pd.date_range(x.index.min(), 
                                                x.index.max(), freq='MS'), method='ffill'))
      .rename_axis(('gvkey','rdq'))
      .reset_index())
del x
# # Conferindo o resultado com a empresa 1001
roe_q[roe_q['gvkey'] == '001001'] # Base Trimestral dos ROEs com data de divulgacao
roe_m[roe_m['gvkey'] == '001001'] # Base Mensal considerando ultimo ROE
roe_m[['gvkey', 'rdq', 'roe']][roe_m['rdq']<='2019-12-01'].dropna().groupby(['rdq'])['gvkey'].count()


## Join most recent announcement dates (item RDQ) from compq
# Hou, Xue, and Zhang (2015) start their sample in January 1972, limited by
# earnings announcement dates and book equity in Compustat quarterly files.
# We follow their procedure from January 1972 onward but extend the sample
# backward to January 1967 following Hou et al. (2019). To overcome the lack
# of quarterly earnings announcement dates, we use the most recent quarterly
# earnings from the fiscal quarter ending at least four months prior to the
# portfolio formation month. To maximize the coverage for quarterly book
# equity, whenever available we first use quarterly book equity from Compustat
# quarterly files. We then supplement the coverage for fiscal quarter four with
# book equity from Compustat annual files.

### -------------------------------------------------------------------------

## Monthly portfolio on ROE ##################################################

# ROE nyse breakdown
ccm_roe1=pd.merge(roe_m, ccm ,how='left',on=['gvkey'])
ccm_roe1['jdate']=ccm_roe1['rdq']+MonthEnd(-1)
# # Conferindo como ficou ccm_roe1
# pd.set_option('max_columns', None)
# ccm_roe1
# pd.reset_option('max_columns')

# set link date bounds
ccm_roe2=ccm_roe1[(ccm_roe1['jdate']>=ccm_roe1['linkdt'])&(ccm_roe1['jdate']<=ccm_roe1['linkenddt'])]
ccm_roe2=ccm_roe2.rename(columns={'rdq':'datadate'})
ccm_roe2=ccm_roe2[['gvkey','permno','datadate', 'jdate', 'roe']]
ccm_roe2.permno=ccm_roe2.permno.astype(int)

# Juntar informacao do exchange code
ccm_roe3 = pd.merge(crsp3[['permno','date', 'jdate', 'shrcd','exchcd','retadj','me','wt','cumretx']].sort_values(by=['permno','jdate']).drop_duplicates(),
                    ccm_roe2, how='inner', on=['permno', 'jdate'])

# select NYSE stocks for bucket breakdown
# exchcd = 1 (Nyse exchange code)
# and positive beme and positive me
# and shrcd in (10,11) / share code of 10 or 11 (CRSP share code of 10 or 11)
nyse_roe=ccm_roe3[(ccm_roe3['exchcd']==1) & 
              (ccm_roe3['roe'].notnull()) &
              ((ccm_roe3['shrcd']==10) | (ccm_roe3['shrcd']==11))]
nyse_roe=nyse_roe.groupby(['jdate'])['roe'].describe(percentiles=[0.3, 0.7]).reset_index()
nyse_roe=nyse_roe[['jdate','30%','70%']].rename(columns={'30%':'roe30', '70%':'roe70'})
# nyse_roe.sort_values('jdate')

# join back size and beme breakdown
ccm_roe3 = pd.merge(ccm_roe3, nyse_roe, how='left', on=['jdate'])



## assign ROE portfolio (monthly) ############################################
# dessa vez usando np.where (bem mais rápido!!!)
ccm_roe3['roeport']=np.where((ccm_roe3['roe'].notnull())&(ccm_roe3['me']>0)&
                              (ccm_roe3['roe']<=ccm_roe3['roe30']), 'r1', '')
ccm_roe3['roeport']=np.where((ccm_roe3['roe'].notnull())&(ccm_roe3['me']>0)&
                              (ccm_roe3['roe']>ccm_roe3['roe70']), 'r3', ccm_roe3['roeport'])
ccm_roe3['roeport']=np.where((ccm_roe3['roe'].notnull())&(ccm_roe3['me']>0)&
                              (ccm_roe3['roe']>ccm_roe3['roe30'])&
                              (ccm_roe3['roe']<=ccm_roe3['roe70']), 'r2', ccm_roe3['roeport'])

# # Formato original
# def roe_bucket(row):
#     if row['roe']<=row['roe30']:
#         value = 'r1'
#     elif row['roe']<=row['roe70']:
#         value='r2'
#     elif row['roe']>row['roe70']:
#         value='r3'
#     else:
#         value=''
#     return value
# ccm_roe3['roeport2']=np.where((ccm_roe3['roe'].notnull())&
#                             (ccm_roe3['me']>0),
#                             ccm_roe3.apply(roe_bucket, axis=1), '')
# # Testando se da o mesmo resultado
# ccm_roe3[ccm_roe3['roeport'] == ccm_roe3['roeport2']]
# del ccm_roe3['roeport2']


# non miss roe port filter
ccm_roe3 = ccm_roe3[((ccm_roe3['roeport']!='')&(ccm_roe3['me']>0))]

# roe not null filter
ccm_roe3 = ccm_roe3[(ccm_roe3['roe'].notnull())]


print(ccm_roe3[(ccm_roe3['permno'] == 93436) & (ccm_roe3['date'].dt.year == 2018)])
print(ccm4[(ccm4['permno'] == 93436) & (ccm4['date'].dt.year == 2018)])


# Passar para nomeclatura de HMXZ
ccm5=ccm3[(ccm3['wt']>0)& (ccm3['ianotnull']==1)& (ccm3['posbm']==1) & (ccm3['nonmissport']==1) & 
          ((ccm3['shrcd']==10) | (ccm3['shrcd']==11))]
ccm5.groupby(by='szport').count()
# DataFrame.copy()
ccm5['szport']=np.where((ccm5['szport']=='S'), 'm1', 'm2')
ccm5['iaport']=np.where((ccm5['iaport']=='L'), 'i1', ccm5['iaport'])
ccm5['iaport']=np.where((ccm5['iaport']=='M'), 'i2', ccm5['iaport'])
ccm5['iaport']=np.where((ccm5['iaport']=='H'), 'i3', ccm5['iaport'])

# Juntar com ccm_roe3
ccm5['jdate']=pd.to_datetime(ccm5['jdate']) #convert datadate to date fmt
ccm5 = pd.merge(ccm5[['permno','jdate','szport', 'iaport', 'retadj', 'wt']], 
        ccm_roe3[['permno','jdate','roeport']],
        how='left', on=['permno','jdate'])


ccm5.groupby(by='szport').count()
ccm5.groupby(by='iaport').count()
ccm5.groupby(by='roeport').count()

## Form Factors from 2x3x3 sort on ME and I/A ##################################

# function to calculate value weighted return
def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan

# value-weigthed return
vwret=ccm5.groupby(['jdate','szport','iaport', 'roeport']).apply(wavg, 'retadj','wt').to_frame().reset_index().rename(columns={0: 'vwret'})
vwret['portsort']=vwret['szport']+vwret['iaport']+vwret['roeport']

# firm count
vwret_n=ccm5.groupby(['jdate','szport','iaport', 'roeport'])['retadj'].count().reset_index().rename(columns={'retadj':'n_firms'})
vwret_n['portsort']=vwret_n['szport']+vwret_n['iaport']+vwret['roeport']

# tranpose
q4_factors=vwret.pivot(index='jdate', columns='portsort', values='vwret').dropna().reset_index()
q4_nfirms=vwret_n.pivot(index='jdate', columns='portsort', values='n_firms').dropna().reset_index()

# create ME, I/A and ROE factors
q4_factors['me_2']=(q4_factors['m2i1r1']+q4_factors['m2i2r1']+q4_factors['m2i3r1']+
                    q4_factors['m2i1r2']+q4_factors['m2i2r2']+q4_factors['m2i3r2']+
                    q4_factors['m2i1r3']+q4_factors['m2i2r3']+q4_factors['m2i3r3'])/9
q4_factors['me_1']=(q4_factors['m1i1r1']+q4_factors['m1i2r1']+q4_factors['m1i3r1']+
                    q4_factors['m1i1r2']+q4_factors['m1i2r2']+q4_factors['m1i3r2']+
                    q4_factors['m1i1r3']+q4_factors['m1i2r3']+q4_factors['m1i3r3'])/9
q4_factors['r_me'] = q4_factors['me_1']-q4_factors['me_2']

q4_factors['ia_3']=(q4_factors['m2i3r1']+q4_factors['m2i3r2']+q4_factors['m2i3r3']+
                    q4_factors['m1i3r1']+q4_factors['m1i3r2']+q4_factors['m1i3r3'])/6
q4_factors['ia_1']=(q4_factors['m2i1r1']+q4_factors['m2i1r2']+q4_factors['m2i1r3']+
                    q4_factors['m1i1r1']+q4_factors['m1i1r2']+q4_factors['m1i1r3'])/6
q4_factors['r_ia'] = q4_factors['ia_1']-q4_factors['ia_3']


q4_factors['roe_3']=(q4_factors['m2i1r3']+q4_factors['m2i2r3']+q4_factors['m2i3r3']+
                     q4_factors['m1i1r3']+q4_factors['m1i2r3']+q4_factors['m1i3r3'])/6
q4_factors['roe_1']=(q4_factors['m2i1r1']+q4_factors['m2i2r1']+q4_factors['m2i3r1']+
                     q4_factors['m1i1r1']+q4_factors['m1i2r1']+q4_factors['m1i3r1'])/6
q4_factors['r_roe'] = q4_factors['roe_3']-q4_factors['roe_1']

q4_factors=q4_factors.rename(columns={'jdate':'date'})


# Comparar com a série de fatores dos autores
os.chdir('C:\Dropbox\Code\Python')
q5_factors = pd.read_csv('hmxz/q5_factors_monthly_2019a.csv')
# print(q5_factors)
# print(q4_factors['r_ia'])

q4_factors['year'] = q4_factors['date'].dt.year
q4_factors['month'] = q4_factors['date'].dt.month
q4_factors[['r_me','r_ia', 'r_roe']] = q4_factors[['r_me','r_ia', 'r_roe']] * 100
qfcomp = pd.merge(q5_factors,
                   q4_factors[['year','month','r_me','r_ia', 'r_roe']],
                   how='inner', on=['year', 'month'])

print(stats.pearsonr(qfcomp['R_ME'], qfcomp['r_me']))
# (0.9905224803611761, 0.0)
# (0.97723366230532, 0.0)

print(stats.pearsonr(qfcomp['R_IA'], qfcomp['r_ia']))
# (0.9438473340470327, 2.90160265201413e-278)
# (0.9674400829890721, 0.0)

print(stats.pearsonr(qfcomp['R_ROE'], qfcomp['r_roe']))
# (0.9327040225769708, 9.657585959879293e-258)
