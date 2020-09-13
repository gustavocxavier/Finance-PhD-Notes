## POR DESENVOLVER: gerar empresas com ROE disponivel


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 12:50:57 2020

@author: gcx
"""

##########################################
# Fama French Factors
# April 2018
# Qingyi (Freda) Song Drechsler
##########################################

import pandas as pd
import numpy as np
import datetime as dt
import wrds
import psycopg2 
import matplotlib.pyplot as plt
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
from scipy import stats

# ## Load from WRDS
# conn=wrds.Connection()

# comp = conn.raw_sql("""
#                     select gvkey, datadate, at, pstkl, txditc,
#                     pstkrv, seq, pstk, ib
#                     from comp.funda
#                     where indfmt='INDL' 
#                     and datafmt='STD'
#                     and popsrc='D'
#                     and consol='C'
#                     and datadate >= '01/01/1966'
#                     """)
                    
# compq = conn.raw_sql("""
#                     select gvkey, datadate, ibq, txditcq, seqq,
#                     ceqq, pstkq, pstkrq, atq, ltq, rdq, fqtr, fyearq                   
#         from comp.fundq
#               where indfmt='INDL' 
#         and datafmt='STD'
#         and popsrc='D'
#         and consol='C'
#         and datadate between '01/01/1966' and '01/01/2019'
#                     """)
#
# del conn

# ## Save pandas locally
# comp.to_pickle('C:/Data/2020_HMXZ/comp.pkl')
# compq.to_pickle('C:/Data/2020_HMXZ/compq.pkl')

# # Em excel
# comp.to_excel('C:/Data/2020_HMXZ/comp.xlsx')
# compq.to_excel('C:/Data/2020_HMXZ/comp.xlsx')

# # No dropbox
# comp.to_pickle('C:/Dropbox/Code/Python/hmxz/comp.pkl')
# compq.to_pickle('C:/Dropbox/Code/Python/hmxz/compq.pkl')

## Load pandas locally
# comp = pd.read_stata('../ccmData/COMP.dta')
comp = pd.read_pickle('C:/Data/2020_HMXZ/comp.pkl')
compq = pd.read_pickle('C:/Data/2020_HMXZ/compq.pkl')
# comp = pd.read_pickle('../../../Data/2020_HMXZ/comp.pkl')
# compq = pd.read_pickle('../../../Data/2020_HMXZ/compq.pkl')


# ## Trying to simplify
# # Get the path
# import os
# print(os.getcwd())
# from pathlib import Path
# # https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f
# print(Path.cwd())
# # On windows : C:\Users\gusta
# # On Linux   :
# ## My Linux
# # comp = pd.read_pickle('Dropbox/Code/Python/hmxz/comp.pkl')
# ## My Windows
# # comp = pd.read_pickle('C:/Dropbox/Code/Python/hmxz/comp.pkl')
# # comp = pd.read_pickle('../../Dropbox/Code/Python/hmxz/comp.pkl')
# # compq = pd.read_pickle('C:/Dropbox/Code/Python/hmxz/compq.pkl')

# # Retirar observacoes duplicadas
# comp = comp.drop_duplicates(['gvkey', 'datadate'],keep= 'last')
# compq = compq.drop_duplicates(['gvkey', 'datadate'],keep= 'last')
# df = compq[compq.duplicated(['gvkey', 'datadate'],keep=False)]

##############################################################################
## Organizar Annual COMPUSTAT
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


##############################################################################

# compq['datadate']=pd.to_datetime(compq['datadate']) #convert datadate to date fmt
# compq['year']=compq['datadate'].dt.year
# compq = compq.sort_values(['gvkey', 'datadate'])

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

## Create stockholders’ equity
#
# Depending on availability, we use stockholders’ equity (item SEQQ), or
# common equity (item CEQQ) plus the carrying value of preferred stock (item
# PSTKQ), or total assets (item ATQ) minus total liabilities (item LTQ) in
# that order as shareholders’ equity.
#
compq['pstkq']=compq['pstkq'].fillna(0)
#    shareholders’ equity = stockholders’ equity (item SEQQ)
# or shareholders’ equity = or common equity (item CEQQ) + the carrying value of preferred stock (item PSTKQ)
compq['se']=np.where(compq['seqq'].isnull(), compq['ceqq']+compq['pstkq'], compq['seqq'])
compq.loc[compq['se'].isnull() & compq['atq'].notnull(),]
# or shareholders’ equity = or total assets (item ATQ) - total liabilities (item LTQ) in that order as 
compq['se']=np.where(compq['se'].isnull(), compq['atq']-compq['ltq'], compq['se'])


## Create Quartely book equity
#
# In particular, book equity is shareholders’ equity, plus balance sheet
# deferred taxes and investment tax credit (item TXDITCQ) if available, minus
# the book value of preferred stock.
#
# be (book equity) = se (shareholders’ equity) 
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

# roe_q = compq[['gvkey', 'datadate', 'ibq', 'lag_be', 'roe', 'rdq']]


# ### -------------------------------------------------------------------------
# ### Create a monthly ROE
# # Earnings data in Compustat quarterly files are used in the months
# # immediately after the most recent public quarterly earnings announcement
# # dates (Compustat quarterly item RDQ). For example, if the earnings for the
# # fourth fiscal quarter of year t − 1 are publicly announced on March 5
# # (or March 25) of year t, we use the announced earnings (divided by the
# # book equity from the third quarter of year t − 1) to form portfolios at
# # the beginning of April of year t.
# # http://global-q.org/uploads/1/2/2/6/122679606/factorstd_2020july.pdf
# roe_m = compq[['gvkey', 'rdq', 'datadate', 'roe']]

# roe_m = roe_m.loc[roe_m['rdq'].notnull(),] # Retirar os NA
# roe_m = roe_m.loc[roe_m['roe'].notnull(),] # Retirar os NA

# roe_m['rdq'] = pd.to_datetime(roe_m['rdq']) # Transformar rdq em data
# roe_m['datadate'] = pd.to_datetime(roe_m['datadate']) # Transformar rdq em data

# # Caso uma empresa tenha duas divulgacao no mesmo dia, ficar apenas com o
# # ROE do trimestre mais recente.
# roe_m = roe_m.sort_values(['gvkey', 'rdq', 'datadate'])
# roe_m = roe_m.drop_duplicates(['gvkey', 'rdq'], keep='last')
# roe_m.drop('datadate', axis='columns', inplace=True)
# # roe_m = roe_m[['gvkey', 'rdq', 'roe']]
# # print(roe_m)
# # roeq.to_csv('C:/Data/2020_HMXZ/roeq.csv')
# # roeq = pd.read_csv('/home/gcx/Dropbox/Code/Python/hmxz/roeq.csv')

# # # Retirar empresas que so tem um balanco na amostra
# # retirar = x[['gvkey','rdq']].groupby('gvkey').count()
# # retirar = retirar[retirar['rdq'] == 1].index
# # x = x[~x['gvkey'].isin(retirar)]

# x = roe_m
# roe_m = (x.set_index('rdq')
#       .groupby('gvkey')['roe']
#       .apply(lambda x: x.reindex(pd.date_range(x.index.min(), 
#                                                x.index.max(), freq='MS'), method='ffill'))
#       .rename_axis(('gvkey','rdq'))
#       .reset_index())

# # # Conferindo o resultado com a empresa 1001
# # print(x[x.gvkey == '001001'])         # Base Trimestral dos ROEs com data de divulgacao
# # print(roe_m[roe_m.gvkey == '001001']) # Base Mensal considerando ultimo ROE

# del x

# # Join most recent announcement dates (item RDQ) from compq
# # Hou, Xue, and Zhang (2015) start their sample in January 1972, limited by earnings announcement dates and book equity in Compustat quarterly files. We follow their procedure from January
# # 1972 onward but extend the sample backward to January 1967 following Hou et al. (2019). To
# # overcome the lack of quarterly earnings announcement dates, we use the most recent quarterly
# # 3
# # earnings from the fiscal quarter ending at least four months prior to the portfolio formation month.
# # To maximize the coverage for quarterly book equity, whenever available we first use quarterly book
# # equity from Compustat quarterly files. We then supplement the coverage for fiscal quarter four with
# # book equity from Compustat annual files.2
# ### -------------------------------------------------------------------------


##############################################################################
# crsp_m = conn.raw_sql("""
#                       select a.permno, a.permco, a.date, b.shrcd, b.exchcd,
#                       a.ret, a.retx, a.shrout, a.prc
#                       from crsp.msf as a
#                       left join crsp.msenames as b
#                       on a.permno=b.permno
#                       and b.namedt<=a.date
#                       and a.date<=b.nameendt
#                       where a.date between '01/01/1966' and '12/31/2019'
#                       and b.exchcd between 1 and 3
#                       """) 

# dlret = conn.raw_sql("""
#                       select permno, dlret, dlstdt 
#                       from crsp.msedelist
#                       """)

# ccm=conn.raw_sql("""
#                   select gvkey, lpermno as permno, linktype, linkprim, 
#                   linkdt, linkenddt
#                   from crsp.ccmxpf_linktable
#                   where substr(linktype,1,1)='L'
#                   and (linkprim ='C' or linkprim='P')
#                   """)
                  
# ## Save pandas locally
# crsp_m.to_pickle('C:/Data/2020_HMXZ/crsp_m.pkl')
# dlret.to_pickle('C:/Data/2020_HMXZ/dlret.pkl')
# ccm.to_pickle('C:/Data/2020_HMXZ/ccm.pkl')

## Load data locally
crsp_m = pd.read_pickle('C:/Data/2020_HMXZ/crsp_m.pkl')
dlret = pd.read_pickle('C:/Data/2020_HMXZ/dlret.pkl')
ccm = pd.read_pickle('C:/Data/2020_HMXZ/ccm.pkl')

# crsp_m = pd.read_pickle('../../../Data/2020_HMXZ/crsp_m.pkl')
# dlret = pd.read_pickle('../../../Data/2020_HMXZ/dlret.pkl')
# ccm = pd.read_pickle('../../../Data/2020_HMXZ/ccm.pkl')



##############################################################################
# CRSP Block      

# change variable format to int
crsp_m[['permco','permno','shrcd','exchcd']]=crsp_m[['permco','permno',
                                                     'shrcd','exchcd']].astype(int)

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
crsp_jun=crsp_jun[['permno','date', 'jdate', 'shrcd','exchcd','retadj','me','wt','cumretx','mebase','lme','dec_me']]
crsp_jun=crsp_jun.sort_values(by=['permno','jdate']).drop_duplicates()

#######################
# CCM Block           #
#######################

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
# and shrcd in (10,11) / share code of 10 or 11
# and at least 2 years in comp
#
# CRSP share code of 10 or 11. We exclude financial firms (SIC between 6000 and 6999) and firms with
# negative book equity
#
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

# def bm_bucket(row):
#     if 0<=row['beme']<=row['bm30']:
#         value = 'L'
#     elif row['beme']<=row['bm70']:
#         value='M'
#     elif row['beme']>row['bm70']:
#         value='H'
#     else:
#         value=''
#     return value

def ia_bucket(row):
    if 0<=row['ia']<=row['ia30']:
        value = 'L'
    elif row['ia']<=row['ia70']:
        value='M'
    elif row['ia']>row['ia70']:
        value='H'
    else:
        value=''
    return value

# assign size portfolio
ccm1_jun['szport']=np.where((ccm1_jun['ia'].notnull())&(ccm1_jun['me']>0)&(ccm1_jun['count']>=1), ccm1_jun.apply(sz_bucket, axis=1), '')
# assign book-to-market portfolio
ccm1_jun['iaport']=np.where((ccm1_jun['ia'].notnull())&(ccm1_jun['me']>0)&(ccm1_jun['count']>=1), ccm1_jun.apply(ia_bucket, axis=1), '')
# create positivebmeme and nonmissport variable
ccm1_jun['ianotnull']=np.where((ccm1_jun['ia'].notnull())&(ccm1_jun['count']>=1), 1, 0)
# ccm1_jun['nonmissport']=np.where((ccm1_jun['ia']!=''), 1, 0)

# store portfolio assignment as of June
june=ccm1_jun[['permno','date', 'jdate', 'iaport','szport','ianotnull']]
june['ffyear']=june['jdate'].dt.year

# merge back with monthly records
crsp3 = crsp3[['date','permno','shrcd','exchcd','retadj','me','wt','cumretx','ffyear','jdate']]
ccm3=pd.merge(crsp3, 
        june[['permno','ffyear','szport','iaport','ianotnull']], how='left', on=['permno','ffyear'])

# keeping only records that meet the criteria
ccm4=ccm3[(ccm3['wt']>0)& (ccm3['ianotnull']==1) & 
          ((ccm3['shrcd']==10) | (ccm3['shrcd']==11))]
# # assign size portfolio
# ccm1_jun['szport']=np.where((ccm1_jun['beme']>0)&(ccm1_jun['me']>0)&(ccm1_jun['count']>=1), ccm1_jun.apply(sz_bucket, axis=1), '')
# # assign book-to-market portfolio
# ccm1_jun['bmport']=np.where((ccm1_jun['beme']>0)&(ccm1_jun['me']>0)&(ccm1_jun['count']>=1), ccm1_jun.apply(bm_bucket, axis=1), '')
# # create positivebmeme and nonmissport variable
# ccm1_jun['posbm']=np.where((ccm1_jun['beme']>0)&(ccm1_jun['me']>0)&(ccm1_jun['count']>=1), 1, 0)
# ccm1_jun['nonmissport']=np.where((ccm1_jun['bmport']!=''), 1, 0)

# # store portfolio assignment as of June
# june=ccm1_jun[['permno','date', 'jdate', 'bmport','szport','posbm','nonmissport']]
# june['ffyear']=june['jdate'].dt.year

# # merge back with monthly records
# crsp3 = crsp3[['date','permno','shrcd','exchcd','retadj','me','wt','cumretx','ffyear','jdate']]
# ccm3=pd.merge(crsp3, 
#         june[['permno','ffyear','szport','bmport','posbm','nonmissport']], how='left', on=['permno','ffyear'])

# # keeping only records that meet the criteria
# ccm4=ccm3[(ccm3['wt']>0)& (ccm3['posbm']==1) & (ccm3['nonmissport']==1) & 
#           ((ccm3['shrcd']==10) | (ccm3['shrcd']==11))]

############################
# Form Fama French Factors #
############################

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

q5_factors = pd.read_csv('hmxz/q5_factors_monthly_2019a.csv')
# print(q5_factors)
# print(ff_factors['r_ia'])


ff_factors['year'] = ff_factors['date'].dt.year
ff_factors['month'] = ff_factors['date'].dt.month
ff_factors[['r_me','r_ia']] = ff_factors[['r_me','r_ia']] * 100
_qfcomp = pd.merge(q5_factors,
                   ff_factors[['year','month','r_me','r_ia']][ff_factors['year']>=1970],
                   how='inner', on=['year', 'month'])
print(stats.pearsonr(_qfcomp['R_ME'], _qfcomp['r_me']))
print(stats.pearsonr(_qfcomp['R_IA'], _qfcomp['r_ia']))

_qfcomp[['year', 'month', 'R_ME', 'r_me', 'R_IA', 'r_ia']].to_excel('qfcomp.xlsx')



# plt.figure(figsize=(16,12))
# plt.suptitle('Comparison of Results', fontsize=20)

# ax1 = plt.subplot(211)
# ax1.set_title('SMB', fontsize=15)
# ax1.set_xlim([dt.datetime(1962,6,1), dt.datetime(2017,12,31)])
# ax1.plot(_qfcomp['R_ME'], 'r--', _qfcomp['WSMB'], 'b-')
# ax1.legend(('R_ME','WSMB'), loc='upper right', shadow=True)

# ax2 = plt.subplot(212)
# ax2.set_title('HML', fontsize=15)
# ax2.plot(_qfcomp['R_IA'], 'r--', _qfcomp['WI_A'], 'b-')
# ax2.set_xlim([dt.datetime(1962,6,1), dt.datetime(2017,12,31)])
# ax2.legend(('R_IA','WI_A'), loc='upper right', shadow=True)

# plt.subplots_adjust(top=0.92, hspace=0.2)

# plt.show()

# ff_factors.to_excel('fatores.xlsx')
##############################################################################
## Testes
# df = comp[:200]
# df['qtd'] = 1
# df.groupby(['gvkey','year']).sum()['qtd']
# df.to_excel('C:/Dropbox/Code/Python/hmxz/df.xlsx')
df = compq
df = df.loc[:,~df.columns.duplicated()]

df = compq[:200]
df.to_pickle('C:/Data/2020_HMXZ/df.pkl')
df = pd.read_pickle('df.pkl')


# # filtrar compq onde pstk é invalido e pstkrq é valido ---------------------
# df = compq[:50]
# print(df)
# print(df[df.pstkq.isnull() & df.pstkrq.notnull()].count()) # 10251
# print(df[df.pstkq.notnull() & df.pstkrq.isnull()].count()) # 10251

# df = pd.merge(compq, compq_ps[['gvkey','datadate','pstkrq']], how='left', on = ['gvkey','datadate'])
# df = df.drop_duplicates(['gvkey', 'datadate'],keep= 'last')