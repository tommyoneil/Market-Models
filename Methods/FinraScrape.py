# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 11:38:11 2021

@author: MI2
"""
import pandas as pd
import numpy as np
import tia.bbg.datamgr as dm
from datetime import date as dt
from bs4 import BeautifulSoup
import re
from urllib import request
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

dataPeriod = "MONTHLY"
startDate = dt(1997,1,1)
today = dt.today()

mgr = dm.BbgDataManager() 

tickers = ['.MRGDBT Index','.MRGCRDT Index','.MRGRTO Index','.DMRGDBT Index','.PMRGDBT Index','.SECCOR1 Index',
           'VIX Index','.NYSEVL2 Index','NYA Index','.NYAMVX Index']

#Only must maintain and update .SECCOR1 Index to get output, unless integrate with SectorCorrelation program
sids = mgr[tickers[5],tickers[6],tickers[8]]
df1 = sids.get_historical(['PX_Last'],startDate,today,dataPeriod)

df1.columns = df1.columns.droplevel(1)
df1Drop1 = pd.DataFrame(df1.iloc[:,0]).dropna().reset_index(drop=True)
df1Drop2 = pd.DataFrame(df1.iloc[:,1]).dropna().reset_index(drop=True)
df1Drop3 = pd.DataFrame(df1.iloc[:,2]).dropna().reset_index(drop=True)
newDF = pd.DataFrame(data=df1Drop1)
newDF[tickers[6]] = df1Drop2
newDF[tickers[8]] = df1Drop3

final_export = pd.DataFrame()

URL = 'https://www.finra.org/investors/learn-to-invest/advanced-investing/margin-statistics'

response = request.urlopen(URL).read()
soup= BeautifulSoup(response, "html.parser")     
links = soup.find_all('a', href=re.compile(r'(.xlsx)'))
url_list = []

for el in links:
    if(el['href'].startswith('http')):
        url_list.append(el['href'])
    else:
        url_list.append('https://www.finra.org/investors/learn-to-invest/advanced-investing/margin-statistics' + el['href'])

for url in url_list:
    file_name = url.rsplit('/', 1)[-1]
    
driver = webdriver.Chrome(ChromeDriverManager().install()) #must be using Chrome or change browser here

driver.get(URL) #find xpath with chrome inspect
btn = driver.find_element_by_xpath('/html/body/div/div/div/div[1]/div/div/div[2]/aside/section/div[2]/div/div/p/a')
btn.click()
df_excel = pd.read_excel('/Users/MI2/Downloads/'+file_name) #change path for user
driver.close()

finalDF = pd.DataFrame(data=df_excel.iloc[:,1:4]).reset_index(drop=True)
finalDF = finalDF.loc[::-1].reset_index(drop=True)
nanDF = pd.DataFrame([np.nan * 12])

final_export[tickers[0]] = finalDF.iloc[:,0]

nansum = finalDF.iloc[:,2].isna().sum()
addFrame = []
for i in range(0,nansum):
    addFrame.append(finalDF.iloc[i,1])
    
for i in range(nansum,len(finalDF)):
    addFrame.append(finalDF.iloc[i,1] + finalDF.iloc[i,2])
#addFrame = [finalDF.iloc[i,1] + finalDF.iloc[i,2] for i in range(0,len(finalDF))]
final_export[tickers[1]] = addFrame
divFrame = [final_export.iloc[i,0] / final_export.iloc[i,1] for i in range(0,len(final_export))]
final_export[tickers[2]] = divFrame
yshift = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
for i in range(0,len(final_export)-12):
    yshift.append(final_export.iloc[i+12,0] - final_export.iloc[i,0])

final_export[tickers[3]] = yshift
yshift2 = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
for i in range(0,len(final_export)-12):
    yshift2.append(((final_export[tickers[0]][i+12] - final_export[tickers[0]][i]) / final_export[tickers[0]][i]) * 100)
    
final_export[tickers[4]] = yshift2
final_export[tickers[5]] = newDF[tickers[5]]
final_export[tickers[6]] = newDF[tickers[6]]
col7 = [final_export[tickers[2]][i] / (final_export[tickers[6]][i]*final_export[tickers[5]][i]) for i in range(0,len(final_export))]
final_export[tickers[7]] = col7
final_export[tickers[8]] = newDF[tickers[8]]
final_export[tickers[9]] = [final_export[tickers[0]][i] / final_export[tickers[8]][i] / final_export[tickers[6]][i] for i in range(0,len(final_export))]

final_export = final_export.set_index(df1.index.to_period('M').drop_duplicates()[0:-1])
final_export.to_excel(r'C:\Users\MI2\Desktop\Excel\MRG Export.xlsx', index = True)