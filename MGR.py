# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 11:38:11 2021

@author: MI2
"""
import pandas as pd
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

sids = mgr[tickers]
df1 = sids.get_historical(['PX_Last'],startDate,today,dataPeriod)

df1.columns = df1.columns.droplevel(1)
df1 = df1[tickers]
df1 = df1.fillna(method='bfill') 
df1 = df1.dropna()

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
    
driver = webdriver.Chrome(ChromeDriverManager().install())

driver.get(URL)
btn = driver.find_element_by_xpath('/html/body/div/div/div/div[1]/div/div/div[2]/aside/section/div[2]/div/div/p/a')
btn.click()
df_excel = pd.read_excel('/Users/MI2/Downloads/'+file_name)
driver.close()

finalDF1 = pd.DataFrame(data=df_excel.iloc[:,1:4]).reset_index(drop=True)
finalDF2 = pd.DataFrame(data=df1.iloc[:,:]).reset_index(drop=True)

for i in range(0,len(finalDF1.columns)):
    finalDF2[finalDF1.columns[i]] = finalDF2.iloc[:,i]

final = finalDF2.set_index(df1.index)
final.to_excel(r'C:\Users\MI2\Desktop\Excel\MRG Export.xlsx', index = True) #check dimensions alight, numbers coordinate!
