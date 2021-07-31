# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 12:22:40 2021

@author: MI2
"""
import matplotlib.pyplot as plt
import tia.bbg.datamgr as dm
from datetime import date as dt
from sklearn.decomposition import PCA
from Norm import *

dataPeriod = "MONTHLY"
startDate = dt(1962,1,1)
endDate = dt(2021,4,30)

mgr = dm.BbgDataManager() 

vbs = ['GEIFOMDE Index',#'GMNEDSCP Index',
       'GMEFOHCP Index','GMEFDSCP Index',
       'GEIFOMOR Index','GCGDOHCP Index','GCGDDSCP Index',#'GMNEBDE6 Index',
       'GINGOHCP Index','GINGDSCP Index'] #'GOSMDSCP Index',...#'GVTSEBE3 Index',
        #'GOGMBDE6 Index', #'GMFPBDE6 Index'
        
# vbs = ['SBOITOTL Index','NAPMPMI Index','NAPMNMI Index','CONSSENT Index',
#       'USHBMIDX Index','OUTFGAF Index']
# 
# vbs = ['OUTFGAF Index','RCHSINDX Index','KCLSSACI Index','EMPRGBCI Index',
#       'DFEDGBA Index']

# vbs = ['GEIFOMBE Index','GMEFBDE6 Index','GINGBDE6 Index']

# vbs = ['GMEFDSCP Index','GINGDSCP Index','GEIFOMDE Index','GCGDDSCP Index']
#       #'GEMGDSCP Index']

# vbs = ['KCLSSACI Index','KCMTLMCI Index','KCMTLMMO Index','KC6SCAPE Index',
#        'KCLSNSAA Index','KCLSNSAC Index','KCLSNSAE Index','KCLSNSAW Index',
#        'KC6SSACI Index','KCLSNSAN Index','KC6SVORD Index','KCLSNSAV Index',
#        'KCVYCAPE Index','KC6SAVGW Index','KC6SBLOG Index','KC6SEMPL Index',
#        'KC6SIFIN Index','KC6SIMAT Index','KC6SNSAA Index','KC6SNSAB Index',
#        'KC6SNSAC Index','KC6SNSAD Index','KC6SNSAE Index','KC6SNSAG Index',
#        'KC6SNSAM Index','KC6SNSAN Index','KC6SNSAP Index','KC6SNSAR Index',
#        'KC6SNSAS Index','KC6SNSAV Index','KC6SNSAW Index','KC6SNSAX Index',
#        'KC6SPP Index','KC6SPR Index','KC6SPROD Index','KC6SSUPD Index',
#        'KC6SVEXP Index','KC6SVSHP Index','KCLSNSAB Index','KCLSNSAD Index',
#        'KCLSNSAG Index','KCLSNSAM Index','KCLSNSAP Index','KCLSNSAR Index',
#        'KCLSNSAS Index','KCVYAVGW Index','KCVYBLOG Index','KCVYEMPL Index',
#        'KCVYIFIN Index','KCVYIMAT Index','KCVYPP Index','KCVYPR Index',
#        'KCVYPROD Index','KCVYSACI Index','KCVYSUPD Index','KCVYVEXP Index',
#        'KCVYVORD Index','KCVYVSHP Index']

sids = mgr[vbs]

df1 = sids.get_historical(['PX_Last'],startDate,endDate,dataPeriod)

df1.columns = df1.columns.droplevel(1)

df1 = df1.fillna(method = 'ffill')

df1 = normal_cal(df1)
print(df1)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df1)
principalDF = pd.DataFrame(data=principalComponents, columns=['Principal Component 1', 'Principal Component 2'], index=df1.index)
print(principalDF)

plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(principalDF['Principal Component 1'],label='Principal Component 1')
plt.plot(principalDF['Principal Component 2'],label='Principal Component 2')
plt.legend(loc=1)
plt.show()

plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(df1)
plt.show()

principalDF['Principal Component 1'].to_excel(r'C:\Users\MI2\Desktop\Excel\IFO3Lead80ThreshGDPMod.xlsx', index = True)
#principalDF['Principal Component 1'].to_excel(r'C:\Users\MI2\Desktop\Excel\FiveFedsPCA.xlsx', index = True) 
#principalDF['Principal Component 1'].to_excel(r'C:\Users\MI2\Desktop\Excel\KCFedPCA1.xlsx', index = True)