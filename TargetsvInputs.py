# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 09:49:14 2021

@author: MI2
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tia.bbg.datamgr as dm
from datetime import date as dt
from Norm import *
import statsmodels.api as sm
from Offsetter import *
from itertools import product
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

startDate = dt(1980,1,1)
endDate = dt(2019,6,30)

# vbs = target, input1, input2
# vbs = ['USGG10YR Index','NAPMNEWO Index','CPI XYOY Index','USGG2YR Index','USGG3M Index'] 
# = ['WGTROVER Index','SBOICOMP Index','SBOICAPS Index','SBOICMPP Index','SBOICAPX Index'] 
#vbs = ['CPI YOY Index','SBOIPRIC Index','OUTFPRF Index'] 
#vbs = ['NAPMPMI Index','OUTFGAF Index','RCHSINDX Index','KCLSSACI Index'] 
#vbs = ['NAPMPMI Index','OUTFNOF Index', 'RCHSBANO Index', 'KCLSVORD Index'] 
#vbs = ['NAPMPMI Index','RCHSINDX Index','KCLSSACI Index'] 
#vbs = ['NFP PCH Index','SLDETGTS Index','SDCLTGTC Index'] 
#vbs = ['CONSSENT Index','COMFCOMF Index','.SPXYOY Index','CL1 Comdty'] 
#vbs = ['CZGDPSAY Index','EUSCCZ Index','CZCCCON Index','EUICCZ Index','EUCOCZ Index'] 
#vbs = ['BCOM Index','DXY Curncy','SPX Index']  
#vbs = ['NAPMPMI Index','DXY Curncy','CL1 Comdty'] 
#vbs = ['CONSSENT Index','3AGSREG Index','SPX Index'] 
# vbs = ['SPNAGDPY Index','EUSCES Index','EUICES Index'] 
#vbs = ['CPI XYOY Index','WGTROVER Index','NAPMPMI Index'] 
# vbs = ['USTW$ Index','FDDSGDP Index','EHCAUS Index'] 
#vbs = ['GDP CYOY Index','CGNOXAY# Index','NAPMPMI Index'] 
#vbs = ['.EUROCMPY Index','EUA2EMU Index','EUAUEMU Index'] 
#vbs = ['LNTN27Y Index','.EURCMPMD Index'] 
#vbs = ['NHSLYOY Index','CONSGOVU Index','CONSHOLR Index', 'CONSVALS Index']#,'CONSIM0 Index'] #  'CONSVELR Index', 'USHBMIDX Index',
#vbs = ['SPX Index','S5AIRLX Index','S5BUILX Index','S5SPRE Index','S5HOTRX Index','S5TEXA Index']  #,'S5CSTMX Index'
#vbs = ['USURTOT Index','USEMPTSW Index','CONCJOBH Index','SBOICOMP Index','ETI INDX Index'] 
#vbs = ['JNLSUCTL Index','JNCICLEI Index','JPSBPRBR Index'] 
#vbs = ['LNTNWDEY Index','GGFKBUS Index','OEDESVAO Index'] 
#vbs = ['LNTNWFRY Index','OEFRSVAM Index','EUA2FR Index'] 
#vbs = ['FRWAHRY Index','OEFRSVAM Index','EUA2FR Index'] 
#vbs = ['FRWAHRY Index','FSUPTOTA Index','EUA2FR Index'] 
#vbs = ['LNTNATYY Index','OEATSVAO Index','EUCCAT Index'] 
#vbs = ['LNTNUKYY Index','EUA2UK Index','EUCCUK Index'] 
#vbs = ['.SWZWGYY Index','SZUEOPEN Index','SZCCFES Index'] 
#vbs = ['.SWEWGYY Index','EUCCSE Index','EUA4SE Index'] 
#vbs = ['USGG2YR Index','WUXIFFRT Index'] 
#vbs = ['SWCPUIFY Index','EPF25SIT Index','EPD27AEA Index'] #,'EPF14MIT Index'] 
#vbs = ['SWCPUIFY Index','EPF23YDE Index','EPF282DE Index'] 
#vbs = ['SWCPUIFY Index','EPF282DE Index','EPF23YDE Index'] 
#vbs = ['CNCPIYOY Index','CB1APKY Index','CHEFTYOY Index',]#,'CHAFPKRM Index'] 
#vbs = ['.SPXYOY Index','NAPMPMI Index']
#vbs = ['SPX Index','134D0... Index']
#vbs = ['CPI XYOY Index','.NYFEDUIG Index','.ATW14LVA Index']
#9-16, Quarterly
#vbs = ['CPI INDX Index','PPIDDH10 Index','PPIDAOTE Index','PPIDNMR1
    #Index']  #6-8, Quarterly

#vbs = ['CPI YOY Index','NAPMNPRC Index']

#vbs = ['CPI XYOY Index','.UIGACT Index'] 

#vbs = ['.EUWGDT Index','UMRTEMU Index'] 

#vbs = ['.SRPFI Index','.MSCIVJPM Index','NBBIYL Index','.SPXYOY Index'] 

#vbs = ['PCE DEFY Index','SBOIPRIC Index','DFEDPPRM Index','.DXYYOY Index']#'KCVYPR Index','OUTFPRF Index',]#,'USGG10YR Index']

#vbs = ['AFFDCMOM Index','ETSLMP Index','PITL Index'] #'ILM3NAVG Index'

#vbs = ['GDX US Equity','SPX Index','XAU Curncy']

#vbs = ['NFP PCH Index','NAPMEMPL Index','NAPMNEMP Index'] 

#vbs = ['UMRTEMU Index','.WUXIMDL Index','EUAUEMU Index'] 

#vbs = ['DXY Index','BICLOISS Index','.USDBS2 Index'] 

#vbs = ['CPI XYOY Index','.OILYOY Index','SBOIHIRE Index','.USDYOY Index']#'.BROAD$YY Index']

#vbs = ['NFCIINDX Index','.MOODBAA Index','.02-10SP Index','.SPXYOY Index','USTW$ Index'] 

#vbs = ['GRCP20MM Index','GRCP2BRM Index','GRCP2HEM Index','GRCP2BVM Index','GRCP2BWM Index','GRCP2NRM Index','GRCP2SAM Index'] 

#vbs = ['CPI XYOY Index','NAPMPMI Index','OUTFDTF Index',]

#vbs = ['CPI XYOY Index','CPTICHNG Index','UIGDDATA Index']#,'NAPMPMI Index']

#vbs = ['CPI XYOY Index','CTRIYOY Index','.PPIDONRY Index','NAPMPMI Index']#'CPI YOY Index','.RBOBYOY Index']#,

#vbs = ['CPI XYOY Index','EMPRNEMP Index','EMPRWORK Index']

#vbs = ['MXUS Index','NFCIINDX Index','DXY Index']

#vbs = ['CPI XYOY Index','.ATLTFIT Index','.CPTRYOY Index']#

#vbs = ['CPI XYOY Index','UIGDDATA Index']

#vbs = ['WGTROVER Index','.WUXIMDL Index']#,'SBOIHIRE Index']

#vbs= ['CPI XYOY Index','KXUSCOP Index']

#vbs = ['CPI XYOY Index','.ATLTFIT Index','PITLYOY Index']#'.CPTRYOY Index',

#vbs = ['CPI XYOY Index','NAPMEMPL Index','SBOIEMPC Index']#'NFP TYOY Index',

#vbs = ['CPI XYOY Index','SBOIPRIC Index','.GASPYOY Index'] 

#vbs = ['NFP P Index','NYA Index'] 

#vbs = ['NAPMPMI Index','LEI AVGW Index','LEI MNO Index','LEI WKIJ Index']

#vbs = ['USURTOT Index','SBOIHIRE Index', 'ETI INDX Index','USEMPTSW Index']  
        #'USEMPTSW Index' =2, CONCJOBH = 0
        #SBOIHIRE = 5, ETI INDX = 4
        
#vbs = ['NAPMNMI Index', 'ETSLYOY Index','CONSHOHR Index']


#vbs = ['RSTAGYOY Index','.SPXYOY Index','SPCS20Y# Index','.GASPYOY Index']#,'PITLYOY Index']

#vbs = ['GDP CYOY Index','.SPXYOY Index'] 

#vbs = ['USURTOT Index','USHBPRSS Index']

# vbs = ['.GLBLEPSP Index','KOEXTOTY Index','.USDYOY Index','.OECDGLEI Index',...
#          'USGG10Y Index'] 

#vbs = ['.SPXSLSY Index','.YOY$ Index']

#vbs = ['USGG10YR Index','SPX Index']

#vbs = ['NAPMPMI Index','RCHSINDX Index'] #'OUTFGAF Index','DFEDGBA Index','EMPRGBCI Index',

#vbs = ['NACCCMI Index','.ECRIWLI2 Index'] 

#vbs = ['USGGBE10 Index','HG2 COMB COMDTY'] 

#vbs = ['GDBR10 Index','SX7E Index']

#vbs = ['NAPMPMI Index','.YOY$ Index'] 

#vbs = ['.GDPRATES Index','.JPMGPMIF Index'] 

#vbs = ['.USCHEMYY Index','.JPMGPMIF Index'] 

#vbs = ['.JPMGPMIF Index','NAPMPMI Index'] 

#vbs = ['.GLCHEMYY Index','.JPMGPMIF Index'] 

#vbs = ['GRGDPPGY Index','GWIGBDE6 Index'] 

#vbs = ['NFP TYOY Index','USWHMNOS Index'] 

#vbs = ['NFP TYOY Index','NAPMPMI Index'] 

#vbs = ['NFP TYOY Index','CGNOXAY# Index'] 

#vbs = ['.UNEMPYOY Index','USWHMNOS Index'] 

#vbs = ['.UNEMPYOY Index','.JOLTYY Index']#'NAPMNEMP Index']#'NAPMEMPL Index']#,, 

#vbs = ['.UNEMPYOY Index','NAPMEMPL Index'] 

#vbs = ['.ISMYOY Index','.2P10YOY Index'] 

#vbs = ['USTRBROA Index','.TWDMDL Index']#'FDDSGDP Index','EHCAUS Index'] 

#vbs = ['.UNEMPYL Index','.CONCLNY Index']

#vbs = ['.JOLTYY Index','.CASSSYOY Index'] 

#vbs = ['NFP TYOY Index','.SBOIEMPY Index'] 

#vbs = ['REDSWYOY Index','NAPMPMI Index'] 

#vbs = ['NAPMPMI Index','.ISM4 Index'] 

#vbs = ['.TEMPEYL Index','NAPMNEWO Index'] 

#vbs = ['JOLTQUIS Index','JOLTTOTL Index'] 

#vbs = ['.SAHMIND Index','.SAHMSALE Index'] 

#vbs = ['INJCSP Index','INJCJC Index'] 

#vbs = ['NAPMPMI Index','.CASSYOY Index'] 

# vbs = ['USURTOT Index', 'USEMPTSW Index', 'CONCJOBH Index',...
#                 'SBOIHIRE Index', 'ETI INDX Index', 'ADP SMLL Index',...
#                 'ADP SML Index', 'ADP MED Index', 'ADP LGE Index',...
#                 'ADP ELGE Index', 'PRUSTOT Index', 'JOLTOPEN Index', 'JOLTLAYS Index',...
#                 'JOLTQUIS Index'] 

#vbs = ['.SPXYOY Index','CPI YOY Index','.USDYOY Index'] #'GDP CURY Index','M2# YOY Index',

#vbs = ['USGG10YR Index','.JPMGPMIF Index'] 
#vbs = ['CPI XYOY Index','.VELO$YY Index'] 
#vbs = ['EWW US Equity','USDMXN Curncy'] 

#vbs = ['NAPMPMI Index','USGGBE30 Index']#'USGGBE05 Index''USGGBE10 Index''USGGBE30 Index''USGG5Y5Y Index'
#'GDP CYOY Index'

#vbs = ['CESIUSD Index','GDGCAFJP Index'] 

#vbs = ['.GOLDSPX Index','DXY Curncy'] 

#vbs = ['SBOIHIRE Index','SBOIEMPC Index'] 

#vbs = ['CPI XYOY Index','SBOIPRIC Index','USRFRUSA Index'] 

#vbs = ['CPI XYOY Index','GDP CYOY Index','M2# YOY Index'] 

#vbs = ['USURTOT Index','USEMPTER Index'] 

#vbs = ['CPI YOY Index','M2# YOY Index','GDP CYOY Index'] 

#vbs = ['DXY Index','.DBTVQEYL Index'] 

#vbs = ['USGGBE05 Index','.BCOMIN Index'] 

#vbs = ['CPRHOERY Index','CONCINFL Index','HVRAHOME Index','HPI YOY# Index','.ASKRNTYY Index'] 

#vbs = ['NAPMPRIC Index','.BCOMIN Index'] 

#vbs = ['PCE CYOY Index','.KCPC1 Index','.BCOMIN Index'] 

#vbs = ['GDP CURY Index','.NETWRTH Index']

#vbs = ['CPRHOERY Index','.OERMDL2 Index'] 

#vbs = ['CPI YOY Index','.PRICESP Index'] 

#vbs = ['PCE DEFY Index','KCLSPR Index'] #,'.GASPYOY Index'] 

#vbs = ['EMPRRECV Index','EMPR6REC Index'] 

#vbs = ['CPI YOY Index','.PRICESP Index'] 

#vbs = ['ECWSPVYY Index','SBOICMPP Index'] 

#vbs = ['CPI XYOY Index','FDDSGDP Index']#'M2# YOY Index'] 

#vbs = ['CPI YOY Index','.ENERYOY2 Index']#,'EMPR6PAY Index','EMPR6REC Index'] 

#vbs =['GDP CURY Index','.EXCESSST Index'] 

#vbs = ['CPI XYOY Index','.COREVPPI Index'] 

#vbs = ['USHEYOY Index','SBOIEMPL Index'] 

vbs = ['CPRHOERY Index','SPCSUSAY Index','CLVIEI01 Index','HVRARENT Index']

maxLag = 8 
minLag = 0 
maLength = 3 
dataPeriod = 'MONTHLY' #EITHER QUARTERLY OR MONTHLY

mgr = dm.BbgDataManager() 
sids = mgr[vbs]
#data = pd.read_excel(r'C:\Users\MI2\Documents\MATLAB\CIXIsfromMatlab\TargetsVInputs.xlsx')
#dataMatrix = data.iloc[1:,1:].set_index(data.iloc[1:,0])
#dataMatrix = dataMatrix[::-1]
dataMatrix = sids.get_historical(['PX_LAST'],startDate, endDate)
dateVector = dataMatrix.index

if dataPeriod == 'QUARTERLY':
    dataMatrix = dataMatrix.dropna()
elif dataPeriod == 'MONTHLY':
    dataMatrix = dataMatrix.fillna(method='ffill')
    dataMatrix = dataMatrix.dropna()
#dataMatrix = dataMatrix.iloc[1:,:] 
#dataMatrix.iloc[:,1:] = np.divide(dataMatrix.iloc[:,1:], dataMatrix.iloc[:,0])
#dataMatrix = np.dot(normal_cal(dataMatrix),100) 
        
dataMatrix = dataMatrix.rolling(maLength,0).mean() #not exactly the same as in Matlab

x1 = range(minLag,maxLag+1) 
x = [str(i) for i in x1]
K = len(dataMatrix.iloc[0,:])-1 
offsets = pd.DataFrame(np.array(list(product(x,x,x))))

resultsList = []
for combo in range(0,len(offsets)):
    for j in range(0,5000000, 200):
        if combo == j:
            print(combo)
    [Yin,Xin,XinCons,Xout,XoutCons] = offsetter(dataMatrix,offsets.iloc[combo,:])
    lm = LinearRegression()
    lm.fit(Xin, Yin)
    y_pred = lm.predict(Xin)
    MeanSqEr = mean_squared_error(Yin, y_pred)
    addOn = [MeanSqEr]
    for i in range(0,len(offsets.iloc[combo,:])):
        addOn.append(int(offsets.iloc[combo,:][i]))
    resultsList.append(addOn)

resultStorage = pd.DataFrame(resultsList)

endResult = resultStorage.sort_values(by=0,ascending=True).reset_index(drop=True)
#endResult.iloc[-6:,:] 
#endResult.iloc[0:4,:] 

best = np.array(endResult.iloc[0,:])
[Yin,Xin,XinCons,Xout,XoutCons] = offsetter(dataMatrix,best[1:]) 

betaB = sm.OLS(np.array(Yin),XinCons).fit().params
Yresult = np.dot(XinCons,betaB)
Rresult = pd.DataFrame(np.corrcoef(Yresult,Yin))

#XConsfull = [XoutCons, 38.80, 1, 8, 1] 

betaB = list(betaB)
Yext = np.dot(XoutCons,betaB)

plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(Yin.reset_index(drop=True))
plt.plot(Yresult)
plt.plot(Yext)
plt.title([vbs[0], str(best[0]), str(best[1]), str(best[2]), str(best[3]), vbs[1:], 
          str(betaB[0]), str(betaB[1]), str(betaB[2]), str(betaB[3]), str(Rresult.iloc[0,1])])
plt.show()

final = pd.DataFrame(data=Yext, index=dateVector[len(dateVector)-len(Yext):])
final.to_excel(r'C:\Users\MI2\Desktop\Excel\CPITransWageCheck.xlsx', index = True)