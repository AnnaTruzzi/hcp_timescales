import nibabel as nib
import numpy as np 
import boto3
from botocore.exceptions import ClientError
import os
from statsmodels.tsa.arima.model import ARIMA

data = np.load('tc_by_roi_afterICA.npy')
model = [1,0,1]
alltau = np.zeros((data.shape[0],data.shape[2]))
allrho1 = np.zeros((data.shape[0],data.shape[2]))
allrho2 = np.zeros((data.shape[0],data.shape[2]))
ar_list = []
ma_list= []

for sub in range(0,data.shape[0]):
    print(sub)
    for ROI in range(0,data.shape[2]):
        mod = ARIMA(endog=data[sub,9:,ROI], order=(model[0],model[1],model[2]), enforce_stationarity=False,enforce_invertibility=False)
        res=mod.fit()
        ar=res.arparams
        rho0 = 1
        ar_list.append(res.arparams)
        ma_list.append(res.maparams)
        try:
            rho1 = ar[0]
            rho2 = ar[0] * rho1
            allrho1[sub,ROI] = rho1
            allrho2[sub,ROI] = rho2
        except:
            allrho1[sub,ROI] = np.nan
            allrho2[sub,ROI] = np.nan
        alltau[sub,ROI]=-1/np.log(rho2 / rho1)

print(alltau.shape)
np.savetxt('hcp_intrinsic_timescales_ARMA_afterICA.txt',alltau)
