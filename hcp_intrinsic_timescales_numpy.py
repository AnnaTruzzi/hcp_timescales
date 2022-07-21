import nibabel as nib
import numpy as np 
import boto3
from botocore.exceptions import ClientError
import os
from statsmodels.tsa.arima.model import ARIMA
from scipy import optimize

def autocorr_decay(dk,A,tau,B):
    return A*(np.exp(-(dk/tau))+B)

data = np.load('tc_by_roi.npy')
model = [1,0,1]
alltau = np.zeros((data.shape[0],data.shape[2]))
allrho1 = np.zeros((data.shape[0],data.shape[2]))
allrho2 = np.zeros((data.shape[0],data.shape[2]))
ar_list = []
ma_list= []
max_lag = 100
removelag0 = 1

for sub in range(0,data.shape[0]):
    print(sub)
    nlags=100
    xdata=np.arange(nlags)

    timescale = np.zeros((data.shape[2], nlags))
    for ROI in range(0,data.shape[2]):
            xc=data[sub,9:,ROI]-np.mean(data[sub,9:,ROI])
            fullcorr=np.correlate(xc, xc, mode='full')
            fullcorr=fullcorr / np.max(fullcorr)
            start=len(fullcorr) // 2
            stop=start+max_lag
            timescale[ROI,:]=fullcorr[start:stop]
    for ROI in range(0,data.shape[2]):
        try:
            A, tau, B = optimize.curve_fit(autocorr_decay,xdata[removelag0:],timescale[ROI,removelag0:],p0=[0,np.random.rand(1)[0]+0.01,0],bounds=(([0,0,-np.inf],[np.inf,np.inf,np.inf])),method='trf',maxfev=1000)[0]
            alltau[sub,ROI] = tau
        except:
            alltau[sub,ROI] = np.nan


print(alltau.shape)
np.savetxt('hcp_intrinsic_timescales_numpy.txt',alltau)
