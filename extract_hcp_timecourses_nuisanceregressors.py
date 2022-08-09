import nibabel as nib
import numpy as np 
import boto3
from botocore.exceptions import ClientError
import os
from statsmodels.tsa.arima.model import ARIMA
import nuisanceRegressionPipeline as nrp

sublist = ['178950']

nsub = len(sublist)
ntp = 1200
nroi = 360

# Set up boto3
session = boto3.Session(profile_name='hcp')
s3 = session.client('s3')

tc_by_roi=np.zeros((nsub, ntp, nroi))

for subind,subj in enumerate(sublist):
    # For each subject
    # Download file from HCP S3. 
    hcpbucket = 'hcp-openaccess'
    sess='1'
    global_mask = f'/HCP_1200/{subj}/MNINonLinear/brainmask_fs.nii.gz'
    localfn_global_mask = f'/home/annatruzzi/hcp_timecourses/{subj}/brainmask_fs.nii.gz'

    white_mask = f'HCP_1200/{subj}/MNINonLinear/wmparc.nii.gz'
    localfn_white_mask = f'/home/annatruzzi/hcp_timecourses/{subj}/wmparc.nii.gz'
    
    fmri = f'HCP_1200/{subj}/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz'
    localfn_fmri = f'/home/annatruzzi/hcp_timecourses/{subj}/rfMRI_REST1_LR.nii.gz'
    
    try:
        s3.download_file(hcpbucket, global_mask, localfn_global_mask)     
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f'Not found on s3 bucket:{hcpbucket} key:{global_mask}')
            raise
        else:
            print("Unexpected error: %s" % e)
            raise
    
    try:
        s3.download_file(hcpbucket, white_mask, localfn_white_mask)     
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f'Not found on s3 bucket:{hcpbucket} key:{white_mask}')
            raise
        else:
            print("Unexpected error: %s" % e)
            raise

    try:
        s3.download_file(hcpbucket, fmri, localfn_fmri)     
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f'Not found on s3 bucket:{hcpbucket} key:{fmri}')
            raise
        else:
            print("Unexpected error: %s" % e)
            raise
  
    nrp.step1_createNuisanceRegressors(nproc=8)
    nrp.step2_nuisanceRegression(nproc=5, model='24pXaCompCorXVolterra',spikeReg=False,zscore=False)
