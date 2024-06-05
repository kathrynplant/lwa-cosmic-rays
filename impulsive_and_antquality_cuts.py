import time
import numpy as np
#import struct
#import matplotlib.pyplot as plt
from cr_data_inspection_functions import *
#from lwa_antpos import reading
#from scipy.optimize import curve_fit
#import scipy.stats as st
#import math
import argparse
import yaml
import os
import numpy.lib.recfunctions as rfn


parser=argparse.ArgumentParser(description='Make preliminary RFI rejection cuts based on output of summarize_events.py.')
parser.add_argument('config',type=str, help='Full path to configuration file')
args=parser.parse_args()
config = args.config

def main():
    starttime=time.time()
    print('Start time ',starttime)
    ############################### set parameters -- read from a config file ######################################
    with open(config, 'r') as file:
        configuration=yaml.safe_load(file)
    #where to save data products
    dataproductsdir=configuration['outdir']
    
    max_ants_bad_kurtosis=configuration['max_ants_bad_kurtosis']
    max_ants_bad_power=configuration['max_ants_bad_power']
    max_ants_saturated=configuration['max_ants_saturated']
    min_pr=configuration['min_pr']
    max_pr=configuration['max_pr']
    fileprefix=configuration['fileprefix']
    nfiles=configuration['nfiles']
    fs=configuration['fs']
    ################################# Load all the summary files and combine into one array #########################
    filelist=[fname for fname in os.listdir(dataproductsdir) if (fname[:2]=='ov' and fname[-3:]=='npy')]
    print(len(filelist), ' summary files to consider')
    arrays_to_merge=[]          
    for i in range(len(filelist)):
        #get the file name of the summary file and of the data file
        summaryfname=filelist[i]
        datafname=summaryfname[:-12]
        #load the summary array
        arr=np.load(dataproductsdir+summaryfname)
        #add a new column with the data file name
        arr2=rfn.append_fields(arr,'datafname',data=np.asarray([datafname]*len(arr)),dtypes='U100',usemask=False)
        #add this array to the list of arrays to merge
        arrays_to_merge.append(arr2)
    
    #merge all the arrays
    merged=np.concatenate(arrays_to_merge)
          
    #print some summary info
    first=np.min(merged['timestamp'])
    last=np.max(merged['timestamp'])
    total_clock_cycles=last-first
    total_seconds=total_clock_cycles/(fs*1e6)
    total_minutes=total_seconds/60
    total_hours=total_minutes/60
    totalevents=len(merged)
    print('observing time in hours: ',round(total_minutes/60,2))
    print('average saved event rate in Hz: ',round(totalevents/total_seconds,2))     
    print('total number of events: ',totalevents)
       
    ################################# Create and Apply cuts #######################################################################
    ok_kurtosis_cut1=merged['n_kurtosis_bad']<max_ants_bad_kurtosis
    ok_power_cut1=merged['n_power_bad']<max_ants_bad_power
    ok_saturation_cut1=merged['n_saturated']<max_ants_saturated
    total_antenna_quality1=combine_cuts([ok_kurtosis_cut1,ok_power_cut1,ok_saturation_cut1])
    impulsive_cut=combine_cuts([merged['power_ratioA']<max_pr,merged['power_ratioB']<max_pr,merged['power_ratioA']>min_pr,merged['power_ratioB']>min_pr])
    selected=merged[combine_cuts([impulsive_cut,total_antenna_quality1])]
    print('number of events passing impulsive cut: ', np.sum(impulsive_cut))      
    print('number of events passing signal quality cuts: ',np.sum(total_antenna_quality1))
    print('number of events passing both cuts: ', len(selected))

    ############################### Save results ###########################################################################
    length=len(selected)
    itemsperfile=(length//nfiles)+1
    print('Saving selected array as ',nfiles, 'files, with ~', itemsperfile, 'rows per file.')
    for i in range(nfiles):
        outfilename=dataproductsdir+fileprefix+'_'+str(i)+'.npy'
        if (i+1)*itemsperfile<length:
            np.save(outfilename,selected[i*itemsperfile:(i+1)*itemsperfile])
        else:
            np.save(outfilename,selected[i*itemsperfile:])

    return

main()
