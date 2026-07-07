#make a new summary file for a subset of events that pass an elevation cut
import time
import numpy as np
import os
import numpy.lib.recfunctions as rfn
import argparse

parser=argparse.ArgumentParser(description='Perform model fits on events specified in a summary file such as that output by summarize_events.py or impulsive_and_antquality_cuts.py. The model fits are a wavefront fit to the observed TOAs, to determine arrival direction parameters, and a Gaussian fit to the spatial distribution of measured SNRs. Results from these fits are appended as additional columns in the summary file.')
parser.add_argument('dataproductsdir',type=str, help='Full path to directory containing the impulsive summary arrays.')
args=parser.parse_args()
dataproductsdir = args.dataproductsdir

#for now I'm not using the endtime option
def load_impulsive_summary(dataproductsdir,endtime):
    #make one summary array from all the impulsive summary files in dataproductsdir, truncating to endtime
    filelist=[fname for fname in os.listdir(dataproductsdir) if (fname[:2]=='fi' and fname[-3:]=='npy')]
    
    if len(filelist):
        arrays_to_merge=[]
        for i in range(len(filelist)):
            #get the file name of the summary file and of the data file
            summaryfname=filelist[i]
            datafname=summaryfname[:-12]
            #load the summary array
            arr=np.load(dataproductsdir+summaryfname)
            #add this array to the list of arrays to merge
            arrays_to_merge.append(arr)

            #merge all the arrays and add a datadir column
            arr = np.concatenate(arrays_to_merge)
            datadir = dataproductsdir[:-14]
            impulsivesummary_raw=rfn.append_fields(arr,'datadir',data=[datadir]*len(arr),dtypes='U100',usemask=False)

            #timecut -- exclude events after 6:30 am local time
            timecut = impulsivesummary_raw['timestamp']<endtime*196*1e6
            impulsivesummary = impulsivesummary_raw[timecut]
    else:
        impulsivesummary=[]
        print('No event summaries available.')
    return impulsivesummary

impulsivesummary = load_impulsive_summary(dataproductsdir,time.time())
if len(impulsivesummary):
    save = impulsivesummary[impulsivesummary['arrival_zenith_angle'] < 75]
    if len(save):
        print('Saving summary file with ', len(save), ' events.')
        np.save(dataproductsdir+'high_el_no_direction_qc.npy',save)
