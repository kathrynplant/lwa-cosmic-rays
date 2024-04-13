import pandas as pd
import time
import numpy as np
import struct
import matplotlib.pyplot as plt
from cr_data_inspection_functions import *
from lwa_antpos import reading
from scipy.optimize import curve_fit
import scipy.stats as st
import math
import argparse
import yaml

parser=argparse.ArgumentParser(description='Parse raw cosmic ray trigger data files and make preliminary RFI rejection cuts.')
parser.add_argument('config',type=str, help='Full path to configuration file')
parser.add_argument('fname', type=str, help='File name of raw data file, not including file path (which is specified in config file)')
args=parser.parse_args()
fname = args.fname 
config = args.config

def main():
    starttime=time.time()
    print('Start time ',starttime)
    ############################### set parameters -- read from a config file ######################################
    with open(config, 'r') as file:
        configuration=yaml.safe_load(file)
    #where to save data products
    outdir=configuration['outdir']
    datadir =configuration['datadir'] 
    shortfname=fname[len(datadir):]
    stop_index=configuration['stop_index']
    #name of csv file with antenna names and coordinates: Columns must have headings 'antname', 'x', 'y', 'elevation'
    array_map_filename=configuration['array_map_filename'] 

    #FIR Filter coefficients
    h=np.asarray(configuration['filter'])

    #parameters for antenna-based cuts
    maximum_ok_power=configuration['maximum_ok_power'] 
    minimum_ok_power=configuration['minimum_ok_power']
    minimum_ok_kurtosis=configuration['minimum_ok_kurtosis']
    maximum_ok_kurtosis=configuration['maximum_ok_kurtosis']
    max_saturated_samples=configuration['maximum_ok_kurtosis']
    known_bad_antennas=configuration['known_bad_antennas']
    
    #inject simulated events? True or False
    simulate=configuration['simulation']

    ########################## load array map ################################
    array_map=pd.read_csv(array_map_filename)
    xdict={}
    ydict={}
    zdict={}
    for i,n in enumerate(array_map['antname']):
        xdict[n]=array_map['x'][i]
        ydict[n]=array_map['y'][i]
        zdict[n]=array_map['elevation'][i]
    arraymapdictionaries=[xdict,ydict,zdict]

    lwa_df = reading.read_antpos()
    namedict=build_mapping_dictionary(lwa_df)

    ####################################### load data ###########################################################
    records = parsefile(fname,end_ind=stop_index) 

    if simulate:
        pulse=configuration['pulse']
        veto_thresh=configuration['veto_thresh']
        ok_vetos_fname=configuration['ok_vetos_fname']
        pulse_antennas=configuration['pulse_antennas']
        records=inject_simulation(records,pulse_antennas,pulse,ok_vetos_fname,veto_thresh,namedict)

    ###################### Organize list of single-antenna records into list of events ############################################
    events=distinguishevents(records,200)
    complete_events=[event for event in events if len(event)==704]

    total=len(events)
    records_per_event=np.asarray([len(e) for e in events])
    complete_events_count=np.sum(records_per_event==704)
    incomplete_events_count=np.sum(records_per_event!=704)

    scrambled_complete_events=0
    for e in complete_events:
        if (e!=[n for n in range(np.min(e),np.min(e)+704)]):
            scrambled_events+=1
    
   ##########Calculate Summary info ####################

    #array to store summary info for each event
    datatypes=dtype={'names':('index_in_file','timestamp','n_good_antennas',
                              'n_saturated','n_kurtosis_bad','n_power_bad','power_ratioA','power_ratioB',
                             'max_core_vs_far_ratio','sum_top_5_core_vs_far_ratio','sum_top_10_core_vs_far_ratio',
                              'meansnr_nearby','meansnr_nearbyA','meansnr_nearbyB',
                              'meansnr','meansnrA','meansnrB'),
                              'formats':(np.intc, np.uint64, np.uintc,
                                         np.uintc,np.uintc,np.uintc,np.single,np.single,
                                         np.single,np.single,np.single,
                                        np.single,np.single,np.single,
                                        np.single,np.single,np.single)}
    summarray = np.zeros(len(complete_events), dtype=datatypes)
    #go through each event
    for j,event_indices in enumerate(complete_events):  
        event=[records[i] for i in event_indices]
        #get summary statistics for each antenna signal
        antenna_summary=summarize_signals(event,h,namedict,xdict,ydict,zdict,details=False)
        #apply per-antenna signal quality cuts
        antenna_summary_flagged,n_saturated,n_kurtosis_bad,n_power_bad=flag_antennas(antenna_summary,                                                                                 maximum_ok_power,                                                                                minimum_ok_power,minimum_ok_kurtosis,                                                                       maximum_ok_kurtosis,                                                                             max_saturated_samples,                                                                                     known_bad_antennas)
        
        n_good_antennas=len(antenna_summary_flagged)
        #get more summary statistics for whole event
        if len(antenna_summary_flagged):
            power_ratioA,power_ratioB,max_core_vs_far_ratio,sum_top_5_core_vs_far_ratio,sum_top_10_core_vs_far_ratio,meansnr_nearby,meansnr_nearbyA,meansnr_nearbyB,meansnr,meansnrA,meansnrB=summarize_event(antenna_summary_flagged)
        else:
            power_ratioA,power_ratioB,max_core_vs_far_ratio,sum_top_5_core_vs_far_ratio,sum_top_10_core_vs_far_ratio,meansnr_nearby,meansnr_nearbyA,meansnr_nearbyB,meansnr,meansnrA,meansnrB=0,0,0,0,0,0,0,0,0,0,0,0
        if (event_indices==[n for n in range(np.min(event_indices),np.min(event_indices)+704)]):
            index_in_file=(event_indices[0])
        else:
            index_in_file=-1e6
        timestamp=event[0]['timestamp']

        summarray[j]=(index_in_file,timestamp,n_good_antennas,
                              n_saturated,n_kurtosis_bad,n_power_bad,power_ratioA,power_ratioB,
                             max_core_vs_far_ratio,sum_top_5_core_vs_far_ratio,sum_top_10_core_vs_far_ratio,
                              meansnr_nearby,meansnr_nearbyA,meansnr_nearbyB,
                              meansnr,meansnrA,meansnrB)
    #save summary array
    np.save(outdir+shortfname+'.summary',summarray)
    #print file summarry
    print(shortfname, config, total, complete_events_count, incomplete_events_count, scrambled_complete_events)
    return

main()
