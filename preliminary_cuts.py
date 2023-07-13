import pandas as pd
import time
import numpy as np
import struct
import matplotlib.pyplot as plt
from cr_data_inspection_functions import *
from lwa_antpos import mapping
from scipy.optimize import curve_fit
import scipy.stats as st
import math
import argparse
import yaml


parser=argparse.ArgumentParser(description='Parse raw cosmic ray trigger data files and make preliminary RFI rejection cuts.')
parser.add_argument('fname', type=str, help='File name of raw data file, not including file path (which is specified in config file)')
#TODO bring back this command line option parser.add_argument('config',type=str, help='Full path to configuration file')
args=parser.parse_args()
fname = args.fname
#TODO bring back this option config = args.config
config='/home/ubuntu/kp/lwa-cosmic-rays/config_preliminary_cuts.yml'

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
minsnr=configuration['minsnr']
minimum_ok_kurtosis=configuration['minimum_ok_kurtosis']
maximum_ok_kurtosis=configuration['maximum_ok_kurtosis']

#parameters for event-based cuts
minmaxstrengthratio=configuration['minmaxstrengthratio']#core vs distant ratio
mintop5ratio=configuration['mintop5ratio']#core vs distant ratio
minstrongdetections2=configuration['minstrongdetections']#Setting this equal to 50 replicates Ryan's cut for strong detections
min_power_ratio=configuration['min_power_ratio']#cut for ratio of power before and after event
max_power_ratio=configuration['max_power_ratio']#cut for ratio of power before and after event

#inject simulated events? True or False
simulate=configuration['simulation']
                       
####################################### load data ###########################################################
records = parsefile(fname,end_ind=stop_index) 

if simulate:
    pulse=configuration['pulse']
    veto_thresh=configuration['veto_thresh']
    ok_vetos_fname=configuration['ok_vetos_fname']
    pulse_antennas=configuration['pulse_antennas']
    records=inject_simulation(records,pulse_antennas,pulse,ok_vetos_fname,veto_thresh)
                       
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

##########Calculate Summary info ####################

#parameters for antenna-based cuts
maximum_ok_power=50**2
minimum_ok_power=30**2
minsnr=5
minimum_ok_kurtosis=-1
maximum_ok_kurtosis=1

print('Start time ',time.time())

#arrays to store summary info for each event
power_ratioA=np.zeros(len(complete_events))
power_ratioB=np.zeros(len(complete_events))

n_strong_detectionsA=np.zeros(len(complete_events))
n_strong_detectionsB=np.zeros(len(complete_events))
n_veto_detections=np.zeros(len(complete_events))

sum_top_5_core_vs_far_ratioA=np.zeros(len(complete_events))
sum_top_5_core_vs_far_ratioB=np.zeros(len(complete_events))
med_core_vs_far_ratioB=np.zeros(len(complete_events))
med_core_vs_far_ratioA=np.zeros(len(complete_events))
max_core_vs_far_ratioB=np.zeros(len(complete_events))
max_core_vs_far_ratioA=np.zeros(len(complete_events))
sum_top_10_snrsA=np.zeros(len(complete_events))
sum_top_10_snrsB=np.zeros(len(complete_events))

#for debugging:
minmeanpowerbeforeA=np.zeros(len(complete_events))
minmeanpowerafterA=np.zeros(len(complete_events))
minmeanpowerbeforeB=np.zeros(len(complete_events))
minmeanpowerafterB=np.zeros(len(complete_events))

#go through each event
for i,event_indices in enumerate(complete_events):  
    event=[records[i] for i in event_indices]
    mergedrecords=mergepolarizations(event,arraymapdictionaries,Filter=h)

    xcoords=np.asarray([record['x'] for record in mergedrecords])
    ycoords=np.asarray([record['y'] for record in mergedrecords])
    zcoords=np.asarray([record['z'] for record in mergedrecords])
    
    #get peak and snr
    index_peakA=np.asarray([record['index_peak_A'] for record in mergedrecords])
    index_peakB=np.asarray([record['index_peak_B'] for record in mergedrecords])

    rmsA=np.asarray([record['rmsA'] for record in mergedrecords])
    peakA=np.asarray([record['peaksmoothedA'] for record in mergedrecords])
    meansmoothedA=np.asarray([record['meansmoothedA'] for record in mergedrecords])

    rmsB=np.asarray([record['rmsB'] for record in mergedrecords])
    peakB=np.asarray([record['peaksmoothedB'] for record in mergedrecords])
    meansmoothedB=np.asarray([record['meansmoothedB'] for record in mergedrecords])

    snrA=peakA/meansmoothedA
    snrB=peakB/meansmoothedB
    
    #get kurtosis before event
    kurtosisA=np.asarray([record['kurtosisA'] for record in mergedrecords])
    kurtosisB=np.asarray([record['kurtosisB'] for record in mergedrecords])

    #calculate mean smoothed power after event (from 10 samples after the peak to 60 samples after the peak)
    meansmoothedpowerafterA=np.asarray([record['powerafterpeakA'] for record in mergedrecords]) 
    meansmoothedpowerafterB=np.asarray([record['powerafterpeakB'] for record in mergedrecords]) 
    
    #for debugging: TODO remove from final script
    minmeanpowerbeforeA[i]=meansmoothedA.min()
    minmeanpowerafterA[i]=meansmoothedpowerafterA.min()
    minmeanpowerbeforeB[i]=meansmoothedB.min()
    minmeanpowerafterB[i]=meansmoothedpowerafterB.min()
    
    #define antenna cut based on power 
    cut_powerA = np.logical_and(meansmoothedA >minimum_ok_power, meansmoothedA <maximum_ok_power)
    cut_powerB = np.logical_and(meansmoothedB >minimum_ok_power, meansmoothedB <maximum_ok_power)
    
    #define antenna cut based on kurtosis
    cut_kurtosisA = np.logical_and(kurtosisA >minimum_ok_kurtosis, kurtosisA <maximum_ok_kurtosis)
    cut_kurtosisB = np.logical_and(kurtosisB >minimum_ok_kurtosis, kurtosisB <maximum_ok_kurtosis)
    
    #define cut to reject stuff that found a peak at the end of the buffer -- typically noise
    cut_tpeakA=index_peakA<3944  #last value where the power calculation after the event will work
    cut_tpeakB=index_peakB<3944

    #define antenna cut based on peak to mean square ratio
    cut_snrA=snrA >minsnr
    cut_snrB=snrB >minsnr

    #combine antenna cuts
    cut_power_kurtosis_A=np.logical_and(cut_powerA,cut_kurtosisA)
    cut_power_kurtosis_B=np.logical_and(cut_powerB,cut_kurtosisB)

    total_cutA=np.logical_and(np.logical_and(cut_power_kurtosis_A,cut_snrA),cut_tpeakA)
    total_cutB=np.logical_and(np.logical_and(cut_power_kurtosis_B,cut_snrB),cut_tpeakB)
    
    #calculate the number of strong detections and the median ratio of power before and after
    n_strong_detectionsA[i]=np.sum(total_cutA)
    n_strong_detectionsB[i]=np.sum(total_cutB)
    
    if len(meansmoothedA[total_cutA]):
        power_ratioA[i]=np.median(meansmoothedpowerafterA[total_cutA]/(meansmoothedA[total_cutA]))
    else:
        power_ratioA[i]=-1
    
    if len(meansmoothedB[total_cutB]):
        power_ratioB[i]=np.median(meansmoothedpowerafterB[total_cutB]/(meansmoothedB[total_cutB]))
    else:
        power_ratioB[i]=-1

    ## apply stronger veto  cut than was running at the time
    
    veto_threshold=np.asarray([(record['veto_power_threshold'][0]) for record in mergedrecords])
    select_veto_antennasA=np.asarray([record['veto_role_A'] for record in mergedrecords])
    select_veto_antennasB=np.asarray([record['veto_role_B'] for record in mergedrecords])
    veto_detectionsA=np.logical_and(peakA>veto_threshold,select_veto_antennasA)
    veto_detectionsB=np.logical_and(peakB>veto_threshold,select_veto_antennasB)
    n_veto_detections[i]=np.sum(veto_detectionsA)+np.sum(veto_detectionsB) #count how many of both polarizations detected it
    
    ## calculate ratio of peak/rms in core vs outriggers
    select_core=(xcoords**2)+(ycoords**2)<(150**2)
    select_far_antennas=(xcoords**2)+(ycoords**2)>(250**2)
    
    core_snrA=snrA[np.logical_and(select_core,cut_power_kurtosis_A)]
    far_antennas_snrA=snrA[np.logical_and(select_far_antennas,cut_power_kurtosis_A)]
    core_snrB=snrB[np.logical_and(select_core,cut_power_kurtosis_B)]
    far_antennas_snrB=snrB[np.logical_and(select_far_antennas,cut_power_kurtosis_B)]
    
    if (len(core_snrA)>10) and (len(far_antennas_snrA)>5):
        med_core_vs_far_ratioA[i]=(np.median(core_snrA))/(np.median(far_antennas_snrA))
        max_core_vs_far_ratioA[i]=(np.max(core_snrA))/(np.max(far_antennas_snrA))
        sort_core_snrA=core_snrA.copy()
        sort_core_snrA.sort()
        sort_far_snrA=far_antennas_snrA.copy()
        sort_far_snrA.sort()
        sum_top_5_core_vs_far_ratioA[i]=(np.sum(sort_core_snrA[-5:]))/(np.sum(sort_far_snrA[-5:]))
        sum_top_10_snrsA[i]=np.sum(sort_core_snrA[-10:])
    else:
        med_core_vs_far_ratioA[i]=-1
        max_core_vs_far_ratioA[i]=-1
        sum_top_5_core_vs_far_ratioA[i]=-1
    
    if (len(core_snrB)>10) and (len(far_antennas_snrB)>5):
        med_core_vs_far_ratioB[i]=(np.median(core_snrB))/(np.median(far_antennas_snrB))
        max_core_vs_far_ratioB[i]=(np.max(core_snrB))/(np.max(far_antennas_snrB))
        sort_core_snrB=core_snrB.copy()
        sort_core_snrB.sort()
        sort_far_snrB=far_antennas_snrB.copy()
        sort_far_snrB.sort()
        sum_top_5_core_vs_far_ratioB[i]=(np.sum(sort_core_snrB[-5:]))/(np.sum(sort_far_snrB[-5:]))
        sum_top_10_snrsB[i]=np.sum(sort_core_snrB[-10:])

    else:
        med_core_vs_far_ratioB[i]=-1
        max_core_vs_far_ratioB[i]=-1
        sum_top_5_core_vs_far_ratioB[i]=-1
        
####################################################################
########Calculate Selection Cuts#####################################

#is the event stronger in A polarization or B polarization?
Apolarized=sum_top_10_snrsA>=sum_top_10_snrsB
Bpolarized=sum_top_10_snrsA<=sum_top_10_snrsB

#number of strong detections
total_strong_detections=n_strong_detectionsA+n_strong_detectionsB
detections_cut2=total_strong_detections>minstrongdetections2

#cut on events based on power change. The dominant polarization cut must satisfy the power change cut
power_change_cutA=np.logical_and(power_ratioA>min_power_ratio,power_ratioA<max_power_ratio)
power_change_cutB=np.logical_and(power_ratioB>min_power_ratio,power_ratioB<max_power_ratio)
Apol_events_that_satisfy_powerchange=np.logical_and(power_change_cutA,Apolarized)
Bpol_events_that_satisfy_powerchange=np.logical_and(power_change_cutB,Bpolarized)
power_change_cut=np.logical_or(Apol_events_that_satisfy_powerchange,Bpol_events_that_satisfy_powerchange)

#cut on total number of veto antenna detections
maxveto=1  #throw away things detected by 2 or more veto antennas
veto_cut=n_veto_detections<=maxveto

#cuts for events that are weaker for the distant antennas
Apol_max_distant_v_core=np.logical_and((max_core_vs_far_ratioA>minmaxstrengthratio),Apolarized)
Bpol_max_distant_v_core=np.logical_and((max_core_vs_far_ratioB>minmaxstrengthratio),Bpolarized)
max_distant_vs_core_cut=np.logical_or(Apol_max_distant_v_core,Bpol_max_distant_v_core)

Apol_top5_distant_v_core=np.logical_and(sum_top_5_core_vs_far_ratioA>mintop5ratio,Bpolarized)
Bpol_top5_distant_v_core=np.logical_and(sum_top_5_core_vs_far_ratioB>mintop5ratio,Bpolarized)

top5_distant_vs_core_cut=np.logical_or(Apol_top5_distant_v_core,Bpol_top5_distant_v_core)

distant_vs_core_cut2=np.logical_or(top5_distant_vs_core_cut,max_distant_vs_core_cut) #I'm combining two metrics 

############################Apply Selection Cuts##########################################

total_events_cut1=np.logical_and(np.logical_and(power_change_cut,detections_cut2),top5_distant_vs_core_cut)
select_events1=[complete_events[i] for i in range(len(complete_events)) if total_events_cut1[i]]
n_selected1=len(select_events1)

total_events_cut2=np.logical_and(np.logical_and(np.logical_and(power_change_cut,detections_cut2),distant_vs_core_cut2),veto_cut)
select_events2=[complete_events[i] for i in range(len(complete_events)) if total_events_cut2[i]]
n_selected2=len(select_events2)

###############Save info on cuts################################

print(shortfname, config, total, complete_events_count, incomplete_events_count, scrambled_complete_events,power_change_cut.sum(),veto_cut.sum(),max_distant_vs_core_cut.sum(),top5_distant_vs_core_cut.sum(),distant_vs_core_cut2.sum(), n_selected1,n_selected2 )


np.save(outdir+shortfname[:-3]+'records_per_event',np.asarray([len(e) for e in events]))
np.save(outdir+shortfname[:-3]+'n_strong_detectionsA',n_strong_detectionsA)
np.save(outdir+shortfname[:-3]+'n_strong_detectionsB',n_strong_detectionsB)
np.save(outdir+shortfname[:-3]+'n_veto_detections',n_veto_detections)
np.save(outdir+shortfname[:-3]+'max_core_vs_far_ratioB',max_core_vs_far_ratioB)
np.save(outdir+shortfname[:-3]+'max_core_vs_far_ratioA',max_core_vs_far_ratioA)

np.save(outdir+shortfname[:-3]+'power_ratioA',power_ratioA)
np.save(outdir+shortfname[:-3]+'power_ratioB',power_ratioB)
np.save(outdir+shortfname[:-3]+'sum_top_5_core_vs_far_ratioA',sum_top_5_core_vs_far_ratioA)
np.save(outdir+shortfname[:-3]+'sum_top_5_core_vs_far_ratioB',sum_top_5_core_vs_far_ratioB)


############################ Save Indices of Selected Events ################################
indices_to_save=[]
for e in select_events1:
    if (e==[n for n in range(np.min(e),np.min(e)+704)]):
        indices_to_save.append(e[0])
np.save(outdir+shortfname[:-3]+'indices_cuts1',np.asarray(indices_to_save))

indices_to_save=[]
for e in select_events2:
    if (e==[n for n in range(np.min(e),np.min(e)+704)]):
        indices_to_save.append(e[0])
np.save(outdir+shortfname[:-3]+'indices_cuts2',np.asarray(indices_to_save))
