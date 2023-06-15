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


parser=argparse.ArgumentParser(description='Parse raw cosmic ray trigger data files and make preliminary RFI rejection cuts.')
parser.add_argument('fname', type=str, help='File name of raw data file')
args=parser.parse_args()
fname = args.fname

#### set parameters -- eventually this could be read from a config file ######################################

#where to save data products
outdir='/data0/cosmic-ray-data/2023May3-dataproducts/'
datadir ='/data0/cosmic-ray-data/2023May3/' #path to fname
shortfname=fname[len(datadir):]
#name of csv file with antenna names and coordinates: Columns must have headings 'antname', 'x', 'y', 'elevation'
array_map_filename='/home/ubuntu/kp/lwa-cosmic-rays/array-map-5-22-2023.csv'

#veto setup at the time the data was collected
veto_names=['LWA-316', 'LWA-334', 'LWA-328', 'LWA-326', 'LWA-322', 'LWA-333']
veto_threshold=200 #200 is what was set at the time the data was recorded

#parameters for antenna-based cuts
maximum_ok_rms=45
minimum_ok_rms=25
minsnr=4.472 #calculated to be the raw voltage equivalent of the smoothed power snr threshold that Ryan used 
minimum_ok_kurtosis=-1
maximum_ok_kurtosis=1

#parameters for event-based cuts
minmaxstrengthratio=1.5 #core vs distant ratio
mintop5ratio=1.5 #core vs distant ratio
minstrongdetections1=20 #number of strong detections for an individual polarization
minstrongdetections2=50 #replicate Ryan's cut for strong detections
min_rms_ratio=0.8 #cut for ratio of rms before and after event
max_rms_ratio=1.2 #cut for ratio of rms before and after event

####################################### load data ###########################################################
records = parsefile(fname)


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

#arrays to store summary info for each event
rms_ratioA=np.zeros(len(complete_events))
rms_ratioB=np.zeros(len(complete_events))

n_strong_detectionsA=np.zeros(len(complete_events))
n_strong_detectionsB=np.zeros(len(complete_events))

n_veto_detections=np.zeros(len(complete_events))

max_core_vs_far_ratioB=np.zeros(len(complete_events))
max_core_vs_far_ratioA=np.zeros(len(complete_events))
sum_top_5_core_vs_far_ratioA=np.zeros(len(complete_events))
sum_top_5_core_vs_far_ratioB=np.zeros(len(complete_events))

#go through each event
for i,event_indices in enumerate(complete_events):  
    event=[records[i] for i in event_indices]
    mergedrecords=mergepolarizations(event,arraymapdictionaries)

    xcoords=np.asarray([record['x'] for record in mergedrecords])
    ycoords=np.asarray([record['y'] for record in mergedrecords])
    zcoords=np.asarray([record['z'] for record in mergedrecords])
    
    #get rms and peak to rms ratio
    rmsA=np.asarray([record['rmsA'] for record in mergedrecords])
    peakA=np.asarray([record['peakA'] for record in mergedrecords])
    rmsB=np.asarray([record['rmsB'] for record in mergedrecords])
    peakB=np.asarray([record['peakB'] for record in mergedrecords])

    peak_to_rmsA=peakA/rmsA
    peak_to_rmsB=peakB/rmsB
    
    #get kurtosis before event
    kurtosisA=np.asarray([st.kurtosis(record['polA_data'][:2000]) for record in mergedrecords])
    kurtosisB=np.asarray([st.kurtosis(record['polB_data'][:2000]) for record in mergedrecords])

    #calculate rms after event (from 10 samples after the peak to 60 samples after the peak)
    rms_afterA=np.asarray([np.std(record['polA_data'][record['index_peak_A']+10:record['index_peak_A']+60]) for record in mergedrecords]) #could speed it up by only doing rms after event for good antennas
    rms_afterB=np.asarray([np.std(record['polB_data'][record['index_peak_B']+10:record['index_peak_B']+60]) for record in mergedrecords])
        
    #define antenna cut based on rms 
    cut_rmsA = np.logical_and(rmsA >minimum_ok_rms, rmsA <maximum_ok_rms)
    cut_rmsB = np.logical_and(rmsB >minimum_ok_rms, rmsB <maximum_ok_rms)
    
    #define antenna cut based on kurtosis
    cut_kurtosisA = np.logical_and(kurtosisA >minimum_ok_kurtosis, kurtosisA <maximum_ok_kurtosis)
    cut_kurtosisB = np.logical_and(kurtosisB >minimum_ok_kurtosis, kurtosisB <maximum_ok_kurtosis)

    #define antenna cut based on peak to rms ratio
    cut_snrA=(peakA/rmsA) >minsnr
    cut_snrB=(peakB/rmsB) >minsnr

    #combine antenna cuts
    cut_rms_kurtosis_A=np.logical_and(cut_rmsA,cut_kurtosisA)
    cut_rms_kurtosis_B=np.logical_and(cut_rmsB,cut_kurtosisB)

    total_cutA=np.logical_and(cut_rms_kurtosis_A,cut_snrA)
    total_cutB=np.logical_and(cut_rms_kurtosis_B,cut_snrB)
    
    #calculate the number of strong detections and the median ratio of rms before and after
    n_strong_detectionsA[i]=np.sum(total_cutA)
    n_strong_detectionsB[i]=np.sum(total_cutB)

    if len(rmsA[total_cutA]):
        rms_ratioA[i]=np.median(rms_afterA[total_cutA]/(rmsA[total_cutA]))
    else:
        rms_ratioA[i]=-1
    
    if len(rmsB[total_cutB]):
        rms_ratioB[i]=np.median(rms_afterB[total_cutB]/(rmsB[total_cutB]))
    else:
        rms_ratioB[i]=-1

    ## calculate number of veto detections among all the veto antennas that were used at the time
    select_veto_antennas=np.asarray([record['antname'] in veto_names for record in mergedrecords])
    veto_detectionsA=np.logical_and(peakA>veto_threshold,select_veto_antennas)
    veto_detectionsB=np.logical_and(peakB>veto_threshold,select_veto_antennas)
    n_veto_detections[i]=np.sum(veto_detectionsA)+np.sum(veto_detectionsB) #count how many of both polarizations detected it
    
    ## calculate ratio of peak/rms in core vs outriggers
    select_core=(xcoords**2)+(ycoords**2)<(150**2)
    select_far_antennas=(xcoords**2)+(ycoords**2)>(250**2)
    
    core_peak_rms_ratioA=peak_to_rmsA[np.logical_and(select_core,cut_rms_kurtosis_A)]
    far_antennas_peak_rms_ratioA=peak_to_rmsA[np.logical_and(select_far_antennas,cut_rms_kurtosis_A)]
    core_peak_rms_ratioB=peak_to_rmsB[np.logical_and(select_core,cut_rms_kurtosis_B)]
    far_antennas_peak_rms_ratioB=peak_to_rmsB[np.logical_and(select_far_antennas,cut_rms_kurtosis_B)]
    
    if len(core_peak_rms_ratioA) and len(far_antennas_peak_rms_ratioA):
        max_core_vs_far_ratioA[i]=(np.max(core_peak_rms_ratioA))/(np.max(far_antennas_peak_rms_ratioA))
        sort_core_snrA=core_peak_rms_ratioA.copy()
        sort_core_snrA.sort()
        sort_far_snrA=far_antennas_peak_rms_ratioA.copy()
        sort_far_snrA.sort()
        sum_top_5_core_vs_far_ratioA[i]=(np.sum(sort_core_snrA[-5:]))/(np.sum(sort_far_snrA[-5:]))

    else:
        max_core_vs_far_ratioA[i]=-1
        sum_top_5_core_vs_far_ratioA[i]=-1
    
    if len(core_peak_rms_ratioB) and len(far_antennas_peak_rms_ratioB):
        max_core_vs_far_ratioB[i]=(np.max(core_peak_rms_ratioB))/(np.max(far_antennas_peak_rms_ratioB))
        sort_core_snrB=core_peak_rms_ratioB.copy()
        sort_core_snrB.sort()
        sort_far_snrB=far_antennas_peak_rms_ratioB.copy()
        sort_far_snrB.sort()
        sum_top_5_core_vs_far_ratioB[i]=(np.sum(sort_core_snrB[-5:]))/(np.sum(sort_far_snrB[-5:]))

    else:
        max_core_vs_far_ratioB[i]=-1
        sum_top_5_core_vs_far_ratioB[i]=-1
        
####################################################################
########Calculate Selection Cuts#####################################

#number of strong detections
total_strong_detections=n_strong_detectionsA+n_strong_detectionsB
detections_cut2=total_strong_detections>minstrongdetections2

#cut on events based on rms change and number of strong detections
rms_change_cutA=np.logical_and(rms_ratioA>min_rms_ratio,rms_ratioA<max_rms_ratio)
rms_change_cutB=np.logical_and(rms_ratioB>min_rms_ratio,rms_ratioB<max_rms_ratio)

#Iff a polarization has strong detections, it must satisfy the rms change cut
rms_change_cut_for_strong_detectionsA=np.logical_or(rms_change_cutA,n_strong_detectionsA<minstrongdetections1)
rms_change_cut_for_strong_detectionsB=np.logical_or(rms_change_cutB,n_strong_detectionsB<minstrongdetections1)

#At least one polarization must satisfy above
rms_change_cut=np.logical_and(rms_change_cut_for_strong_detectionsA,rms_change_cut_for_strong_detectionsB)

#cut on total number of veto antenna detections
maxveto=1  #throw away things detected by 2 or more veto antennas
veto_cut=n_veto_detections<=maxveto

#cut for events that are weaker for the distant antennas
max_distant_vs_core_cut=np.logical_or(max_core_vs_far_ratioA>minmaxstrengthratio,max_core_vs_far_ratioB>minmaxstrengthratio)
top5_distant_vs_core_cut=np.logical_or(sum_top_5_core_vs_far_ratioA>mintop5ratio,sum_top_5_core_vs_far_ratioB>mintop5ratio)

distant_vs_core_cut2=np.logical_or(top5_distant_vs_core_cut,max_distant_vs_core_cut) #I'm combining two metrics since they select very non overlapping subsets so far

############################Apply Selection Cuts##########################################

total_events_cut1=np.logical_and(np.logical_and(rms_change_cut,detections_cut2),top5_distant_vs_core_cut)
select_events1=[complete_events[i] for i in range(len(complete_events)) if total_events_cut1[i]]
n_selected1=len(select_events1)

total_events_cut2=np.logical_and(np.logical_and(np.logical_and(rms_change_cut,detections_cut2),distant_vs_core_cut2),veto_cut)
select_events2=[complete_events[i] for i in range(len(complete_events)) if total_events_cut2[i]]
n_selected2=len(select_events2)

###############Save info on cuts################################

print(shortfname, total, complete_events_count, incomplete_events_count, scrambled_complete_events,rms_change_cut.sum(),veto_cut.sum(),max_distant_vs_core_cut.sum(),top5_distant_vs_core_cut.sum(),distant_vs_core_cut2.sum(), n_selected1,n_selected2 )


np.save(outdir+shortfname[:-3]+'records_per_event',np.asarray([len(e) for e in events]))
np.save(outdir+shortfname[:-3]+'n_strong_detectionsA',n_strong_detectionsA)
np.save(outdir+shortfname[:-3]+'n_strong_detectionsB',n_strong_detectionsB)
np.save(outdir+shortfname[:-3]+'n_veto_detections',n_veto_detections)
np.save(outdir+shortfname[:-3]+'max_core_vs_far_ratioB',max_core_vs_far_ratioB)
np.save(outdir+shortfname[:-3]+'max_core_vs_far_ratioA',max_core_vs_far_ratioA)

np.save(outdir+shortfname[:-3]+'rms_ratioA',rms_ratioA)
np.save(outdir+shortfname[:-3]+'rms_ratioB',rms_ratioB)
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
