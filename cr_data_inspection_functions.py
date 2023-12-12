#!/usr/bin/env python
import os
import pandas as pd
import time
import numpy as np
import struct
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
import scipy.stats as st
from scipy import signal
from lwa_antpos import mapping

### functions for building fast way to lookup antenna names
#@profile
def lookup_antname(lwa_df,snap,inp):
    #Returns the antenna name string given an array configuration dataframe, a snap board number, and an input number.
    #adapted from lwa_antpos.mapping.snap2_to_antpol
    #lwa_df is the pandas dataframe object describing the array configuration, as obtained from lwa_antpos.reading.read_antpos_etcd()
    #snap is the snap board number, from [1,11]
    #inp is the input index, from [0,63]
    sel=np.where((lwa_df['snap2_location']==snap)&((lwa_df['pola_fpga_num']==inp)|(lwa_df['polb_fpga_num']==inp)))
    if len(sel[0]) != 1:
        print('Did not find exactly one antpol.')
        return
    elif lwa_df.iloc[sel]['pola_fpga_num'][0]==inp:
        return lwa_df.iloc[sel].index.to_list()[0] + 'A'
    else:
        return lwa_df.iloc[sel].index.to_list()[0] + 'B'
#@profile
def build_mapping_dictionary(lwa_df):
    fpgamappingdictionary={}
    for s in range(1,12):
        for i  in range(0,64):
            fpgamappingdictionary['s'+str(s)+'i'+str(i)]=lookup_antname(lwa_df,s,i)
    return fpgamappingdictionary

#@profile
def lookup_antname_in_dictionary(namedict,snap,inp):
    return namedict['s'+str(snap)+'i'+str(inp)]
        
#functions for loading and parsing data
def printheader(rawpacketdata):
    #parse the header
    #From MSB to LSB the header is:
    #telescope time: 8 bytes to be interpreted as a 64-bit integer
    headerword=rawpacketdata[-32:]
    print('telescope time',headerword[0:8])

    # 5 bytes zero padding
    print('zero padding',headerword[8:13])

    # 1 byte board id
    print('board id', headerword[13])

    # 1 byte this board triggered
    print('this board triggered', headerword[14])

    # 1 byte coincidence threshold
    print('coincidence threshold', headerword[15])

    # 1 byte veto threshold
    print('veto threshold', headerword[16])

    # 4 bytes (32-bit integer) power threshold for trigger antennas
    print('trigger power threshold', headerword[17:21])

    # 4 bytes (32-bit integer) power threshold for veto antennas
    print('veto power threshold', headerword[21:25])

    # 2 bytes (16 bit integer) coincidence window
    print('coincidence window',headerword[25:27])

    # 2 bytes (16 bit integer) veto window
    print('veto coincidence window',headerword[27:29])

    # 1 byte antenna id
    print('antenna id', headerword[29])

    # 1 byte trigger role
    print('trigger role', headerword[30])

    # 1 byte veto role
    print('veto role',headerword[31])
    return

#@profile
def parseheader(rawpacketdata):
    #rawpacketdata is the entire raw data payload previously extracted from a cosmic ray udp packet
    #returns a dictionary containing the metadata from the packet header, which it should be noted is the last 256 bits of the UDP data payload
    headerword=rawpacketdata[-32:]

    headerdictionary={}
    headerdictionary['timestamp'] = struct.unpack('>Q',headerword[0:8])[0]
    headerdictionary['board_id'] = int(headerword[13])
    headerdictionary['this_board_triggered'] =  int(headerword[14])
    headerdictionary['coincidence_threshold'] =  int(headerword[15])
    headerdictionary['veto_threshold'] =  int(headerword[16])
    headerdictionary['trigger_power_threshold'] = struct.unpack('>I',headerword[17:21])
    headerdictionary['veto_power_threshold'] = struct.unpack('>I',headerword[21:25])
    headerdictionary['coincidence_window'] = struct.unpack('>H',headerword[25:27])
    headerdictionary['veto_coincidence_window'] = struct.unpack('>H',headerword[27:29])
    headerdictionary['antenna_id'] =  int(headerword[29])
    headerdictionary['trigger_role'] = int(headerword[30])
    headerdictionary['veto_role'] = int(headerword[31])
    return headerdictionary

#@profile
def read_in_chunks(file_object, chunk_size):
    """Lazy function (generator) to read a file piece by piece.
    taken from https://stackoverflow.com/questions/519633/lazy-method-for-reading-big-file-in-python"""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data

#@profile
def parsefile(fname, start_ind=None, end_ind=None):
    #get data and header from a file (fname) with raw data from an arbitrary number of cosmic ray packets
    #returns a list of dictionaries, with one dictionary per packet
    #the data key contains a numpy array of the timeseries and the other keys are header info as specified by the parseheader function
    packet_size = 8192
    if os.path.getsize(fname)%packet_size: 
        print("File contains incomplete packets")
        return
    records = []
    i = 0
    if start_ind == None or start_ind < 0:
        start_ind = 0
    byte_start = start_ind * packet_size
    with open(fname, mode="rb") as datafile:   #read the file
        datafile.seek(byte_start)
        for piece in read_in_chunks(datafile, packet_size):
            data_dict = parseheader(piece)
            data_dict['data'] =np.frombuffer(piece[:-32],dtype='>h')[16:-32]
            records.append(data_dict)
            i+=1
            if i == end_ind:
                break
    return records

#@profile
def packet_ant_id_2_snap_input(i):
    #This function returns the snap input number corresponding to a packet with integer i in the antenna_id field in the packet header
    #someday the need for this remapping could probably be addressed in the firmware
    return (i&0b111000)+((i+1)&0b000111)

#@profile
def distinguishevents(records,maxoffset):
    #This function turns a list of single-antenna records into a list of events.
    #records is a list of single-antenna records, such as that returned by parsefile.
    #maxoffset is the maximum timestamp difference (in number of clockcycles) for two records to be considered part of the same event.
    #The function distinguish events returns a list of events, where each event is a list of indices of the records (single-antenna dictionaries) that belong to that event.

    #start an empty list which will ultimately have one element per event
    events=[]

    currentevent_indices=[]

    currenteventtimestamp=records[0]['timestamp']

    maxtimestamp=currenteventtimestamp+maxoffset #all
    mintimestamp=currenteventtimestamp-maxoffset
    for i,record in enumerate(records):
        recordtimestamp=record['timestamp']
        if mintimestamp<recordtimestamp<maxtimestamp:
            #print(recordtimestamp, maxtimestamp,True)
            #currentevent_records.append(record)
            currentevent_indices.append(i)
        else: #start a new event
            #print(recordtimestamp, maxtimestamp,False)
            #eventcount+=1
            events.append(currentevent_indices)
            #currentevent_records=[record]
            currentevent_indices=[i]

            currenteventtimestamp=record['timestamp']
            maxtimestamp=currenteventtimestamp+maxoffset
    events.append(currentevent_indices)
    return events

#@profile
def summarize_signals(event,Filter,namedict,xdict,ydict,zdict):
    #make an array that will have one row per antenna signal, and one column per element of summary info
    datatypes={'names':('antname','pol','x', 'y', 'z','distance','index_raw_peak','index_smooth_peak',
                        'raw_peak','smooth_peak','mean_power','mean_power_after','powerratio','kurtosis','snr',
                        'veto_power_threshold','veto_role','nsaturate'),
                          'formats':('U10', 'U10',np.single,np.single,np.single,np.single,np.uintc, np.uintc,np.intc,
                                     np.single,np.single,np.single,np.single,np.single,np.single,np.uint,np.uintc,np.int)}
    single_event_summarray=np.zeros(len(event), dtype=datatypes)
    
    #loop over all the records in the event, calculating the summary info
    average_window=(1/4)*np.ones(4)

    for r,record in enumerate(event):
        record['antname']=lookup_antname_in_dictionary(namedict,record['board_id'],packet_ant_id_2_snap_input(record['antenna_id']))
        antname=record['antname'][:-1]
        pol=record['antname'][-1]
        x=xdict[antname]
        y=ydict[antname]
        z=zdict[antname]
        distance=(x**2)+(y**2)
        data = record['data'].astype(np.int32)
        index_raw_peak = np.argmax(data)
        raw_peak = data[index_raw_peak]
        nsaturate = np.sum(np.abs(data)>510)
        veto_power_threshold = record['veto_power_threshold'][0]
        veto_role = record['veto_role']
        if type(Filter)==np.ndarray:
            data=signal.convolve(data,Filter,mode='valid')
        powertimeseries=np.square(data)
        smoothed=signal.convolve(powertimeseries,average_window,mode='valid')
        index_smooth_peak = np.argmax(smoothed)
        smooth_peak = smoothed[index_smooth_peak]
        mean_power=np.mean(smoothed[:2000])
        if index_smooth_peak<3944:
            mean_power_after = np.mean(smoothed[index_smooth_peak+10:index_smooth_peak+60])
        else:
            mean_power_after = 0
        powerratio = mean_power_after/mean_power
        kurtosis=st.kurtosis(data[:2000])
        snr=smooth_peak/mean_power
        #save the summary info from this record
        single_event_summarray[r]=(antname,pol,x,y,z,distance,index_raw_peak,index_smooth_peak,raw_peak,
                                  smooth_peak,mean_power,mean_power_after,powerratio,kurtosis,snr,veto_power_threshold,veto_role,nsaturate)
    #return the structured array of summary info from all the antennas
    return single_event_summarray

def flag_antennas(antenna_summary,maximum_ok_power,minimum_ok_power, minimum_ok_kurtosis,maximum_ok_kurtosis):
    flagged=antenna_summary[antenna_summary['mean_power']<maximum_ok_power]
    flagged=flagged[flagged['mean_power']>minimum_ok_power]
    flagged=flagged[flagged['kurtosis']<maximum_ok_kurtosis]
    flagged=flagged[flagged['index_smooth_peak']<3944]  #last value where valid mean_power_after can be calculated
    flagged=flagged[flagged['nsaturate']<10]
    return flagged

def summarize_event(antenna_summary_array):
    #make selection to separate A and B polarizations
    Bpol_cut=antenna_summary_array['pol']=='B'
    Apol_cut=antenna_summary_array['pol']=='A'

    #compare before and after ratio
    if np.sum(Apol_cut)>0:
        power_ratioA=np.median(antenna_summary_array['powerratio'][Apol_cut])
    else:
        power_ratioA=0
    if np.sum(Bpol_cut)>0:
        power_ratioB=np.median(antenna_summary_array['powerratio'][Bpol_cut])
    else:
        power_ratioB=0

    #how many veto antennas detect the event
    peak_exceeds_veto_threshold=antenna_summary_array['raw_peak']>antenna_summary_array['veto_power_threshold']
    veto_antennas=antenna_summary_array['veto_role']==1
    n_veto_detections=np.sum(np.logical_and(peak_exceeds_veto_threshold,veto_antennas))

    #compare power in core to power in distant antennas
    select_core=antenna_summary_array['distance']<(150**2)
    select_far=antenna_summary_array['distance']>(250**2)
    core_snrs=antenna_summary_array[select_core]['snr']
    far_snrs=antenna_summary_array[select_far]['snr']
    ranked_core_snrs=np.sort(np.copy(core_snrs))
    ranked_far_snrs=np.sort(np.copy(far_snrs))

    max_core_vs_far_ratio=ranked_core_snrs[-1]/ranked_far_snrs[-1]
    if len(ranked_core_snrs)> 10:
        sum_top_5_core_vs_far_ratio=np.sum(ranked_core_snrs[-5:])/np.sum(ranked_far_snrs[-5:])
        sum_top_10_core_vs_far_ratio=np.sum(ranked_core_snrs[-10:])/np.sum(ranked_far_snrs[-10:])
    else:
        sum_top_5_core_vs_far_ratio=0
        sum_top_10_core_vs_far_ratio=0
        
    return power_ratioA,power_ratioB,n_veto_detections,max_core_vs_far_ratio,sum_top_5_core_vs_far_ratio,sum_top_10_core_vs_far_ratio




#@profile
def mergepolarizations(event,arraymapdictionary,namedict,Filter='None'):
    #this function takes single-polarization dictionaries such as that output by parsefile and merges polarization pairs into a single dictionary for each antenna stand
    #event is a list of records, in the format output by parsefile, that all belong with one event
    #arraymap dictionary contains antenna position information
    #namedict is a dictionary of antenna names as output by the build_mapping_dictionary function

    #Filter can be None or a 1D numpy array of coefficients for a time-domain FIR. If filter is not none, the timeseries will be convolved with the provided coefficients during the mergepolarizations function.

    xdict,ydict,zdict=arraymapdictionary
    average_window=(1/4)*np.ones(4)

    for record in event:
        record['antname']=lookup_antname_in_dictionary(namedict,record['board_id'],packet_ant_id_2_snap_input(record['antenna_id']))

    merging=[record for record in event if record['antname'][-1]=='A']
    polB=[record for record in event if record['antname'][-1]=='B']

    mergedrecords=[]
    for record in merging:
        newrecord={} #make a new dictionary to merge the polarizations together
        # Information that is the same for both polarizations can be filled in from the PolA record
        antname=record['antname'][:-1]
        newrecord['antname']=antname
        newrecord['x']=xdict[antname]
        newrecord['y']=ydict[antname]
        newrecord['z']=zdict[antname]
        newrecord['timestamp']=record['timestamp']
        newrecord['board_id'] = record['board_id']
        newrecord['this_board_triggered'] =  record['this_board_triggered']
        newrecord['coincidence_threshold'] =  record['coincidence_threshold'] 
        newrecord['veto_threshold'] =   record['veto_threshold'] 
        newrecord['trigger_power_threshold'] = record['trigger_power_threshold']
        newrecord['veto_power_threshold'] = record['veto_power_threshold']
        newrecord['coincidence_window'] = record['coincidence_window']
        newrecord['veto_coincidence_window'] = record['veto_coincidence_window'] 

        #Information that is specific to POl A
        newrecord['trigger_role_A'] = record['trigger_role']
        newrecord['antenna_id_A'] =  record['antenna_id'] 
        newrecord['veto_role_A'] = record['veto_role']
        Adata = record['data'].astype(np.int32)
        if Filter!='None':
            Adata=signal.convolve(Adata,Filter,mode='valid')
        newrecord['polA_data']=Adata
        newrecord['rmsA']=np.std(Adata[:2000])
        newrecord['kurtosisA']=st.kurtosis(Adata[:2000])
        powertimeseriesA=np.square(Adata)
        smoothedA=signal.convolve(powertimeseriesA,average_window,mode='valid')
        newrecord['meansmoothedA']=np.mean(smoothedA[:2000])
        index_peak_A=np.argmax(smoothedA)
        newrecord['index_peak_A'] =index_peak_A
        newrecord['peaksmoothedA']=smoothedA[index_peak_A]
        newrecord['powerafterpeakA']=np.mean(smoothedA[index_peak_A+10:index_peak_A+60])
        #find the polB data
        for Brecord in polB:
            if Brecord['antname'][:-1]==antname:
                #Information that is specific to POlB
                newrecord['trigger_role_B'] = Brecord['trigger_role']
                newrecord['antenna_id_B'] =  Brecord['antenna_id'] 
                newrecord['veto_role_B'] = Brecord['veto_role']
                Bdata = Brecord['data'].astype(np.int32)
                if Filter!='None':
                    Bdata=signal.convolve(Bdata,Filter,mode='valid')
                newrecord['polB_data']=Bdata
                newrecord['rmsB']=np.std(Bdata[:2000]) #use only the first half to calculate the rms, before the event starts
                newrecord['kurtosisB']=st.kurtosis(Bdata[:2000])
                powertimeseriesB=np.square(Bdata)
                smoothedB=signal.convolve(powertimeseriesB,average_window,mode='valid')
                newrecord['meansmoothedB']=np.mean(smoothedB[:2000])
                index_peak_B=np.argmax(smoothedB)
                newrecord['index_peak_B'] =index_peak_B
                newrecord['peaksmoothedB']=np.abs(smoothedB[index_peak_B])       
                newrecord['powerafterpeakB']=np.mean(smoothedB[index_peak_B+10:index_peak_B+60])

        mergedrecords.append(newrecord)
    return mergedrecords

def inject_simulation(records,pulse_antennas,pulse,ok_vetos_fname,veto_thresh,namedict):
    #Simulate an event by adding a delta function pulse to the timeseries for certain antennas
    #This is designed to add pulses to data from untriggered snapshots, as a quick test of selection cuts 
    #records is a list of single-antenna records such as that output by parsefile
    #pulse_antennas is the list of antenna names (specifying which polarization) to add the pulse to
    #pulse is the pulse height
    #ok_vetos_fname is the file name of a numpy file with an array indicating which signals are veto antennas (same format as used by the code to run the detector)
    #veto_thresh is the desired veto threshold to use
    #returns a list of records that have the pulses added to them
    #namedict is a dictionary of antenna names as output by the build_mapping_dictionary function
    ok_vetos=np.load(ok_vetos_fname)
    for r in records:
        snap=r['board_id']
        snapinput=packet_ant_id_2_snap_input(r['antenna_id'])
        antname=lookup_antname_in_dictionary(namedict,snap,snapinput)
        if ok_vetos[snap-1,snapinput]:
            r['veto_role']=1
            r['veto_power_threshold']=[veto_thresh]
        if antname in pulse_antennas:
            data=r['data']
            sample=data[2500]
            if np.abs(sample+pulse)<512:
                data[2500]=sample+pulse
            else:
                data[2500]=512
    return records


### Arrival direction fitting
def toa_plane(ant_coords,theta,phi):
    #This calculates the arrival times at each antenna of a plane wave moving across the array
    #The TOAs are returned in number of clock cycles relative to the arrival time at the zero,zero,zero coordinate
    #theta and phi in degrees, coordinates in meters
    #theta is the angle between the source direction and zenith. Theta=0 for a zenith source
    #phi is the azimuth angle of the source
    c=3e8
    sample_rate=1.97e8 #MHz
    phi_rad=(phi*math.pi/180)
    theta_rad=theta*math.pi/180
    x,y,z=ant_coords
    
    #calculate cartesian unit vector in the direction of the source
    yhat=math.sin(theta_rad)*math.cos(phi_rad)
    xhat=math.sin(theta_rad)*math.sin(phi_rad)
    zhat=math.cos(theta_rad)
    
    #project all the antenna coordinates into the source direction
    dot_product=((xhat*x)+(yhat*y)+(zhat*z))
    
    #convert distance to a time offset in number of clock cycles
    time_diff=(sample_rate/c)*dot_product  
    return time_diff

def grad_toa_plane(ant_coords,theta,phi):
    #This is the gradient of toa_plane w.r.t theta and phi
    c=3e8
    sample_rate=1.97e8 #MHz
    phi_rad=(phi*math.pi/180)
    theta_rad=theta*math.pi/180
    x,y,z=ant_coords

    dtdtheta=(math.pi/180)*(sample_rate/c)*((y*math.cos(theta_rad)*math.cos(phi_rad))+(x*math.cos(theta_rad)*math.cos(phi_rad))-(z*math.sin(theta_rad)))
    dtdphi=(math.pi/180)*(sample_rate/c)*((-y*math.sin(theta_rad)*math.sin(phi_rad)) +(x*math.sin(theta_rad)*math.cos(phi_rad)))
    return np.asarray([dtdtheta,dtdphi]).transpose()

def rank_by_snr(event,arraymapdictionaries,namedict,minimum_ok_rms=25,maximum_ok_rms=45,minimum_ok_kurtosis=-1,maximum_ok_kurtosis=1,Filter='None'):
    # Return a list of antenna names and snrs in order from strongest snr to smallest , in separate rankings for each polarization and for core and distant antennas
    #Event is a list of records (single-packet dictionaries) belonging to the same event
    #Only antennas with signals (in the first half of the buffer) that satisfy the specified rms and kurtosis cuts are included in the ranking
    #Filter can be None or a 1D numpy array of coefficients for a time-domain FIR. If filter is not none, the timeseries will be convolved with the provided coefficients during the mergepolarizations function.
    #namedict is a dictionary of antenna names as output by the build_mapping_dictionary function

    mergedrecords=mergepolarizations(event,arraymapdictionaries,namedict,Filter)
    
    xcoords=np.asarray([record['x'] for record in mergedrecords])
    ycoords=np.asarray([record['y'] for record in mergedrecords])
    zcoords=np.asarray([record['z'] for record in mergedrecords])
    antnames=[record['antname'] for record in mergedrecords]
    #get snr ratio
    noiseA=np.asarray([record['meansmoothedA'] for record in mergedrecords])
    peakA=np.asarray([record['peaksmoothedA'] for record in mergedrecords])
    noiseB=np.asarray([record['meansmoothedB'] for record in mergedrecords])
    peakB=np.asarray([record['peaksmoothedB'] for record in mergedrecords])

    snrA=peakA/noiseA
    snrB=peakB/noiseB
    
    #get kurtosis before event
    kurtosisA=np.asarray([record['kurtosisA'] for record in mergedrecords])
    kurtosisB=np.asarray([record['kurtosisB'] for record in mergedrecords])

    #define antenna cut based on rms 
    cut_rmsA = np.logical_and(noiseA >minimum_ok_rms**2, noiseA <maximum_ok_rms**2)
    cut_rmsB = np.logical_and(noiseB >minimum_ok_rms**2, noiseB <maximum_ok_rms**2)
    
    #define antenna cut based on kurtosis
    cut_kurtosisA = np.logical_and(kurtosisA >minimum_ok_kurtosis, kurtosisA <maximum_ok_kurtosis)
    cut_kurtosisB = np.logical_and(kurtosisB >minimum_ok_kurtosis, kurtosisB <maximum_ok_kurtosis)

    #combine antenna cuts
    cutA=np.logical_and(cut_rmsA,cut_kurtosisA)
    cutB=np.logical_and(cut_rmsB,cut_kurtosisB)
    
    select_core_antennas=(xcoords**2)+(ycoords**2)<(150**2)
    select_far_antennas=(xcoords**2)+(ycoords**2)>(250**2)  #note the core vs far cuts used in plotting are different than what's used for estimating if event is concentrated on core
    
    #apply antenna quality cuts
    snrA_good_core=snrA[np.logical_and(cutA,select_core_antennas)]    
    ants_good_core_A=[antnames[i]+'A' for i in range(len(antnames)) if (cutA[i] and select_core_antennas[i]) ]
    snrB_good_core=snrB[np.logical_and(cutB,select_core_antennas)]
    ants_good_core_B=[antnames[i]+'B' for i in range(len(antnames)) if (cutB[i] and select_core_antennas[i]) ]
    snrA_good_far=snrA[np.logical_and(cutA,select_far_antennas)]
    ants_good_far_A=[antnames[i]+'A' for i in range(len(antnames)) if (cutA[i] and select_far_antennas[i])]
    snrB_good_far=snrB[np.logical_and(cutB,select_far_antennas)]
    ants_good_far_B=[antnames[i]+'B' for i in range(len(antnames)) if (cutB[i] and select_far_antennas[i])]

    # sort by snr and take the top 5 in each category
    
    ranked_core_A_pol=[pair for pair in sorted(zip(snrA_good_core,ants_good_core_A),reverse=True)]
    ranked_core_B_pol=[pair for pair in sorted(zip(snrB_good_core,ants_good_core_B),reverse=True)]
    ranked_far_A_pol=[pair for pair in sorted(zip(snrA_good_far,ants_good_far_A),reverse=True)]
    ranked_far_B_pol=[pair for pair in sorted(zip(snrB_good_far,ants_good_far_B),reverse=True)]
    
    return ranked_core_A_pol,ranked_core_B_pol,ranked_far_A_pol,ranked_far_B_pol

########################## snapshot plotting functions  ##########################################################

def plot_timeseries(event,antenna_names,zoom='peak',Filter='None'):
    #Event is a list of records (single-packet dictionaries) belonging to the same event
    #antennas is a list where each element in the list is a tuple of format (s,a) where s is the index of the snap board and a is the index of the antenna to plot
    #If a requested antenna to plot is not in the list (which happens if that packet has been lost), the missing antenna is skipped
    #The requested antennas are plotted in the order they appear in event, not in the order of the input list
    #zoom is either the string 'peak' or a tuple specifying the range of samples to restrict the x axis to
    #Filter can be None or a 1D numpy array of coefficients for a time-domain FIR. If filter is not 'None', the timeseries will be convolved with the provided coefficients.
    for record in event:
        s=record['board_id']
        a=packet_ant_id_2_snap_input(record['antenna_id'])
        antname=mapping.snap2_to_antpol(s,a)
        if antname in antenna_names:
            timeseries=record['data']
            if Filter!='None':
                timeseries=signal.convolve(timeseries,Filter,mode='valid')
            rms=np.std(timeseries[:2000])
            kurtosis=st.kurtosis(timeseries[:2000])
            peak=np.max(np.abs(timeseries))
            plt.figure(figsize=(20,5))
            plt.suptitle(antname + ' snap input '+ str(s) +','+ str(a) + ' peak='+str(peak)+' rms='+str(round(rms,3))+' kurtosis='+str(round(kurtosis,3))+' peak/rms='+str(round(peak/rms,3)))
            
            plt.subplot(121)
            plt.plot(timeseries)
            plt.xlabel('time sample')
            plt.ylabel('voltage [ADC units]')

            plt.subplot(122)
            plt.plot(timeseries)
            plt.xlabel('time sample')
            plt.ylabel('voltage [ADC units]')
            if zoom=='peak':
                plt.xlim(np.argmax(np.abs(timeseries))-50,np.argmax(np.abs(timeseries))+150)
            else:
                plt.xlim(zoom[0],zoom[1])            
    return

def plot_spectra(event,antenna_names,zoom='peak',Filter='None'):
    #Event is a list of records (single-packet dictionaries) belonging to the same event
    #Antennas is a list where each element in the list is a tuple of format (s,a) where s is the index of the snap board and a is the index of the antenna to plot
    #If a requested antenna to plot is not in the list (which happens if that packet has been lost), the missing antenna is skipped
    #The requested antennas are plotted in the order they appear in event, not in the order of the input list
    #Filter can be None or a 1D numpy array of coefficients for a time-domain FIR. If filter is not 'None', the timeseries will be convolved with the provided coefficients.
    fs=196.30822126189432
    for record in event:
        s=record['board_id']
        a=record['antenna_id']
        antname=mapping.snap2_to_antpol(s,a)
        if antname in antenna_names:
            timeseries=record['data']
            if Filter!='None':
                timeseries=signal.convolve(timeseries,Filter,mode='valid')
            rms=np.std(timeseries[:2000])
            kurtosis=st.kurtosis(timeseries[:2000])
            peak=np.max(np.abs(timeseries))
            plt.figure(figsize=(20,5))
            plt.title(antname + ' snap input '+ str(s) +','+ str(a) + ' peak='+str(peak)+' rms='+str(round(rms,3))+' kurtosis='+str(round(kurtosis,3))+' peak/rms='+str(round(peak/rms,3)))
            spec = np.square(np.abs(np.fft.rfft(timeseries)))
            f = np.linspace(0, fs/2, len(spec))
            plt.plot(f, 10*np.log(spec), '.-')
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Relative Power [dB]")
           
            
    return

def plot_power_timeseries(event,antenna_names,zoom='peak',Filter1='None',Filter2='None'):
    #Plots the square of the voltage timeseries for the selected antennas, optionally filtered and smoothed
    #Event is a list of records (single-packet dictionaries) belonging to the same event
    #Antennas is a list where each element in the list is a tuple of format (s,a) where s is the index of the snap board and a is the index of the antenna to plot
    #If a requested antenna to plot is not in the list (which happens if that packet has been lost), the missing antenna is skipped
    #The requested antennas are plotted in the order they appear in event, not in the order of the input list
    #Filter1 and Filter2 each can be 'None' or a 1D numpy array of coefficients for a time-domain FIR. If filter is not 'None', the timeseries will be convolved with the provided coefficients.
    #Filter1 is applied to the raw voltage timeseries
    #Filter2 is a smoothing kernel to apply AFTER squaring the voltage timeseries to obtain a power timeseries
    for record in event:
        s=record['board_id']
        a=packet_ant_id_2_snap_input(record['antenna_id'])
        antname=mapping.snap2_to_antpol(s,a)
        if antname in antenna_names:
            timeseries=record['data']
            
            rms=np.std(timeseries[:2000])
            kurtosis=st.kurtosis(timeseries[:2000])
            peak=np.max(np.abs(timeseries))

            if Filter1!='None':
                timeseries=signal.convolve(timeseries,Filter1,mode='valid')
            timeseries=np.square(timeseries.astype(single))
            if Filter2!='None':
                timeseries=signal.convolve(timeseries,Filter2,mode='valid')
            powerpeak=np.max(timeseries)
            powermean=np.mean(timeseries[:2000])

           
            plt.figure(figsize=(20,5))
            plt.suptitle(antname + ' snap input '+ str(s) +','+ str(a) + ' peak='+str(peak)+' rms='+str(round(rms,3))+' kurtosis='+str(round(kurtosis,3))+' peak/rms='+str(round(peak/rms,3))+'power peak'+str(round(powerpeak,3)) +'power mean'+str(round(powerpeak,3)) +'snr'+str(round(powerpeak/powermean,3)))
            
            plt.subplot(121)
            plt.plot(timeseries)
            plt.xlabel('time sample')
            plt.ylabel('voltage [ADC units]')

            plt.subplot(122)
            plt.plot(timeseries)
            plt.xlabel('time sample')
            plt.ylabel('voltage [ADC units]')
            if zoom=='peak':
                plt.xlim(np.argmax(timeseries)-50,np.argmax(timeseries)+150)
            else:
                plt.xlim(zoom[0],zoom[1])
    return

def plot_event_snr(event,arraymapdictionaries,namedict,minimum_ok_rms=25,maximum_ok_rms=45,minimum_ok_kurtosis=-1,maximum_ok_kurtosis=1,annotate=False,Filter='None'):
    #cmin=0
    #cmax=35
    #Plots the snr for each polarization of an event, over the antenna positions of the array
    #snr is defined as peak of smoothed power divided by mean smoothed power
    #Event is a list of records (single-packet dictionaries) belonging to the same event
    #Antennas are filtered to only plot antennas whose signals (in the first half of the buffer) are within the
    #bounds set by minimum_ok_rms, maximum_ok_rms, minimum_ok_kurtosis, maximum_ok_kurtosis
    #Antennas are labelled if annotate=True
    #Filter can be None or a 1D numpy array of coefficients for a time-domain FIR. If filter is not none, the timeseries will be convolved with the provided coefficients during the mergepolarizations function.
    #namedict is a dictionary of antenna names as output by the build_mapping_dictionary function

    mergedrecords=mergepolarizations(event,arraymapdictionaries,namedict,Filter)

    xcoords=np.asarray([record['x'] for record in mergedrecords])
    ycoords=np.asarray([record['y'] for record in mergedrecords])
    zcoords=np.asarray([record['z'] for record in mergedrecords])
    antnames=[record['antname'] for record in mergedrecords]

    #get snr ratio
    noiseA=np.asarray([record['meansmoothedA'] for record in mergedrecords])
    peakA=np.asarray([record['peaksmoothedA'] for record in mergedrecords])
    noiseB=np.asarray([record['meansmoothedB'] for record in mergedrecords])
    peakB=np.asarray([record['peaksmoothedB'] for record in mergedrecords])

    snrA=peakA/noiseA
    snrB=peakB/noiseB
    
    #get kurtosis before event
    kurtosisA=np.asarray([record['kurtosisA'] for record in mergedrecords])
    kurtosisB=np.asarray([record['kurtosisB'] for record in mergedrecords])

    #define antenna cut based on rms 
    cut_rmsA = np.logical_and(noiseA >minimum_ok_rms**2, noiseA <maximum_ok_rms**2)
    cut_rmsB = np.logical_and(noiseB >minimum_ok_rms**2, noiseB <maximum_ok_rms**2)
    
    #define antenna cut based on kurtosis
    cut_kurtosisA = np.logical_and(kurtosisA >minimum_ok_kurtosis, kurtosisA <maximum_ok_kurtosis)
    cut_kurtosisB = np.logical_and(kurtosisB >minimum_ok_kurtosis, kurtosisB <maximum_ok_kurtosis)

    #combine antenna cuts
    cutA=np.logical_and(cut_rmsA,cut_kurtosisA)
    cutB=np.logical_and(cut_rmsB,cut_kurtosisB)
    
    select_core_antennas=(xcoords**2)+(ycoords**2)<(115**2)
    select_far_antennas=(xcoords**2)+(ycoords**2)>(115**2)  #note the core vs far cuts used in plotting are different than what's used for estimating if event is concentrated on core
    
    sizescale=5
    plt.figure(figsize=(15,15))
    plt.suptitle('Signal to noise ratio, good antennas only')
    plt.subplot(221)
    plt.title("Polarization A ")
    plt.scatter(xcoords[cutA],ycoords[cutA],c=snrA[cutA],s=sizescale*(snrA[cutA]))
    plt.colorbar()
    #plt.clim(0,35)
    plt.ylabel('North-South position [m]')
    if annotate:
        for i in range(len(antnames)):
            if cutA[i] and select_far_antennas[i]:
                txt=antnames[i]
                x=xcoords[i]
                y=ycoords[i]
                plt.annotate(txt[3:], (x, y))
    
    plt.subplot(222)
    plt.title("Polarization A --zoom in")
    plt.scatter(xcoords[cutA],ycoords[cutA],c=snrA[cutA],s=sizescale*(snrA[cutA]))
    plt.xlim(-105,105)
    plt.ylim(-105,105)
    plt.colorbar(label='snr')
   # plt.clim(cmin,cmax)

    #plt.clim(cmin,cmax)
    if annotate:
        for i in range(len(antnames)):
            if cutA[i] and select_core_antennas[i]:
                txt=antnames[i]
                x=xcoords[i]
                y=ycoords[i]
                plt.text(x,y,txt[3:],fontsize='x-small')
                
    plt.subplot(223)
    plt.title("Polarization B ")
    plt.scatter(xcoords[cutB],ycoords[cutB],c=snrB[cutB],s=sizescale*(snrB[cutB]))
    plt.colorbar()
   # plt.clim(cmin,cmax)

    plt.xlabel('East-West position [m]')
    plt.ylabel('North-South position [m]')
    if annotate:
        for i in range(len(antnames)):
            if cutB[i] and select_far_antennas[i]:
                txt=antnames[i]
                x=xcoords[i]
                y=ycoords[i]
                plt.annotate(txt[3:], (x, y))

    plt.subplot(224)
    plt.title("Polarization B -- zoom in")
    plt.scatter(xcoords[cutB],ycoords[cutB],c=snrB[cutB],s=sizescale*(snrB[cutB]))
    plt.xlim(-105,105)
    plt.ylim(-105,105)
    plt.colorbar(label='snr')
    plt.xlabel('East-West position [m]')
   # plt.clim(cmin,cmax)

    if annotate:
        for i in range(len(antnames)):
            if cutB[i] and select_core_antennas[i]:
                txt=antnames[i]
                x=xcoords[i]
                y=ycoords[i]
                plt.text(x,y,txt[3:],fontsize='x-small')

    return

def plot_event_total_power_snr(event,arraymapdictionaries,namedict,minimum_ok_rms=25,maximum_ok_rms=45,minimum_ok_kurtosis=-1,maximum_ok_kurtosis=1,annotate=False,Filter='None'):
    #cmin=0
    #cmax=35
    #Plots the snr over the antenna positions of the array
    #snr is defined as peak of smoothed power divided by mean smoothed power, and power timeseries of both polarizations are summed before smoothing
    #Event is a list of records (single-packet dictionaries) belonging to the same event
    #Antennas are filtered to only plot antennas whose signals (in the first half of the buffer) are within the
    #bounds set by minimum_ok_rms, maximum_ok_rms, minimum_ok_kurtosis, maximum_ok_kurtosis
    #Antennas are labelled if annotate=True
    #namedict is a dictionary of antenna names as output by the build_mapping_dictionary function
    #Filter can be None or a 1D numpy array of coefficients for a time-domain FIR. If filter is not none, the timeseries will be convolved with the provided coefficients during the mergepolarizations function.
    mergedrecords=mergepolarizations(event,arraymapdictionaries,namedict,Filter)

    xcoords=np.asarray([record['x'] for record in mergedrecords])
    ycoords=np.asarray([record['y'] for record in mergedrecords])
    zcoords=np.asarray([record['z'] for record in mergedrecords])
    antnames=[record['antname'] for record in mergedrecords]
    
    #get snr ratio
    average_window=0.25*np.ones(4)
    snr=np.zeros(len(mergedrecords))
    for i,record in enumerate(mergedrecords):
        #note that in the mergedrecords output the timeseries has already been filtered
        Adata=record['polA_data']
        Bdata=record['polB_data']
        powertimeseriesA=np.square(Adata.astype(np.int32))
        powertimeseriesB=np.square(Bdata.astype(np.int32))
        powertimeseries=0.5*(powertimeseriesA+powertimeseriesB)
        smoothed=signal.convolve(powertimeseries,average_window,mode='valid')
        noise=np.mean(smoothed[:2000])
        peak=np.max(smoothed)
        snr[i]=peak/noise
    
    #get kurtosis before event
    kurtosisA=np.asarray([record['kurtosisA'] for record in mergedrecords])
    kurtosisB=np.asarray([record['kurtosisB'] for record in mergedrecords])

    #define antenna cut based on rms 
    noiseA=np.asarray([record['meansmoothedA'] for record in mergedrecords])
    noiseB=np.asarray([record['meansmoothedB'] for record in mergedrecords])
    cut_rmsA = np.logical_and(noiseA >minimum_ok_rms**2, noiseA <maximum_ok_rms**2)
    cut_rmsB = np.logical_and(noiseB >minimum_ok_rms**2, noiseB <maximum_ok_rms**2)
    
    #define antenna cut based on kurtosis
    cut_kurtosisA = np.logical_and(kurtosisA >minimum_ok_kurtosis, kurtosisA <maximum_ok_kurtosis)
    cut_kurtosisB = np.logical_and(kurtosisB >minimum_ok_kurtosis, kurtosisB <maximum_ok_kurtosis)

    #combine antenna cuts
    cutA=np.logical_and(cut_rmsA,cut_kurtosisA)
    cutB=np.logical_and(cut_rmsB,cut_kurtosisB)
    cut_total=np.logical_and(cutA,cutB)
    
    select_core_antennas=(xcoords**2)+(ycoords**2)<(115**2)
    select_far_antennas=(xcoords**2)+(ycoords**2)>(115**2)  #note the core vs far cuts used in plotting are different than what's used for estimating if event is concentrated on core
    
    sizescale=5
    plt.figure(figsize=(15,15))
    plt.suptitle('Signal to noise ratio in polarization-combined timeseries')
    plt.subplot(221)
    #plt.title("Polarization A ")
    plt.scatter(xcoords[cut_total],ycoords[cut_total],c=snr[cut_total],s=sizescale*(snr[cut_total]))
    plt.colorbar()
    #plt.clim(0,35)
    plt.ylabel('North-South position [m]')
    if annotate:
        for i in range(len(antnames)):
            if cut_total[i] and select_far_antennas[i]:
                txt=antnames[i]
                x=xcoords[i]
                y=ycoords[i]
                plt.annotate(txt[3:], (x, y))
    
    plt.subplot(222)
    #plt.title("Polarization A --zoom in")
    plt.scatter(xcoords[cut_total],ycoords[cut_total],c=snr[cut_total],s=sizescale*(snr[cut_total]))
    plt.xlim(-105,105)
    plt.ylim(-105,105)
    plt.colorbar(label='snr')
   # plt.clim(cmin,cmax)

    #plt.clim(cmin,cmax)
    if annotate:
        for i in range(len(antnames)):
            if cut_total[i] and select_core_antennas[i]:
                txt=antnames[i]
                x=xcoords[i]
                y=ycoords[i]
                plt.text(x,y,txt[3:],fontsize='x-small')
                
    return

def plot_event_snr_polaverage(event,arraymapdictionaries,namedict,minimum_ok_rms=25,maximum_ok_rms=45,minimum_ok_kurtosis=-1,maximum_ok_kurtosis=1,annotate=False,Filter='None'):
    #cmin=0
    #cmax=35
    #Plots the average of the pol A and pol B snrs, over the antenna positions of the array
    #snr is defined as peak of smoothed power divided by mean smoothed power
    #Event is a list of records (single-packet dictionaries) belonging to the same event
    #Antennas are filtered to only plot antennas whose signals (in the first half of the buffer) are within the
    #bounds set by minimum_ok_rms, maximum_ok_rms, minimum_ok_kurtosis, maximum_ok_kurtosis
    #Antennas are labelled if annotate=True
    #namedict is a dictionary of antenna names as output by the build_mapping_dictionary function
    #Filter can be None or a 1D numpy array of coefficients for a time-domain FIR. If filter is not none, the timeseries will be convolved with the provided coefficients during the mergepolarizations function.
    mergedrecords=mergepolarizations(event,arraymapdictionaries,namedict,Filter)

    xcoords=np.asarray([record['x'] for record in mergedrecords])
    ycoords=np.asarray([record['y'] for record in mergedrecords])
    zcoords=np.asarray([record['z'] for record in mergedrecords])
    antnames=[record['antname'] for record in mergedrecords]
        
    #get snr ratio
    noiseA=np.asarray([record['meansmoothedA'] for record in mergedrecords])
    peakA=np.asarray([record['peaksmoothedA'] for record in mergedrecords])
    noiseB=np.asarray([record['meansmoothedB'] for record in mergedrecords])
    peakB=np.asarray([record['peaksmoothedB'] for record in mergedrecords])

    snrA=peakA/noiseA
    snrB=peakB/noiseB
    snr=(snrA+snrB)/2
    
    #get kurtosis before event
    kurtosisA=np.asarray([record['kurtosisA'] for record in mergedrecords])
    kurtosisB=np.asarray([record['kurtosisB'] for record in mergedrecords])

    #define antenna cut based on rms 
    cut_rmsA = np.logical_and(noiseA >minimum_ok_rms**2, noiseA <maximum_ok_rms**2)
    cut_rmsB = np.logical_and(noiseB >minimum_ok_rms**2, noiseB <maximum_ok_rms**2)
    
    #define antenna cut based on kurtosis
    cut_kurtosisA = np.logical_and(kurtosisA >minimum_ok_kurtosis, kurtosisA <maximum_ok_kurtosis)
    cut_kurtosisB = np.logical_and(kurtosisB >minimum_ok_kurtosis, kurtosisB <maximum_ok_kurtosis)

    #combine antenna cuts
    cutA=np.logical_and(cut_rmsA,cut_kurtosisA)
    cutB=np.logical_and(cut_rmsB,cut_kurtosisB)
    cut_total=np.logical_and(cutA,cutB)
    
    select_core_antennas=(xcoords**2)+(ycoords**2)<(115**2)
    select_far_antennas=(xcoords**2)+(ycoords**2)>(115**2)  #note the core vs far cuts used in plotting are different than what's used for estimating if event is concentrated on core
    
    sizescale=5
    plt.figure(figsize=(15,15))
    plt.suptitle('Signal to noise ratio, mean of both polarizations')
    plt.subplot(221)
    #plt.title("Polarization A ")
    plt.scatter(xcoords[cut_total],ycoords[cut_total],c=snr[cut_total],s=sizescale*(snr[cut_total]))
    plt.colorbar()
    #plt.clim(0,35)
    plt.ylabel('North-South position [m]')
    if annotate:
        for i in range(len(antnames)):
            if cut_total[i] and select_far_antennas[i]:
                txt=antnames[i]
                x=xcoords[i]
                y=ycoords[i]
                plt.annotate(txt[3:], (x, y))
    
    plt.subplot(222)
    #plt.title("Polarization A --zoom in")
    plt.scatter(xcoords[cut_total],ycoords[cut_total],c=snr[cut_total],s=sizescale*(snr[cut_total]))
    plt.xlim(-105,105)
    plt.ylim(-105,105)
    plt.colorbar(label='snr')
   # plt.clim(cmin,cmax)

    #plt.clim(cmin,cmax)
    if annotate:
        for i in range(len(antnames)):
            if cut_total[i] and select_core_antennas[i]:
                txt=antnames[i]
                x=xcoords[i]
                y=ycoords[i]
                plt.text(x,y,txt[3:],fontsize='x-small')
                
    return

def plot_event_toas(event,arraymapdictionaries,namedict,minimum_ok_rms=25,maximum_ok_rms=45,minimum_ok_kurtosis=-1,maximum_ok_kurtosis=1,annotate=False,Filter='None'):
    #Plots the time of arrival for each antenna and polarization of an event, over the antenna positions of the array
    #time of arrival is in units of clock cycles with respect to the earliest packet timestamp in the event
    #Event is a list of records (single-packet dictionaries) belonging to the same event
    #Antennas are filtered to only plot antennas whose signals (in the first half of the buffer) are within the
    #bounds set by minimum_ok_rms, maximum_ok_rms, minimum_ok_kurtosis, maximum_ok_kurtosis
    #Antennas are labelled if annotate=True
    #namedict is a dictionary of antenna names as output by the build_mapping_dictionary function
    #Filter can be None or a 1D numpy array of coefficients for a time-domain FIR. If filter is not none, the timeseries will be convolved with the provided coefficients.
    
    mergedrecords=mergepolarizations(event,arraymapdictionaries,namedict,Filter)

    xcoords=np.asarray([record['x'] for record in mergedrecords])
    ycoords=np.asarray([record['y'] for record in mergedrecords])
    zcoords=np.asarray([record['z'] for record in mergedrecords])
    antnames=[record['antname'] for record in mergedrecords]
    #get rms 
    rmsA=np.asarray([(record['meansmoothedA'])**0.5 for record in mergedrecords])
    rmsB=np.asarray([(record['meansmoothedB'])**0.5 for record in mergedrecords])

    peakA=np.asarray([record['peaksmoothedA'] for record in mergedrecords])
    peakB=np.asarray([record['peaksmoothedB'] for record in mergedrecords])

    snrA=peakA/(rmsA**2)
    snrB=peakB/(rmsB**2)
    
    #get time of peak
    index_peak_A=np.asarray([record['index_peak_A'] for record in mergedrecords])
    index_peak_B=np.asarray([record['index_peak_B'] for record in mergedrecords])
    timestamps=np.asarray([record['timestamp'] for record in mergedrecords])
    min_time=np.min(timestamps)

    t_rel_A=index_peak_A + timestamps - min_time
    t_rel_B=index_peak_B + timestamps - min_time
    
    
    #get kurtosis before event
    kurtosisA=np.asarray([record['kurtosisA'] for record in mergedrecords])
    kurtosisB=np.asarray([record['kurtosisB'] for record in mergedrecords])

    #define antenna cut based on rms 
    cut_rmsA = np.logical_and(rmsA >minimum_ok_rms, rmsA <maximum_ok_rms)
    cut_rmsB = np.logical_and(rmsB >minimum_ok_rms, rmsB <maximum_ok_rms)
    
    #define antenna cut based on kurtosis
    cut_kurtosisA = np.logical_and(kurtosisA >minimum_ok_kurtosis, kurtosisA <maximum_ok_kurtosis)
    cut_kurtosisB = np.logical_and(kurtosisB >minimum_ok_kurtosis, kurtosisB <maximum_ok_kurtosis)

    #combine antenna cuts
    cutA=np.logical_and(cut_rmsA,cut_kurtosisA)
    cutB=np.logical_and(cut_rmsB,cut_kurtosisB)

    
    select_core_antennas=(xcoords**2)+(ycoords**2)<(115**2)
    select_far_antennas=(xcoords**2)+(ycoords**2)>(115**2)  #note the core vs far cuts used in plotting are different than what's used for estimating if event is concentrated on core
    
    sizescale=5
    plt.figure(figsize=(15,15))
    plt.suptitle('Time of Peak, good antennas only')
    plt.subplot(221)
    plt.title("Polarization A ")
    plt.scatter(xcoords[cutA],ycoords[cutA],c=t_rel_A[cutA],s=sizescale*(snrA[cutA]))
    plt.colorbar()
    plt.clim(np.median(t_rel_B[cutB])-700,np.median(t_rel_B[cutB])+700)
    plt.ylabel('North-South position [m]')
    if annotate:
        for i in range(len(antnames)):
            if cutA[i] and select_far_antennas[i]:
                txt=antnames[i]
                x=xcoords[i]
                y=ycoords[i]
                plt.annotate(txt[3:], (x, y))
    
    plt.subplot(222)
    plt.title("Polarization A --zoom in")
    plt.scatter(xcoords[cutA],ycoords[cutA],c=t_rel_A[cutA],s=sizescale*(snrA[cutA]))
    plt.xlim(-105,105)
    plt.ylim(-105,105)
    plt.colorbar(label='time sample')
    plt.clim(np.median(t_rel_A[cutA])-75,np.median(t_rel_A[cutA])+75)
    if annotate:
        for i in range(len(antnames)):
            if cutA[i] and select_core_antennas[i]:
                txt=antnames[i]
                x=xcoords[i]
                y=ycoords[i]
                plt.text(x,y,txt[3:],fontsize='x-small')
                
    plt.subplot(223)
    plt.title("Polarization B ")
    plt.scatter(xcoords[cutB],ycoords[cutB],c=t_rel_B[cutB],s=sizescale*(snrB[cutB]))
    plt.colorbar()
    plt.clim(np.median(t_rel_B[cutB])-700,np.median(t_rel_B[cutB])+700)

    plt.xlabel('East-West position [m]')
    plt.ylabel('North-South position [m]')
    if annotate:
        for i in range(len(antnames)):
            if cutB[i] and select_far_antennas[i]:
                txt=antnames[i]
                x=xcoords[i]
                y=ycoords[i]
                plt.annotate(txt[3:], (x, y))

    plt.subplot(224)
    plt.title("Polarization B -- zoom in")
    plt.scatter(xcoords[cutB],ycoords[cutB],c=t_rel_B[cutB],s=sizescale*(snrB[cutB]))
    plt.xlim(-105,105)
    plt.ylim(-105,105)
    plt.colorbar(label='time sample')
    plt.clim(np.median(t_rel_B[cutB])-75,np.median(t_rel_B[cutB])+75)
    plt.xlabel('East-West position [m]')
    
    if annotate:
        for i in range(len(antnames)):
            if cutB[i] and select_core_antennas[i]:
                txt=antnames[i]
                x=xcoords[i]
                y=ycoords[i]
                plt.text(x,y,txt[3:],fontsize='x-small')
    return

def plot_all_timeseries(event):
    for b in range(11):
        singleboard=[record for record in event if record['board_id']==b+1]
        fig=plt.figure(figsize=(20,15))
        plt.suptitle("Board "+str(b+1)+" timestamp "+str(singleboard[0]['timestamp']))
        for i in range(64):
            ax=fig.add_subplot(8,8,1+i)
            if i<len(singleboard):
                record=singleboard[i]
                antenna=packet_ant_id_2_snap_input(record['antenna_id']) #Get the snap2 input number
                antname=mapping.snap2_to_antpol(b,antenna) 

                timeseries=record['data']
                plt.plot(timeseries)
                ax.text(.5,.5,antname,horizontalalignment='center',transform=ax.transAxes)
            if i > 55:
                plt.xlabel('time sample')
            if i%8==0:
                plt.ylabel('voltage [ADC units]')
    return


def plot_all_spectra(event):
    for b in range(11):
        singleboard=[record for record in event if record['board_id']==b+1]
        fig=plt.figure(figsize=(20,15))
        plt.suptitle("Board "+str(b+1)+" timestamp "+str(singleboard[0]['timestamp']))
        for i in range(64):
            ax=fig.add_subplot(8,8,1+i)
            if i<len(singleboard):
                record=singleboard[i]
                antenna=packet_ant_id_2_snap_input(record['antenna_id']) #Get the snap2 input number
                antname=mapping.snap2_to_antpol(b,antenna)
                timeseries=record['data']
                spec=np.fft.rfft(timeseries)
                plt.plot(np.log(np.square(np.abs(spec))))
                ax.text(.5,.5,antname,horizontalalignment='center',transform=ax.transAxes)
            if i > 55:
                plt.xlabel('frequency channel')
            if i%8==0:
                plt.ylabel('power')
    return

def plot_all_histograms(event):
    for b in range(11):
        singleboard=[record for record in event if record['board_id']==b+1]
        fig=plt.figure(figsize=(20,15))
        plt.suptitle("Board "+str(b+1)+" timestamp "+str(singleboard[0]['timestamp']))
        for i in range(64):
            ax=fig.add_subplot(8,8,1+i)
            if i<len(singleboard):
                record=singleboard[i]
                antenna=packet_ant_id_2_snap_input(record['antenna_id']) #Get the snap2 input number
                antname=mapping.snap2_to_antpol(b,antenna)

                timeseries=record['data']
                plt.hist(timeseries)
                ax.text(.5,.5,antname,horizontalalignment='center',transform=ax.transAxes)

            if i > 55:
                plt.xlabel('Voltage [ADC units]')
            if i%8==0:
                plt.ylabel('Counts')
    return

def plot_select_antennas(event,antennas):
    #Event is a list of records (single-packet dictionaries) belonging to the same event
    #antennas is a list where each element in the list is a tuple of format (s,a) where s is the index of the snap board and a is the index of the antenna to plot
    #If a requested antenna to plot is not in the list (which happens if that packet has been lost), the missing antenna is skipped
    #The requested antennas are plotted in the order they appear in event, not in the order of the input list
    for record in event:
        s=record['board_id']
        a=packet_ant_id_2_snap_input(record['antenna_id'])
        antname=mapping.snap2_to_antpol(s,a)
        if antname in antennas:
            timeseries=record['data']
            plt.figure(figsize=(20,5))
            plt.suptitle(antname + ' snap '+ str(s) + ' antenna ' + str(a))
            
            plt.subplot(131)
            plt.plot(timeseries)
            plt.xlabel('time sample')
            plt.ylabel('voltage [ADC units]')

            plt.subplot(132)
            plt.hist(timeseries)
            plt.xlabel('Voltage [ADC units]')
            plt.ylabel('Counts')

            plt.subplot(133)
            plt.xlabel('frequency channel')
            plt.ylabel('power')
            spec=np.fft.rfft(timeseries)
            plt.plot(np.log(np.square(np.abs(spec))))     
    return


def plot_peak_to_rms_ratio(peak_to_rmsA,cutA,peak_to_rmsB,cutB,xcoords,ycoords,cmin,cmax):
    #Do I still need this as well as plot_event_peak_to_rms??
    plt.figure(figsize=(10,10))
    plt.suptitle('Ratio of Peak absolute value to RMS, good antennas only')
    plt.subplot(221)
    plt.title("Polarization A ")
    plt.scatter(xcoords[cutA],ycoords[cutA],c=peak_to_rmsA[cutA])
    plt.colorbar()
    plt.clim(cmin,cmax)
    plt.ylabel('North-South position [m]')

    plt.subplot(222)
    plt.title("Polarization A --zoom in")
    plt.scatter(xcoords[cutA],ycoords[cutA],c=peak_to_rmsA[cutA])
    plt.xlim(-200,200)
    plt.ylim(-200,200)
    plt.colorbar(label='ADC units')
    plt.clim(cmin,cmax)

    plt.subplot(223)
    plt.title("Polarization B ")
    plt.scatter(xcoords[cutB],ycoords[cutB],c=peak_to_rmsB[cutB])
    plt.colorbar()
    plt.clim(cmin,cmax)
    plt.xlabel('East-West position [m]')
    plt.ylabel('North-South position [m]')

    plt.subplot(224)
    plt.title("Polarization B -- zoom in")
    plt.scatter(xcoords[cutB],ycoords[cutB],c=peak_to_rmsB[cutB])
    plt.xlim(-200,200)
    plt.ylim(-200,200)
    plt.colorbar(label='ADC units')
    plt.clim(cmin,cmax)
    plt.xlabel('East-West position [m]')
    return


def plot_fit(x,y,toa_data,best_model_toas,residual,czoom_min,czoom_max,title):
    plt.figure(figsize=(15,10))
    plt.suptitle(title)
    plt.subplot(231)
    plt.axes='equal'
    plt.scatter(x,y,c=toa_data)
    plt.colorbar()
    plt.title('Observed relative TOAs')

    plt.subplot(232)
    plt.axes='equal'
    plt.scatter(x,y,c=best_model_toas)
    plt.colorbar()
    plt.title('Best fit model toas')

    plt.subplot(233)
    plt.axes='equal'
    plt.scatter(x,y,c=residual)
    plt.colorbar()
    plt.title('Residual')

    plt.subplot(234)
    plt.axes='equal'
    plt.scatter(x,y,c=toa_data)
    plt.colorbar()
    plt.xlim(-200,200)
    plt.ylim(-200,200)
    plt.clim(czoom_min,czoom_max)
    plt.title('Observed relative TOAs')

    plt.subplot(235)
    plt.axes='equal'
    plt.scatter(x,y,c=best_model_toas)
    plt.colorbar()
    plt.xlim(-200,200)
    plt.ylim(-200,200)
    plt.clim(czoom_min,czoom_max)
    plt.title('Best fit model toas')

    plt.subplot(236)
    plt.axes='equal'
    plt.scatter(x,y,c=residual)
    plt.colorbar()
    plt.xlim(-200,200)
    plt.ylim(-200,200)
    #plt.clim(czoom_min,czoom_max)
    plt.title('Residual')


    
### NOTE THAT THIS FUNCTION IS FOR DATA FORMAT FROM OLD PACKETIZER I'm leaving it here in case old-format data ever needs to be used in commissioning
def single_board_snapshot_summary_plots(fname,boardnumber):
    #plot spectra
    fbins=np.linspace(0,197/2,int(1+4096/2))
    chanmap=np.loadtxt('channelmap.txt')
    snapshot=np.load(fname)


    fig1 = plt.figure(figsize=(20,15))
    for i in range(64):
        ax=fig1.add_subplot(8,8,1+i)
        fpgachan=chanmap[1,i]
        antname=mapping.snap2_to_antpol(boardnumber,fpgachan)
        ax.text(.5,.1,antname,horizontalalignment='center',transform=ax.transAxes)

        spec=np.fft.rfft(snapshot[:,i+4])
        plt.plot(fbins,np.log(np.square(np.abs(spec))))
        plt.ylim(0,25)
        if i > 55:
            plt.xlabel('frequency [MHz]')
        if i%8==0:
            plt.ylabel('power')

    fbins=np.linspace(0,197/2,int(1+4096/2))
    #isnormal=np.zeros(64)  

    #plot histogram
    fig2 = plt.figure(figsize=(20,15))
    for i in range(64):
        ax=fig2.add_subplot(8,8,1+i)
        #ax.text(.5,.5,int(chanmap[1,i]),horizontalalignment='center',transform=ax.transAxes)
        fpgachan=chanmap[1,i]
        antname=mapping.snap2_to_antpol(boardnumber,fpgachan)
        ax.text(.5,.5,antname,horizontalalignment='center',transform=ax.transAxes)


        plt.hist(snapshot[:,i+4])
        #isnormal[i] = st.normaltest(snapshot[:,i+4])[1]
        if i > 55:
            plt.xlabel('voltage [ADC units]')
        if i%8==0:
            plt.ylabel('Counts')
        #plt.xlim(-200,200)

    #plot timeseries
    fig3 = plt.figure(figsize=(20,15))
    for i in range(64):
        ax=fig3.add_subplot(8,8,1+i)
        #plt.title(i)
        plt.plot(snapshot[:,i+4])
        fpgachan=chanmap[1,i]
        antname=mapping.snap2_to_antpol(boardnumber,fpgachan)
        ax.text(.5,.5,antname,horizontalalignment='center',transform=ax.transAxes)
        if i > 55:
            plt.xlabel('time sample')
        if i%8==0:
            plt.ylabel('voltage [ADC units]')
    return
