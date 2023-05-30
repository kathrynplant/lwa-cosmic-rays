#!/usr/bin/env python
import os
import pandas as pd
import time
import numpy as np
import struct
import matplotlib.pyplot as plt
from lwa_antpos import mapping  #TODO: install lwa_antpos on Delphinium

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

def unpackdata(rawpacketdata,datatype):
    #parse raw bytes into a numpy array
    #pairs of bytes are interpreted as 16 bit integers, and the last 32 bytes are omitted as they are the header
    #raw packet data is the data payload of a UDP packet from the cosmic ray firmware, including the header word
    #datatype can be signed or unsigned and big endian or little endian, but must be a 16-bit data type. This argument was only for debugging
    nbytes=len(rawpacketdata)
    unpackeddata=np.zeros(int((nbytes-32)/2),dtype=datatype)
    for i in range(int((nbytes-32)/2)):
        value=struct.unpack(datatype,rawpacketdata[(2*i):(2*i +2)]) #read a pair of bytes as an integer. Last 32 bytes are header
        unpackeddata[i]=value[0] 
    return unpackeddata

def read_in_chunks(file_object, chunk_size):
    """Lazy function (generator) to read a file piece by piece.
    taken from https://stackoverflow.com/questions/519633/lazy-method-for-reading-big-file-in-python"""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data

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
            data_dict['data'] = unpackdata(piece, '>h')[16:-32]
            records.append(data_dict)
            i+=1
            if i == end_ind:
                break
    return records

def packet_ant_id_2_snap_input(i):
    #This function returns the snap input number corresponding to a packet with integer i in the antenna_id field in the packet header
    #someday the need for this remapping could probably be addressed in the firmware
    return (i&0b111000)+((i+1)&0b000111)

def distinguishevents(records,maxoffset):
    #This function turns a list of single-antenna records, into a list of events.
    #records is a list of single-antenna records, such as that returned by parsefile.
    #maxoffset is the maximum timestamp difference (in number of clockcycles) for two records to be considered part of the same event.
    #The function distinguish events returns a list of events, where each event is a list of indices of the records (single-antenna dictionaries) that belong to that event.

    #start an empty list which will ultimately have one element per event
    events=[]
    #eventcount=1  #keep track of how many separate events there are

    #start an list for the first event. The first record is the first element of the first event
    #currentevent_records=[]
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


def mergepolarizations(event,arraymapdictionary):
    xdict,ydict,zdict=arraymapdictionary
    for record in event:
        record['antname']=mapping.snap2_to_antpol(record['board_id'],record['antenna_id'])

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
        Adata = record['data']
        newrecord['polA_data']=Adata
        newrecord['rmsA']=np.std(Adata[:2000])
        index_peak_A=np.argmax(np.abs(Adata))
        newrecord['index_peak_A'] =index_peak_A
        newrecord['peakA']=np.abs(Adata[index_peak_A]) 

        #find the polB data
        for Brecord in polB:
            if Brecord['antname'][:-1]==antname:
                #Information that is specific to POlB
                newrecord['trigger_role_B'] = Brecord['trigger_role']
                newrecord['antenna_id_B'] =  Brecord['antenna_id'] 
                newrecord['veto_role_B'] = Brecord['veto_role']
                Bdata = Brecord['data']
                newrecord['polB_data']=Bdata
                newrecord['rmsB']=np.std(Bdata[:2000]) #use only the first half to calculate the rms, before the event starts
                index_peak_B=np.argmax(np.abs(Bdata))
                newrecord['index_peak_B'] =index_peak_B
                newrecord['peakB']=np.abs(Bdata[index_peak_B])   

        mergedrecords.append(newrecord)
    return mergedrecords

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

def grad_toa_plane(antcoords,theta,phi):
    #This is the gradient of toa_plane w.r.t theta and phi
    c=3e8
    sample_rate=1.97e8 #MHz
    phi_rad=(phi*math.pi/180)
    theta_rad=theta*math.pi/180
    x,y,z=ant_coords

    dtdtheta=(math.pi/180)*(sample_rate/c)*((y*math.cos(theta_rad)*math.cos(phi_rad))+(x*math.cos(theta_rad)*math.cos(phi_rad))-(z*math.sin(theta_rad)))
    dtdphi=(math.pi/180)*(sample_rate/c)*((-y*math.sin(theta_rad)*math.sin(phi_rad)) +(x*math.sin(theta_rad)*math.cos(phi_rad)))
    return np.asarray([dtdtheta,dtdphi]).transpose()


###Handy snapshot plotting functions
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
                antname=mapping.snap2_to_antpol(b+1,antenna) #TODO zero index the boards or 1-index??

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
                antname=mapping.snap2_to_antpol(b+1,antenna) #TODO zero index the boards or 1-index??
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
                antname=mapping.snap2_to_antpol(b+1,antenna) #TODO zero index the boards or 1-index??

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
        a=record['antenna_id']
        antname=mapping.snap2_to_antpol(s,a)
        if (s,a) in antennas:
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
    plt.clim(czoom_min,czoom_max)
    plt.title('Residual')



### NOTE THAT THIS FUNCTION IS FOR OLD FORMAT I'm leaving it here in case old-format data still needs to be used in commissioning
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
