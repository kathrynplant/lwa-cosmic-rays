#!/usr/bin/env python
import pandas as pd
import time
import numpy as np
import struct
from lwa_f import snap2_fengine
from lwa_f import blocks
import matplotlib.pyplot as plt
from lwa_antpos import mapping

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


def parsefile(fname):
    #get data and header from a file (fname) with raw data from an arbitrary number of cosmic ray packets
    #returns a list of dictionaries, with one dictionary per packet
    #the data key contains a numpy array of the timeseries and the other keys are header info as specified by the parseheader function
    with open(fname, mode="rb") as datafile:   #read the file
         rawfiledata = datafile.read()
    records=[]  #start a list of all the packets
    
    #check if there are any incomplete packets and quit if so
    if len(rawfiledata)%8192: 
        print("File contains incomplete packets")
        return
    else:  #if there are no incomplete packets, proceed to parsing the packets
        npackets=len(rawfiledata)//8192
        print("File contains ",npackets," packets")
        for p in range(npackets):
            rawpacketdata=rawfiledata[8192*p:8192*(p+1)]
            dictionary=parseheader(rawpacketdata)
            timeseries=unpackdata(rawpacketdata,'>h')
            dictionary['data']=timeseries
            records.append(dictionary)
    return records

def packet_ant_id_2_snap_input(i):
    #This function returns the snap input number corresponding to a packet with integer i in the antenna_id field in the packet header
    #someday the need for this remapping could probably be addressed in the firmware
    return (i&0b111000)+((i+1)&0b000111)

def distinguishevents(records,maxoffset):
    #This function turns a list of single-antenna records, into a list of events.
    #records is a list of single-antenna records, such as that returned by parsefile.
    #maxoffset is the maximum timestamp difference (in number of clockcycles) for two records to be considered part of the same event.
    #The function distinguish events returns a list of events, where each event is a list of the records (single-antenna dictionaries) that belong to that event.

    #start an empty list which will ultimately have one element per event
    events=[]
    eventcount=1  #keep track of how many separate events there are

    #start an list for the first event. The first record is the first element of the first event
    currentevent=[records[0]]
    currenteventtimestamp=records[0]['timestamp']

    maxtimestamp=currenteventtimestamp+maxoffset #all 
    for record in records:
        recordtimestamp=record['timestamp']
        if recordtimestamp<maxtimestamp:
            #print(recordtimestamp, maxtimestamp,True)
            currentevent.append(record)
        else: #start a new event
            #print(recordtimestamp, maxtimestamp,False)
            eventcount+=1
            events.append(currentevent)
            currentevent=[record]
            currenteventtimestamp=record['timestamp']
            maxtimestamp=currenteventtimestamp+maxoffset
    events.append(currentevent)
    return events


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
