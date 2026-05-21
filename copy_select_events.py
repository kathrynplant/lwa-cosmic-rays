# This script takes a summary array file for a selection of events and then extracts the raw data for those events and copies it into newfiles.  The summary array file is updated to specify where each event is copied to (new file name and index in file).
#The new files contain up to 100 events per file.
#The new files store raw data in the same format as the original files produced by the data capture software.

import numpy as np
#from cr_data_inspection_functions import *
import os
import numpy.lib.recfunctions as rfn
from pathlib import Path
import argparse

parser=argparse.ArgumentParser(description='Copy the raw data for a selection of events into new files. The data format of the new files is the same as the format that datacapture.py writes. The selection of events is described in a summary array file of the same format as that produced by summarize_events.py (but typically contains only a subset of that output, such as a selection of all the events passing the impulsive and antenna quality cuts). The new files contain 100 events per file until the last file which contains any remainder less than or equal to 100 events.')
parser.add_argument('datadir',type=str,help='Path to directory containing the data files. All the data files must be in one directory, and the new data files will be written here.')
parser.add_argument('fname', type=str, help='Full path to summary file')
args=parser.parse_args()
datadir = args.datadir
summary_fname = args.fname 

def copy_events(source_file_names,indices_in_source,new_file_name):
    
    #check for correct number of inputs
    assert len(source_file_names) == len(indices_in_source)
    
    #Each packet is 8192 Bytes and there are 704 packets per event
    packet_size = 8192
    event_size = 704*packet_size
    
    #open the new file to hold the output. This will overwrite if it already exists.
    with open(new_file_name,mode="wb") as outfileobject:
        #loop over events
        for i in range(len(source_file_names)):
            source_file_name = source_file_names[i]
            start_ind = indices_in_source[i]
            #read the data from the source file, for one event, and then write it into the new file
            with open(source_file_name, mode="rb") as sourcefile:
                byte_start = start_ind * packet_size
                assert byte_start % packet_size == 0  
                sourcefile.seek(byte_start)        #find the start of the event
                data = sourcefile.read(event_size) #read the event data
                #check that it hasn't hit the end of the file (if indexing is correct, it never will)
                if len(data) != event_size: 
                    raise IOError(f"Incomplete event starting at packet {start_ind}")
                
                #write the data
                outfileobject.write(data)
    return

#load the summary array
summarray=np.load(summary_fname)
p = Path(summary_fname)
shortname = p.name[:-4]

#For a chosen number of events in the new files, determine how many files there will be, there names, and the new event indices
Ntot = len(summarray) #total number of events
Nf = 100 #desired maximimum number of events per new file
Nnewfiles = (Ntot//Nf)+1 #resulting number of new datafiles files
place_in_new_file = np.asarray([n%Nf for n in range(Ntot)])
which_new_file = np.asarray([n//Nf for n in range(Ntot)])
new_fnames = np.asarray([shortname+'_'+str(n//Nf) + '.dat' for n in range(Ntot)])

#copy the columns with the original data file names and indices to new columns in the summary array
original_indices_in_file = np.copy(summarray['index_in_file'])
summarray=rfn.append_fields(summarray,'original_index_in_file',data=original_indices_in_file,dtypes=np.intc,usemask=False)
original_datafnames = np.copy(summarray['datafname'])
summarray=rfn.append_fields(summarray,'original_datafname',data=original_datafnames,dtypes='U100',usemask=False)

#populate the index_in_file and datafname columns with the new values
summarray['index_in_file'] = 704*place_in_new_file 
summarray['datafname']=new_fnames

#save the updated summary file
np.save(summary_fname,summarray)

#Now copy the events into each new file
unique_names = set(new_fnames)
for new_file_name in unique_names: #loop over each of the new files
    print('Starting on new file ',datadir+new_file_name)
    #find the events that need to go in that file
    events_for_file = summarray[summarray['datafname']==new_file_name]
    source_file_names = [datadir + f for f in events_for_file['original_datafname']]
    indices_in_source = events_for_file['original_index_in_file']
    #copy the data
    copy_events(source_file_names,indices_in_source,datadir+new_file_name)