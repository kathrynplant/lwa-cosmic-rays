#!/bin/bash

#Runs the first stage of processing on any data in the /data0/cosmic-ray-data/newdata and /data1/cosmic-ray-data/newdata directories
#This script takes one optional commandline input which is the directory name that will replace "newdata" after processing. Typically this is chosen to be the date of the observation, e.g. 2024June21.
#If no argument is given, the directory name will be the date the processing was run.

#Run this script in the "deployment" conda environment

if [[ $# -ge 1 ]]; then
    DATESPEC="$1"
else
    DATESPEC=$(date +%Y%b%-d)
fi


for drive in "/data0" "/data1";
do
	#Check if there is data to process
	if [ -z "$(ls -A "$drive/cosmic-ray-data/newdata")" ]; then
        	echo "No data in $drive/cosmic-ray-data/newdata"
            	continue
    	fi

	#prepare directories
	mkdir $drive/cosmic-ray-data/new-dataproducts
	cp /home/ubuntu/kp/lwa-cosmic-rays/config-gpu9.yml $drive/cosmic-ray-data/new-dataproducts/config.yml
	cd /home/ubuntu/kp/lwa-cosmic-rays

	#Compute summary statistics for all complete events
	echo "Computing summary statistics for all events."
	ls $drive/cosmic-ray-data/newdata/*.dat | /usr/local/bin/parallel --jobs 18 /home/ubuntu/anaconda3/envs/deployment/bin/python -u summarize_events.py     $drive/cosmic-ray-data/new-dataproducts/config.yml $drive/cosmic-ray-data/new-dataproducts/

	#Make first cuts based on the summaries
	echo "Applying quality and impulsivity cut."
	/home/ubuntu/anaconda3/envs/deployment/bin/python -u impulsive_and_antquality_cuts.py $drive/cosmic-ray-data/new-dataproducts/config.yml $drive/cosmic-ray-data/new-dataproducts/

	#Do model fits on events that pass those first cuts
	echo "Starting model fits."
	ls $drive/cosmic-ray-data/new-dataproducts/first_cut_* | /usr/local/bin/parallel --jobs 18  /home/ubuntu/anaconda3/envs/deployment/bin/python -u  model_fits.py $drive/cosmic-ray-data/new-dataproducts/config.yml $drive/cosmic-ray-data/newdata/
	
	#Select events to save
	echo "Selecting events to copy."
	/home/ubuntu/anaconda3/envs/deployment/bin/python -u select_high_el_no_qc.py $drive/cosmic-ray-data/new-dataproducts/
	#I want to replace this with something that reads the selection criteria from a config file

	#copy select events to new files
	echo "Copying select events to new files."
	/home/ubuntu/anaconda3/envs/deployment/bin/python -u copy_select_events.py $drive/cosmic-ray-data/newdata/ $drive/cosmic-ray-data/new-dataproducts/high_el_no_direction_qc.npy  #TO DO update file name after I make the more general selection script  
	
	#Generate pdf summary
	#TBD
	
	# Rename directory
	cd $drive/cosmic-ray-data
	mv newdata $DATESPEC
	mv new-dataproducts $DATESPEC-dataproducts
	mkdir newdata
	

done
