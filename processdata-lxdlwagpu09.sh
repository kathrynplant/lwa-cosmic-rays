#Runs the first stage of processing on any data in the /data0/cosmic-ray-data/newdata and /data1/cosmic-ray-data/newdata directories
#This script takes one commandline input which is the directory name that will replace "newdata" after processing. Typically this is chosen to be the date of the observation, e.g. 2024June21.

#Run this script in the "deployment" conda environment


#Check for the correct number of arguments
if [[ $# -ne 1 ]]; then
	echo "Please provide a name for the final directory. Usage: $0 dir_name"
	exit 2
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
	ls $drive/cosmic-ray-data/newdata/*.dat | parallel --jobs 18 python summarize_events.py     $drive/cosmic-ray-data/new-dataproducts/config.yml $drive/cosmic-ray-data/new-dataproducts/

	#Make first cuts based on the summaries
	python impulsive_and_antquality_cuts.py $drive/cosmic-ray-data/new-dataproducts/config.yml $drive/cosmic-ray-data/new-dataproducts/

	#Do model fits on events that pass those first cuts
	ls $drive/cosmic-ray-data/new-dataproducts/first_cut_* | parallel --jobs 18  python model_fits.py $drive/cosmic-ray-data/new-dataproducts/config.yml $drive/cosmic-ray-data/newdata/

	# Rename directory
	cd $drive/cosmic-ray-data
	mv newdata $1
	mv new-dataproducts $1-dataproducts
	mkdir newdata

done
