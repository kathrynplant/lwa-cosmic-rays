#Runs the first stage of processing on any data in the /lustre/kplant/newdata/ directory
#This script takes one commandline input which is the directory name that will replace "newdata" after processing. Typically this is chosen to be the date of the observation, e.g. 2024June21.

#Run this script in the "deployment" conda environment

mkdir /lustre/kplant/new-dataproducts
cp /home/kplant/lwa-cosmic-rays/config_preliminary_cuts.yml /lustre/kplant/new-dataproducts/
cd /home/kplant/lwa-cosmic-rays
ls /lustre/kplant/newdata/*.dat | parallel --jobs 50 python summarize_events.py     /lustre/kplant/new-dataproducts/config_preliminary_cuts.yml

mv /lustre/kplant/newdata /lustre/kplant/$1
mv /lustre/kplant/new-dataproducts/* /lustre/kplant/$1-dataproducts
