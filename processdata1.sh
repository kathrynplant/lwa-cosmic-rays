#Runs the first stage of processing on any data in the /lustre/kplant/newdata/ directory
conda activate deployment
mkdir /lustre/kplant/new-dataproducts
cp /home/kplant/lwa-cosmic-rays/config_preliminary_cuts.yml /lustre/kplant/newdata-dataproducts/

cd /home/kplant/lwa-cosmic-rays
ls /lustre/kplant/newdata/*.dat | parallel --jobs 50 python summarize_events.py     /lustre/kplant/newdata-dataproducts/config_preliminary_cuts.yml

mv newdata $1
mv new-dataproducts $1-dataproducts
