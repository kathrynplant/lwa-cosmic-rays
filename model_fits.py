import time
import numpy as np
#import struct
#import matplotlib.pyplot as plt
from cr_data_inspection_functions import *
from lwa_antpos import reading
from scipy.optimize import curve_fit
#import scipy.stats as st
import math
import argparse
import yaml
import numpy.lib.recfunctions as rfn


parser=argparse.ArgumentParser(description='Perform model fits on events specified in a summary file such as that output by summarize_events.py or impulsive_and_antquality_cuts.py. The model fits are a wavefront fit to the observed TOAs, to determine arrival direction parameters, and a Gaussian fit to the spatial distribution of measured SNRs. Results from these fits are appended as additional columns in the summary file.')
parser.add_argument('config',type=str, help='Full path to configuration file')
parser.add_argument('fname', type=str, help='Full path to summary file')
args=parser.parse_args()
fname = args.fname 
config = args.config

def main():
    starttime=time.time()
    print('Start time ',starttime)
    ############################### set parameters -- read from a config file ######################################
    with open(config, 'r') as file:
        configuration=yaml.safe_load(file)
    #where to save data products
    datadir =configuration['datadir'] 
    #name of csv file with antenna names and coordinates: Columns must have headings 'antname', 'x', 'y', 'elevation'
    array_map_filename=configuration['array_map_filename'] 
    
    #FIR Filter coefficients and sample rate
    h=np.asarray(configuration['filter'])
    fs=configuration['fs']
    
    #parameters for antenna-based cuts
    maximum_ok_power=configuration['maximum_ok_power'] 
    minimum_ok_power=configuration['minimum_ok_power']
    minimum_ok_kurtosis=configuration['minimum_ok_kurtosis']
    maximum_ok_kurtosis=configuration['maximum_ok_kurtosis']
    max_saturated_samples=configuration['maximum_ok_kurtosis']
    known_bad_antennas=configuration['known_bad_antennas']
    
    #parameters for model fits

    minsnr= configuration['minsnr']# Only antennas with snr >minsnr will be included in the model fits
    niter_toa= configuration['niter_toa']#The time of arrival model fit will be repeated for niter_toa iterations, with outliers flagged after each iteration.
    niter_gauss= configuration['niter_gauss']#The Gaussian model fit will be repeated for niter_gauss iterations, with outliders flagged after each iteration. Setting it to 1 iteration means that no flagging will be performed.
    maxdev= configuration['maxdev'] #outliers with residuals more than maxdev median absolute deviations from the median residual will be flagged on each flagging iteration
    
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

    lwa_df = reading.read_antpos()
    namedict=build_mapping_dictionary(lwa_df)

    ##################################################################################################

    #load summary array         
    summarray=np.load(fname)
    
    #create dictionary to hold lists of results
    fitstats={}
    fitstats['arrival_zenith_angle']=[]
    fitstats['arrival_zenith_angle_err']=[]
    fitstats['arrival_azimuth']=[]
    fitstats['arrival_azimuth_err']=[]
    fitstats['source_distance']=[]
    fitstats['source_distance_err']=[]
    fitstats['toa_fit_rms_res']=[]
    fitstats['weightedres']=[]
    fitstats['gauss_amp']=[]
    fitstats['gauss_amp_err']=[]
    fitstats['gauss_azimuth']=[]
    fitstats['gauss_azimuth_err']=[]
    fitstats['gauss_lateral_scale']=[]
    fitstats['gauss_lateral_scale_err']=[]
    fitstats['gauss_aspect_ratio']=[]
    fitstats['gauss_aspect_ratio_err']=[]
    fitstats['gauss_center_x']=[]
    fitstats['gauss_center_x_err']=[]
    fitstats['gauss_center_y']=[]
    fitstats['gauss_center_y_err']=[]
    fitstats['gauss_fit_rms_res']=[]
    
    #load each event and do fits
    for i in range(len(summarray)):
        #load data
        datafile=datadir+summarray[i]['datafname']
        index_in_file=summarray[i]['index_in_file']
        event_records=parsefile(datafile,start_ind=index_in_file,end_ind=704 )
        event_summary=summarize_signals(event_records,np.asarray(configuration['filter']),namedict,xdict,ydict,zdict)
        event_summary_flagged=flag_antennas(event_summary,configuration['maximum_ok_power'], configuration['minimum_ok_power'],
                                            configuration['minimum_ok_kurtosis'],configuration['maximum_ok_kurtosis'],1,
        configuration['known_bad_antennas'])[0]

        #TOA fit
        try:
                poptt,pcovt,rms_res_t,weightedresidual,array_toa_fit,reference=robust_direction_fit(event_summary_flagged[event_summary_flagged['snr']>minsnr],niter=3,outlier_limit=maxdev,plot=False,toa_func=toa_sphere,fitbounds=([0,0,1],[90,360,1e8]),weightbysnr=True)
        except RuntimeError:
            poptt=np.zeros(3)
            pcovt=np.zeros((3,3))
            rms_res_t=0
            weightedresidual=0
            array_toa_fit=0
            reference=np.zeros(4)
        rms_weightedresidual=np.sqrt(np.mean(np.square(weightedresidual)))

        #Gaussian fit -- choose the brighter polarization
        meansnrA=np.mean(event_summary_flagged[event_summary_flagged['pol']=='A']['snr'])
        meansnrB=np.mean(event_summary_flagged[event_summary_flagged['pol']=='B']['snr'])
        if meansnrA>meansnrB:
            summary_array = event_summary_flagged[np.logical_and(event_summary_flagged['snr']>minsnr,event_summary_flagged['pol']=='A')]
        else:
            summary_array = event_summary_flagged[np.logical_and(event_summary_flagged['snr']>minsnr,event_summary_flagged['pol']=='B')]
        try:
            poptg,pcovg,rms_res_g,array_spatial_fit=robust_spatial_fit(summary_array,1,maxdev,gauss2d,([0,0,0,1,-10000,-10000],[50,180,1000,100,10000,10000]),plot=False)
        except RuntimeError:
            poptg=np.zeros(6)
            pcovg=np.zeros((6,6))
            rms_res_g=0
            array_spatial_fit=0

        #summarize results
        fitstats['arrival_zenith_angle'].append(poptt[0])
        fitstats['arrival_zenith_angle_err'].append(math.sqrt(pcovt[0,0]))

        fitstats['arrival_azimuth'].append(poptt[1])
        fitstats['arrival_azimuth_err'].append(math.sqrt(pcovt[1,1]))

        fitstats['source_distance'].append(poptt[2])
        fitstats['source_distance_err'].append(math.sqrt(pcovt[2,2]))

        fitstats['toa_fit_rms_res'].append(rms_res_t)
        fitstats['weightedres'].append(rms_weightedresidual)

        fitstats['gauss_amp'].append(poptg[0])
        fitstats['gauss_amp_err'].append(math.sqrt(pcovg[0,0]))

        fitstats['gauss_azimuth'].append(poptg[1])
        fitstats['gauss_azimuth_err'].append(math.sqrt(pcovg[1,1]))

        fitstats['gauss_lateral_scale'].append(poptg[2])
        fitstats['gauss_lateral_scale_err'].append(math.sqrt(pcovg[2,2]))

        fitstats['gauss_aspect_ratio'].append(poptg[3])
        fitstats['gauss_aspect_ratio_err'].append(math.sqrt(pcovg[3,3]))

        fitstats['gauss_center_x'].append(poptg[4])
        fitstats['gauss_center_x_err'].append(math.sqrt(pcovg[4,4]))

        fitstats['gauss_center_y'].append(poptg[5])
        fitstats['gauss_center_y_err'].append(math.sqrt(pcovg[5,5]))

        fitstats['gauss_fit_rms_res'].append(rms_res_g)

   ######### Add results to summary array and save results #####################################################################

    summarray=rfn.append_fields(summarray,'arrival_zenith_angle',data=fitstats['arrival_zenith_angle'],dtypes=float,usemask=False)
    summarray=rfn.append_fields(summarray,'arrival_zenith_angle_err',data=fitstats['arrival_zenith_angle_err'],dtypes=float,usemask=False)
    summarray=rfn.append_fields(summarray,'arrival_azimuth',data=fitstats['arrival_azimuth'],dtypes=float,usemask=False)
    summarray=rfn.append_fields(summarray,'arrival_azimuth_err',data=fitstats['arrival_azimuth_err'],dtypes=float,usemask=False)
    summarray=rfn.append_fields(summarray,'source_distance',data=fitstats['source_distance'],dtypes=float,usemask=False)
    summarray=rfn.append_fields(summarray,'source_distance_err',data=fitstats['source_distance_err'],dtypes=float,usemask=False)
    summarray=rfn.append_fields(summarray,'toa_fit_rms_res',data=fitstats['toa_fit_rms_res'],dtypes=float,usemask=False)
    summarray=rfn.append_fields(summarray,'weightedres',data=fitstats['weightedres'],dtypes=float,usemask=False)
    summarray=rfn.append_fields(summarray,'gauss_amp',data=fitstats['gauss_amp'],dtypes=float,usemask=False)
    summarray=rfn.append_fields(summarray,'gauss_amp_err',data=fitstats['gauss_amp_err'],dtypes=float,usemask=False)
    summarray=rfn.append_fields(summarray,'gauss_azimuth',data=fitstats['gauss_azimuth'],dtypes=float,usemask=False)
    summarray=rfn.append_fields(summarray,'gauss_azimuth_err',data=fitstats['gauss_azimuth_err'],dtypes=float,usemask=False)
    summarray=rfn.append_fields(summarray,'gauss_lateral_scale',data=fitstats['gauss_lateral_scale'],dtypes=float,usemask=False)
    summarray=rfn.append_fields(summarray,'gauss_lateral_scale_err',data=fitstats['gauss_lateral_scale_err'],dtypes=float,usemask=False)
    summarray=rfn.append_fields(summarray,'gauss_aspect_ratio',data=fitstats['gauss_aspect_ratio'],dtypes=float,usemask=False)
    summarray=rfn.append_fields(summarray,'gauss_aspect_ratio_err',data=fitstats['gauss_aspect_ratio_err'],dtypes=float,usemask=False)
    summarray=rfn.append_fields(summarray,'gauss_center_x',data=fitstats['gauss_center_x'],dtypes=float,usemask=False)
    summarray=rfn.append_fields(summarray,'gauss_center_x_err',data=fitstats['gauss_center_x_err'],dtypes=float,usemask=False)
    summarray=rfn.append_fields(summarray,'gauss_center_y',data=fitstats['gauss_center_y'],dtypes=float,usemask=False)
    summarray=rfn.append_fields(summarray,'gauss_center_y_err',data=fitstats['gauss_center_y_err'],dtypes=float,usemask=False)
    summarray=rfn.append_fields(summarray,'gauss_fit_rms_res',data=fitstats['gauss_fit_rms_res'],dtypes=float,usemask=False)

    #save summary array
    np.save(fname,summarray)
   
    return

main()
