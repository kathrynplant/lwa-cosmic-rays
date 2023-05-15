import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math
import pylab
from scipy.interpolate import interp1d
import time
from lsl import skymap, astro
from lsl.common import stations
from lsl.common.paths import DATA as dataPath
from lsl.misc import parser as aph

from lsl.misc import telemetry

def driftcurve(pol = 'EW', lfsm = True, frequency = 74.e6, verbose=True, time_start=time.time(), \
               empirical=False, do_plot=False, time_step = 10.0, savedir=None):
    # Validate
    if pol not in ('EW', 'NS'):
        raise ValueError("Invalid polarization: %s" % pol)

    #OVRO info
    nam = 'ovro'
    sta = stations.lwa1
    sta.lat, sta.lon, sta.elev = ('37.23977727', '-118.2816667', 1182.89)

    # Read in the skymap (GSM or LF map @ 74 MHz)
    if not lfsm:
        smap = skymap.SkyMapGSM(freq_MHz=frequency/1e6)
        if verbose:
            print("Read in GSM map at %.2f MHz of %s pixels; min=%f, max=%f" % (frequency/1e6, len(smap.ra), smap._power.min(), smap._power.max()))
    else:
        smap = skymap.SkyMapLFSM(freq_MHz=frequency/1e6)
        if verbose:
            print("Read in LFSM map at %.2f MHz of %s pixels; min=%f, max=%f" % (frequency/1e6, len(smap.ra), smap._power.min(), smap._power.max()))

    # Get the emperical model of the beam and compute it for the correct frequencies
    beamDict = np.load(os.path.join(dataPath, 'lwa1-dipole-emp.npz'))
    if pol == 'EW':
        beamCoeff = beamDict['fitX']
    else:
        beamCoeff = beamDict['fitY']
    try:
        beamDict.close()
    except AttributeError:
        pass
    alphaE = np.polyval(beamCoeff[0,0,:], frequency)
    betaE =  np.polyval(beamCoeff[0,1,:], frequency)
    gammaE = np.polyval(beamCoeff[0,2,:], frequency)
    deltaE = np.polyval(beamCoeff[0,3,:], frequency)
    alphaH = np.polyval(beamCoeff[1,0,:], frequency)
    betaH =  np.polyval(beamCoeff[1,1,:], frequency)
    gammaH = np.polyval(beamCoeff[1,2,:], frequency)
    deltaH = np.polyval(beamCoeff[1,3,:], frequency)
    if verbose:
        print("Beam Coeffs. X: a=%.2f, b=%.2f, g=%.2f, d=%.2f" % (alphaH, betaH, gammaH, deltaH))
        print("Beam Coeffs. Y: a=%.2f, b=%.2f, g=%.2f, d=%.2f" % (alphaE, betaE, gammaE, deltaE))

    if empirical:
        corrDict = np.load(os.path.join(dataPath, 'lwa1-dipole-cor.npz'))
        cFreqs = corrDict['freqs']
        cAlts  = corrDict['alts']
        if corrDict['degrees'].item():
            cAlts *= np.pi / 180.0
        cCorrs = corrDict['corrs']
        corrDict.close()

        if frequency/1e6 < cFreqs.min() or frequency/1e6 > cFreqs.max():
            print("WARNING: Input frequency of %.3f MHz is out of range, skipping correction" % (frequency/1e6,))
            corrFnc = None
        else:
            fCors = cAlts*0.0
            for i in range(fCors.size):
                ffnc = interp1d(cFreqs, cCorrs[:,i], bounds_error=False)
                fCors[i] = ffnc(frequency/1e6)
            corrFnc = interp1d(cAlts, fCors, bounds_error=False)

    else:
        corrFnc = None

    def compute_beam_pattern(az, alt, corr=corrFnc):
        zaR = np.pi/2 - alt*np.pi / 180.0
        azR = az*np.pi / 180.0

        c = 1.0
        if corrFnc is not None:
            c = corrFnc(alt*np.pi / 180.0)
            c = np.where(np.isfinite(c), c, 1.0)

        pE = (1-(2*zaR/np.pi)**alphaE)*np.cos(zaR)**betaE + gammaE*(2*zaR/np.pi)*np.cos(zaR)**deltaE
        pH = (1-(2*zaR/np.pi)**alphaH)*np.cos(zaR)**betaH + gammaH*(2*zaR/np.pi)*np.cos(zaR)**deltaH

        return c*np.sqrt((pE*np.cos(azR))**2 + (pH*np.sin(azR))**2)

    if do_plot:
        az = np.zeros((90,360))
        alt = np.zeros((90,360))
        for i in range(360):
            az[:,i] = i
        for i in range(90):
            alt[i,:] = i
        pylab.figure(1)
        pylab.title("Beam Response: %s pol. @ %0.2f MHz" % (pol, frequency/1e6))
        pylab.imshow(compute_beam_pattern(az, alt), extent=(0,359, 0,89), origin='lower')
        pylab.xlabel("Azimuth [deg]")
        pylab.ylabel("Altitude [deg]")
        pylab.grid(1)
        pylab.draw()

    # Calculate times in both site LST and UTC
    t0 = astro.unix_to_utcjd(time_start)
    lst = astro.get_local_sidereal_time(sta.long*180.0/math.pi, t0) / 24.0
    t0 -= lst*(23.933/24.0) # Compensate for shorter sidereal days
    times = np.arange(0.0, 1.0, time_step/1440.0) + t0

    lstList = []
    powListAnt = []

    for t in times:
        # Project skymap to site location and observation time
        pmap = skymap.ProjectedSkyMap(smap, sta.lat*180.0/math.pi, sta.long*180.0/math.pi, t)
        lst = astro.get_local_sidereal_time(sta.long*180.0/math.pi, t)
        lstList.append(lst)

        # Convolution of user antenna pattern with visible skymap
        gain = compute_beam_pattern(pmap.visibleAz, pmap.visibleAlt)
        powerAnt = (pmap.visiblePower * gain).sum() / gain.sum()
        powListAnt.append(powerAnt)

        if verbose:
            lstH = int(lst)
            lstM = int((lst - lstH)*60.0)
            lstS = ((lst - lstH)*60.0 - lstM)*60.0
            sys.stdout.write("LST: %02i:%02i:%04.1f, Power_ant: %.1f K\r" % (lstH, lstM, lstS, powerAnt))
            sys.stdout.flush()
    sys.stdout.write("\n")

    # plot results
    if do_plot:
        pylab.figure(2)
        pylab.title("Driftcurve: %s pol. @ %0.2f MHz - %s" % \
            (pol, frequency/1e6, nam.upper()))
        pylab.plot(lstList, powListAnt, "ro", label="Antenna Pattern")
        pylab.xlabel("LST [hours]")
        pylab.ylabel("Temp. [K]")
        pylab.grid(2)
        pylab.draw()
        pylab.show()
    
    if savedir != None:
        outputFile = "%sdriftcurve_%s_%s_%.2f_%.2f.txt" % (savedir, nam, pol, frequency/1e6, t0)
        print("Writing driftcurve to file '%s'" % outputFile)
        mf = open(outputFile, "w")
        for lst,pow in zip(lstList, powListAnt):
            mf.write("%f  %f\n" % (lst, pow))
        mf.close()
    return lstList, powListAnt

def single_time_driftcurve(pol = 'EW', lfsm = True, frequency = 74.e6, time_start=time.time(), empirical=False):
    # Validate
    if pol not in ('EW', 'NS'):
        raise ValueError("Invalid polarization: %s" % pol)

    #OVRO info
    nam = 'ovro'
    sta = stations.lwa1
    sta.lat, sta.lon, sta.elev = ('37.23977727', '-118.2816667', 1182.89)

    # Read in the skymap (GSM or LF map @ 74 MHz)
    if not lfsm:
        smap = skymap.SkyMapGSM(freq_MHz=frequency/1e6)
    else:
        smap = skymap.SkyMapLFSM(freq_MHz=frequency/1e6)

    # Get the emperical model of the beam and compute it for the correct frequencies
    beamDict = np.load(os.path.join(dataPath, 'lwa1-dipole-emp.npz'))
    if pol == 'EW':
        beamCoeff = beamDict['fitX']
    else:
        beamCoeff = beamDict['fitY']
    try:
        beamDict.close()
    except AttributeError:
        pass
    alphaE = np.polyval(beamCoeff[0,0,:], frequency)
    betaE =  np.polyval(beamCoeff[0,1,:], frequency)
    gammaE = np.polyval(beamCoeff[0,2,:], frequency)
    deltaE = np.polyval(beamCoeff[0,3,:], frequency)
    alphaH = np.polyval(beamCoeff[1,0,:], frequency)
    betaH =  np.polyval(beamCoeff[1,1,:], frequency)
    gammaH = np.polyval(beamCoeff[1,2,:], frequency)
    deltaH = np.polyval(beamCoeff[1,3,:], frequency)

    if empirical:
        corrDict = np.load(os.path.join(dataPath, 'lwa1-dipole-cor.npz'))
        cFreqs = corrDict['freqs']
        cAlts  = corrDict['alts']
        if corrDict['degrees'].item():
            cAlts *= np.pi / 180.0
        cCorrs = corrDict['corrs']
        corrDict.close()

        if frequency/1e6 < cFreqs.min() or frequency/1e6 > cFreqs.max():
            print("WARNING: Input frequency of %.3f MHz is out of range, skipping correction" % (frequency/1e6,))
            corrFnc = None
        else:
            fCors = cAlts*0.0
            for i in range(fCors.size):
                ffnc = interp1d(cFreqs, cCorrs[:,i], bounds_error=False)
                fCors[i] = ffnc(frequency/1e6)
            corrFnc = interp1d(cAlts, fCors, bounds_error=False)

    else:
        corrFnc = None

    def compute_beam_pattern(az, alt, corr=corrFnc):
        zaR = np.pi/2 - alt*np.pi / 180.0
        azR = az*np.pi / 180.0

        c = 1.0
        if corrFnc is not None:
            c = corrFnc(alt*np.pi / 180.0)
            c = np.where(np.isfinite(c), c, 1.0)

        pE = (1-(2*zaR/np.pi)**alphaE)*np.cos(zaR)**betaE + gammaE*(2*zaR/np.pi)*np.cos(zaR)**deltaE
        pH = (1-(2*zaR/np.pi)**alphaH)*np.cos(zaR)**betaH + gammaH*(2*zaR/np.pi)*np.cos(zaR)**deltaH

        return c*np.sqrt((pE*np.cos(azR))**2 + (pH*np.sin(azR))**2)

    # Calculate times in both site LST and UTC
    t0 = astro.unix_to_utcjd(time_start)
    lst = astro.get_local_sidereal_time(sta.long*180.0/math.pi, t0) / 24.0
    t0 -= lst*(23.933/24.0) # Compensate for shorter sidereal days

    # Project skymap to site location and observation time
    pmap = skymap.ProjectedSkyMap(smap, sta.lat*180.0/math.pi, sta.long*180.0/math.pi, t0)

    # Convolution of user antenna pattern with visible skymap
    gain = compute_beam_pattern(pmap.visibleAz, pmap.visibleAlt)
    powerAnt = (pmap.visiblePower * gain).sum() / gain.sum()

    return lst, powerAnt
