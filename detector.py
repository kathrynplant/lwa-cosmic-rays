"""
Created on Tue Sep 18 13:14:16 2018

@author: romerowo

slighlty modified by andrew ludwig
"""

import numpy as np
from scipy.interpolate import Akima1DInterpolator

class Detector:
    def __init__(self):
        self.kB = 1.3806488e-23
        self.c = 299792458. # in m
        self.Z0 = 120.*np.pi# Ohms
        '''Antenna zenithal gain pattern'''
        zen =      np.array([0.,  15.,  30.,  45.,  60.,  75., 80.,  85.,   90. ])
        zen_gain = np.array([8.6, 8.03, 7.28, 5.84, 4.00,  0., -2.9, -8.3, -40.])
        self.zen_gain_interp = Akima1DInterpolator(zen, zen_gain-3.) # -3 is the conversion from dBic to dBi in linear polarization

        '''Antenna impedance'''
        fr =   [24., 34.,    44.,  54.,  64.,  74.,   84.,   94.]
        Z_re = [1.8,  10.,   40.,  128., 317., 271.,  158.,  129.]
        Z_im = [-68.5, 12.6, 91.5, 169., 88.2, -80.3, -89.4, -58]
        
        self.Z_re_interp = Akima1DInterpolator(fr, Z_re)
        self.Z_im_interp = Akima1DInterpolator(fr, Z_im)
        self.Z_in = 100. # Ohms, the impedance seen by the terminals of the antenna.

        f = np.load('data/filter_function.npz')
        self.filter_interp = Akima1DInterpolator(np.arange(30., 80.1, 0.1), f['filter_array'])
        
       
    def calculate_noise_power(self, freqs, sky_temps, df, use_filter_datasheet=True):
        #freqs in MHz, sky_temps come from driftcurve script and already in K
        self.fr_fine = freqs
        if use_filter_datasheet:
            self.filter = self.filter_interp(freqs)
        
        '''P_div is the power from the voltage divider'''
        P_div = np.abs(self.Z_in)**2/np.abs(self.Z_in + self.Z_re_interp(self.fr_fine)+1j*self.Z_im_interp(self.fr_fine))**2
        self.Noise = 4. * self.kB * sky_temps * self.Z_re_interp(self.fr_fine) * P_div
        if use_filter_datasheet:
            self.Noise *= self.filter
        self.Noise += self.kB*250.*np.real(self.Z_in) # 250. Kelvin internal noise
        
        self.f_c = 55.*1e6
        lam_c = self.c/self.f_c
        
        self.h_eff = 4. * self.Z_re_interp(self.fr_fine) / self.Z0 * lam_c**2 / 4. / np.pi 
        self.h_eff *= np.abs(self.Z_in)**2 / np.abs(self.Z_re_interp(self.fr_fine)+1j*self.Z_im_interp(self.fr_fine)+self.Z_in)**2
        if use_filter_datasheet:
            self.h_eff *= self.filter
        self.h_eff = np.sqrt(self.h_eff)
        
        self.h_0 = np.mean(self.h_eff) # assume flat spectrum for CR pulse
        
        self.V_rms = np.sqrt(np.sum(self.Noise * df*1.e6))
        return self.Noise #comes out in V^2/Hz
        
    ####################################################################################

    def Efield_2_Voltage(self, E_field, theta_zenith_deg): # input in V/m
        # Get directivity
        D = self.zen_gain_interp(theta_zenith_deg)
        return E_field * self.h_0 * np.sqrt( 10**(D/10.))
        
