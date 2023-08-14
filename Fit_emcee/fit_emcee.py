#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#"""
#Created on Wed Apr 19 12:30:16 2023
#
#@author: mapet
#"""
import astropy.units as u
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy 
import corner
import emcee
from multiprocessing import Pool
import timeit
import sys
import os 

if len(sys.argv) != 3:
    print('missing code module')
    print('try something like this')
    print('python fit_uplim_emcee.py LeMoC 1000')
    quit()

# Read from terminal code module -- leptonic or hadronic
flag_c = sys.argv[1] 

if flag_c == 'LeMoC': 
    from Code import LeMoC as numcode 
if flag_c == 'LeHaMoC': 
    from Code import LeHaMoC as numcode 

# Modify if needed
D = 3262   # luminosity distance (Mpc)
z = 0.557  # redshift

# Create directory for saving chains
directory = 'chains/'+flag_c
path = os.path.join(directory) 
try:
    os.makedirs(path, exist_ok = True)
    print("Directory '%s' created successfully" % directory)
except OSError as error:
    print("Directory '%s' can not be created" % directory)

#######################
#SED file reader # 
#######################
def read_data(filename1, filename2):
    # read gamma-ray data
    data_fermi = np.loadtxt(filename2, usecols=range(0, 4))
    dat_fermi = pd.DataFrame(data_fermi, columns=['E0', 'vFv0', 'vFve', 'flag'])
    dat_fermi['E'] = np.log10(dat_fermi['E0'])
    dat_fermi['vFv'] = np.log10(dat_fermi['vFv0']*1e6*(u.eV).to(u.erg)) # MeV/cm2/s to erg/cm2/s
    vFve_hi_log = np.log10(dat_fermi['vFv0']+dat_fermi['vFve']) - np.log10(dat_fermi['vFv0'])
    vFve_lo_log = -np.log10(dat_fermi['vFv0']-dat_fermi['vFve']) + np.log10(dat_fermi['vFv0'])
    
    # read multi-wavelength data up to X-rays
    data = np.loadtxt(filename1, usecols=range(0, 4))
    dat = pd.DataFrame(data, columns=['E', 'eB', 'vFv', 'vFve'])
    
    # merge the data 
    xd = np.append(dat['E'].to_numpy(), 
                    dat_fermi['E'].to_numpy())
    yd = np.append(dat['vFv'].to_numpy(), 
                    dat_fermi['vFv'].to_numpy())
    yderr_hi = np.append(dat['vFve'].to_numpy(), 
                        vFve_hi_log.to_numpy())
    yderr_lo = np.append(dat['vFve'].to_numpy(), 
                        vFve_lo_log.to_numpy())
    flags = np.append(np.zeros(len(dat['E'])), 
                        dat_fermi['flag'].to_numpy())
    
    return xd, yd, yderr_hi, yderr_lo, flags
############################################################################
############################################################################

#######################
#Model# 
#######################
class Model:
    
    def __init__(self, D = 3262, z = 0.557):
         
        self.D = D*1e6*(u.pc).to(u.cm) # D in Mpc
        self.z = z
        self.distance_norm = np.log10(4 * np.pi *self.D**2) 

    def __call__(self, x, theta):
        doppler = theta[-1]
        boost = 4 * doppler
        params = theta[:-1]
        flux_norm = -self.distance_norm + boost
        x_pred, y_pred = numcode(params, 'Parameters.txt') 
        xp = np.log10(x_pred) + doppler - np.log10(1.+self.z)
        y_pred_d = np.log10(y_pred) + flux_norm
        if min(y_pred) == 0.:
            print('warning: zero model')
            print('parameters', params)
        # Convert the result to numpy and interpolate
        return np.interp(x, xp, y_pred_d)

#######################
#Likelihood functions# 
#######################

def log_likelihood_LeMoC(theta, x, y, yerr, yerr1, flags):
    mask0 = (flags == 0) ## detections
    mask1 = (flags == 1) ## upper limits 
    y_det = y[mask0]
    y_ul = y[mask1]
    yerr_det = yerr[mask0] ## upper errors
    yerr1_det = yerr1[mask0] ## lower errors
    P1, P2, P3, P4, P5, P6, P7, log_f = theta
    y_model = model(x, [P1, P2, P3, P4, P5, P6, P7])
    y_model_ul = y_model[mask1]
    y_model_det = y_model[mask0] 
    
    if (y_model_det[-2] < y_det[-2]):
        yerr_det[-2] = yerr1_det[-2]
    if (y_model_det[-1] < y_det[-1]):
        yerr_det[-1] = yerr1_det[-1]    
    
    sigma2 = yerr_det**2 + np.exp(2 * log_f)  
    
    rms = 0.05
    temp = scipy.special.erf(((y_ul-y_model_ul)/((2**0.5)*rms)))
    loglike_lim = 1.*np.sum(np.log((np.pi/2.)**0.5*rms*(1.+temp)))
    loglike_det = -0.5*np.sum((y_det - y_model_det)**2/sigma2 + np.log(sigma2))
    
    return  loglike_det +  loglike_lim 

def log_likelihood_LeHaMoC(theta, x, y, yerr, yerr1, flags):
    mask0 = (flags == 0) ## detections
    mask1 = (flags == 1) ## upper limits 
    y_det = y[mask0]
    y_ul = y[mask1]
    yerr_det = yerr[mask0] ## upper errors
    yerr1_det = yerr1[mask0] ## lower errors 
    P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, log_f = theta 
    y_model = model(x, [P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11])
    y_model_ul = y_model[mask1]
    y_model_det = y_model[mask0] 
    
    if (y_model_det[-2] < y_det[-2]):
        yerr_det[-2] = yerr1_det[-2]
    if (y_model_det[-1] < y_det[-1]):
        yerr_det[-1] = yerr1_det[-1]    
    
    sigma2 = yerr_det**2 + np.exp(2 * log_f)  
    
    rms = 0.05
    temp = scipy.special.erf(((y_ul-y_model_ul)/((2**0.5)*rms)))
    loglike_lim = 1.*np.sum(np.log((np.pi/2.)**0.5*rms*(1.+temp)))
    loglike_det = -0.5*np.sum((y_det - y_model_det)**2/sigma2 + np.log(sigma2))
    
    return  loglike_det +  loglike_lim 

def log_prior_LeMoC(theta):
    P1, P2, P3, P4, P5, P6, P7, log_f = theta
    if (14 < P1 < 17) and (-2 < P2 < 2) and (0. < P3 < 4.5) and\
        (5.0 < P4 < 7) and (-6 < P5 < -2) and (1.5 < P6 < 3) and\
        (0 < P7 < 3) and (-6.0 < log_f < 1.0):
        return 0.0
    return -np.inf

def log_prior_LeHaMoC(theta):
    P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, log_f = theta
    if (14 < P1 < 17) and (-1 < P2 < 3) and (0. < P3 < 3) and\
        (4. < P4 < 6) and (-5 < P5 < 0) and (1.1 < P6 < 3) and\
        (0. < P7 < 5) and (6. < P8 < 10.) and (-5 < P9 < 0) and\
        (1.1 < P10 < 3) and (0 < P11 < 3) and (-7.0 < log_f < 1.0):
        return 0.0
    return -np.inf

def log_probability(theta, x, y, yerr, yerr1, flags, flag_c):
    if flag_c == 'LeMoC':
        lp = log_prior_LeMoC(theta)
        ll = log_likelihood_LeMoC(theta, x, y, yerr, yerr1, flags)
    if flag_c == 'LeHaMoC':
        lp = log_prior_LeHaMoC(theta)
        ll = log_likelihood_LeHaMoC(theta, x, y, yerr, yerr1, flags)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ll

############################################################################
############################################################################


# Initial guess of parameters# 
if flag_c == 'LeMoC':
## LEPTONIC MODEL
    params = [15.25808, 0.038818, 3., 6.081445, -4., 2., 1.3]
    # labels = ['log(R [cm])', 'log(B [G])', '$log(\gamma_{min})$', '$log(\gamma_{max})$', '$log(l_e)$', 'p', '$log(\delta)$', '$log(f)$']
if flag_c == 'LeHaMoC':    
## HADRONIC MODEL
    params = [15.5, 1., 0.5, 5., -3., 1.2,  0.5, 7.3, -4.5, 2., 1.]
    # labels = ['log(R [cm])', 'log(B [G])', '$log(\gamma_{e,min})$', '$log(\gamma_{e,max})$', '$log(l_e)$', '$p_e$', '$log(\gamma_{p,min})$', '$log(\gamma_{p,max})$', '$log(l_p)$', 'p_p', '$log(\delta)$','logf']

# construct Model instance
x = np.linspace(7, 30, 100)
model = Model(D, z)   
y_model = model(x, params)  

# read data files (from https://ui.adsabs.harvard.edu/abs/2020ApJ...899..113P/abstract)
xd, yd, yderr_hi, yderr_lo, flags = read_data('OUsed_5BZBJ09553551-11jan2020.txt','SED_fermi_data.txt' )

# initialization of chains
init = [*params,-2] 
pos = init + 1e-2 * np.random.randn(48, len(init))
nwalkers, ndim = pos.shape     

# define output filename
filename = directory+"/out.h5" 
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)


############################################################################
############################################################################

# Run MCMC with multiprocessors 
steps = int(sys.argv[2]) # integer
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,  args=(xd, yd, yderr_hi, yderr_lo, flags, flag_c), pool=pool, backend = backend)
    start = timeit.default_timer()
    sampler.run_mcmc(pos, steps, progress=True)
    end = timeit.default_timer()
    multi_time = end - start
    print("Multiprocessing took {0:.1f} seconds".format(multi_time)) 
    
 