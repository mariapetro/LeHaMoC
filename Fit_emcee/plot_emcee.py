#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 15:47:10 2023

@author: mapet
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 14:16:58 2023

@author: mapet
"""
import astropy.units as u
from astropy import constants as const
import corner
import emcee
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd 
import sys
import os 

if len(sys.argv) != 4:
    print('missing code module')
    print('try something like this:')
    print('python plot_emcee.py LeMoC 5000 100')
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

# Directory for saving chains
out_dir = 'chains/'+flag_c
# Directory for saving plots 
log_dir = 'plots'


# Define plot labels 
if flag_c == 'LeMoC':
## LEPTONIC MODEL
    labels = ['log(R [cm])', 'log(B [G])', '$log(\gamma_{min})$', '$log(\gamma_{max})$', '$log(l_e)$', 'p', '$log(\delta)$', '$log(f)$']
if flag_c == 'LeHaMoC':    
## HADRONIC MODEL
    labels = ['log(R [cm])', 'log(B [G])', '$log(\gamma_{e,min})$', '$log(\gamma_{e,max})$', '$log(l_e)$', '$p_e$', '$log(\gamma_{p,min})$', '$log(\gamma_{p,max})$', '$log(l_p)$', 'p_p', '$log(\delta)$','logf']


#######################
#SED file reader # 
#######################
def read_data(filename2, filename1):
    # read gamma-ray data
    data_fermi = np.loadtxt(filename1, usecols=range(0, 4))
    dat_fermi = pd.DataFrame(data_fermi, columns=['E0', 'vFv0', 'vFve', 'flag'])
    dat_fermi['E'] = np.log10(dat_fermi['E0'])
    dat_fermi['vFv'] = np.log10(dat_fermi['vFv0']*1e6*(u.eV).to(u.erg)) # MeV/cm2/s to erg/cm2/s
    vFve_hi_log = np.log10(dat_fermi['vFv0']+dat_fermi['vFve']) - np.log10(dat_fermi['vFv0'])
    vFve_lo_log = -np.log10(dat_fermi['vFv0']-dat_fermi['vFve']) + np.log10(dat_fermi['vFv0'])
    
    # read multi-wavelength data up to X-rays
    data = np.loadtxt(filename2, usecols=range(0, 4))
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

############################################################################
############################################################################

## Read HDF5 file with MCMC results
#reader = emcee.backends.HDFBackend( out_dir + "/out.h5" , read_only=True)  
reader = emcee.backends.HDFBackend("chains/psyn-lim.h5" , read_only=True)  

burnin = int(sys.argv[2]) # integer
thin = int(sys.argv[3]) # integer

samples = reader.get_chain(discard=burnin, thin=thin)
flat_samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin) 

print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
print("flat log prob shape: {0}".format(log_prob_samples.shape)) 
 
steps, nwalkers, ndim =  samples.shape

############################################################################
############################################################################

## Create plot of chains ----------------------------------------------- 
fig, axes = plt.subplots(ndim, figsize=(12, 12), sharex=True)
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.2)
    ax.set_xlim(0, len(samples)+1)
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

ax.set_xlabel('Steps')
plt.savefig("{}/chain.png".format(log_dir),bbox_inches='tight',dpi=150)    
plt.show()


## Create SED plot with 100 randomly selected samples from posterior ----------------------------------------------- 
inds = np.random.randint(len(flat_samples), size=100)

## Read data files (from https://ui.adsabs.harvard.edu/abs/2020ApJ...899..113P/abstract)
xd, yd, yderr_hi, yderr_lo, flags = read_data('OUsed_5BZBJ09553551-11jan2020.txt','SED_fermi_data.txt' )

## Construct Model instance
x = np.linspace(7, 30, 100)
model = Model(D, z)   

plt.ylim([-14.5,-10.5])
plt.xlim([11,29])
sample = flat_samples[5] 
y_model = model(x, sample[:-1])

for ind in inds:
    sample = flat_samples[ind] 
    y_model = model(x, sample[:-1])
    plt.plot(x, y_model, "C1",  alpha=0.1)

plt.errorbar(xd[len(xd)-7:len(xd)], yd[len(xd)-7:len(xd)], yerr=[yderr_lo[len(xd)-7:len(xd)],yderr_hi[len(xd)-7:len(xd)]], uplims = flags[len(xd)-7:len(xd)] , marker = '', linestyle = 'None', color='m')
plt.errorbar(xd[0:len(xd)-7], yd[0:len(xd)-7],yerr = yderr_lo[0:len(xd)-7],marker='',linestyle='None', color='k')
plt.plot(xd[0:len(xd)-7],  yd[0:len(xd)-7], 'ok', alpha=0.5)
plt.plot(xd[len(xd)-7:len(xd)],  yd[len(xd)-7:len(xd)], 'om', alpha = 0.5)
plt.xlabel(r'$log(\nu$ [Hz])')
plt.ylabel(r'$log(\nu \, F_{\nu}$ [$erg\, cm^{-2} \, s^{-1}$])')
plt.legend(loc='upper left')
plt.savefig("{}/SED.png".format(log_dir),bbox_inches='tight',dpi=150)
plt.show()


# ## Creater corner plot ----------------------------------------------- 
fig = corner.corner(flat_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], plot_datapoints=False, show_titles=True, color='darkslategrey')
 
# Extract the axes
axes = np.array(fig.axes).reshape((ndim, ndim))

# Loop over the histograms
for yi in range(ndim-1):
    for xi in range(yi):
        ax = axes[yi, xi]
plt.savefig("{}/cornerplot.png".format(log_dir),bbox_inches='tight',dpi=150)
plt.show()