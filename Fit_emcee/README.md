# Emcee modeling of blazar SED with LeHaMoC

As an indicative application we choose 3HSP J095507.9+355101 (Giommi et al. 2020; Paliya et al. 2020) at redshift z = 0.557 (Paiano et al. 2020; Paliya et al.
2020) that belongs to the rare class of extreme blazars (See Section 4.1 in http://arxiv.org/abs/2308.06174). Observational data are from https://ui.adsabs.harvard.edu/abs/2020ApJ...899..113P/abstract 


## Files Included

1. **Code.py**: This file contains the leptonic (LeMoC) and hadronic (LeHaMoC) radiative modules of the code. It takes user-defined parameters from `Parameters*.txt` to set up the simulation. 
**Note**: Currently LeHaMoC includes only proton synchrotron radiation; other hadronic processes will be add to the repository soon. 

2. **LeHaMoC_f.py**: This file includes all the necessary formulas for calculating emissivities and energy loss rates for various processes such as Electron and Proton Synchrotron, Inverse Compton (IC), gamma-gamma absorption, pair creation, and Synchrotron self-absorption.

3. **fit_emcee.py**: This script performs MCMC fitting to the blazar SED (see *txt files) using emcee and multiprocessing. The script creates a directory "chains/flag_c" where the output file (out.h5) is saved. 

Here, flag_c = "LeMoC" or "LeHaMoC" determines if leptonic or hadronic SED fitting will be performed.

Example of running the script: 

- *python fit_emcee.py flag_c steps*

where steps (integer) is the number of steps that each chain is propagated. The default number of walkers is 48.

4. **plot_emcee.py**: This script loads the observational data and the MCMC results (from out.h5) and creates 3 plots:

- Chain plot 

- SED plot with 100 random realizations from posterior distributions

- Corner plot

## Free Parameters Used in fit_emcee

- For leptonic fitting: log10(R0), log10(B0), log10(g_min_el), log10(g_min_pr), log10(comp_el), p_el, log10(doppler) 

- For hadronic fitting: log10(R0), log10(B0), log10(g_min_el), log10(g_max_el), log10(comp_el), p_el, log10(g_min_pr), log10(g_max_pr), log10(comp_pr), p_pr, log10(doppler) 


## Simulation Parameters

You can customize various parameters in the `Parameters.txt` file:

- **time_init**: Initial time of the simulation, measured in units of the initial radius over the speed of light (R0/c).

- **time_end**: The final time of the simulation, also measured in units of the initial radius over the speed of light (R0/c).

- **step_alg**: Step size used in the algorithm, expressed in units of the initial radius over the speed of light (R0/c).

- **grid_g_pr**: Number of grid points between g_min_el and g_max_el.

- **grid_g_pr**: Number of grid points between g_min_pr and g_max_pr.

- **grid_nu**: Number of grid points for photons' frequency.

- **Vexp**: Expansion velocity in units of the speed of light (c).

- **m**: Power-law index of the magnetic field due to source expansion.

- **Ad_l_flag**: Adiabatic losses flag (1 to include, 0 to exclude).

- **inj_flag**: Electron injection profile (1 for continuous, 0 for instantaneous)

- **Syn_l_flag**: Synchrotron losses flag (1 to include, 0 to exclude).

- **Syn_emis_flag**: Synchrotron emission flag (1 to include, 0 to exclude).

- **IC_l_flag**: Inverse Compton scattering losses flag (1 to include, 0 to exclude).

- **IC_emis_flag**: Inverse Compton scattering emission flag (1 to include, 0 to exclude).

- **SSA_l_flag**: Synchrotron Self-absorption losses flag (1 to include, 0 to exclude).

- **gg_flag**: Gamma-gamma absorption-emission flag (1 to include, 0 to exclude).

- **esc_flag**: Escape of electrons and photons flag (1 to include, 0 to exclude).

- **BB_flag**: Black body flag (1 to include, 0 to exclude).

- **BB_temperature**: Common logarithm of the Black body temperature in Kelvin (K).

- **GB_ext**: External Grey Body photon field flag (1 to include, 0 to exclude).

- **PL_flag**: External power-law photon field flag.

- **dE_dV_ph**: Energy density in erg cm^{-3} of the external power-law photon field.

- **nu_min_ph**: Minimum frequency of the power-law photon field.

- **nu_max_ph**: Maximum frequency of the power-law photon field.

- **s_ph**: Power-law index of the power-law photon field.

- **User_ph**: External user photon field flag (1 to include, 0 to exclude). If included, provide a `.txt` file named 'Photons_spec_user.txt' with columns (nu[Hz],dN/dVdnu[cm^{-3}Hz^{-1}]). 

## Dependencies

- Python 3

 
