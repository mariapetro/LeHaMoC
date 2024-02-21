# LeHaMoC: Leptonic-Hadronic Modeling Code for High-Energy Astrophysical Sources

LeHaMoC is a leptohadronic modeling code designed for simulating high-energy astrophysical sources. It simulates the behavior of relativistic pairs, protons interacting with magnetic fields and photons in a spherical region. The physical processes that are included are:


1. Synchrotron emission and self-absorption
2. Inverse Compton scattering
3. Photon-photon pair production
4. Adiabatic losses 
5. Proton-photon pion production 
6. Proton-photon (Bethe-Heitler) pair production
7. Proton-proton collisions 

The user can also model expanding spherical sources with a variable magnetic field strength. The user can also define 3 types of external radiation fields:

-Grey body or black body

-Power-law

-Tabulated

## Files Included

1. **LeHaMoC.py**: This file contains the main code for the LeMoC simulation. It takes user-defined parameters from `Parameters*.txt` to set up the simulation.

2. **Parameters.txt**: Users should edit this file to specify the parameters for their simulation. This includes setting up the initial conditions, time interval, and other relevant parameters. There are currently two input files ready for use that correspond to Tests 1 and 3 described in Stathopoulos et al.

3. **LeHaMoC_f.py**: This file includes all the necessary formulas for calculating emissivities and energy loss rates for various processes such as Synchrotron, Inverse Compton (IC), gamma-gamma absorption, pair creation, and Synchrotron self-absorption.


## Getting Started

i. Clone this repository to your local machine using `git clone <repository-url>`

ii. Edit the `Parameters.txt` file to set up the initial conditions and simulation parameters according to your needs.

## Simulation Parameters

To configure the LeHaMoC simulation, you can customize various parameters in the `Parameters.txt` file:

1. **time_init**: Initial time of the simulation, measured in units of the initial radius over the speed of light (R0/c).

2. **time_end**: The final time of the simulation, also measured in units of the initial radius over the speed of light (R0/c).

3. **step_alg**: Step size used in the algorithm, expressed in units of the initial radius over the speed of light (R0/c).

4. **PL_inj**: Power law injection flag (1 to include, 0 to use distribution with exponential cut-offs).

5. **g_min_el**: Minimum Lorentz factor of electrons on the grid.

6. **g_max_el**: Maximum Lorentz factor of electrons on the grid.

7. **g_el_PL_min**: Minimum Lorentz factor of power-law electrons.

8. **g_el_PL_max**: Maximum Lorentz factor of power-law electrons.

9. **grid_g_el**: Number of grid points between g_min_el and g_max_el.

10. **g_min_pr**: Minimum Lorentz factor of protons on the grid.

11. **g_max_pr**: Maximum Lorentz factor of protons on the grid.

12. **g_pr_PL_min**: Minimum Lorentz factor of power-law protons.

13. **g_pr_PL_max**: Maximum Lorentz factor of power-law protons.

14. **grid_g_pr**: Number of grid points between g_min_pr and g_max_pr.

15. **grid_nu**: Number of grid points for photons' frequency.

16. **p_el**: Power-law index of the electron distribution.

17. **L_el**: Log10 of luminosity of electrons in erg s^{-1}.

18. **p_pr**: Power-law index of the proton distribution.

19. **L_pr**: Log10 of luminosity of protons in erg s^{-1}.

20. **Vexp**: Expansion velocity in units of the speed of light (c).

21. **R0**: Log10 of the initial radius of the spherical blob in centimeters (cm).

22. **B0**: Magnetic field intensity in Gauss (G).

23. **m**: Power-law index of the magnetic field due to source expansion.

24. **delta**: Doppler factor.

25. **inj_flag**: Electron injection profile (1 for continuous, 0 for instantaneous).

26. **Ad_l_flag**: Adiabatic losses flag (1 to include, 0 to exclude).

27. **Syn_l_flag**: Synchrotron losses flag (1 to include, 0 to exclude).

28. **Syn_emis_flag**: Synchrotron emission flag (1 to include, 0 to exclude).

29. **IC_l_flag**: Inverse Compton scattering losses flag (1 to include, 0 to exclude).

30. **IC_emis_flag**: Inverse Compton scattering emission flag (1 to include, 0 to exclude).

31. **SSA_l_flag**: Synchrotron Self-absorption losses flag (1 to include, 0 to exclude).

32. **gg_flag**: Gamma-gamma absorption-emission flag (1 to include, 0 to exclude).

33. **pg_pi_l_flag**: Photopion losses flag (1 to include, 0 to exclude).

34. **pg_pi_emis_flag**: Photopion emission flag (1 to include, 0 to exclude).

35. **pg_BH_l_flag**: Bethe-Heitler losses flag (1 to include, 0 to exclude).

36. **pg_BH_emis_flag**: Bethe-Heitler losses flag (1 to include, 0 to exclude).

37. **n_H**: Number density of cold protons in #/cm^{3}.

38. **pp_l_flag**: Proton-proton (pp) losses flag (1 to include, 0 to exclude).

39. **pp_ee_emis_flag**: Pairs emission from pp interactions flag (1 to include, 0 to exclude).

40. **pp_g_emis_flag**: Photon emission from pp interactions flag (1 to include, 0 to exclude).

41. **pp_nu_emis_flag**: Neutrino emission from pp interactions flag (1 to include, 0 to exclude).

42. **neutrino_flag**: Neutrino flag for photopion interactions flag (1 to include, 0 to exclude).

43. **esc_flag_el**: Escape of pairs flag (1 to include, 0 to exclude).

44. **esc_flag_pr**: Escape of protons flag (1 to include, 0 to exclude).

45. **BB_flag**: Black body flag (1 to include, 0 to exclude).

46. **BB_temperature**: Black body temperature in Kelvin (K).

47. **GB_ext**: External Grey Body photon field flag (1 to include, 0 to exclude).

48. **PL_flag**: External power-law photon field flag.

49. **dE_dV_ph**: Energy density in erg cm^{-3} of the external power-law photon field.

50. **nu_min_ph**: Minimum frequency of the power-law photon field.

51. **nu_max_ph**: Maximum frequency of the power-law photon field.

52. **s_ph**: Power-law index of the power-law photon field.

53. **User_ph**: External user photon field flag (1 to include, 0 to exclude). If included, provide a .txt file named 'Photons_spec_user.txt' with columns (nu[Hz],dN/dVdnu[cm^{-3}Hz^{-1}]).

iii. Run the `LeHaMoC.py` code using a compatible Python interpreter. Make sure to have all necessary dependencies installed. Example:

- *python LeHaMoC.py Parameters_Test3.txt Test3*


iv. Once the simulation is complete, open the `Plotting_Tool.ipynb` notebook to visualize and analyze the simulation results. Follow the instructions provided in the notebook.

## Output files
The default output files are:

1. Pairs_Distribution.txt (1st col: log10(gamma_e), 2nd col: log10(dN_e/(dV dgamma_e)) [cm^(-3)]
2. Photons_Distribution.txt (1st col: log10(v), 2nd col: log10(dL_ph/(dv)) [erg s^(-1)]
3. Protons_Distribution.txt (1st col: log10(gamma_p), 2nd col: log10(dN_p/(dV dgamma_p)) [cm^(-3)]
4. Neutrinos_Distribution.txt (1st col: log10(v), 2nd col: log10(dN_nu/(dv dV)) [neutrinos cm^(-3) Hz^(-1)]

## Dependencies

- Python 3 (check also the .yml file)


## Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit pull requests.

## License

This project is licensed under the [GNU GPLv3]
