# LeHaMoC: Leptonic-Hadronic Modeling Code for High-Energy Astrophysical Sources

LeHaMoC is a leptohadronic modeling code designed for simulating high-energy astrophysical sources. It simulates the behavior of relativistic pairs protons interacting with magnetic fields and photons in a spherical region. The physical processes that are included are:

-Synchrotron emission and self-absorption
-Inverse Compton scattering
-Photon-photon pair production
-Adiabatic losses
-Proton-photon pion production 
-Proton-photon (Bethe-Heitler) pair production 
-Proton-proton collisions 

The user can also model expanding spherical sources with a variable magnetic field strength. The user can also define 3 types of external radiation fields:

-Grey body or black body
-Power-law
-Tabulated

## Files Included
1. **LeHaMoC.py**: This file contains the main code for the LeHaMoC simulation. It takes user-defined parameters from Parameters*.txt to set up the simulation.

2. **Parameters.txt**: Users should edit this file to specify the parameters for their simulation. This includes setting up the initial conditions, time interval, and other relevant parameters. There are currently two input files ready for use that correspond to Tests 1 and 3 described in Stathopoulos et al.

3. **LeHaMoC_f.py**: This file includes all the necessary formulas for calculating emissivities and energy loss rates for various processes such as Synchrotron, Inverse Compton (IC), gamma-gamma absorption, pair creation, and Synchrotron self-absorption and all proton-photon interactions.

4. **Tests.ipynb**: This Jupyter Notebook provides a simple interface for users to visualize and analyze the simulation tests as presented in Section 3  in Stathopoulos et al. 2023. Users can generate plots to better understand the behavior of the simulated astrophysical scenarios.

5. **Phi_g_K&A.txt**: TABLE I from (Kelner & Aharonian 2008)

6. **Phi_e-nu_e_K&A.txt**: TABLE II from (Kelner & Aharonian 2008)

7. **Phi_g_leptons_K&A.txt**: TABLE III from (Kelner & Aharonian 2008)

8. **kp_pg.txt**: Energy-dependent inelasticity of a proton from (Stecker 1968)

9. **cross_section.csv**: Energy-dependent cross-section of photopion process from (Morejon et al. 2019)

10. **f(xi).csv**: \phi(\xi)/\xi^2 from (Blumenthal 1970)

11. **Test_1_el_dis_W_SSA.txt**: Test 1 pair distribution as obtained from the ATHEνΑ code 

12. **Test_1_ph_dis_W_SSA.txt**: Test 1 photon distribution as obtained from the ATHEνΑ code  

13. **pr_spec_E_p24_C_C.txt**: Test 2 protons distribution used for the test

14. **ph_spec_E_p24_C_C.txt**: Test 2 target photon field used for the simulation  

15. **el_spec_E_p24_C_C.txt**: Test 2 pair spectrum from photopion production as obtained from the ATHEνΑ code  

16. **BH_spec_E_p24_C_C.txt**: Test 2 pair spectrum from Bethe-Heitler process as obtained from the ATHEνΑ code

17. **pi0_spec_E_p24_C_C.txt**: Test 2 photon spectrum produced by the photopion process as obtained from the ATHEνΑ code

18. **nu_spec_E_p24_C_C.txt**: Test 2 neutrino spectrum produced by the photopion process as obtained from the ATHEνΑ code

19. **Photons_test_4_new.txt**: Test 4 photon spectrum as obtained from the ATHEνΑ code 

20. **neutrino_spec_obs_test4.txt**: Test 4 neutrino spectrum as obtained from the ATHEνΑ code  

## Getting Started
Clone this repository to your local machine using git clone <repository-url>

Edit the Parameters.txt file to set up the initial conditions and simulation parameters according to your needs.

## Simulation Parameters
To configure the LeHaMoC simulation, you can customize various parameters in the Parameters.txt file:

## Numbered Parameters

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

17. **L_el**: Luminosity of electrons in erg s^{-1}.

18. **p_pr**: Power-law index of the proton distribution.

19. **L_pr**: Luminosity of protons in erg s^{-1}.

20. **Vexp**: Expansion velocity in units of the speed of light (c).

21. **R0**: Common logarithm of the initial radius of the spherical blob in centimeters (cm).

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

46. **BB_temperature**: Common logarithm of the Black body temperature in Kelvin (K).

47. **GB_ext**: External Grey Body photon field flag (1 to include, 0 to exclude).

48. **PL_flag**: External power-law photon field flag.

49. **dE_dV_ph**: Energy density in erg cm^{-3} of the external power-law photon field.

50. **nu_min_ph**: Minimum frequency of the power-law photon field.

51. **nu_max_ph**: Maximum frequency of the power-law photon field.

52. **s_ph**: Power-law index of the power-law photon field.

53. **User_ph**: External user photon field flag (1 to include, 0 to exclude). If included, provide a .txt file named 'Photons_spec_user.txt' with columns (nu[Hz],dN/dVdnu[cm^{-3}Hz^{-1}]).

Run the LeHaMoC.py code using a compatible Python interpreter. Make sure to have all necessary dependencies installed. Example:
python LeHaMoC.py Parameters_Test.txt
