# LeMoC: Leptonic Modeling Code for High-Energy Astrophysical Sources

LeMoC is the leptonic module of LeHaMoC, a code designed for modeling/simulating high-energy astrophysical sources. It simulates the behavior of relativistic pairs interacting with magnetic fields and photons in a spherical region. The physical processes that are included are:


1. Synchrotron emission and self-absorption
2. Inverse Compton scattering
3. Photon-photon pair production
4. Adiabatic losses 

The user can also model expanding spherical sources with a variable magnetic field strength. The user can also define 3 types of external radiation fields:

-Grey body or black body

-Power-law

-Tabulated

## Files Included

1. **LeMoC.py**: This file contains the main code for the LeMoC simulation. It takes user-defined parameters from `Parameters*.txt` to set up the simulation.

2. **Parameters.txt**: Users should edit this file to specify the parameters for their simulation. This includes setting up the initial conditions, time interval, and other relevant parameters. There are currently two input files ready for use that correspond to Tests 1 and 3 described in Stathopoulos et al.

3. **LeHaMoC_f.py**: This file includes all the necessary formulas for calculating emissivities and energy loss rates for various processes such as Synchrotron, Inverse Compton (IC), gamma-gamma absorption, pair creation, and Synchrotron self-absorption.


## Getting Started

i. Clone this repository to your local machine using `git clone <repository-url>`

ii. Edit the `Parameters.txt` file to set up the initial conditions and simulation parameters according to your needs.

## Simulation Parameters

To configure the LeMoC simulation, you can customize various parameters in the `Parameters.txt` file:

1. **time_init**: Initial time of the simulation, measured in units of the initial radius over the speed of light (R0/c).

2. **time_end**: The final time of the simulation, also measured in units of the initial radius over the speed of light (R0/c).

3. **step_alg**: Step size used in the algorithm, expressed in units of the initial radius over the speed of light (R0/c).

4. **g_min_el**: Minimum Lorentz factor of electrons on the grid in log10.

5. **g_max_el**: Maximum Lorentz factor of electrons on the grid in log10.

6. **g_el_PL_min**: Minimum Lorentz factor of power-law electrons in log10.

7. **g_el_PL_max**: Maximum Lorentz factor of power-law electrons in log10.

8. **grid_g_el**: Number of grid points between g_min_el and g_max_el.

9. **grid_nu**: Number of grid points for photons' frequency.

10. **p_el**: Power-law index of the electron distribution.

11. **L_el**: Log10 of luminosity of electrons in erg s^{-1}.

12. **Vexp**: Expansion velocity in units of the speed of light (c).

13. **R0**: Log10 of the initial radius of the spherical blob in centimeters (cm).

14. **B0**: Magnetic field intensity in Gauss (G).

15. **m**: Power-law index of the magnetic field due to source expansion.

16. **delta**: Doppler factor.

17. **inj_flag**: Electron injection profile (1 for continuous, 0 for instantaneous).

18. **Ad_l_flag**: Adiabatic losses flag (1 to include, 0 to exclude).

19. **Syn_l_flag**: Synchrotron losses flag (1 to include, 0 to exclude).

20. **Syn_emis_flag**: Synchrotron emission flag (1 to include, 0 to exclude).

21. **IC_l_flag**: Inverse Compton scattering losses flag (1 to include, 0 to exclude).

22. **IC_emis_flag**: Inverse Compton scattering emission flag (1 to include, 0 to exclude).

23. **SSA_l_flag**: Synchrotron Self-absorption losses flag (1 to include, 0 to exclude).

24. **gg_flag**: Gamma-gamma absorption-emission flag (1 to include, 0 to exclude).

25. **esc_flag**: Escape of pairs flag (1 to include, 0 to exclude).

26. **BB_flag**: Black body flag (1 to include, 0 to exclude).

27. **BB_temperature**: Black body temperature in Kelvin (K).

28. **GB_ext**: External Grey Body photon field flag (1 to include, 0 to exclude).

29. **PL_flag**: External power-law photon field flag.

30. **dE_dV_ph**: Energy density in erg cm^{-3} of the external power-law photon field.

31. **nu_min_ph**: Minimum frequency of the power-law photon field.

32. **nu_max_ph**: Maximum frequency of the power-law photon field.

33. **s_ph**: Power-law index of the power-law photon field.

34. **User_ph**: External user photon field flag (1 to include, 0 to exclude). If included, provide a .txt file named 'Photons_spec_user.txt' with columns (log(nu[Hz]),log(dN/dVdnu[cm^{-3}Hz^{-1}])).

iii. Run the `LeMoC.py` code using a compatible Python interpreter. Make sure all necessary dependencies are installed. Example:

- *python LeMoC.py Parameters_Test3.txt Test3*


iv. Once the simulation is complete, open the `Plotting_Tool.ipynb` notebook to visualize and analyze the simulation results. Follow the instructions provided in the notebook.

## Output files
The default output files are:

1. Particles_Distribution.txt (1st col: log10(gamma_e), 2nd col: log10(dN_e/(dV dgamma_e)) [cm^(-3)]
2. Photons_Distribution.txt (1st col: log10(v), 2nd col: log10(dL_ph/(dv)) [erg s^(-1)]

## Dependencies

- Python 3 (check also the .yml file)


## Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit pull requests.

## License

This project is licensed under the [GNU GPLv3]
