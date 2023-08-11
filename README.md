# LeMoC: Leptonic Modeling Code for High-Energy Astrophysical Sources

LeMoC is a versatile time-dependent leptonic modeling code designed for simulating high-energy astrophysical sources. It simulates the behavior of relativistic electrons interacting with magnetic fields and photons in a spherical region.

## Files Included

1. **LeMoC.py**: This file contains the main code for the LeMoC simulation. It takes user-defined parameters from `Parameters.txt` to set up the simulation.

2. **Parameters.txt**: Users should edit this file to specify the parameters for their simulation. This includes setting up the initial conditions, time interval, and other relevant parameters.

3. **LeMoC_f.py**: This file includes all the necessary formulas for calculating emissivities and energy loss rates for various processes such as Synchrotron, Inverse Compton (IC), gamma-gamma absorption, pair creation, and Synchrotron self-absorption.

4. **Plotting_Tool.ipynb**: This Jupyter Notebook provides a tool for users to visualize and analyze the simulation results. Users can generate plots to better understand the behavior of the simulated astrophysical source.

## Getting Started

1. Clone this repository to your local machine using `git clone <repository-url>`

2. Edit the `Parameters.txt` file to set up the initial conditions and simulation parameters according to your needs.

## Simulation Parameters

To configure the LeMoC simulation, you can customize various parameters in the `Parameters.txt` file:

- **time_init**: Initial time of the simulation, measured in units of the initial radius over the speed of light (R0/c).

- **time_end**: The final time of the simulation, also measured in units of the initial radius over the speed of light (R0/c).

- **step_alg**: Step size used in the algorithm, expressed in units of the initial radius over the speed of light (R0/c).

- **g_min_el**: Minimum Lorentz factor of electrons on the grid.

- **g_max_el**: Maximum Lorentz factor of electrons on the grid.

- **g_PL_min**: Minimum Lorentz factor of power-law electrons.

- **g_PL_max**: Maximum Lorentz factor of power-law electrons.

- **grid_g_el**: Number of grid points between g_min_el and g_max_el.

- **grid_nu**: Number of grid points for photons' frequency.

- **p_el**: Power-law index of the electron distribution.

- **L_el**: Luminosity of electrons in erg s^{-1}.

- **Vexp**: Expansion velocity in units of the speed of light (c).

- **R0**: Common logarithm of the initial radius of the spherical blob in centimeters (cm).

- **B0**: Magnetic field intensity in Gauss (G).

- **m**: Power-law index of the magnetic field due to source expansion.

- **delta**: Doppler factor.

- **Ad_l_flag**: Adiabatic losses flag (1 to include, 0 to exclude).

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


3. Run the `LeMoC.py` code using a compatible Python interpreter. Make sure to have all necessary dependencies installed.

4. Once the simulation is complete, open the `Plotting_Tool.ipynb` notebook to visualize and analyze the simulation results. Follow the instructions provided in the notebook.

## Dependencies

- Python 3


## Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit pull requests.

## License

This project is licensed under the [LICENSE]
