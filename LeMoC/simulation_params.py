from dataclasses import dataclass, field, fields
from typing import Any
import numpy as np

@dataclass
class SimulationParams:
    time_init: float = field(default=0.0, metadata={'comment': 'Initial time of the simulation (R0/c)'})
    time_end: float = field(default=10.0, metadata={'comment': 'Final time of the simulation (R0/c)'})
    step_alg: float = field(default=2.0, metadata={'comment': 'Step size in the algorithm (R0/c)'})
    PL_inj: float = field(default=0.0, metadata={'comment': 'Power law injection flag (1 to include, 0 otherwise)'})
    g_min_el: float = field(default=0.0, metadata={'comment': 'Minimum Lorentz factor of electrons'})
    g_max_el: float = field(default=11.0, metadata={'comment': 'Maximum Lorentz factor of electrons'})
    g_el_PL_min: float = field(default=0.0, metadata={'comment': 'Min Lorentz factor for power-law electrons'})
    g_el_PL_max: float = field(default=5.5, metadata={'comment': 'Max Lorentz factor for power-law electrons'})
    grid_g_el: float = field(default=300, metadata={'comment': 'Grid points between g_min_el and g_max_el'})
    g_min_pr: float = field(default=0.0, metadata={'comment': 'Minimum Lorentz factor of protons'})
    g_max_pr: float = field(default=8.0, metadata={'comment': 'Maximum Lorentz factor of protons'})
    g_pr_PL_min: float = field(default=0.0, metadata={'comment': 'Min Lorentz factor for power-law protons'})
    g_pr_PL_max: float = field(default=6.93, metadata={'comment': 'Max Lorentz factor for power-law protons'})
    grid_g_pr: float = field(default=160, metadata={'comment': 'Grid points between g_min_pr and g_max_pr'})
    grid_nu: float = field(default=100.0, metadata={'comment': 'Grid points for photons\' frequency'})
    p_el: float = field(default=2.01, metadata={'comment': 'Power-law index of electron distribution'})
    L_el: float = field(default=40.6, metadata={'comment': 'Electron luminosity in erg s^{-1}'})
    p_pr: float = field(default=2.01, metadata={'comment': 'Power-law index of proton distribution'})
    L_pr: float = field(default=46.46, metadata={'comment': 'Proton luminosity in erg s^{-1}'})
    Vexp: float = field(default=0.0, metadata={'comment': 'Expansion velocity (c)'})
    R0: float = field(default=1e16, metadata={'comment': 'Log of initial radius of the blob (cm)'})
    B0: float = field(default=0.1, metadata={'comment': 'Magnetic field intensity (G)'})
    m: float = field(default=0.0, metadata={'comment': 'Magnetic field power-law index due to expansion'})
    delta: float = field(default=1.0, metadata={'comment': 'Doppler factor'})
    inj_flag: float = field(default=1.0, metadata={'comment': 'Electron injection profile flag'})
    Ad_l_flag: float = field(default=1.0, metadata={'comment': 'Adiabatic losses flag'})
    Syn_l_flag: float = field(default=1.0, metadata={'comment': 'Synchrotron losses flag'})
    Syn_emis_flag: float = field(default=1.0, metadata={'comment': 'Synchrotron emission flag'})
    IC_l_flag: float = field(default=1.0, metadata={'comment': 'Inverse Compton losses flag'})
    IC_emis_flag: float = field(default=1.0, metadata={'comment': 'Inverse Compton emission flag'})
    SSA_l_flag: float = field(default=1.0, metadata={'comment': 'SSA losses flag'})
    gg_flag: float = field(default=1.0, metadata={'comment': 'Gamma-gamma absorption-emission flag'})
    pg_pi_l_flag: float = field(default=1.0, metadata={'comment': 'Photopion losses flag'})
    pg_pi_emis_flag: float = field(default=1.0, metadata={'comment': 'Photopion emission flag'})
    pg_BH_l_flag: float = field(default=1.0, metadata={'comment': 'Bethe-Heitler losses flag'})
    pg_BH_emis_flag: float = field(default=1.0, metadata={'comment': 'Bethe-Heitler emission flag'})
    n_H: float = field(default=0.0, metadata={'comment': 'Cold proton number density (#/cm^3)'})
    pp_l_flag: float = field(default=0.0, metadata={'comment': 'Proton-proton losses flag'})
    pp_ee_emis_flag: float = field(default=0.0, metadata={'comment': 'Pairs emission from pp interactions flag'})
    pp_g_emis_flag: float = field(default=0.0, metadata={'comment': 'Photon emission from pp interactions flag'})
    pp_nu_emis_flag: float = field(default=0.0, metadata={'comment': 'Neutrino emission from pp interactions flag'})
    neutrino_flag: float = field(default=1.0, metadata={'comment': 'Neutrino from photopion interactions flag'})
    esc_flag_el: float = field(default=1.0, metadata={'comment': 'Escape of pairs flag'})
    esc_flag_pr: float = field(default=1.0, metadata={'comment': 'Escape of protons flag'})
    BB_flag: float = field(default=0.0, metadata={'comment': 'Black body flag'})
    temperature: float = field(default=5.0, metadata={'comment': 'Log of Black body temperature (K)'})
    GB_ext: float = field(default=1.0, metadata={'comment': 'External Grey Body photon field flag'})
    PL_flag: float = field(default=0.0, metadata={'comment': 'External power-law photon field flag'})
    dE_dV_ph: float = field(default=0.0, metadata={'comment': 'Energy density of external power-law photon field (erg cm^{-3})'})
    nu_min_ph: float = field(default=0.0, metadata={'comment': 'Min frequency of power-law photon field'})
    nu_max_ph: float = field(default=0.0, metadata={'comment': 'Max frequency of power-law photon field'})
    s_ph: float = field(default=2.01, metadata={'comment': 'Power-law index of the photon field'})
    User_ph: float = field(default=0.0, metadata={'comment': 'External user photon field flag'})
    out_name: str = field(default="test", metadata={'comment': 'Output file name'})


def create_param_file(params: SimulationParams, file_name: str,with_comment=False):
    with open(file_name, 'w') as file:
        for f in fields(params):
            comment = f.metadata.get('comment', '')
            value = getattr(params, f.name)
            if(f.name=="R0"):
                value = np.log10(value)
            line = f"{f.name} = {value}"
            if(with_comment):
                if comment:
                    line += f"  # {comment}"
            file.write(line + "\n")

def parse_value(value: str, field_type: Any) -> Any:
    if field_type == float:
        return float(value)
    elif field_type == int:
        return int(value)
    elif field_type == str:
        return value
    else:
        raise ValueError(f"Unsupported field type: {field_type}")

def load_param_file(file_name: str) -> SimulationParams:
    params = SimulationParams()
    with open(file_name, 'r') as f:
        for line in f:
            if not line.strip() or line.strip().startswith('#'):
                continue
            param_name, param_value = line.split('=', 1)
            param_name = param_name.strip()
            param_value = param_value.split('#')[0].strip()
            for field in fields(params):
                if field.name == param_name:
                    setattr(params, param_name, parse_value(param_value, field.type))
                    break
    return params