import astropy.units as u
from astropy import constants as const
import numpy as np

G = (const.G).cgs.value
c = (const.c).cgs.value
Ro = (const.R_sun).cgs.value
Mo = (const.M_sun).cgs.value
yr = (u.yr).to(u.s)
kpc = (u.kpc).to(u.cm)
pc = (u.pc).to(u.cm)
m_pr = (u.M_p).to(u.g)
m_el = (u.M_e).to(u.g)
kb = (const.k_B).cgs.value
h = (const.h).cgs.value
q = (const.e.gauss).value
sigmaT = (const.sigma_T).cgs.value
eV = (u.eV).to(u.erg)
B_cr = 2*np.pi*m_el**2*c**3/(h*q)