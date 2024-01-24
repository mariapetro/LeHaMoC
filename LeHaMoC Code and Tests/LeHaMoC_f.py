import numpy as np
from shapely.geometry import  LineString
from astropy import units as u
from astropy import constants as const 
import scipy.integrate as integrate
from scipy import interpolate
import pandas as pd
from numpy import inf
from scipy import stats

#######################
#tables# 
#######################
Phi_g_tab = pd.read_csv("Phi_g_K&A.txt", names=["eta_div_eta_0","s_g","d_g","B_g"], sep=" ",  skiprows=0)
Phi_el_tab = pd.read_csv("Phi_g_leptons_K&A.txt", names=["eta_div_eta_0","s_e+","d_e+","B_e+","s_anti_nu_mu","d_anti_nu_mu","B_anti_nu_mu","s_nu_mu","d_nu_mu","B_nu_mu","s_nu_e","d_nu_e","B_nu_e"], sep=" ",  skiprows=0)
Phi_el_1_tab = pd.read_csv("Phi_e-nu_e_K&A.txt", names=["eta_div_eta_0","s_e-","d_e-","B_e-","s_anti_nu_e","d_anti_nu_e","B_anti_nu_e"], sep=" ",  skiprows=0)
f_k_i = pd.read_csv('f(xi).csv',names=("k_i","fk_i"))
Cross_Section_pg = pd.read_csv('cross_section.csv',names=('Ph_En','C_S'))
kp_pg = pd.read_csv('kp_pg.txt',names=('e','k'),sep=" ")

#######################
#constants# 
#######################
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
h_0 = 0.313
r = 0.1458
E_th_pi = 1.22*10**(-3.) #in TeV

#Unints of rest mass in TeV
x_m_pr = 938.272046*10**(-6.)
x_m_el = 0.511*10**(-6.)
x_m_pi = 139.57*10**(-6.) #charged pion 
x_m_pi0 = 134.9766*10**(-6.) #neutral pion 
x_m_mu = 105.66*10**(-6.)
K_pi = 0.17

E_nu_space = np.logspace(10.,22.,50)*eV

#Read parameters file
fileName = "Parameters.txt"
fileObj = open(fileName)
params = {}
for line in fileObj:
    line=line.strip()
    key_value = line.split("=")
    params[key_value[0].strip()] = float(key_value[1].strip())
    
time_init = float(params['time_init']) #R0/c
time_end = float(params['time_end']) #R0/c
step_alg = float(params['step_alg']) #R0/c
g_min_el = float(params['g_min_el']) #log
g_max_el = float(params['g_max_el']) #log
g_el_PL_min = float(params['g_el_PL_min']) #log
g_el_PL_max = float(params['g_el_PL_max']) #log
grid_g_el = float(params['grid_g_el'])
g_min_pr = float(params['g_min_pr']) #log
g_max_pr = float(params['g_max_pr']) #log
g_pr_PL_min = float(params['g_pr_PL_min']) #log
g_pr_PL_max = float(params['g_pr_PL_max']) #log
grid_g_pr = float(params['grid_g_pr'])
grid_nu = float(params['grid_nu'])
p_el = float(params['p_el'])
L_el = float(params['L_el']) #log
p_pr = float(params['p_pr'])
L_pr = float(params['L_pr']) #log
Vexp = float(params['Vexp'])*c #/c
R0 = 10**float(params['R0']) #log
B0 = float(params['B0'])
m = float(params['m'])
delta = float(params["delta"])
Ad_l_flag = float(params['Ad_l_flag'])
Syn_l_flag = float(params['Syn_l_flag'])
Syn_emis_flag = float(params['Syn_emis_flag'])
IC_l_flag = float(params['IC_l_flag'])
IC_emis_flag = float(params['IC_emis_flag'])
SSA_l_flag = float(params['SSA_l_flag'])
gg_flag = float(params['gg_flag'])
pg_pi_l_flag = float(params['pg_pi_l_flag'])
pg_pi_emis_flag = float(params['pg_pi_emis_flag'])
pg_BH_l_flag = float(params['pg_BH_l_flag'])
pg_BH_emis_flag = float(params['pg_BH_emis_flag'])
n_H = float(params['n_H']) 
pp_l_flag = float(params['pp_l_flag'])
pp_ee_emis_flag = float(params['pp_ee_emis_flag'])
pp_g_emis_flag = float(params['pp_g_emis_flag'])
neutrino_flag = float(params['neutrino_flag'])
esc_flag_el = float(params['esc_flag_el'])
esc_flag_pr = float(params['esc_flag_pr'])
BB_flag = float(params['BB_flag'])
temperature = 10**float(params['temperature']) #log 
GB_ext = float(params['GB_ext'])
PL_flag = float(params['PL_flag'])
dE_dV_ph = float(params['dE_dV_ph'])
nu_min_ph = float(params['nu_min_ph'])
nu_max_ph = float(params['nu_max_ph'])
s_ph = float(params['s_ph'])
User_ph = float(params['User_ph']) 
grid_size_pr = grid_g_pr
g_pr = np.logspace(g_min_pr,g_max_pr,int(grid_size_pr))
g_pr_mp = np.array([(g_pr[im+1]+g_pr[im-1])/2. for im in range(0,len(g_pr)-1)])
dg_pr = np.array([((g_pr[im+1])-(g_pr[im-1]))/2. for im in range(1,len(g_pr)-1)])
dg_l_pr = np.log(g_pr[1])-np.log(g_pr[0])

#######################
#functions# 
#######################

# expanding source radius 
def R(R0,t,t_i,Vexp):
    return R0+Vexp*(t-t_i)

# magnetic field strength
def B(B0,R0,R,m):
    return B0*(R0/R)**m

# volume of spherical source
def Volume(R):
    return 4.*np.pi/3.*R**3. 

# electron normalization for a power-law injection with sharp off
def Q_el_Lum(L_el,p_el,g_min_el,g_max_el):
    if p_el == 2.:
        return L_el/(m_el*c**2.*np.log(g_max_el/g_min_el)) 
    else:
        return L_el*(-p_el+2.)/(m_el*c**2.*(g_max_el**(-p_el+2.)-g_min_el**(-p_el+2.)))
    
# proton Normalization for a power-law injection with sharp off
def Q_pr_Lum(L_pr,p_pr,g_min_pr,g_max_pr):
    if p_pr == 2.:
        return L_pr/(m_pr*c**2.*np.log(g_max_pr/g_min_pr)) 
    else:
        return L_pr*(-p_pr+2.)/(m_pr*c**2.*(g_max_pr**(-p_pr+2.)-g_min_pr**(-p_pr+2.)))

def Q_norm_exp_c(R0,comp,g,g_max_ind,p_i):
    return 4.*np.pi*R0*c*(comp)/(sigmaT*np.trapz(g[:g_max_ind]**(-p_i+2.)*np.exp(-g[:g_max_ind]/g[g_max_ind]),np.log(g[:g_max_ind])))

# converts electron compactness to luminosity
def Lum_e_inj(comp,R):
    return 4.*np.pi*R*m_el*c**3.*comp/(sigmaT)

# convert proton compactness to luminosity
def Lum_p_inj(comp,R0):
    return 4.*np.pi*R0*m_pr*c**3.*comp/(sigmaT)

#Synchrotron critical frequency (Radiative Processes in Astrophysics, by George B. Rybicki, Alan P. Lightman, Wiley-VCH , June 1986.)
def nu_c(gamma,B):
    return 3./(4.*np.pi)*q*B/(m_el*c)*gamma**2.

#Synchrotron emissivity dN/dVdνdtdΩ (Relativistic Jets from Active Galactic Nuclei, by M. Boettcher, D.E. Harris, ahd H. Krawczynski, Berlin: Wiley, 2012)
def Q_syn_space(Np,B,nu,a_cr,C_syn,g):                       
    syn_em = C_syn*B**(2./3.)*nu**(-2./3.)*(np.trapz(Np*g**(1./3.)*np.exp(-nu/(a_cr*g**2.)), np.log(g)))
    return syn_em

#Synchrotron self absorption coefficient (High Energy Radiation from Black Holes: Gamma Rays, Cosmic Rays, and Neutrinos by Charles D. Dermer and Govind Menon. Princeton Univerisity Press, 2009)
def aSSA(N_el,B,nu,g,dg_l_el):
    return q**3.*B*nu**(-5./3.)/(2.*m_el**2.*c**2.*0.8975)*np.trapz((2.*nu_c(g[1:-1],B))**(-1./3.)*np.exp(-nu/nu_c(g[1:-1],B))*(np.diff(N_el[1:])/dg_l_el-2.*N_el[1:-1]),np.log(g[1:-1]))

#Synchrotron self absorption coefficient (High Energy Radiation from Black Holes: Gamma Rays, Cosmic Rays, and Neutrinos by Charles D. Dermer and Govind Menon. Princeton Univerisity Press, 2009) delta approximation
def aSSA_delta_approx(N_el,B,nu,g,dg_l_el):
    gamma = np.sqrt(nu*np.pi*m_el*c/(q*B))
    return np.pi*c*q**2./(36.*m_el*c**2.*nu*gamma)*np.interp(np.log10(gamma),np.log10(g[1:-1]),np.diff(N_el[1:])/dg_l_el-N_el[1:-1])

#Synchrotron self absorption coefficient (Eq. 40 in Mastichiadis A., Kirk J. G., 1995, A\&A, 295, 613)
def aSSA_delta_approx_M_K95(N_el,B,nu,g,dg_l_el):
    b = B/B_cr
    x = h*nu/(m_el*c**2.)
    gamma = np.sqrt(x/b)
    return 137.*np.pi/(6.)*(x*b)**(-1./2.)*gamma**(-3.)*np.interp(np.log10(gamma),np.log10(g[1:-1]),np.diff(N_el[1:])/dg_l_el-2.*N_el[1:-1])*sigmaT

#Synchrotron self absorption frequency (determined by solving tau_ssa = 1)
def SSA_frequency(index_SSA,nu,aSSA_space,R):
    if index_SSA>0:
        line_1 = LineString(np.column_stack((np.log10(nu[index_SSA-2:index_SSA+1]),np.log10(np.multiply(aSSA_space,-R)[index_SSA-2:index_SSA+1]))))
        line_2 = LineString(np.column_stack((np.log10(nu[index_SSA-2:index_SSA+1]),np.zeros([3]))))
        int_pt = line_1.intersection(line_2)
        return (10**int_pt.x) 

#Inverse Compton scattering emissivity  (from Blumenthal & Gould 1970, Rev. Mod. Phys. 42, 237)  
def Q_IC(Np, g_el, nu_ic_temp, photons, nu_targ, index):
    temp_int_in_en = []
    for g_ind in range(0,len(g_el)):
        q_IC = nu_ic_temp/(4.*nu_targ*g_el[g_ind]*(g_el[g_ind]-h*nu_ic_temp/(m_el*c**2.)))
        if q_IC[0] > 0.:
            function_IC = 2.*q_IC*np.log(q_IC)+(1.+2.*q_IC)*(1.-q_IC)+(h*nu_ic_temp/(g_el[g_ind]*m_el*c**2.-h*nu_ic_temp))**2./(1.+h*nu_ic_temp/(g_el[g_ind]*m_el*c**2.-h*nu_ic_temp))*(1.-q_IC)/2.
            function_IC[function_IC < 0.] = 0.
            temp_int_in_en.append(np.trapz(np.multiply(photons[:index],function_IC[:index]),np.log(nu_targ[:index])))
        else:
            temp_int_in_en.append(0.)
    IC_em = sigmaT*c*0.75*np.trapz(Np*g_el**(-1.)*temp_int_in_en,np.log(g_el))       
    return IC_em

#gamma-gamma absorption coefficient  ( Coppi P. S., Blandford R. D., 1990, MNRAS, 245, 453. doi:10.1093/mnras/245.3.453) 
def a_gg(nu_ic,nu_target,photons_target):   
    t_gg_m=[]
    x = h*nu_target/(m_el*c**2.)
    for nu_ic_el in nu_ic:
        x_IC = h*nu_ic_el/(m_el*c**2.)
        x_space = np.logspace(np.log10(1.3/x_IC),np.log10(max(x)),50)
        if x_space[0] < x_space[1]:                   
            photons_gg = 10**np.interp(np.log10(x_space),np.log10(x),np.log10(photons_target/(h)*m_el*c**2.))
            t_gg_m.append(np.trapz(0.652*sigmaT*((x_space*x_IC)**2.-1.)/(x_space*x_IC)**3.*np.log(x_space*x_IC)*x_space*photons_gg,np.log(x_space)))
        else: t_gg_m.append(0.)
    return(t_gg_m)
            
#Pair creation from gamma gamma absorption dN/dVdtdg (Eq. 57 in Mastichiadis A., Kirk J. G., 1995, A\&A, 295, 613)
def Q_ee_f(nu_target,photons_target,nu_ic,photons_IC,g,R0):
    Q_ee_temp=[]
    for g_e in g: 
        if (2.*g_e) > 1.:
            x_prime = np.logspace(np.log10((2.*g_e)**(-1.)),np.log10(h*nu_target[-1]/(m_el*c**2.)))
            n_ph_prime = 10**np.interp(np.log10(x_prime),np.log10(h*nu_target/(m_el*c**2.)),np.log10(photons_target*m_el*c**2./h))
            n_d_u = 2.*g_e*x_prime
            n_g = 10.**np.interp(np.log10(2.*g_e*m_el*c**2./h),np.log10(nu_ic),np.log10(photons_IC))
            R_gg = 2.61*(n_d_u**2.-1)/n_d_u**3.*np.log(n_d_u)
            Q_ee_temp.append(n_g*np.trapz(n_ph_prime*R_gg*x_prime,np.log(x_prime)))            
        else: Q_ee_temp.append(0.)
    return(np.multiply(c*sigmaT*m_el*c**2./h,Q_ee_temp))

#computes energy density of target photons for ICS scattering in Thomson limit                                                 
def U_ph_f(g,nu_target,photons_target,R):
    U_ph_temp=[]
    U_ph_tot=m_el*c**2./(sigmaT*R)*np.trapz(nu_target[:-1]*photons_target[:-1]*(sigmaT*R*m_el*c**2./h),nu_target[:-1])
    for l_f in g:
        nu_T = 3.*m_el*c**2./(4.*h*l_f)                                    
        if nu_T < nu_target[-1]:
            nu_temp = np.logspace(np.log10(nu_target[0]),np.log10(nu_T),50) 
            photons_temp = 10.**np.interp(np.log10(nu_temp),np.log10(nu_target),np.log10(photons_target))  
            n_d_targ = h*nu_temp/(m_el*c**2.)
            U_ph_temp.append(m_el*c**2./(sigmaT*R)*np.trapz(n_d_targ*photons_temp*(sigmaT*R*m_el*c**2./h),n_d_targ))
        else: U_ph_temp.append(U_ph_tot)
    return(U_ph_temp)

#proton loss rate from BH pair creation (Eq. 13 in Blumenthal G.R. 1970)
def dg_dt_BH(g_pr,nu,photons,f_k_i): 
    C_BH = 3.*sigmaT*c*m_el/(8*np.pi*137.*m_pr)
    dg_dt_pg = []
    for g_p in g_pr:
        if g_p*h*nu[-1] > (2.1)*m_el*c**2.:
            kappa_int = np.logspace(np.log10(2.), np.log10(g_p*h*nu[-1]/(m_el*c**2.)), 50)
            photons_BH = m_el*c**2./h*10**np.interp(np.log10(kappa_int/(2.*g_p)),np.log10(h*nu/(m_el*c**2)),np.log10(photons))
            f_k = np.interp(np.log10(kappa_int),f_k_i.k_i,f_k_i.fk_i)
            dg_dt_pg.append(np.trapz(10**f_k*photons_BH*kappa_int,np.log10(kappa_int)))
        else:
            dg_dt_pg.append(0.) 
    return(np.multiply(dg_dt_pg,C_BH))

#proton loss rate from pg interaction (Eq. B3 in Begelman M.C. et al. 1990)
def dg_dt_pg(g_pr,nu,photons,E_th=145.*10**6.*eV):
    C_pion = 5.*10**(-31.)*c
    dg_dt_pg = []
    for g_p in g_pr:
        if g_p*h*nu[-1] > E_th: #photon's energy in proton's rest frame threshold argument
            int_pg_losses = []
            ε_bar_space = np.logspace(np.log10(E_th),np.log10(2.*g_p*h*nu[-1])+6,100)
            Cross_Section_pg_int = np.interp(np.log10(ε_bar_space),np.log10(np.array(Cross_Section_pg.Ph_En)[1:]*10**9.*eV),np.array(Cross_Section_pg.C_S)[1:]*10**9.*eV)
            kp_pg_int = np.interp(np.log10(ε_bar_space),np.log10(np.array(kp_pg.e)*10**9.*eV),np.array(kp_pg.k)*10**9.*eV)
            for ε_bar in ε_bar_space:
                if ε_bar/(2.*g_p) < h*nu[-1]/2.:
                    ε_prime_space = np.logspace(np.log10(ε_bar/(2.*g_p)),np.log10(h*nu[-1]/2.),30)
                    dN_dVdε_prime = np.interp(np.log10(ε_prime_space),np.log10(h*np.array(nu)),np.array(photons/h))
                    int_pg_losses.append(np.trapz(dN_dVdε_prime/ε_prime_space,np.log(ε_prime_space)))
                else:
                    int_pg_losses.append(0.)
            dg_dt_pg.append(1./(g_p)*np.trapz(Cross_Section_pg_int*kp_pg_int*int_pg_losses,np.log(ε_bar_space)))
        else: 
            dg_dt_pg.append(0.) 
    return(np.multiply(dg_dt_pg,C_pion))

#computes Bethe-Heitler pair production differential cross section d\sigma/(d\theta_dE_) (Blumenthal G. R. 1970)
def cs_BH_diff(E_,g_pr,g_el,k): 
    E_p = k-E_
    E_p = np.where(np.isnan(E_p), 0, E_p)
    E_p[E_p < 0] = 0

    p_ = np.sqrt(E_**2-1.)
    p_ [p_ < 0] = 0.
    
    p_p = np.sqrt(E_p**2.-1.)
    p_p = np.where(np.isnan(p_p), 0, p_p)
    p_p [p_p < 0] = 0

    cos_th_ = (g_pr*E_-g_el)/(g_pr*p_)
    
    T = np.sqrt(k**2.+p_**2.-2.*k*p_*cos_th_)
    T[T < 0] = 0.

    D_ = (E_-p_*cos_th_)
    D_ = np.where(np.isnan(D_), 0, D_)
    D_ [D_ < 0. ] = 0.
    
    Y = (2./p_**2.)*np.log((E_*E_p+p_*p_p+1.)/k)
    Y = np.where(np.isnan(Y), 0, Y)
    Y[Y < 0] = 0.
    
    y_plus = p_p**(-1.)*np.log((E_p+p_p)/(E_p-p_p))
    y_plus = np.where(np.isnan(y_plus), 0, y_plus)
    y_plus[y_plus<0.] = 0.
    
    d_plus_T = np.log((T+p_p)/(T-p_p))
    d_plus_T = np.where(np.isnan(d_plus_T), 0, d_plus_T)
    d_plus_T[d_plus_T < 0.] = 0.
    
    return 3./(8.*np.pi*137.)*sigmaT*p_p*p_/k**3.*(-4.*(np.sin(np.arccos(cos_th_)))**2.*(2.*E_**2.+1.)/(p_**2.*D_**4.)+(5.*E_**2.-2.*E_*E_p+3.)/(p_**2.*D_**2.)+(p_**2.-k**2.)/(T**2.*D_**2.)+2.*E_p/(p_**2.*D_)+
                        Y/(p_*p_p)*(2.*E_*(np.sin(np.arccos(cos_th_)))**2.*(3.*k+p_**2.*E_p)/D_**4.+(2.*E_**2*(E_**2.+E_p**2.)-7.*E_**2.-3.*E_*E_p-E_p**2.+1.)/D_**2.+k*(E_**2.-E_*E_p-1.)/D_)-
                        d_plus_T/(p_p*T)*(2./D_**2.-3.*k/D_-k*(p_**2.-k**2.)/(T**2.*D_))-2.*y_plus/D_)/p_

#Interpolated integral of cross section 
def interp_cs_BH_int(g_pr,g_el,nu_int_min, nu_int_max):
    
    g_pr_interp = np.logspace(np.log10(g_pr[0]),np.log10(g_pr[-1]),30)                          # Protons
    g_el_interp = np.logspace(np.log10(g_el[0]),np.log10(g_el[-1]),30)                          # Electrons
    x = np.logspace(np.log10(h*nu_int_min/(m_el*c**2.)),np.log10(h*nu_int_max/(m_el*c**2.)),30) #dimensionless photon energy x
        
    #Integral limits+integral value initial arrays
    Ee_min_array_p = []
    Ee_max_array_p = []
    Total_int_p = []
    Ee_min_array_m = []
    Ee_max_array_m = []
    Total_int_m = []
    
    # Calculation of the integral for the given inputs
    for i in range(0,len(g_pr_interp)):
        g_pr_rand = g_pr_interp[i]
        for g_el_rand in g_el_interp:
            for j in range(0,len(x)):
                w_rand = (2.*g_pr_rand*x[j]) 
                E_min = (g_pr_rand**2.+g_el_rand**2.)/(2.*g_pr_rand*g_el_rand)
                E_array = np.logspace(np.log10(E_min),np.log10(w_rand-1.))
                if E_array[0] < E_array[-1] and g_pr_rand > g_el_rand :
                    Ee_min_array_p.append(E_array[0])
                    Ee_max_array_p.append(E_array[-1])  
                    sBH = cs_BH_diff(E_array,g_pr_rand,g_el_rand,w_rand)
                    sBH = np.where(np.isnan(sBH), 0., sBH)
                    res = integrate.simpson(sBH*E_array,np.log(E_array))  
                    Total_int_p.append(res)
                elif E_array[0] < E_array[-1] and g_pr_rand < g_el_rand:
                    Ee_min_array_m.append(E_array[0])
                    Ee_max_array_m.append(E_array[-1])  
                    sBH = cs_BH_diff(E_array,g_pr_rand,g_el_rand,w_rand)
                    sBH = np.where(np.isnan(sBH), 0., sBH)
                    res = integrate.simpson(sBH*E_array,np.log(E_array))   
                    Total_int_m.append(res)
                    
    #surface for gamma_p < gamma_e
    z_m = np.where(np.isnan(np.log10(Total_int_m)), 0.1, np.log10(Total_int_m))
    z_m[z_m == -inf] = 0.
    mask = np.where(z_m < -15)
    x_m = np.log10(Ee_min_array_m.copy())
    y_m = np.log10(Ee_max_array_m.copy())
    x_m = [x_m[i] for i in mask[0]]
    y_m = [y_m[i] for i in mask[0]]
    z_m = [z_m[i] for i in mask[0]]
    
    #surface for gamma_p > gamma_e
    z_p = np.where(np.isnan(np.log10(Total_int_p)), 0.1, np.log10(Total_int_p))
    z_p[z_p == -inf] = 0.
    mask = np.where(z_p < -15)
    x_p = np.log10(Ee_min_array_p.copy())
    y_p = np.log10(Ee_max_array_p.copy())
    x_p = [x_p[i] for i in mask[0]]
    y_p = [y_p[i] for i in mask[0]]
    z_p = [z_p[i] for i in mask[0]]
    
    intrp_cs_m_f = interpolate.bisplrep(x_m, y_m, z_m, s=2) 
    intrp_cs_p_f = interpolate.bisplrep(x_p, y_p, z_p, s=2)
    
    #maximum allowed value of surface gamma_p < gamma_e
    max_intrp_cs_m_f = max(z_m) 
    #maximum allowed value of surface gamma_p > gamma_e
    max_intrp_cs_p_f = max(z_p)
    
    return (intrp_cs_m_f,intrp_cs_p_f,max_intrp_cs_m_f,max_intrp_cs_p_f)

# Eq. (21) Kelner S.R. and Aharonian F. A. 2009
def x_plus(eta,r):
    return 1./(2.*(1.+eta))*(eta+r**2.+np.sqrt((eta-r**2.-2.*r)*(eta-r**2.+2.*r)))

def x_minus(eta,r):
    return 1./(2.*(1.+eta))*(eta+r**2.-np.sqrt((eta-r**2.-2.*r)*(eta-r**2.+2.*r)))

# Eq. (40) Kelner S.R. and Aharonian F. A. 2009
def x_plus_e(eta,r):
    return 1./(2.*(1.+eta))*(eta-2*r+np.sqrt(eta*(eta-4.*r*(1.+r))))

def x_minus_e(eta,r):
    return 1./(2.*(1.+eta))*(eta-2*r-np.sqrt(eta*(eta-4.*r*(1.+r))))

# Emissivities of the secondary particles from Eq. (30) Kelner S.R. and Aharonian F. A. 2009                
def Qp_g_opt(g_el,nu_ic,N_pr,g_pr,photons_targ,nu_target,flag_product,h_0=0.313,r=0.1458):
    if flag_product == "2_g": 
        sum_Qp_g = []
        g_pr_min_int = h_0*m_pr*c**2./(4.*h*nu_target[-1])
        g_pr_temp = np.logspace(np.log10(max(g_pr_min_int,g_pr[0])),np.log10(g_pr[-1]),50)
        N_pr_temp = 10**np.interp(np.log10(g_pr_temp),np.log10(g_pr),np.log10(N_pr))
        for nu_g in nu_ic:
            Qp_g_temp = []
            for g_ind in range(len(g_pr_temp)):
                c_temp = m_pr*c**2./(4.*g_pr_temp[g_ind])
                eta_0_max = 4.*h*nu_target[-1]*g_pr_temp[g_ind]/(m_pr*c**2.)
                x = h*nu_g/(g_pr_temp[g_ind]*m_pr*c**2)
                epsilon_0 = h_0*c_temp
                if np.log10(epsilon_0/h) < np.log10(nu_target[-1]):
                    eta_space = np.logspace(np.log10(h_0),np.log10(eta_0_max),25)
                    nu_target_temp = eta_space*c_temp/h
                    photons_targ_temp = 10**np.interp(np.log10(nu_target_temp),np.log10(nu_target),np.log10(photons_targ))
                    Qp_g_temp.append(np.trapz(photons_targ_temp*Phi_g(eta_space,x,"2_g")*nu_target_temp,np.log(h*nu_target_temp)))
                else:
                    Qp_g_temp.append(0.)
            sum_Qp_g.append(np.trapz(h*N_pr_temp*Qp_g_temp/(m_pr*c**2.),np.log(g_pr_temp)))
                    
    elif flag_product == "e-" :
        sum_Qp_g = []
        g_pr_min_int = 2.14*h_0*m_pr*c**2./(4.*g_el[-1]*m_el*c**2.)
        g_pr_temp = np.logspace(np.log10(max(g_pr_min_int,g_pr[0])),np.log10(g_pr[-1]),50)
        N_pr_temp = 10**np.interp(np.log10(g_pr_temp),np.log10(g_pr),np.log10(N_pr))
        for g in g_el:
            Qp_g_temp = []
            for g_ind in range(len(g_pr_temp)):
                c_temp = m_pr*c**2./(4.*g_pr_temp[g_ind])
                eta_0_max = 4.*h*nu_target[-1]*g_pr_temp[g_ind]/(m_pr*c**2.)
                x = g*m_el*c**2./(g_pr_temp[g_ind]*m_pr*c**2)
                epsilon_0 = 2.14*h_0*c_temp
                if np.log10(epsilon_0/h) < np.log10(nu_target[-1]):
                    eta_space = np.logspace(np.log10(2.14*h_0),np.log10(eta_0_max),25)
                    nu_target_temp = eta_space*c_temp/h
                    photons_targ_temp = 10**np.interp(np.log10(nu_target_temp),np.log10(nu_target),np.log10(photons_targ))
                    Qp_g_temp.append(np.trapz(photons_targ_temp*np.nan_to_num(Phi_g(eta_space,x,flag_product))*nu_target_temp,np.log(h*nu_target_temp)))
                else:
                    Qp_g_temp.append(0.)
            sum_Qp_g.append(np.trapz(m_el*c**2.*N_pr_temp*Qp_g_temp/(m_pr*c**2.),np.log(g_pr_temp)))
                
    elif flag_product == "e+":     
        sum_Qp_g = []
        g_pr_min_int = h_0*m_pr*c**2./(4.*g_el[-1]*m_el*c**2.)
        g_pr_temp = np.logspace(np.log10(max(g_pr_min_int,g_pr[0])),np.log10(g_pr[-1]),50)
        N_pr_temp = 10**np.interp(np.log10(g_pr_temp),np.log10(g_pr),np.log10(N_pr))
        for g in g_el:
            Qp_g_temp = []
            for g_ind in range(len(g_pr_temp)):
                c_temp = m_pr*c**2./(4.*g_pr_temp[g_ind])
                eta_0_max = 4.*h*nu_target[-1]*g_pr_temp[g_ind]/(m_pr*c**2.)
                x = g*m_el*c**2./(g_pr_temp[g_ind]*m_pr*c**2)
                epsilon_0 = h_0*c_temp
                if np.log10(epsilon_0/h) < np.log10(nu_target[-1]):
                    eta_space = np.logspace(np.log10(h_0),np.log10(eta_0_max),25)
                    nu_target_temp = eta_space*c_temp/h
                    photons_targ_temp = 10**np.interp(np.log10(nu_target_temp),np.log10(nu_target),np.log10(photons_targ))
                    Qp_g_temp.append(np.trapz(photons_targ_temp*np.nan_to_num(Phi_g(eta_space,x,flag_product))*nu_target_temp,np.log(h*nu_target_temp)))
                else:
                    Qp_g_temp.append(0.)
            sum_Qp_g.append(np.trapz(m_el*c**2.*N_pr_temp*Qp_g_temp/(m_pr*c**2.),np.log(g_pr_temp)))
                    
    elif flag_product == "\bar_nu_e":
        sum_Qp_g = []
        g_pr_min_int = 2.14*h_0*m_pr*c**2./(4.*E_nu_space[-1])
        g_pr_temp = np.logspace(np.log10(max(g_pr_min_int,g_pr[0])),np.log10(g_pr[-1]),50)
        N_pr_temp = 10**np.interp(np.log10(g_pr_temp),np.log10(g_pr),np.log10(N_pr))
        for E_nu_ind in range(0,len(E_nu_space)):
            Qp_g_temp = []
            for g_ind in range(len(g_pr_temp)):
                c_temp = m_pr*c**2./(4.*g_pr_temp[g_ind])
                eta_0_max = 4.*h*nu_target[-1]*g_pr_temp[g_ind]/(m_pr*c**2.)
                x = E_nu_space[E_nu_ind]/(g_pr_temp[g_ind]*m_pr*c**2)
                epsilon_0 = 2.14*h_0*c_temp
                if np.log10(epsilon_0/h) < np.log10(nu_target[-1]):
                    eta_space = np.logspace(np.log10(2.14*h_0),np.log10(eta_0_max),25)
                    nu_target_temp = eta_space*c_temp/h
                    photons_targ_temp = 10**np.interp(np.log10(nu_target_temp),np.log10(nu_target),np.log10(photons_targ))
                    Qp_g_temp.append(np.trapz(photons_targ_temp*np.nan_to_num(Phi_g(eta_space,x,flag_product))*nu_target_temp,np.log(h*nu_target_temp)))
                else:
                    Qp_g_temp.append(0.)                
            sum_Qp_g.append(np.trapz(h*N_pr_temp*Qp_g_temp/(m_pr*c**2.),np.log(g_pr_temp)))

    else:
        sum_Qp_g = []
        g_pr_min_int = h_0*m_pr*c**2./(4.*E_nu_space[-1])
        g_pr_temp = np.logspace(np.log10(max(g_pr_min_int,g_pr[0])),np.log10(g_pr[-1]),50)
        N_pr_temp = 10**np.interp(np.log10(g_pr_temp),np.log10(g_pr),np.log10(N_pr))
        for E_nu_ind in range(0,len(E_nu_space)):
            Qp_g_temp = []
            for g_ind in range(len(g_pr_temp)):
                c_temp = m_pr*c**2./(4.*g_pr_temp[g_ind])
                eta_0_max = 4.*h*nu_target[-1]*g_pr_temp[g_ind]/(m_pr*c**2.)
                x = E_nu_space[E_nu_ind]/(g_pr_temp[g_ind]*m_pr*c**2)
                epsilon_0 = h_0*c_temp
                if np.log10(epsilon_0/h) < np.log10(nu_target[-1]):
                    eta_space = np.logspace(np.log10(h_0),np.log10(eta_0_max),25)
                    nu_target_temp = eta_space*c_temp/h
                    photons_targ_temp = 10**np.interp(np.log10(nu_target_temp),np.log10(nu_target),np.log10(photons_targ))
                    Qp_g_temp.append(np.trapz(photons_targ_temp*np.nan_to_num(Phi_g(eta_space,x,flag_product))*nu_target_temp,np.log(h*nu_target_temp)))
                else:
                    Qp_g_temp.append(0.)                
            sum_Qp_g.append(np.trapz(h*N_pr_temp*Qp_g_temp/(m_pr*c**2.),np.log(g_pr_temp)))
            
    return(np.array(sum_Qp_g))

def Phi_g(eta_space,x,flag_product,h_0=0.313,r=0.1458):
    Phi_g_temp = []
     
    if flag_product=="2_g":
        for eta in eta_space:
            h_h_0 = eta/h_0
            x_p_plus = x_plus(eta,r)
            x_p_minus = x_minus(eta,r)          
            y=(x-x_p_minus)/(x_p_plus-x_p_minus)
            s_g = np.interp(h_h_0, Phi_g_tab["eta_div_eta_0"], Phi_g_tab["s_g"] )
            d_g = np.interp(h_h_0, Phi_g_tab["eta_div_eta_0"], Phi_g_tab["d_g"] )
            B_g = np.interp(h_h_0, Phi_g_tab["eta_div_eta_0"], Phi_g_tab["B_g"] )       
            psi = 2.5+0.4*np.log(h_h_0)
        
            if x < x_p_minus:
                Phi_g_temp.append(B_g*(np.log(2.))**(psi))
            elif x_p_plus > x > x_p_minus:
                Phi_g_temp.append(B_g*np.exp(-s_g*np.log(x/x_p_minus)**d_g)*np.log(2./(1+y**2.))**(psi))
            else :
                Phi_g_temp.append(0.)
            
    elif flag_product=="e+" or flag_product=="\bar_nu_mu" or flag_product=="nu_e":
        for eta in eta_space:
            h_h_0 = eta/h_0        
            x_p_plus = x_plus(eta,r)
            x_p_minus = x_minus(eta,r)/4. 
            
            y=(x-x_p_minus)/(x_p_plus-x_p_minus)
            if flag_product=="e+":
                s_e_nu = np.interp(h_h_0, Phi_el_tab["eta_div_eta_0"], Phi_el_tab["s_e+"] )
                d_e_nu = np.interp(h_h_0, Phi_el_tab["eta_div_eta_0"], Phi_el_tab["d_e+"] )
                B_e_nu = np.interp(h_h_0, Phi_el_tab["eta_div_eta_0"], Phi_el_tab["B_e+"] )
            elif flag_product=="\bar_nu_mu":
                s_e_nu = np.interp(h_h_0, Phi_el_tab["eta_div_eta_0"], Phi_el_tab["s_anti_nu_mu"] )
                d_e_nu = np.interp(h_h_0, Phi_el_tab["eta_div_eta_0"], Phi_el_tab["d_anti_nu_mu"] )
                B_e_nu = np.interp(h_h_0, Phi_el_tab["eta_div_eta_0"], Phi_el_tab["B_anti_nu_mu"] )      
            else:
                s_e_nu = np.interp(h_h_0, Phi_el_tab["eta_div_eta_0"], Phi_el_tab["s_nu_e"] )
                d_e_nu = np.interp(h_h_0, Phi_el_tab["eta_div_eta_0"], Phi_el_tab["d_nu_e"] )
                B_e_nu = np.interp(h_h_0, Phi_el_tab["eta_div_eta_0"], Phi_el_tab["B_nu_e"] )                      
            psi = 2.5+1.4*np.log(h_h_0)
            
            if x < x_p_minus:
                Phi_g_temp.append(B_e_nu*(np.log(2.))**(psi))
            elif x_p_plus> x > x_p_minus:
                Phi_g_temp.append(B_e_nu*np.exp(-s_e_nu*np.log(x/x_p_minus)**d_e_nu)*np.log(2./(1+y**2.))**(psi))
            else :
                Phi_g_temp.append(0.)
    
    elif flag_product=="nu_mu":
        for eta in eta_space:
            h_h_0 = eta/h_0           
            if h_h_0<2.17:
                x_p_plus = 0.427*x_plus(eta,r)
            elif 10 > h_h_0 > 2.17:
                x_p_plus = (0.427+0.0729*(h_h_0-2.14))*x_plus(eta,r)
            else:
                x_p_plus = x_plus(eta,r)
            x_p_minus = 0.427*x_minus(eta,r)  
        
            y=(x-x_p_minus)/(x_p_plus-x_p_minus)
            s_e_nu = np.interp(eta/h_0, Phi_el_tab["eta_div_eta_0"], Phi_el_tab["s_nu_mu"] )
            d_e_nu = np.interp(eta/h_0, Phi_el_tab["eta_div_eta_0"], Phi_el_tab["d_nu_mu"] )
            B_e_nu = np.interp(eta/h_0, Phi_el_tab["eta_div_eta_0"], Phi_el_tab["B_nu_mu"] )
            psi = 2.5+1.4*np.log(h_h_0)
        
            if x < x_p_minus:
                Phi_g_temp.append(B_e_nu*(np.log(2.))**(psi))
            elif x_p_plus> x > x_p_minus:
                Phi_g_temp.append(B_e_nu*np.exp(-s_e_nu*np.log(x/x_p_minus)**d_e_nu)*np.log(2./(1+y**2.))**(psi))
            else :
                Phi_g_temp.append(0.)

    elif flag_product=="e-" or flag_product=="\bar_nu_e":
        for eta in eta_space:
            h_h_0 = eta/h_0       
            x_p_plus = x_plus_e(eta,r)
            x_p_minus = x_minus_e(eta,r)/2.  
            y=(x-x_p_minus)/(x_p_plus-x_p_minus)
            if flag_product=="e-":                    
                s_e = np.interp(h_h_0, Phi_el_1_tab["eta_div_eta_0"], Phi_el_1_tab["s_e-"] )
                d_e = np.interp(h_h_0, Phi_el_1_tab["eta_div_eta_0"], Phi_el_1_tab["d_e-"] )
                B_e = np.interp(h_h_0, Phi_el_1_tab["eta_div_eta_0"], Phi_el_1_tab["B_e-"] )
            else:
                s_e = np.interp(h_h_0, Phi_el_1_tab["eta_div_eta_0"], Phi_el_1_tab["s_anti_nu_e"] )
                d_e = np.interp(h_h_0, Phi_el_1_tab["eta_div_eta_0"], Phi_el_1_tab["d_anti_nu_e"] )
                B_e = np.interp(h_h_0, Phi_el_1_tab["eta_div_eta_0"], Phi_el_1_tab["B_anti_nu_e"] )                
            if h_h_0 > 4.: 
                psi = 6.*(1-np.exp(1.5*(4.-h_h_0)))
            else:
                psi = 0.
            if x < x_p_minus:
                Phi_g_temp.append(B_e*(np.log(2.))**(psi))
            elif x_p_plus > x > x_p_minus:
                Phi_g_temp.append(B_e*np.exp(-s_e*np.log(x/x_p_minus)**d_e)*np.log(2./(1+y**2.))**(psi))
            else :
                Phi_g_temp.append(0.)   
    return Phi_g_temp

#function that calculates BH emissivity
def Q_BH_sol(g_el,g_pr,N_pr,nu_trgt,photons_trgt,intrp_cs_m,intrp_cs_p,max_intrp_cs_m,max_intrp_cs_p):
    dnde = np.zeros(len(g_el))
    for i in range(0, len(g_el)):
        g_el_item = g_el[i]
        g_pr_int = []
        for j in range(0,len(g_pr)):
            g_pr_item = g_pr[j]
            Emin = (g_pr_item+g_el_item)**2/(4.*g_pr_item**2*g_el_item)
            Emax = m_pr/(g_pr_item*m_el)
            if Emin < Emax :
                E_arr = np.logspace(np.log10(1.001*Emin), np.log10(Emax), 10)
                f_ph_int = 10**np.interp(E_arr,nu_trgt,np.log10(photons_trgt))*m_el*c**2./(h*E_arr**2)  # photons in dN/dVdν
                temp_ph = []
                for Eph in E_arr:
                    wmin = (g_pr_item+g_el_item)**2/(2*g_pr_item*g_el_item)
                    wmax = 2*g_pr_item*Eph
                    if (wmin < wmax):
                        w_arr = np.logspace(np.log10(wmin), np.log10(wmax),10)
                        temp_w = []
                        for w in w_arr:
                            Ee_min = (g_pr_item**2+g_el_item**2)/(2*g_pr_item*g_el_item)
                            Ee_max = w-1
                            if Ee_min < Ee_max and g_pr_item > g_el_item and Ee_min < 80./100.*Ee_max:
                                znew = interpolate.bisplev(np.log10(Ee_min), np.log10(Ee_max), intrp_cs_p)
                                if znew>(max_intrp_cs_p):
                                    znew = 0.
                                res = w*10**znew
                                temp_w.append(res)
                            elif Ee_min < Ee_max and g_pr_item < g_el_item and Ee_min < 80./100.*Ee_max :
                                  
                                znew = interpolate.bisplev(np.log10(Ee_min), np.log10(Ee_max), intrp_cs_m)
                                if znew>(max_intrp_cs_m):
                                    znew = 0.
                                res = w*10**znew
                                temp_w.append(res) 
                            else:
                                temp_w.append(0)
                        temp_ph.append(integrate.simpson(temp_w*w_arr,np.log(w_arr)))
                    else:
                        temp_ph.append(0)
                    temp_ph = np.array(temp_ph)
                    temp_ph[temp_ph < 0.] = 0.
                    temp_ph = list(temp_ph)
                g_pr_int.append(N_pr[j]*integrate.simpson(f_ph_int*temp_ph*E_arr,np.log(E_arr))/(2.*g_pr_item**3.))    
            else:
                g_pr_int.append(0.)    
        if len(g_pr_int) < 1:
            dnde[i] = 0.
        else:
            dnde[i] = integrate.simpson(g_pr_int*g_pr,np.log(g_pr)) 
    return c*dnde  # A factor of 2 should be included to account for energy conservation                               

#computes correction factor in electron injection luminosity by checking the energy balance in a fast synchrotron cooling scenario
def cor_factor_syn_el(g_el_space,R0,B0,p_el,Lum_e_injected):
    #Constants
    time_init = 0.
    time_end = 20.
    step_alg = 1.

    g_el = g_el_space
    g_el_mp = np.array([(g_el[im+1]+g_el[im-1])/2. for im in range(0,len(g_el)-1)])
    const_el = 4./3.*sigmaT/(8.*np.pi*m_el*c)
    dg_el = np.array([((g_el[im+1])-(g_el[im-1]))/2. for im in range(1,len(g_el)-1)])

    #gamma-nu space
    nu_syn = np.logspace(7.5,np.log10(7.*nu_c(g_el[-1],B0))+1.4,100)
    day_counter=0.

    #Initialize values for particles and photons
    N_el = np.zeros(len(g_el))
    el_inj = np.zeros(len(g_el))
    photons_syn = np.ones(len(nu_syn))*10**(-260.)
    N_el[0] = N_el[-1] = 10**(-160.)
    
    El_lum = []
    Ph_lum = []
    t = []
    t_plot = []
    time_real = time_init
    el_inj = Q_el_Lum(Lum_e_injected,p_el,g_el[1],g_el[-2])*g_el**(-p_el)
    C_syn_el = sigmaT*c/(h*24.*np.pi**2.*0.8975)*(4.*np.pi*m_el*c/(3.*q))**(4./3.)
    while time_real <  time_end*R0/c:
        dt = 0.1001*R0/c
        time_real += dt    
        t.append(time_real)
        Radius = R0
        B = B0
        a_cr_el = 3.*q*B/(4.*np.pi*m_el*c)
        
        b_syn_el = const_el*B**2.
        dgdt_Syn_el_m = b_syn_el*np.divide(np.power(g_el_mp[0:-1],2.),dg_el)
        dgdt_Syn_el_p = b_syn_el*np.divide(np.power(g_el_mp[1:],2.),dg_el)

        V1 = np.zeros(len(g_el)-2)
        V2 = 1.+dt*(c/Radius+dgdt_Syn_el_m)
        V3 = -dt*(dgdt_Syn_el_p) 
        S_ij = N_el[1:-1]+el_inj[1:-1].copy()*dt
        N_el[1:-1] = thomas(V1, V2, V3, S_ij)

        Q_Syn_el = [Q_syn_space(N_el/Volume(Radius),B,nu_syn[nu_ind],a_cr_el,C_syn_el,g_el) for nu_ind in range(len(nu_syn)-1)] 

        V1 = np.zeros(len(nu_syn)-2)
        V2 = 1.+dt*(c/R0*np.ones(len(nu_syn)-2))
        V3 = np.zeros(len(nu_syn)-2)
        S_ij = photons_syn[1:-1]+4.*np.pi*np.multiply(Q_Syn_el,dt)[1:]*Volume(Radius)
        photons_syn[1:-1] = thomas(V1, V2, V3, S_ij )  
        
        if day_counter < time_real:
            day_counter=day_counter+step_alg*R0/c
            t_plot.append(time_real)
            Syn_temp_plot = np.multiply(photons_syn/Volume(Radius),h*nu_syn**2.)*4.*np.pi/3.*Radius**2.*c
            El_lum.append(np.trapz(el_inj*g_el**2.*m_el*c**2.,np.log(g_el)))
            Ph_lum.append(np.trapz(Syn_temp_plot,np.log(nu_syn)))
    return np.divide(Ph_lum,El_lum)[-1]

p_pr_list = [2.,2.5,3.]
eta_list = [1.1 , 0.86, 0.91]

def n_cor(tau,Radius,J_e):
    return tau*np.sqrt(3.)/(J_e*sigmaT*Radius)

def cs_pp_inel(E_p):
    mask = np.array(E_p) > E_th_pi
    L = np.log(np.array(E_p))
    cs_pp_temp = np.zeros(len(E_p))
    cs_pp_temp[mask] = 10**(-27.)*(34.3+1.88*L[mask]+0.25*np.array(L[mask])**2.)*(1.-(np.divide(E_th_pi,E_p[mask]))**4.)**2 
    return cs_pp_temp

def q_pi(E_pi,p_pr,species,N_pr_TeV,n_H,g_pr):
    if species == "g":
        if p_pr == 2.:
            eta = 1.1
        elif p_pr == 2.5:
            eta = 0.86
        elif p_pr == 3.:
            eta = 0.91
        else:
            eta = np.interp(p_pr,eta_list,p_pr_list)
    else:
        if p_pr == 2.:
            eta = 0.77
        elif p_pr == 2.5:
            eta = 0.62
        elif p_pr == 3.:
            eta = 0.67        
        else:
            eta = np.interp(p_pr,eta_list,p_pr_list)
            
    return eta*c*n_H/K_pi*cs_pp_inel(x_m_pr+E_pi/K_pi)*10.**(np.interp(np.log10(x_m_pr+E_pi/K_pi),np.log10(g_pr*x_m_pr),np.log10(N_pr_TeV)))


def g_nu_mu(x,r):
    return (3.-2.*r)/(9.*(1.-r)**2.)*(9.*x**2.-6.*np.log(x)-4.*x**3.-5.)

def h_nu_mu_1(x,r):
    return (3.-2.*r)/(9.*(1.-r)**2.)*(9.*r**2.-6.*np.log(r)-4.*r**3.-5.)

def h_nu_mu_2(x,r):
    return (1.+2.*r)*(r-x)/(9.*r**2.)*(9.*(r+x)-4.*(r**2.+r*x+x**2.))

def g_nu_e(x,r):
    return 2.*(1.-x)/(3.*(1.-r)**2.)*(6.*(1.-x)**2.+r*(5.+5.*x-4.*x**2.)+6.*r*np.log(x))

def h_nu_e_1(x,r):
    return 2./(3.*(1.-r)**2.)*((1.-r)*(6.-7.*r+11.*r**2.-4.*r**3.)+6.*r*np.log(r))

def h_nu_e_2(x,r):
    return 2.*(r-x)/(3.*r**2.)*(7.*r**2.-4.*r**3.+7.*x*r**2.-2.*x**2.-4.*x**2.*r)

def Q_pp_sub2(x_space,species,E_p,E,p_pr,N_pr_TeV,n_H,g_pr):
    L = np.log(E_p)
    
    if species == "g":
        B_g = 1.3+0.14*L+0.011*L**2.
        beta_g = 1./(1.79+0.11*L+0.008*L**2.)
        k_g = 1./(0.801+0.049*L+0.014*L**2.)
        F_g = B_g*np.log(x_space)/x_space*((1.-x_space**beta_g)/(1.+k_g*x_space**beta_g*(1.-x_space**beta_g)))**4.*(1./np.log(x_space)-4.*beta_g*x_space**beta_g/(1.-x_space**beta_g)-
                (4.*k_g*beta_g*x_space**beta_g*(1.-2.*x_space**beta_g))/(1.+k_g*x_space**beta_g*(1.-x_space**beta_g)))
        return np.trapz(c*n_H*cs_pp_inel(E/x_space)*10.**np.interp(np.log10((E/x_space)),np.log10(g_pr*x_m_pr),np.log10(N_pr_TeV))*F_g,np.log(x_space))
    
    elif species == "nu_mu_1":
        y_space = np.divide(x_space,0.427)
        B_g = 1.75+0.204*L+0.01*L**2.
        beta_g = 1./(1.67+0.111*L+0.0038*L**2.)
        k_g = 1.07-0.086*L+0.002*L**2.

        B_e = 1./(69.5+2.65*L+0.3*L**2.)
        beta_e = 1./(0.201+0.062*L+0.00042*L**2.)**(1./4.)
        k_e = (0.279+0.141*L+0.0172*L**2.)/(0.3+(2.3+L)**2.)
        F_nu_mu_1 = B_g*np.log(y_space)/y_space*((1.-y_space**beta_g)/(1.+k_g*y_space**beta_g*(1.-y_space**beta_g)))**4.*(1./np.log(y_space)-4.*beta_g*y_space**beta_g/(1-y_space**beta_g)-
                (4*k_g*beta_g*y_space**beta_g*(1.-2.*y_space**beta_g))/(1.+k_g*y_space**beta_g*(1.-y_space**beta_g)))
        return np.trapz(c*n_H*cs_pp_inel(E/x_space)*10.**np.interp(np.log10((E/x_space)),np.log10(g_pr*x_m_pr),np.log10(N_pr_TeV))*F_nu_mu_1,np.log(x_space))

    elif species == "nu_mu_2":
        y_space = np.divide(x_space,0.427)
        B_g = 1.75+0.204*L+0.01*L**2.
        beta_g = 1./(1.67+0.111*L+0.0038*L**2.)
        k_g = 1.07-0.086*L+0.002*L**2.

        B_e = 1./(69.5+2.65*L+0.3*L**2.)
        beta_e = 1./(0.201+0.062*L+0.00042*L**2.)**(1./4.)
        k_e = (0.279+0.141*L+0.0172*L**2.)/(0.3+(2.3+L)**2.)
        F_nu_mu_2 = B_e*(1.+k_e*(np.log(x_space))**2.)**3./(x_space*(1.+0.3/x_space**beta_e))*(-np.log(x_space))**5.
        return np.trapz(c*n_H*cs_pp_inel(E/x_space)*10.**np.interp(np.log10((E/x_space)),np.log10(g_pr*x_m_pr),np.log10(N_pr_TeV))*F_nu_mu_2,np.log(x_space))


    elif species == "e" or species == "nu_e" :
        B_e = 1./(69.5+2.65*L+0.3*L**2.)
        beta_e = 1./(0.201+0.062*L+0.00042*L**2.)**(1./4.)
        k_e = (0.279+0.141*L+0.0172*L**2.)/(0.3+(2.3+L)**2.)
        F_e =  B_e*(1.+k_e*(np.log(x_space))**2.)**3./(x_space*(1.+0.3/x_space**beta_e))*(-np.log(x_space))**5.
        return np.trapz(c*n_H*cs_pp_inel(E/x_space)*10.**np.interp(np.log10((E/x_space)),np.log10(g_pr*x_m_pr),np.log10(N_pr_TeV))*F_e,np.log(x_space))


    else:
        raise ValueError("Unkown particles species")
        
def Q_pp_sub1(x,species,E_p,E,p_pr,N_pr_TeV,n_H,g_pr):
    if species == "g": 
        E_g = np.multiply(x,E_p)
        E_min  = E_g+x_m_pi0**2./(4.*E_g)
        E_pi = np.logspace(np.log10(min(E_min)),3.,40)
        return 2.*np.trapz(q_pi(E_pi,p_pr,species,N_pr_TeV,n_H,g_pr)/np.sqrt(E_pi**2.-0.*x_m_pi0**2.)*E_pi,np.log(E_pi))
    
    elif species == "nu_mu_1":
        E_nu_mu_1 = np.multiply(x,E_p)
        E_max = max(x)*(max(E_p)-x_m_pr)
        E_min_1 = E_nu_mu_1/(1.-(x_m_mu/x_m_pi)**2.)
        E_min_2 = E_nu_mu_1+x_m_pi**2./(4.*E_nu_mu_1) 
        E_min = max(max(E_min_1),max(E_min_2))
        E_pi = np.logspace(np.log10(E_min),np.log10(E_max),40)
        return 2./0.427*np.trapz(q_pi(E_pi,p_pr,species,N_pr_TeV,n_H,g_pr)/np.sqrt(E_pi**2.-0.*x_m_pi**2.)*E_pi,np.log(E_pi))
    

    elif species == "nu_mu_2":
        f_nu_mu_2 = []
        r = (x_m_mu/x_m_pi)**2.
        E_nu_mu_2 = np.multiply(x,E_p)
        E_min = E_nu_mu_2+x_m_mu**2./(4.*E_nu_mu_2)
        E_pi = np.logspace(np.log10(max(E_min)),np.log10(1000.*max(E_min)),40)
        x_new = sorted(max(E_nu_mu_2)/E_pi)
        for x_element in x_new:
            if x_element > r:
                f_nu_mu_2.append(g_nu_mu(x_element,r))
            else:
                f_nu_mu_2.append(h_nu_mu_1(x_element,r)+h_nu_mu_2(x_element,r))
        f_nu_mu_2_norm = (1./np.trapz(f_nu_mu_2,x_new))
        return 2.*np.trapz(np.multiply(f_nu_mu_2_norm,f_nu_mu_2)*q_pi(E_pi,p_pr,species,N_pr_TeV,n_H,g_pr)/np.sqrt(E_pi**2.-0.*x_m_pi**2.)*E_pi,np.log(E_pi))
    
    elif species == "e":
        f_nu_mu_2 = []
        r = (x_m_mu/x_m_pi)**2.
        E_e = np.multiply(x,E_p)
        E_min = E_e+x_m_el**2./(4.*E_e)
        E_pi = np.logspace(np.log10(max(E_min)),np.log10(1000.*max(E_min)),40)
        x_new = sorted(min(np.unique(E_e))/E_pi)
        for x_element in x_new:
            if x_element > r:
                if g_nu_mu(x_element,r)<0. :            
                    f_nu_mu_2.append(0.)
                else :   
                    f_nu_mu_2.append(g_nu_mu(x_element,r))
            else:
                if h_nu_mu_1(x_element,r)<0.:
                    flag_h_nu_mu_1 = 0.
                else:
                    flag_h_nu_mu_1  = 1.
                if h_nu_mu_2(x_element,r)<0.:
                    flag_h_nu_mu_1 = 0.
                else:
                    flag_h_nu_mu_2 = 1.
                f_nu_mu_2.append((flag_h_nu_mu_1*h_nu_mu_1(x_element,r)+flag_h_nu_mu_2*h_nu_mu_2(x_element,r)))
        f_nu_mu_2_norm = (1./np.trapz(f_nu_mu_2,x_new))
        return 2.*np.trapz(np.multiply(f_nu_mu_2_norm,f_nu_mu_2)*q_pi(E_pi,p_pr,species,N_pr_TeV,n_H,g_pr)/np.sqrt(E_pi**2.-0.*x_m_pi**2.)*E_pi,np.log(E_pi))
    
    elif species == "nu_e":
        f_nu_e = []
        r = (x_m_mu/x_m_pi)**2.
        E_nu_e = np.multiply(x,E_p)
        E_min = E_nu_e+x_m_el**2./(4.*E_nu_e)
        E_pi = np.logspace(np.log10(max(E_min)),np.log10(100.*max(E_min)),40)
        x_new = sorted(max(E_nu_e)/E_pi)
        for x_element in x_new:
            if x_element > r:
                f_nu_e.append(g_nu_e(x_element,r))
            else:
                f_nu_e.append(h_nu_e_1(x_element,r)+h_nu_e_2(x_element,r))
        f_nu_e_norm = (1./np.trapz(f_nu_e,x_new))
        
        return 2.*np.trapz(np.multiply(f_nu_e_norm,f_nu_e)*q_pi(E_pi,p_pr,species,N_pr_TeV,n_H,g_pr)/np.sqrt(E_pi**2.-0*x_m_pi**2.)*E_pi,np.log(E_pi))
    
    else:
        raise ValueError("Unkown particles species")   
        
def Q_e_pp(g_el,g_pr,N_pr_TeV,p_pr,n_H):
    Q_pp_e_list = []
    norm_ind = 0
    E_species = g_el*m_el*c**2.*0.624151 #Electrons energies in TeV
    E_p_space = g_pr*m_pr*c**2.*0.624151 #Protons energies in TeV
    for E in E_species:
        x_sub1 = []
        x_sub2 = []
        x_sub3 = []
        Q_pp_temp_e = 0.
        for i in range(0,len(E_p_space)):
            x_element = E/E_p_space[i]
            if   E > 0.1 and x_element<0.427:  
                x_sub2.append(x_element)
            elif  E < 0.1 and x_element<1.: 
                x_sub1.append(x_element)
            else:
                x_sub3.append(0.)
                Q_pp_temp_e += 0.

        if len(x_sub2) > 0.:
            x_sub2 = sorted(x_sub2)
            Q_pp_temp_e += Q_pp_sub2(x_sub2,"e",E/x_sub2,E,p_pr,N_pr_TeV,n_H,g_pr)
        else:
            Q_pp_temp_e += 0.

        if len(x_sub1) > 0.:
            norm_ind += 1
            x_sub1 = sorted(x_sub1)    
            Q_pp_temp_e += Q_pp_sub1(x_sub1,"e",E/x_sub1,E,p_pr,N_pr_TeV,n_H,g_pr)
        else:
            Q_pp_temp_e += 0.
        Q_pp_e_list.append(Q_pp_temp_e)
    slope_e, intercept_e, r_value_e, p_value_e, std_err_e = stats.linregress(np.log10(E_species[norm_ind:norm_ind+2]),np.log10(Q_pp_e_list[norm_ind:norm_ind+2]))
    y_fit_e = 10**(slope_e*np.log10(E_species[norm_ind-1])+intercept_e)
    norm_e = y_fit_e/Q_pp_e_list[norm_ind-1]
    Q_pp_e_list[:norm_ind] = np.multiply(norm_e,Q_pp_e_list[:norm_ind])
    return Q_pp_e_list

def Q_g_pp(nu_ic,g_pr,N_pr_TeV,p_pr,n_H):
    norm_ind = 0
    E_species = h*nu_ic*0.624151 #Photons energies in TeV
    E_p_space = g_pr*m_pr*c**2.*0.624151 #Protons energies in TeV   
    Q_pp_g_list = [] 
    for E in E_species:
        x_sub1 = []
        x_sub2 = []
        x_sub3 = []
        Q_pp_temp_g = 0.
        for i in range(0,len(E_p_space)):
            x_element = E/E_p_space[i]
            if   E > 0.1 and x_element < 0.427  :
                x_sub2.append(x_element)
            elif  E < 0.1 and x_element < 1.:
                x_sub1.append(x_element)
            else:
                x_sub3.append(0.)
                Q_pp_temp_g += 0.

        if len(x_sub2) > 0.:
            x_sub2 = sorted(x_sub2)
            Q_pp_temp_g += Q_pp_sub2(x_sub2,"g",E/x_sub2,E,p_pr,N_pr_TeV,n_H,g_pr)
        else:
            Q_pp_temp_g += 0.

        if len(x_sub1) > 0.:
            norm_ind += 1
            x_sub1 = sorted(x_sub1)    
            Q_pp_temp_g += Q_pp_sub1(x_sub1,"g",E/x_sub1,E,p_pr,N_pr_TeV,n_H,g_pr)
        else:
            Q_pp_temp_g += 0.
        Q_pp_g_list.append(Q_pp_temp_g)
    slope_g, intercept_g, r_value_g, p_value_g, std_err_g = stats.linregress(np.log10(E_species[norm_ind:norm_ind+2]),np.log10(Q_pp_g_list[norm_ind:norm_ind+2]))
    y_fit_g = 10**(slope_g*np.log10(E_species[norm_ind-1])+intercept_g)
    norm_g = y_fit_g/Q_pp_g_list[norm_ind-1]
    Q_pp_g_list[:norm_ind] = np.multiply(norm_g,Q_pp_g_list[:norm_ind])
    return Q_pp_g_list

def Q_nu_mu_pp(nu_nu,g_pr,N_pr_TeV,p_pr,n_H):
    norm_ind = 0
    E_species = h*nu_nu*0.624151 #Neutrinos energies in TeV
    E_p_space = g_pr*m_pr*c**2.*0.624151 #Protons energies in TeV   
    Q_pp_nu_mu_list = []
    for E in E_species:
        x_sub1 = []
        x_sub2 = []
        x_sub3 = []
        Q_pp_temp_nu_mu = 0.
        for i in range(0,len(E_p_space)):
            x_element = E/E_p_space[i]
            if   E > 0.1 and x_element < 0.427 and x_element > 10**(-3.) :  #Neutrinos from the deay of pions continue up to 0.427Epi
                x_sub2.append(x_element)
            elif  E < 0.1 and x_element < 10**(-3.):   #delta function approximation
                x_sub1.append(x_element)
            else:
                x_sub3.append(0.)
                Q_pp_temp_nu_mu += 0.

        if len(x_sub2) > 0.:
            x_sub2 = sorted(x_sub2)
            Q_pp_temp_nu_mu += Q_pp_sub2(x_sub2,"nu_mu_1",E/x_sub2,E,p_pr,N_pr_TeV,n_H,g_pr)+Q_pp_sub2(x_sub2,"nu_mu_2",E/x_sub2,E,p_pr,N_pr_TeV,n_H,g_pr)  
        else:
            Q_pp_temp_nu_mu += 0.

        if len(x_sub1) > 0.:
            norm_ind += 1
            x_sub1 = sorted(x_sub1)    
            Q_pp_temp_nu_mu += Q_pp_sub1(x_sub1,"nu_mu_1",E/x_sub1,E,p_pr,N_pr_TeV,n_H,g_pr)+Q_pp_sub1(x_sub1,"nu_mu_2",E/x_sub1,E,p_pr,N_pr_TeV,n_H,g_pr)
        else:
            Q_pp_temp_nu_mu += 0.
        Q_pp_nu_mu_list.append(Q_pp_temp_nu_mu)
    slope_nu_mu, intercept_nu_mu, r_value_nu_mu, p_value_nu_mu, std_err_nu_mu = stats.linregress(np.log10(E_species[norm_ind:norm_ind+2]),np.log10(Q_pp_nu_mu_list[norm_ind:norm_ind+2]))
    y_fit_nu_mu = 10**(slope_nu_mu*np.log10(E_species[norm_ind-1])+intercept_nu_mu)
    norm_nu_mu = y_fit_nu_mu/Q_pp_nu_mu_list[norm_ind-1]
    Q_pp_nu_mu_list[:norm_ind] = np.multiply(norm_nu_mu,Q_pp_nu_mu_list[:norm_ind])
    return Q_pp_nu_mu_list

def Q_nu_e_pp(nu_nu,g_pr,N_pr_TeV,p_pr,n_H):
    norm_ind = 0
    E_species = h*nu_nu*0.624151 #Photons energies in TeV
    E_p_space = g_pr*m_pr*c**2.*0.624151 #Protons energies in TeV   
    Q_pp_nu_e_list = []
    for E in E_species:
        x_sub1 = []
        x_sub2 = []
        x_sub3 = []
        Q_pp_temp_nu_e = 0.
        for i in range(0,len(E_p_space)):
            x_element = E/E_p_space[i]
            if   E > 0.1 and x_element < 0.427  and x_element > 10**(-3.):
                x_sub2.append(x_element)
            elif  E < 0.1 and x_element < 10**(-3.):
                x_sub1.append(x_element)
            else:
                x_sub3.append(0.)
                Q_pp_temp_nu_e += 0.

        if len(x_sub2) > 0.:
            x_sub2 = sorted(x_sub2)
            Q_pp_temp_nu_e += Q_pp_sub2(x_sub2,"nu_e",E/x_sub2,E,p_pr,N_pr_TeV,n_H,g_pr) 
        else:
            Q_pp_temp_nu_e += 0.

        if len(x_sub1) > 0.:
            norm_ind += 1
            x_sub1 = sorted(x_sub1)    
            Q_pp_temp_nu_e += Q_pp_sub1(x_sub1,"nu_e",E/x_sub1,E,p_pr,N_pr_TeV,n_H,g_pr)
        else:
            Q_pp_temp_nu_e += 0.
        Q_pp_nu_e_list.append(Q_pp_temp_nu_e)
    slope_nu_e, intercept_nu_e, r_value_nu_e, p_value_nu_e, std_err_nu_e = stats.linregress(np.log10(E_species[norm_ind:norm_ind+2]),np.log10(Q_pp_nu_e_list[norm_ind:norm_ind+2]))
    y_fit_nu_e = 10**(slope_nu_e*np.log10(E_species[norm_ind-1])+intercept_nu_e)
    norm_nu_e = y_fit_nu_e/Q_pp_nu_e_list[norm_ind-1]
    Q_pp_nu_e_list[:norm_ind] = np.multiply(norm_nu_e,Q_pp_nu_e_list[:norm_ind])
    return Q_pp_nu_e_list

# converts flux to luminosity
def nuL_nu_obs(nu_F_nu,Dist_in_pc,delta,R0):
    return np.multiply(nu_F_nu,1.)*(4.*np.pi*(Dist_in_pc*pc)**2.)/delta**4.

# converts luminosity to flux
def nuF_nu_obs(nu_L_nu,Dist_in_pc,delta,R0):
    return np.multiply(nu_L_nu,delta**4.)/(4.*np.pi*(Dist_in_pc*pc)**2.)

#computes total photon spectrum by adding different spectral components
def photons_tot(nu_syn,nu_bb,photons_syn,nu_ic,photons_IC,nu_tot,photons_bb,photons_pl,photons_user):
    return 10**(np.interp(np.log10(nu_tot),np.log10(nu_bb),np.log10(photons_bb)))+10**(np.interp(np.log10(nu_tot),np.log10(nu_syn),np.log10(photons_syn)))+10**(np.interp(np.log10(nu_tot),np.log10(nu_ic),np.log10(photons_IC)))+photons_pl+photons_user

def thomas(a,b,c,d):
    """ A is the tridiagnonal coefficient matrix and d is the RHS matrix"""
    N = len(a)
    cp = np.zeros(N,dtype='float64') # store tranformed c or c'
    dp = np.zeros(N,dtype='float64') # store transformed d or d'
    X = np.zeros(N,dtype='float64') # store unknown coefficients
    
    # Perform Forward Sweep
    # Equation 1 indexed as 0 in python
    cp[0] = c[0]/b[0]  
    dp[0] = d[0]/b[0]
    # Equation 2, ..., N (indexed 1 - N-1 in Python)
    for i in np.arange(1,(N),1):
        dnum = b[i] - a[i]*cp[i-1]
        cp[i] = c[i]/dnum
        dp[i] = (d[i]-a[i]*dp[i-1])/dnum
    
    # Perform Back Substitution
    X[(N-1)] = dp[N-1]  # Obtain last xn 

    for i in np.arange((N-2),-1,-1):  # use x[i+1] to obtain x[i]
        X[i] = (dp[i]) - (cp[i])*(X[i+1])
    
    return(X)
