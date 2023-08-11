# This is the leptonic version of a radiative transfer code LeHaMoC developed by S. I. Stathopoulos 
# in collaboration with M. Petropoulou. When using this code, make reference to the
# following publication: Stathopoulos et al., 2023, A&A
 

import numpy as np
import astropy.units as u
from astropy import constants as const 
from astropy.modeling.models import BlackBody
import pandas as pd
import sys
from tqdm import tqdm


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
################################

if len(sys.argv) != 3:
    print('incorrect parameters passed')
    print('try something like this')
    print('python LeMoC.py Parameters.txt out1')
    quit()

#Read parameters file
fileName = sys.argv[1] 
out_s = sys.argv[2]

#Define output files
out1 = "Particles_Distribution_"+out_s+".txt"
out2 = "Photons_Distribution_"+out_s+".txt"


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
g_PL_min = float(params['g_PL_min']) #log
g_PL_max = float(params['g_PL_max']) #log
grid_g_el = float(params['grid_g_el'])
grid_nu = float(params['grid_nu'])
p_el = float(params['p_el'])
L_el = float(params['L_el']) #log
Vexp = float(params['Vexp'])*c #/c
R0 = 10**float(params['R0']) #log
B0 = float(params['B0'])
m = float(params['m'])
delta = float(params["delta"])
inj_flag = float(params['inj_flag'])
Ad_l_flag = float(params['Ad_l_flag'])
Syn_l_flag = float(params['Syn_l_flag'])
Syn_emis_flag = float(params['Syn_emis_flag'])
IC_l_flag = float(params['IC_l_flag'])
IC_emis_flag = float(params['IC_emis_flag'])
SSA_l_flag = float(params['SSA_l_flag'])
gg_flag = float(params['gg_flag'])
esc_flag = float(params['esc_flag'])
BB_flag = float(params['BB_flag'])
temperature = 10**float(params['BB_temperature']) #log 
GB_ext = float(params['GB_ext'])
PL_flag = float(params['PL_flag'])
dE_dV_ph = float(params['dE_dV_ph'])
nu_min_ph = float(params['nu_min_ph'])
nu_max_ph = float(params['nu_max_ph'])
s_ph = float(params['s_ph'])
User_ph = float(params['User_ph'])

import LeMoC_f as f # imports functions
 
import time
start_time = time.time()

time_real = time_init
dt = step_alg*R0/c # time step used for solving the PDE
day_counter = 0.
comp_el = sigmaT*10**L_el/(4.*np.pi*R0*m_el*c**3) # intial electron compactness
Radius = R0

# initialization of the electron Lorentz factor array
grid_size = grid_g_el
g_el = np.logspace(g_min_el,g_max_el,int(grid_size))
g_el_mp = np.array([(g_el[im+1]+g_el[im-1])/2. for im in range(0,len(g_el)-1)])
dg_el = np.array([((g_el[im+1])-(g_el[im-1]))/2. for im in range(1,len(g_el)-1)])   # delta gamma 
dg_l_el = np.log(g_el[1])-np.log(g_el[0]) # logarithmic delta gamma

if g_PL_max == g_max_el:
    index_PL_max = -1
else:
    index_PL_max = min(min(np.where(g_el > 10**g_PL_max)))
    
if g_PL_min == 0.:
    index_PL_min = 1
else: 
    index_PL_min = max(max(np.where(g_el < 10**g_PL_min)))

# initialization of photon frequency arrays    
nu_syn = np.logspace(7.5,np.log10(7.*f.nu_c(g_el[-1],B0))+1.4,int(grid_size/2))
nu_ic = np.logspace(10.,30.,int(grid_size/2))
nu_tot = np.logspace(np.log10(nu_syn[0]),np.log10(nu_ic[-1]),int(grid_nu))
a_gg_f = np.zeros(len(nu_ic))

#External grey body (GB) photon field (if GB_ext = 1 then photon spectrum is BB with the given temperature)
#Units (nu,dN/dVdnu)
if BB_flag == 0.:
    dN_dVdnu_BB = np.zeros(2)
    nu_bb = np.array([nu_syn[0], nu_syn[-1]])
else:
    bb = BlackBody(temperature*u.K)
    nu_bb = np.array(np.logspace(np.log10(5.879*10**10*temperature)-6., np.log10(5.879*10**10*temperature)+1.5,60)*u.Hz)
    photons_bb = np.array(4.*np.pi/c*bb(nu_bb)/(h*nu_bb))                       
    GB_norm = np.trapz(photons_bb*h*nu_bb**2.,np.log(nu_bb))/(GB_ext) 
    dN_dVdnu_BB = photons_bb/GB_norm
                     
#External power law (PL) photon field      
#Units (nu,dN/dVdnu)        
if PL_flag == 0.:
    dN_dVdnu_pl = np.zeros(len(nu_tot))
else:
    nu_ph_ext_sp = np.logspace(nu_min_ph,nu_max_ph,100)
    k_ph = (np.trapz(dE_dV_ph*nu_ph_ext_sp**(-s_ph+1.)))**(-1.)
    nu_ph_ext_sp[-1] = 0.
    dN_dVdnu_pl = 10**np.interp(np.log10(nu_tot),np.log10(nu_ph_ext_sp),np.log10(k_ph*nu_ph_ext_sp**(-s_ph)))

#External user-defined photon field              
if User_ph == 0.:
    dN_dVdnu_user = np.zeros(len(nu_tot))
else:  
    #Units (nu,dN/dVdnu)
    Photons_spec_user = pd.read_csv('Photons_spec_user.txt',names=('logx','logy'),sep=",")
    nu_user = 10**np.array(Photons_spec_user.logx)
    dN_dVdnu_user_temp = 10**np.array(Photons_spec_user.logy)
    dN_dVdnu_user_temp[-1] = 10**(-160.)
    dN_dVdnu_user = 10**np.interp(np.log10(nu_tot),np.log10(nu_user),np.log10(dN_dVdnu_user_temp))
                    
#Initialize arrays for particles and photons
N_el = np.zeros(len(g_el)) # Number of electrons & positrons
Q_ee = np.zeros(len(g_el)-1) # Pair production rate
el_inj = np.ones(len(g_el))*10**(-260.) # Primary electron injection rate 
el_inj[index_PL_min:index_PL_max] = f.Q_el_Lum(f.Lum_e_inj(comp_el,Radius),p_el,g_el[index_PL_min],g_el[index_PL_max])*g_el[index_PL_min:index_PL_max]**(-p_el)
N_el = el_inj.copy()
N_el[0] = N_el[-1] = 10**(-260.) # boundary conditions 

photons_syn = np.ones(len(nu_syn))*10**(-260.)
photons_IC = np.ones(len(nu_ic))*10**(-260.)

photons_syn = np.append(photons_syn,10**(-260.))
dN_dVdnu_BB = np.append(dN_dVdnu_BB,10**(-260.))

nu_syn = np.append(nu_syn,nu_tot[-1])
nu_bb = np.append(nu_bb,nu_tot[-1])

nu_syn_mp = np.array([(nu_syn[im+1]+nu_syn[im-1])/2. for im in range(0,len(nu_syn)-1)])
nu_ic_mp = np.array([(nu_ic[im+1]+nu_ic[im-1])/2. for im in range(0,len(nu_ic)-1)])
dnu = np.array([(nu_syn[nu_ind+1]-nu_syn[nu_ind-1])/2. for nu_ind in range(1,len(nu_syn)-1)])
dnu_ic = np.array([(nu_ic[nu_ind+1]-nu_ic[nu_ind-1])/2. for nu_ind in range(1,len(nu_ic)-1)])

 
# Solution of the PDEs
with open(out1,'w') as f1, open(out2,'w') as f2:
    for i in tqdm(range(int(time_end)),desc="Progress..."):
    # while time_real <  time_end*R0/c: 
        time_real += dt    
        Radius = f.R(R0,time_real,time_init,Vexp)
        M_F = f.B(B0,R0,Radius,m)
        a_cr_el = 3.*q*M_F/(4.*np.pi*m_el*c)
        
        # Calculate total dN/dVdν
        photons = f.photons_tot(nu_syn,nu_bb,photons_syn,nu_ic,photons_IC,nu_tot,dN_dVdnu_BB*f.Volume(Radius),dN_dVdnu_pl*f.Volume(Radius),dN_dVdnu_user*f.Volume(Radius))/f.Volume(Radius)
        
        if Ad_l_flag == 1.:
            b_ad = Vexp/Radius
            dgdt_ad_el_m = b_ad*np.divide(np.power(g_el_mp[0:-1],1.),dg_el)
            dgdt_ad_el_p = b_ad*np.divide(np.power(g_el_mp[1:],1.),dg_el)
            dnudt_ad_syn_m = b_ad*np.divide(nu_syn_mp[0:-1],dnu)
            dnudt_ad_syn_p = b_ad*np.divide(nu_syn_mp[1:],dnu)
            dnudt_ad_IC_m = b_ad*np.divide(nu_ic_mp[0:-1]-nu_ic[:-2],dnu_ic)
            dnudt_ad_IC_p = b_ad*np.divide(nu_ic[:1],dnu_ic)
        else:
            dgdt_ad_el_m = np.zeros(len(g_el)-2)
            dgdt_ad_el_p = np.zeros(len(g_el)-2)
            dnudt_ad_syn_m = np.zeros(len(nu_syn)-2)
            dnudt_ad_syn_p = np.zeros(len(nu_syn)-2)
            dnudt_ad_IC_m = np.zeros(len(nu_ic)-2)
            dnudt_ad_IC_p = np.zeros(len(nu_ic)-2)
        
        if Syn_l_flag == 1.:
            b_syn_el = (4./3.)*sigmaT/(8.*np.pi*m_el*c)*M_F**2.
            dgdt_Syn_el_m = b_syn_el*np.divide(np.power(g_el_mp[0:-1],2.),dg_el)
            dgdt_Syn_el_p = b_syn_el*np.divide(np.power(g_el_mp[1:],2.),dg_el)
        else :
            dgdt_Syn_el_m = np.zeros(len(g_el)-2)      
            dgdt_Syn_el_p = np.zeros(len(g_el)-2)  
    
        if IC_l_flag == 1.:
            U_ph = f.U_ph_f(g_el,nu_tot,photons,Radius)
            b_Com_el = 4./3.*sigmaT*np.multiply(c,U_ph)/(m_el*c**2.)
            dgdt_IC_el_m = b_Com_el[1:-1]*np.divide(np.power(g_el_mp[0:-1],2.),dg_el)
            dgdt_IC_el_p = b_Com_el[2:]*np.divide(np.power(g_el_mp[1:],2.),dg_el)
        else:
            dgdt_IC_el_m = np.zeros(len(g_el)-2)    
            dgdt_IC_el_p = np.zeros(len(g_el)-2) 
    
        V1 = np.zeros(len(g_el)-2)
        V2 = 1.+dt*(c/Radius*esc_flag+dgdt_Syn_el_m+dgdt_IC_el_m+dgdt_ad_el_m)
        V3 = -dt*(dgdt_Syn_el_p+dgdt_IC_el_p+dgdt_ad_el_p) 
        if inj_flag == 1.:
            S_ij = N_el[1:-1]+np.multiply(el_inj[1:-1],dt)+np.multiply(Q_ee[1:],dt)*f.Volume(Radius)
        if inj_flag == 0.:
            S_ij = N_el[1:-1]+np.multiply(Q_ee[1:],dt)*f.Volume(Radius)
            
        N_el[1:-1] = f.thomas(V1, V2, V3, S_ij)    
        dN_el_dVdg_el = np.array(N_el/f.Volume(Radius))
    
        if Syn_emis_flag == 1.:
            Q_Syn_el = np.divide([f.Q_syn_space(dN_el_dVdg_el,M_F,nu_syn[nu_ind],a_cr_el,g_el) for nu_ind in range(len(nu_syn)-1)], f.cor_factor_syn_el(g_el,R0,10**4.,p_el,f.Lum_e_inj(comp_el,R0)))
        else: 
            Q_Syn_el = np.zeros(len(nu_syn)-1)
    
        if IC_emis_flag == 1.:
            Q_IC = [f.Q_IC_space_optimized(dN_el_dVdg_el,g_el,nu_ic[nu_ind],photons,nu_tot,len(nu_tot)-1) for nu_ind in range(0,len(nu_ic)-1)]
        else:
            Q_IC = np.zeros(len(nu_ic)-1)  
    
        if SSA_l_flag == 1.:
            aSSA_space_syn = [-np.absolute(f.aSSA(dN_el_dVdg_el,M_F,nu_syn[nu_ind],g_el,dg_l_el)) for nu_ind in range(0,len(nu_syn-1))]
            aSSA_space_ic = [-np.absolute(f.aSSA(dN_el_dVdg_el,M_F,nu_ic[nu_ind],g_el,dg_l_el)) for nu_ind in range(0,len(nu_ic-1))]
        else:
            aSSA_space_syn = np.zeros(len(nu_syn-1)) 
            aSSA_space_ic = np.zeros(len(nu_ic-1)) 
    
        V1 = np.zeros(len(nu_syn)-2)
        V2 = 1.+dt*(c/Radius+dnudt_ad_syn_m-np.multiply(aSSA_space_syn[1:-1],1)*c)
        V3 = -dt*dnudt_ad_syn_p
        S_ij = photons_syn[1:-1]+4.*np.pi*np.multiply(Q_Syn_el,dt)[1:]*f.Volume(Radius)
        photons_syn[1:-1] = f.thomas(V1, V2, V3, S_ij)  
    
        V1 = np.zeros(len(nu_ic)-2)
        V2 = 1.+dt*(c/Radius+dnudt_ad_IC_m+np.multiply(a_gg_f[1:-1],c)-np.multiply(aSSA_space_ic[1:-1],1)*c)
        V3 = -dt*dnudt_ad_IC_p
        S_ij = photons_IC[1:-1]+np.multiply(Q_IC,dt)[1:]*f.Volume(Radius)
        photons_IC[1:-1] = f.thomas(V1, V2, V3, S_ij )
        
        if gg_flag == 0.:
            a_gg_f = np.zeros(len(nu_ic))
        else: 
            a_gg_f = f.a_gg(nu_ic,nu_tot,photons)
            Q_ee = f.Q_ee_f(nu_tot,photons,nu_ic,photons_IC/f.Volume(Radius),g_el,Radius)                  

        if day_counter<time_real:
            day_counter=day_counter+step_alg*R0/c
            photons = f.photons_tot(nu_syn,nu_bb,photons_syn,nu_ic,photons_IC,nu_tot,dN_dVdnu_BB*f.Volume(Radius),dN_dVdnu_pl*f.Volume(Radius),dN_dVdnu_user*f.Volume(Radius))/f.Volume(Radius)
            Spec_temp_tot = np.multiply(photons,h*nu_tot**2.)*4.*np.pi/3.*Radius**2.*c  
            pr1 = [[str(el_list) for el_list in np.log10(g_el) ],[str(el_list) for el_list in np.log10(N_el/f.Volume(Radius)) ]]
            pr2 = [[str(el_list) for el_list in np.log10(nu_tot) ],[str(el_list) for el_list in np.log10(Spec_temp_tot) ]]
            # here is where you unpack everything
            for row in zip(*pr1):
                f1.write(' '.join(row) + '\n')
            for row in zip(*pr2):
                f2.write(' '.join(row) + '\n')
        
print("--- %s seconds ---" % "{:.2f}".format((time.time() - start_time)))