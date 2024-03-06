# This is the leptohadronic version of a radiative transfer code LeHaMoC. 
# Copyright (C) 2023  S. I. Stathopoulos, M. Petropoulou.  
# When using this code, refer to the following 
# publication: Stathopoulos et al., 2023, A&A    

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation (version 3).

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.


import numpy as np
import astropy.units as u
from astropy import constants as const 
from astropy.modeling.models import BlackBody
import pandas as pd
import sys
from tqdm import tqdm
import LeHaMoC_f as f # imports functions
import time

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
################################

if len(sys.argv) != 3:
    print('incorrect parameters passed')
    print('try something like this')
    print('python LeHaMoC.py Parameters.txt _fileName')
    quit()

out_s = sys.argv[2]    
    
#Define output files
out1 = "Pairs_Distribution"+out_s+".txt"
out2 = "Photons_Distribution"+out_s+".txt"
out3 = "Protons_Distribution"+out_s+".txt"
out4 = "Neutrinos_Distribution"+out_s+".txt"

fileName = sys.argv[1] 
fileObj = open(fileName)
params = {}
for line in fileObj:
    line=line.strip()
    key_value = line.split("=")
    params[key_value[0].strip()] = float(key_value[1].strip())
    
time_init = float(params['time_init']) #R0/c
time_end = float(params['time_end']) #R0/c
step_alg = float(params['step_alg']) #R0/c
PL_inj = float(params['PL_inj']) #flag
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
inj_flag = float(params["inj_flag"])
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
pp_nu_emis_flag = float(params['pp_nu_emis_flag'])
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

start_time = time.time()
time_real = time_init
dt = step_alg*R0/c #time step used for solving the PDE
day_counter = 0.
comp_el = sigmaT*10**L_el/(4.*np.pi*R0*m_el*c**3) # intial electron compactness
comp_pr = sigmaT*10**L_pr/(4.*np.pi*R0*m_pr*c**3) # intial proton compactness
Radius = R0

# initialization of the electron Lorentz factor array
grid_size_el = grid_g_el
g_el = np.logspace(g_min_el,g_max_el,int(grid_size_el))
g_el_mp = np.array([(g_el[im+1]+g_el[im-1])/2. for im in range(0,len(g_el)-1)])
dg_el = np.array([((g_el[im+1])-(g_el[im-1]))/2. for im in range(1,len(g_el)-1)])
dg_l_el = np.log(g_el[1])-np.log(g_el[0])

grid_size_pr = grid_g_pr
g_pr = np.logspace(g_min_pr,g_max_pr,int(grid_size_pr))
g_pr_mp = np.array([(g_pr[im+1]+g_pr[im-1])/2. for im in range(0,len(g_pr)-1)])
dg_pr = np.array([((g_pr[im+1])-(g_pr[im-1]))/2. for im in range(1,len(g_pr)-1)])
dg_l_pr = np.log(g_pr[1])-np.log(g_pr[0])

if g_el_PL_max == g_max_el:
    index_PL_max_el = -1
else:
    index_PL_max_el = min(min(np.where(g_el > 10**g_el_PL_max)))   
if g_el_PL_min == 0.:
    index_PL_min_el = 1
else: 
    index_PL_min_el = max(max(np.where(g_el < 10**g_el_PL_min)))

if g_pr_PL_max == g_max_pr:
    index_PL_max_pr = -1
else:
    index_PL_max_pr = int(min(min(np.where(g_pr > 10**g_pr_PL_max)))+1) 
if g_pr_PL_min == 0.:
    index_PL_min_pr = 1
else: 
    index_PL_min_pr = int(max(max(np.where(g_pr < 10**g_pr_PL_min)))+1)
    
nu_syn = np.logspace(7.5,np.log10(7.*f.nu_c(g_el[-1],B0))+1.4,int(grid_size_el/2))
nu_ic = np.logspace(15.,32.,int(grid_size_el/2))
nu_nu = np.logspace(10.,22.,50)*eV/h
nu_tot = np.logspace(np.log10(nu_syn[0]),np.log10(max(nu_ic[-1],nu_syn[-1])),int(grid_nu))

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
    k_ph = dE_dV_ph/np.trapz(nu_ph_ext_sp**(-s_ph+1.),nu_ph_ext_sp)/h
    dN_dVdnu_pl_temp = k_ph*nu_ph_ext_sp**(-s_ph)
    dN_dVdnu_pl_temp[-1] = 10**(-260.)
    dN_dVdnu_pl = 10**np.interp(np.log10(nu_tot),np.log10(nu_ph_ext_sp),np.log10(dN_dVdnu_pl_temp))

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
                  
#Initialize values for particles and photons
N_el = np.zeros(len(g_el)) #Electron-positron number
Q_ee = np.zeros(len(g_el)-2) #Electron-positron from gg
el_inj = np.ones(len(g_el))*10**(-260.) #Electron injection term
N_pr = np.zeros(len(g_pr)) #Protons number
pr_inj = np.ones(len(g_pr))*10**(-260.) #Proton injection term
N_nu = np.zeros(len(nu_nu))
a_gg_f = np.zeros(len(nu_ic))

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

a_gg_f_syn = np.zeros(len(nu_syn)-2)
a_gg_f_ic = np.zeros(len(nu_ic)-2)

if PL_inj == 1.:
    el_inj[index_PL_min_el:index_PL_max_el] = f.Q_el_Lum(f.Lum_e_inj(comp_el,Radius),p_el,g_el[index_PL_min_el],g_el[index_PL_max_el])*g_el[index_PL_min_el:index_PL_max_el]**(-p_el)/f.Volume(Radius)
    pr_inj[index_PL_min_pr:index_PL_max_pr] = f.Q_pr_Lum(f.Lum_p_inj(comp_pr,Radius),p_pr,g_pr[index_PL_min_pr],g_pr[index_PL_max_pr])*g_pr[index_PL_min_pr:index_PL_max_pr]**(-p_pr)/f.Volume(Radius)
else:
    pr_inj = f.Q_norm_exp_c(R0,comp_pr,g_pr,index_PL_max_pr,p_pr)*g_pr**(-p_pr)*np.exp(-g_pr/g_pr[index_PL_max_pr])/f.Volume(Radius)
    el_inj = f.Q_norm_exp_c(R0,comp_el,g_el,index_PL_max_el,p_el)*g_el**(-p_el)*np.exp(-g_el/g_el[index_PL_max_el])/f.Volume(Radius)

Spec_list = []

#Constants
C_syn_el = sigmaT*c/(h*24.*np.pi**2.*0.8975)*(4.*np.pi*m_el*c/(3.*q))**(4./3.)/f.cor_factor_syn_el(g_el,R0,10**4.,p_el,f.Lum_e_inj(comp_el,R0))
C_syn_pr = sigmaT*c/(h*24.*np.pi**2.*0.8975)*(4.*np.pi*m_pr*c/(3.*q))**(4./3.)*(m_el/m_pr)**2.

const_el = 4./3.*sigmaT/(8.*np.pi*m_el*c)
const_pr = const_el*(m_el/m_pr)**3.

Pairs_lum = []
Protons_lum = []

N_el = el_inj.copy()*f.Volume(R0)
N_el[0] = N_el[-1] = 10**(-260.) # boundary conditions 

if pg_BH_emis_flag == 1.:
    intrp_cs_m,intrp_cs_p,max_intrp_cs_m,max_intrp_cs_p = f.interp_cs_BH_int(np.logspace(0.,8.,50),np.logspace(0.,10.,50),nu_tot[0],nu_tot[-1])

# Solution of the PDEs
with open(out1,'w') as f1, open(out2,'w') as f2, open(out3,'w') as f3, open(out4,'w') as f4:
    for i in tqdm(range(int(time_end/step_alg)),desc="Progress..."):
        time_real += dt    
        Radius = f.R(R0,time_real,time_init,Vexp)
        M_F = f.B(B0,R0,Radius,m)
        a_cr_el = 3.*q*M_F/(4.*np.pi*m_el*c)
        a_cr_pr = 3.*q*M_F/(4.*np.pi*m_pr*c)  
        # Calculate total dN/dVdnu
        photons = f.photons_tot(nu_syn,nu_bb,photons_syn,nu_ic,photons_IC,nu_tot,dN_dVdnu_BB*f.Volume(Radius),dN_dVdnu_pl*f.Volume(Radius),dN_dVdnu_user*f.Volume(Radius))/f.Volume(Radius)
        if Ad_l_flag == 1.:
            b_ad = Vexp/Radius
            dgdt_ad_el_m = b_ad*np.divide(np.power(g_el_mp[0:-1],1.),dg_el)
            dgdt_ad_el_p = b_ad*np.divide(np.power(g_el_mp[1:],1.),dg_el)
            dgdt_ad_pr_m = b_ad*np.divide(np.power(g_pr_mp[0:-1],1.),dg_pr)
            dgdt_ad_pr_p = b_ad*np.divide(np.power(g_pr_mp[1:],1.),dg_pr)
            dnudt_ad_syn_m = b_ad*np.divide(nu_syn_mp[0:-1],dnu)
            dnudt_ad_syn_p = b_ad*np.divide(nu_syn_mp[1:],dnu)
            dnudt_ad_IC_m = b_ad*np.divide(nu_ic_mp[0:-1]-nu_ic[:-2],dnu_ic)
            dnudt_ad_IC_p = b_ad*np.divide(nu_ic[:1],dnu_ic)
        else:
            dgdt_ad_el_m = dgdt_ad_el_p = np.zeros(len(g_el)-2)
            dgdt_ad_pr_m = dgdt_ad_pr_p = np.zeros(len(g_pr)-2)
            dnudt_ad_syn_m = dnudt_ad_syn_p = np.zeros(len(nu_syn)-2)
            dnudt_ad_IC_m = dnudt_ad_IC_p = np.zeros(len(nu_ic)-2)
        
        if Syn_l_flag == 1.:
            b_syn_el = const_el*M_F**2.
            b_syn_pr = const_pr*M_F**2.
            dgdt_Syn_el_m = b_syn_el*np.divide(np.power(g_el_mp[0:-1],2.),dg_el)
            dgdt_Syn_el_p = b_syn_el*np.divide(np.power(g_el_mp[1:],2.),dg_el)
            dgdt_Syn_pr_m = b_syn_pr*np.divide(np.power(g_pr_mp[0:-1],2.),dg_pr)
            dgdt_Syn_pr_p = b_syn_pr*np.divide(np.power(g_pr_mp[1:],2.),dg_pr)
        else :
            dgdt_Syn_el_m = dgdt_Syn_el_p = np.zeros(len(g_el)-2)  
            dgdt_Syn_pr_m = dgdt_Syn_pr_p = np.zeros(len(g_pr)-2)  
    
        if IC_l_flag == 1.:
            U_ph = f.U_ph_f(g_el,nu_tot,photons,Radius)
            b_Com_el = 4./3.*sigmaT*np.multiply(c,U_ph)/(m_el*c**2.)
            dgdt_IC_el_m = b_Com_el[1:-1]*np.divide(np.power(g_el_mp[0:-1],2.),dg_el)
            dgdt_IC_el_p = b_Com_el[2:]*np.divide(np.power(g_el_mp[1:],2.),dg_el)
        else:
            dgdt_IC_el_m = dgdt_IC_el_p = np.zeros(len(g_el)-2)    
            
        if pg_pi_l_flag == 1.:
            dgdt_pg_pi_m = np.array(f.dg_dt_pg_approx(g_pr_mp[0:-1],nu_tot,photons))/dg_pr
            dgdt_pg_pi_p = np.array(f.dg_dt_pg_approx(g_pr_mp[1:],nu_tot,photons))/dg_pr
        else:
            dgdt_pg_pi_m = dgdt_pg_pi_p = np.zeros(len(g_pr)-2)
            
        if pg_BH_l_flag == 1.:
            dgdt_pg_BH_m = np.array(f.dg_dt_BH(g_pr_mp[0:-1],nu_tot,photons,f_k_i))/dg_pr
            dgdt_pg_BH_p = np.array(f.dg_dt_BH(g_pr_mp[1:],nu_tot,photons,f_k_i))/dg_pr
        else:
            dgdt_pg_BH_m = dgdt_pg_BH_p = np.zeros(len(g_pr)-2)
           
        if pp_l_flag == 1.:
            dgdt_pp_pi_m = []
            dgdt_pp_pi_p = []  
            dgdt_pp_pi_m=np.divide(0.65*c*n_H*f.cs_pp_inel(g_pr_mp[0:-1]*m_pr*c**2.*0.624151-m_pr*c**2.*0.624151)/(m_pr*c**2.),dg_pr)
            dgdt_pp_pi_p=np.divide(0.65*c*n_H*f.cs_pp_inel(g_pr_mp[1:]*m_pr*c**2.*0.624151-m_pr*c**2.*0.624151)/(m_pr*c**2.),dg_pr)        
        else:
            dgdt_pp_pi_m = dgdt_pp_pi_p = np.zeros(len(g_pr)-2)

        V1 = np.zeros(len(g_pr)-2)
        V2 = 1.+dt*(c/Radius*esc_flag_pr+dgdt_Syn_pr_m+dgdt_pg_pi_m+dgdt_pg_BH_m+dgdt_pp_pi_m)
        V3 = -dt*(dgdt_Syn_pr_p+dgdt_pg_pi_p+dgdt_pg_BH_p+dgdt_pp_pi_p)
        S_ij = N_pr[1:-1]+np.multiply(pr_inj[1:-1],dt)*f.Volume(Radius)
        N_pr[1:-1] = f.thomas(V1, V2, V3, S_ij)
        dN_pr_dVdg_pr = np.array(N_pr/f.Volume(Radius))

        if pg_BH_emis_flag == 1.:
            Q_pg_BH = f.Q_BH_sol(g_el,g_pr,dN_pr_dVdg_pr,nu_tot*h/(m_el*c**2.),np.array(photons),intrp_cs_m,intrp_cs_p,max_intrp_cs_m,max_intrp_cs_p)[1:-1]
        else:
            Q_pg_BH = np.zeros(len(g_el)-2)

        if pg_pi_emis_flag == 1.:
            Q_pg_pi = f.Qp_g_opt(g_el,nu_ic,dN_pr_dVdg_pr,g_pr,np.array(photons),nu_tot,"e+")[1:-1]+f.Qp_g_opt(g_el,nu_ic,dN_pr_dVdg_pr,g_pr,np.array(photons),nu_tot,"e-")[1:-1]
            Q_pg_g = f.Qp_g_opt(g_el,nu_ic,dN_pr_dVdg_pr,g_pr,photons,nu_tot,"2_g")[1:-1] 
            if neutrino_flag == 1.:
                Q_pg_nu = np.multiply(f.Qp_g_opt(g_el,nu_ic,N_pr,g_pr,photons,nu_tot,"nu_mu")+f.Qp_g_opt(g_el,nu_ic,N_pr,g_pr,photons,nu_tot,"\bar_nu_mu")+f.Qp_g_opt(g_el,nu_ic,N_pr,g_pr,photons,nu_tot,"nu_e")+f.Qp_g_opt(g_el,nu_ic,N_pr,g_pr,photons,nu_tot,"\bar_nu_e"),dt)[1:-1]    
            else:
                Q_pg_nu = np.zeros(len(nu_nu)-2)
        else:
            Q_pg_pi = np.zeros(len(g_el)-2)
            Q_pg_g = np.zeros(len(nu_ic)-2)
            Q_pg_nu = np.zeros(len(nu_nu)-2)
            
        if pp_ee_emis_flag == 1.:
            Q_pp_ee = np.multiply(f.Q_e_pp(g_el,g_pr,dN_pr_dVdg_pr/(m_pr*c**2.*0.624151),p_pr,n_H)[1:-1],m_el*c**2.)
        else:
            Q_pp_ee = np.zeros(len(g_el)-2)

        V1 = np.zeros(len(g_el)-2)
        V2 = 1.+dt*(c/Radius*esc_flag_el+dgdt_Syn_el_m+dgdt_IC_el_m+dgdt_ad_el_m)
        V3 = -dt*(dgdt_Syn_el_p+dgdt_IC_el_p+dgdt_ad_el_p) 
        if inj_flag == 1.:
            S_ij = N_el[1:-1]+np.multiply(el_inj[1:-1]+Q_ee+Q_pg_pi+Q_pg_BH+Q_pp_ee,dt)*f.Volume(Radius)
        if inj_flag == 0.:
            S_ij = N_el[1:-1]+(Q_ee+Q_pg_pi+Q_pg_BH+Q_pp_ee)*dt*f.Volume(Radius)
        N_el[1:-1] = f.thomas(V1, V2, V3, S_ij)    
        dN_el_dVdg_el = np.array(N_el/f.Volume(Radius))
         
        if Syn_emis_flag == 1.:
            Q_Syn_el = np.array([f.Q_syn_space(dN_el_dVdg_el,M_F,nu_syn[nu_ind],a_cr_el,C_syn_el,g_el) for nu_ind in range(1,len(nu_syn)-1)]) 
            Q_Syn_pr = np.array([f.Q_syn_space(dN_pr_dVdg_pr,M_F,nu_syn[nu_ind],a_cr_pr,C_syn_pr,g_pr) for nu_ind in range(1,len(nu_syn)-1)])
        else: 
            Q_Syn_el = Q_Syn_pr = np.zeros(len(nu_syn)-2)
            
        if IC_emis_flag == 1.:
            Q_IC = np.array([f.Q_IC(dN_el_dVdg_el,g_el,nu_ic[nu_ind],photons,nu_tot,len(nu_tot)-1) for nu_ind in range(0,len(nu_ic)-1)])[1:]
        else:
            Q_IC = np.zeros(len(nu_ic)-2)  
    
        if SSA_l_flag == 1.:
            aSSA_space_syn = np.array([-np.absolute(f.aSSA(dN_el_dVdg_el,M_F,nu_syn[nu_ind],g_el,dg_l_el)) for nu_ind in range(1,len(nu_syn)-1)])
            aSSA_space_ic = np.array([-np.absolute(f.aSSA(dN_el_dVdg_el,M_F,nu_ic[nu_ind],g_el,dg_l_el)) for nu_ind in range(1,len(nu_ic)-1)])
        else:
            aSSA_space_syn = np.zeros(len(nu_syn)-2) 
            aSSA_space_ic = np.zeros(len(nu_ic)-2)   
            
        if pp_g_emis_flag == 1.:
            Q_pp_g = np.multiply(f.Q_g_pp(nu_ic,g_pr,dN_pr_dVdg_pr/(m_pr*c**2.*0.624151),p_pr,n_H)[1:-1],h)
        else:
            Q_pp_g = np.zeros(len(nu_ic)-2)

        if pp_nu_emis_flag == 1.:
            Q_pp_nu = 2.*np.multiply(f.Q_nu_e_pp(nu_nu,g_pr,dN_pr_dVdg_pr/(m_pr*c**2.*0.624151),p_pr,n_H)[1:-1],dt)*h*f.Volume(Radius)+2.*np.multiply(f.Q_nu_mu_pp(nu_nu,g_pr,dN_pr_dVdg_pr/(m_pr*c**2.*0.624151),p_pr,n_H)[1:-1],dt)*h*f.Volume(Radius)
        else:
            Q_pp_nu = np.zeros(len(nu_nu)-2)

        V1 = np.zeros(len(nu_syn)-2)
        V2 = 1.+dt*(c/Radius+dnudt_ad_syn_m-aSSA_space_syn*c+a_gg_f_syn*c)
        V3 = -dt*dnudt_ad_syn_p
        S_ij = photons_syn[1:-1]+4.*np.pi*(Q_Syn_el+Q_Syn_pr)*dt*f.Volume(Radius)
        photons_syn[1:-1] = f.thomas(V1, V2, V3, S_ij)  
        
        V1 = np.zeros(len(nu_ic)-2)
        V2 = 1.+dt*(c/Radius+dnudt_ad_IC_m-aSSA_space_ic*c+a_gg_f_ic*c)
        V3 = -dt*dnudt_ad_IC_p
        S_ij = photons_IC[1:-1]+(Q_IC+Q_pg_g+Q_pp_g)*dt*f.Volume(Radius)
        photons_IC[1:-1] = f.thomas(V1, V2, V3, S_ij)
        
        if gg_flag == 0.:
            a_gg_f_ic = np.zeros(len(nu_ic)-2)
            a_gg_f_syn = np.zeros(len(nu_syn)-2)
            Q_ee = np.zeros(len(g_el)-2)
        else: 
            a_gg_f_ic = np.array(f.a_gg(nu_ic,nu_tot,photons)[1:-1])
            a_gg_f_syn = np.array(f.a_gg(nu_syn,nu_tot,photons)[1:-1])
            Q_ee = f.Q_ee_f(nu_tot,photons,nu_tot,photons,g_el,Radius)[1:-1] 
            
        V1 = np.zeros(len(nu_nu)-2)
        V2 = 1.+dt*(c/Radius*np.ones(len(nu_nu)-2))
        V3 = np.zeros(len(nu_nu)-2)
        S_ij = N_nu[1:-1]+Q_pp_nu+Q_pg_nu
        N_nu[1:-1] = f.thomas(V1, V2, V3, S_ij)
        
        if day_counter<time_real:            
            day_counter=day_counter+step_alg*R0/c
            photons = f.photons_tot(nu_syn,nu_bb,photons_syn,nu_ic,photons_IC,nu_tot,dN_dVdnu_BB*f.Volume(Radius),dN_dVdnu_pl*f.Volume(Radius),dN_dVdnu_user*f.Volume(Radius))/f.Volume(Radius)
            Spec_temp_tot = np.multiply(photons,h*nu_tot**2.)*4.*np.pi/3.*Radius**2.*c  
            pr1 = [[str(el_list) for el_list in np.log10(g_el) ],[str(el_list) for el_list in np.log10(dN_el_dVdg_el) ]] #np.log10(\gamma_{el}), np.log10(dN_{el}/(dV d\gamma_{el}))
            pr2 = [[str(el_list) for el_list in np.log10(nu_tot) ],[str(el_list) for el_list in np.log10(Spec_temp_tot) ]] #np.log10(\nu), np.log10(\nu L_{\nu))
            pr3 = [[str(el_list) for el_list in np.log10(g_pr) ],[str(el_list) for el_list in np.log10(dN_pr_dVdg_pr) ]] #np.log10(\gamma_{pr}), np.log10(dN_{pr}/(dV d\gamma_{pr}))
            pr4 = [[str(el_list) for el_list in np.log10(nu_nu) ],[str(el_list) for el_list in np.log10(N_nu) ]] #np.log10(E_{\nu}/h), np.log10(dN/d(E_{\nu}/h)) 
            # Here is where you unpack everything
            for row in zip(*pr1):
                f1.write(' '.join(row) + '\n')
            for row in zip(*pr2):
                f2.write(' '.join(row) + '\n') 
            for row in zip(*pr3):
                f3.write(' '.join(row) + '\n')
            for row in zip(*pr4):
                f4.write(' '.join(row) + '\n')         
        
print("--- %s seconds ---" % "{:.2f}".format((time.time() - start_time)))
