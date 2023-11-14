#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 19:09:51 2023

@author: mapet
"""

# This is the leptohadronic version of a radiative transfer code LeHaMoC. 
# Copyright (C) 2023  S. I. Stathopoulos, M. Petropoulou.  
# When using this code, make reference to the following 
# publication: Stathopoulos et al., 2023, A&A    

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation (check licence).

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
 

import numpy as np
import astropy.units as u
from astropy import constants as const 
from astropy.modeling.models import BlackBody
import pandas as pd
import LeHaMoC_f as f # imports functions

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
#########################

def LeMoC(params, fileName):
# Free model parameters    
    R0 = 10**params[0]
    B0 = 10**params[1]
    g_PL_min = params[2]
    g_PL_max = params[3]
    comp_el = params[4]
    p_el = params[5]
    
    g_min_el = g_PL_min - 1.
    g_max_el = g_PL_max + 1.
    
    fileObj = open(fileName)
    params_frozen = {} 
    for line in fileObj:
        line=line.strip() 
        key_value = line.split("=")
        params_frozen[key_value[0].strip()] = float(key_value[1].strip())
    
    time_init = float(params_frozen['time_init']) #R0/c
    time_end = float(params_frozen['time_end']) #R0/c
    step_alg = float(params_frozen['step_alg']) #R0/c
    grid_g_el = float(params_frozen['grid_g_el'])
    grid_nu = float(params_frozen['grid_nu'])  
    Vexp = float(params_frozen['Vexp'])*c #/c
    m = float(params_frozen['m']) 
    inj_flag = float(params_frozen['inj_flag'])
    Ad_l_flag = float(params_frozen['Ad_l_flag'])
    Syn_l_flag = float(params_frozen['Syn_l_flag'])
    Syn_emis_flag = float(params_frozen['Syn_emis_flag'])
    IC_l_flag = float(params_frozen['IC_l_flag'])
    IC_emis_flag = float(params_frozen['IC_emis_flag'])
    SSA_l_flag = float(params_frozen['SSA_l_flag'])
    gg_flag = float(params_frozen['gg_flag'])
    esc_flag = float(params_frozen['esc_flag'])
    BB_flag = float(params_frozen['BB_flag'])
    temperature = 10**float(params_frozen['BB_temperature']) #log 
    GB_ext = float(params_frozen['GB_ext'])
    PL_flag = float(params_frozen['PL_flag'])
    dE_dV_ph = float(params_frozen['dE_dV_ph'])
    nu_min_ph = float(params_frozen['nu_min_ph'])
    nu_max_ph = float(params_frozen['nu_max_ph'])
    s_ph = float(params_frozen['s_ph'])
    User_ph = float(params_frozen['User_ph'])
  
    time_real = time_init
    dt = step_alg*R0/c # time step used for solving the PDE
    day_counter = 0.
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
#Units (nu,dN/dVdnu)           
    if User_ph == 0.:
        dN_dVdnu_user = np.zeros(len(nu_tot))
    else: 
        Photons_spec_user = pd.read_csv('Photons_spec_user.txt',names=('logx','logy'),sep=",")
        nu_user = 10**np.array(Photons_spec_user.logx)
        dN_dVdnu_user_temp = 10**np.array(Photons_spec_user.logy)
        dN_dVdnu_user_temp[-1] = 10**(-160.)
        dN_dVdnu_user = 10**np.interp(np.log10(nu_tot),np.log10(nu_user),np.log10(dN_dVdnu_user_temp))
                        
#Initialize arrays for particles and photons
    N_el = np.zeros(len(g_el)) # Number of electrons & positrons
    Q_ee = np.zeros(len(g_el)-1) # Pair production rate
    el_inj = np.ones(len(g_el))*10**(-260.) # Primary electron injection rate 
    el_inj[index_PL_min:index_PL_max] = f.Q_el_Lum(f.Lum_e_inj(10**comp_el,Radius),p_el,g_el[index_PL_min],g_el[index_PL_max])*g_el[index_PL_min:index_PL_max]**(-p_el)
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
    while time_real <  time_end*R0/c: 
        time_real += dt    
        Radius = f.R(R0,time_real,time_init,Vexp)
        M_F = f.B(B0,R0,Radius,m)
        a_cr_el = 3.*q*M_F/(4.*np.pi*m_el*c)
        
# Calculate total dN/dVdnu
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
            day_counter=day_counter+dt
            photons = f.photons_tot(nu_syn,nu_bb,photons_syn,nu_ic,photons_IC,nu_tot,dN_dVdnu_BB*f.Volume(Radius),dN_dVdnu_pl*f.Volume(Radius),dN_dVdnu_user*f.Volume(Radius))/f.Volume(Radius)
            Spec_temp_tot = np.multiply(photons,h*nu_tot**2.)*4.*np.pi/3.*Radius**2.*c  
            
#returns v'[Hz] and (vLv)' [erg s^{-1}] in the comoving frame
    return (nu_tot,Spec_temp_tot)


def LeHaMoC(params, fileName):
## CAUTION ##
# this hadronic module has only proton synchrotron included, no adiabatic losses 
    
# Free model parameters    
    R0 = 10**params[0]
    B0 = 10**params[1]
    g_PL_min = params[2]
    g_PL_max = params[3]
    comp_el = params[4]
    p_el = params[5]
    g_PL_min_pr = params[6]
    g_PL_max_pr = params[7]
    comp_pr = params[8]
    p_pr = params[9]    
    
    g_min_el = g_PL_min - 0.5
    g_max_el = g_PL_max + 0.5
    
    g_min_pr = g_PL_min_pr - 0.1 
    g_max_pr = g_PL_max_pr + 0.5
        
    fileObj = open(fileName)
    params_frozen = {} 
    for line in fileObj:
        line=line.strip() 
        key_value = line.split("=")
        params_frozen[key_value[0].strip()] = float(key_value[1].strip())
    
    time_init = float(params_frozen['time_init']) #R0/c
    time_end = float(params_frozen['time_end']) #R0/c
    step_alg = float(params_frozen['step_alg']) #R0/c
    grid_g_el = float(params_frozen['grid_g_el'])
    grid_g_pr = float(params_frozen['grid_g_pr'])
    grid_nu = float(params_frozen['grid_nu'])  
    Vexp = float(params_frozen['Vexp'])*c #/c
    m = float(params_frozen['m']) 
    inj_flag = float(params_frozen['inj_flag']) 
    Syn_l_flag = float(params_frozen['Syn_l_flag'])
    Syn_emis_flag = float(params_frozen['Syn_emis_flag'])
    IC_l_flag = float(params_frozen['IC_l_flag'])
    IC_emis_flag = float(params_frozen['IC_emis_flag'])
    SSA_l_flag = float(params_frozen['SSA_l_flag'])
    gg_flag = float(params_frozen['gg_flag'])
    esc_flag = float(params_frozen['esc_flag'])
    BB_flag = float(params_frozen['BB_flag'])
    temperature = 10**float(params_frozen['BB_temperature']) #log 
    GB_ext = float(params_frozen['GB_ext'])
    PL_flag = float(params_frozen['PL_flag'])
    dE_dV_ph = float(params_frozen['dE_dV_ph'])
    nu_min_ph = float(params_frozen['nu_min_ph'])
    nu_max_ph = float(params_frozen['nu_max_ph'])
    s_ph = float(params_frozen['s_ph'])
    User_ph = float(params_frozen['User_ph'])
  
    time_real = time_init
    dt = step_alg*R0/c # time step used for solving the PDE
    day_counter = 0.
    Radius = R0  

# initialization of the electron Lorentz factor array
    grid_size = grid_g_el
    g_el = np.logspace(g_min_el,g_max_el,int(grid_size))
    g_el_mp = np.array([(g_el[im+1]+g_el[im-1])/2. for im in range(0,len(g_el)-1)])
    dg_el = np.array([((g_el[im+1])-(g_el[im-1]))/2. for im in range(1,len(g_el)-1)])   # delta gamma 
    dg_l_el = np.log(g_el[1])-np.log(g_el[0]) # logarithmic delta gamma
 
# initialization of the proton Lorentz factor array
    grid_size_pr = grid_g_pr
    g_pr = np.logspace(g_min_pr,g_max_pr,int(grid_size_pr))
    g_pr_mp = np.array([(g_pr[im+1]+g_pr[im-1])/2. for im in range(0,len(g_pr)-1)])
    dg_pr = np.array([((g_pr[im+1])-(g_pr[im-1]))/2. for im in range(1,len(g_pr)-1)])
    
    if g_PL_max == g_max_el:
        index_PL_max = -1
    else:
        index_PL_max = min(min(np.where(g_el > 10**g_PL_max)))
        
    if g_PL_min == 0.:
        index_PL_min = 1
    else: 
        index_PL_min = max(max(np.where(g_el < 10**g_PL_min)))        
        
    if g_PL_max_pr == g_max_pr:
        index_PL_max_pr = -1
    else:
        index_PL_max_pr = min(min(np.where(g_pr > 10**g_PL_max_pr)))
        
    if g_PL_min_pr == 0.:
        index_PL_min_pr = 1
    else: 
        index_PL_min_pr = max(max(np.where(g_pr < 10**g_PL_min_pr)))        

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
#Units (nu,dN/dVdnu)           
    if User_ph == 0.:
        dN_dVdnu_user = np.zeros(len(nu_tot)) 
    else: 
        Photons_spec_user = pd.read_csv('Photons_spec_user.txt',names=('logx','logy'),sep=",")
        nu_user = 10**np.array(Photons_spec_user.logx)
        dN_dVdnu_user_temp = 10**np.array(Photons_spec_user.logy)
        dN_dVdnu_user_temp[-1] = 10**(-160.)
        dN_dVdnu_user = 10**np.interp(np.log10(nu_tot),np.log10(nu_user),np.log10(dN_dVdnu_user_temp))
                       
#Initialize arrays for particles and photons
    N_el = np.zeros(len(g_el)) # Number of electrons & positrons
    Q_ee = np.zeros(len(g_el)-1) # Pair production rate
    el_inj = np.ones(len(g_el))*10**(-260.) # Primary electron injection rate 
    el_inj[index_PL_min:index_PL_max] = f.Q_el_Lum(f.Lum_e_inj(10**comp_el,Radius),p_el,g_el[index_PL_min],g_el[index_PL_max])*g_el[index_PL_min:index_PL_max]**(-p_el)
    N_el = el_inj.copy()
    N_el[0] = N_el[-1] = 10**(-260.) # boundary conditions 
    
#Initialize arrays for particles and photons
    N_pr = np.zeros(len(g_pr)) # Number of electrons & positrons
    pr_inj = np.ones(len(g_pr))*10**(-260.) # Primary electron injection rate 
    pr_inj[index_PL_min_pr:index_PL_max_pr] = f.Q_pr_Lum(f.Lum_pr_inj(10**comp_pr,Radius),p_pr,g_pr[index_PL_min_pr],g_pr[index_PL_max_pr])*g_pr[index_PL_min_pr:index_PL_max_pr]**(-p_pr)
    N_pr = pr_inj.copy()
    N_pr[0] = N_pr[-1] = 10**(-260.) # boundary conditions 

    photons_syn = np.ones(len(nu_syn))*10**(-260.)
    photons_IC = np.ones(len(nu_ic))*10**(-260.)
    
    photons_syn = np.append(photons_syn,10**(-260.))
    dN_dVdnu_BB = np.append(dN_dVdnu_BB,10**(-260.))
    
    nu_syn = np.append(nu_syn,nu_tot[-1])
    nu_bb = np.append(nu_bb,nu_tot[-1])

# Solution of the PDEs
    while time_real <  time_end*R0/c: 
        time_real += dt    
        Radius = f.R(R0,time_real,time_init,Vexp)
        M_F = f.B(B0,R0,Radius,m)
        a_cr_el = 3.*q*M_F/(4.*np.pi*m_el*c)
        a_cr_pr = 3.*q*M_F/(4.*np.pi*m_pr*c)
        
# Calculate total dN/dVdnu
        photons = f.photons_tot(nu_syn,nu_bb,photons_syn,nu_ic,photons_IC,nu_tot,dN_dVdnu_BB*f.Volume(Radius),dN_dVdnu_pl*f.Volume(Radius),dN_dVdnu_user*f.Volume(Radius))/f.Volume(Radius)
        
        if Syn_l_flag == 1.:
            b_syn_el = (4./3.)*sigmaT/(8.*np.pi*m_el*c)*M_F**2.
            b_syn_pr = b_syn_el*(m_el/m_pr)**3.
            dgdt_Syn_el_m = b_syn_el*np.divide(np.power(g_el_mp[0:-1],2.),dg_el)
            dgdt_Syn_el_p = b_syn_el*np.divide(np.power(g_el_mp[1:],2.),dg_el)
            dgdt_Syn_pr_m = b_syn_pr*np.divide(np.power(g_pr_mp[0:-1],2.),dg_pr)
            dgdt_Syn_pr_p = b_syn_pr*np.divide(np.power(g_pr_mp[1:],2.),dg_pr)
        else :
            dgdt_Syn_el_m = np.zeros(len(g_el)-2)      
            dgdt_Syn_el_p = np.zeros(len(g_el)-2)
            dgdt_Syn_pr_m = np.zeros(len(g_pr)-2)      
            dgdt_Syn_pr_p = np.zeros(len(g_pr)-2) 
    
        if IC_l_flag == 1.:
            U_ph = f.U_ph_f(g_el,nu_tot,photons,Radius)
            b_Com_el = 4./3.*sigmaT*np.multiply(c,U_ph)/(m_el*c**2.)
            dgdt_IC_el_m = b_Com_el[1:-1]*np.divide(np.power(g_el_mp[0:-1],2.),dg_el)
            dgdt_IC_el_p = b_Com_el[2:]*np.divide(np.power(g_el_mp[1:],2.),dg_el)
        else:
            dgdt_IC_el_m = np.zeros(len(g_el)-2)    
            dgdt_IC_el_p = np.zeros(len(g_el)-2) 
    
        V1 = np.zeros(len(g_el)-2)
        V2 = 1.+dt*(c/Radius*esc_flag+dgdt_Syn_el_m+dgdt_IC_el_m)
        V3 = -dt*(dgdt_Syn_el_p+dgdt_IC_el_p) 
        if inj_flag == 1.:
            S_ij = N_el[1:-1]+np.multiply(el_inj[1:-1],dt)+np.multiply(Q_ee[1:],dt)*f.Volume(Radius)
        if inj_flag == 0.:
            S_ij = N_el[1:-1]+np.multiply(Q_ee[1:],dt)*f.Volume(Radius)
            
        N_el[1:-1] = f.thomas(V1, V2, V3, S_ij)    
        dN_el_dVdg_el = np.array(N_el/f.Volume(Radius))
        
        
        V1 = np.zeros(len(g_pr)-2)
        V2 = 1.+dt*(c/Radius*esc_flag+dgdt_Syn_pr_m)
        V3 = -dt*(dgdt_Syn_pr_p)
        if inj_flag == 1.:
            S_ij = N_pr[1:-1]+np.multiply(pr_inj[1:-1],dt)
        if inj_flag == 0.:
            S_ij = N_pr[1:-1]            
             
        N_pr[1:-1] = f.thomas(V1, V2, V3, S_ij) 
        dN_pr_dVdg_pr = np.array(N_pr/f.Volume(Radius))
    
        if Syn_emis_flag == 1.:
            Q_Syn_el = np.divide([f.Q_syn_space(dN_el_dVdg_el,M_F,nu_syn[nu_ind],a_cr_el,g_el) for nu_ind in range(len(nu_syn)-1)], f.cor_factor_syn_el(g_el,R0,10**4.,p_el,f.Lum_e_inj(comp_el,R0)))
            Q_Syn_pr = [f.Q_syn_space_pr(dN_pr_dVdg_pr,M_F,nu_syn[nu_ind],a_cr_pr,g_pr) for nu_ind in range(len(nu_syn)-1)] 
        else: 
            Q_Syn_el = np.zeros(len(nu_syn)-1)
            Q_Syn_pr = np.zeros(len(nu_syn)-1)
    
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
        V2 = 1.+dt*(c/Radius-np.multiply(aSSA_space_syn[1:-1],1)*c)
        V3 =  np.zeros(len(nu_syn)-2)
        S_ij = photons_syn[1:-1]+4.*np.pi*np.multiply(Q_Syn_el,dt)[1:]*f.Volume(Radius)+4.*np.pi*np.multiply(Q_Syn_pr,dt)[1:]*f.Volume(Radius)
        photons_syn[1:-1] = f.thomas(V1, V2, V3, S_ij)  
    
        V1 = np.zeros(len(nu_ic)-2)
        V2 = 1.+dt*(c/Radius+np.multiply(a_gg_f[1:-1],c)-np.multiply(aSSA_space_ic[1:-1],1)*c)
        V3 =  np.zeros(len(nu_ic)-2)
        S_ij = photons_IC[1:-1]+np.multiply(Q_IC,dt)[1:]*f.Volume(Radius)
        photons_IC[1:-1] = f.thomas(V1, V2, V3, S_ij)
        
        if gg_flag == 0.:
            a_gg_f = np.zeros(len(nu_ic))
        else: 
            a_gg_f = f.a_gg(nu_ic,nu_tot,photons)
            Q_ee = f.Q_ee_f(nu_tot,photons,nu_ic,photons_IC/f.Volume(Radius),g_el,Radius)                  

        if day_counter<time_real:
            day_counter=day_counter+dt
            photons = f.photons_tot(nu_syn,nu_bb,photons_syn,nu_ic,photons_IC,nu_tot,dN_dVdnu_BB*f.Volume(Radius),dN_dVdnu_pl*f.Volume(Radius),dN_dVdnu_user*f.Volume(Radius))/f.Volume(Radius)
            Spec_temp_tot = np.multiply(photons,h*nu_tot**2.)*4.*np.pi/3.*Radius**2.*c  
        
#returns nu'[Hz] and (nuL_nu)' [erg s^{-1}] in the comoving frame
    return (nu_tot,Spec_temp_tot) 
