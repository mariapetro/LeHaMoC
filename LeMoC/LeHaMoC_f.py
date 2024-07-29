#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 19:32:57 2023

@author: mapet
"""

from shapely.geometry import LineString
from constants import *

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


#Electron normalization for a power-law injection with sharp off
def Q_el_Lum(L_el,p_el,g_min_el,g_max_el):
    if p_el == 2.:
        return L_el/(m_el*c**2.*np.log(g_max_el/g_min_el)) 
    else:
        return L_el*(-p_el+2.)/(m_el*c**2.*(g_max_el**(-p_el+2.)-g_min_el**(-p_el+2.)))
    
#Proton normalization for a power-law injection with sharp off
def Q_pr_Lum(L_pr,p_pr,g_min_pr,g_max_pr):
    if p_pr == 2.:
        return L_pr/(m_pr*c**2.*np.log(g_max_pr/g_min_pr)) 
    else:
        return L_pr*(-p_pr+2.)/(m_pr*c**2.*(g_max_pr**(-p_pr+2.)-g_min_pr**(-p_pr+2.)))    

# Converts electron compactness to luminosity
def Lum_e_inj(comp,R):
    return 4.*np.pi*R*m_el*c**3.*comp/(sigmaT)

# Converts proton compactness to luminosity
def Lum_pr_inj(comp,R):
    return 4.*np.pi*R*m_pr*c**3.*comp/(sigmaT)

# Converts luminosity to flux
def nuF_nu_obs(nu_L_nu,Dist_in_pc,delta):
    return np.multiply(nu_L_nu,delta**4.)/(4.*np.pi*(Dist_in_pc*pc)**2.)

#Synchrotron critical frequency for electrons (Radiative Processes in Astrophysics, by George B. Rybicki, Alan P. Lightman, Wiley-VCH , June 1986.)
def nu_c(gamma,B):
    return 3./(4.*np.pi)*q*B/(m_el*c)*gamma**2.

#Synchrotron critical frequency for electrons (Radiative Processes in Astrophysics, by George B. Rybicki, Alan P. Lightman, Wiley-VCH , June 1986.)
def nu_c_pr(gamma,B):
    return 3./(4.*np.pi)*q*B/(m_pr*c)*gamma**2.

#Synchrotron emissivity dN/dVd\nudtd\Omega (Relativistic Jets from Active Galactic Nuclei, by M. Boettcher, D.E. Harris, ahd H. Krawczynski, Berlin: Wiley, 2012)
def Q_syn_space(Np,B,nu,a_cr,g):                      
    C_syn = sigmaT*c/(h*24.*np.pi**2.*0.8975)*(4.*np.pi*m_el*c/(3.*q))**(4./3.)
    syn_em = C_syn*B**(2./3.)*nu**(-2./3.)*(np.trapz(Np*g**(1./3.)*np.exp(-nu/(a_cr*g**2.)), np.log(g)))
    return syn_em

#Synchrotron emissivity dN/dVd\nudtd\Omega for protons (Relativistic Jets from Active Galactic Nuclei, by M. Boettcher, D.E. Harris, ahd H. Krawczynski, Berlin: Wiley, 2012)
def Q_syn_space_pr(Np,B,nu,a_cr,g):                      
    C_syn = sigmaT*c/(h*24.*np.pi**2.*0.8975)*(4.*np.pi*m_pr*c/(3.*q))**(4./3.)*(m_el/m_pr)**2.
    syn_em = C_syn*B**(2./3.)*nu**(-2./3.)*(np.trapz(Np*g**(1./3.)*np.exp(-nu/(a_cr*g**2.)), np.log(g)))
    return syn_em

#Synchrotron self absorption coefficient (High Energy Radiation from Black Holes: Gamma Rays, Cosmic Rays, and Neutrinos by Charles D. Dermer and Govind Menon. Princeton Univerisity Press, 2009)
def aSSA(N_el,B,nu,g,dg_l_el):
    return q**3.*B*nu**(-5./3.)/(2.*m_el**2.*c**2.*0.8975)*np.trapz((2.*nu_c(g[1:-1],B))**(-1./3.)*np.exp(-nu/nu_c(g[1:-1],B))*(np.diff(N_el[1:])/dg_l_el-2.*N_el[1:-1]),np.log(g[1:-1]))

#Synchrotron self absorption frequency (determined by solving tau_ssa = 1)
def SSA_frequency(index_SSA,nu,aSSA_space,R):
    if index_SSA>0:
        line_1 = LineString(np.column_stack((np.log10(nu[index_SSA-2:index_SSA+1]),np.log10(np.multiply(aSSA_space,-R)[index_SSA-2:index_SSA+1]))))
        line_2 = LineString(np.column_stack((np.log10(nu[index_SSA-2:index_SSA+1]),np.zeros([3]))))
        int_pt = line_1.intersection(line_2)
        return (10**int_pt.x) 

# Inverse Compton scattering emissivity  (from Blumenthal & Gould 1970, Rev. Mod. Phys. 42, 237)  
def Q_IC_space_optimized(Np, g_el, nu_ic_temp, photons, nu_targ, index):
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

#gamma gamma absorption coefficient  ( Coppi P. S., Blandford R. D., 1990, MNRAS, 245, 453. doi:10.1093/mnras/245.3.453) 
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
    return(np.multiply(c*sigmaT*m_el*c**2./h,Q_ee_temp[:-1]))

#computes total photon spectrum by adding different spectral components
def photons_tot(nu_syn,nu_bb,photons_syn,nu_ic,photons_IC,nu_tot,photons_bb,photons_pl,photons_user):
    with np.errstate(divide='ignore'):
        return  10**(np.interp(np.log10(nu_tot),np.log10(nu_bb),np.log10(photons_bb)))+10**(np.interp(np.log10(nu_tot),np.log10(nu_syn),np.log10(photons_syn)))+10**(np.interp(np.log10(nu_tot),np.log10(nu_ic),np.log10(photons_IC)))+photons_pl+photons_user 
       

#computes energy density of target photons for ICS scattering in Thomson limit   
def U_ph_f(g,nu_target,photons_target,R):
    U_ph_temp=[]
    U_ph_tot=m_el*c**2./(sigmaT*R)*np.trapz(nu_target[:-1]*photons_target[:-1]*(sigmaT*R*m_el*c**2./h),nu_target[:-1])
    for l_f in g:
        if (3.*m_el*c**2.)/(4.*h*l_f)<nu_target[-1]:
            photons_temp_v=np.interp(np.log10((3.*m_el*c**2.)/(4.*h*l_f)),np.log10(nu_target),np.log10(photons_target))
            index_temp=max(max(np.where(np.log10(nu_target) < np.log10((3.*m_el*c**2.)/(4.*h*l_f))))+1)
            photons_temp=np.insert(photons_target,index_temp, 10**photons_temp_v)
            nu_temp=np.insert(nu_target,index_temp, (3.*m_el*c**2.)/(4.*h*l_f))
            n_d_targ=h*nu_temp/(m_el*c**2.)
            U_ph_temp.append(m_el*c**2./(sigmaT*R)*np.trapz(n_d_targ[:index_temp+1]*photons_temp[:index_temp+1]*(sigmaT*R*m_el*c**2./h),n_d_targ[:index_temp+1]))
        else: U_ph_temp.append(U_ph_tot)
    return(U_ph_temp)

#computes correction factor in electron injection luminosity by checking the energy balance in a fast synchrotron cooling scenario
def cor_factor_syn_el(g_el_space,R0,B0,p_el,Lum_e_injected):
    #Constants
    time_init = 0.
    time_end = 20.
    step_alg = 1.

    g_el = g_el_space
    g_el_mp = np.array([(g_el[im+1]+g_el[im-1])/2. for im in range(0,len(g_el)-1)])
    C_syn_el = sigmaT*c/(h*24.*np.pi**2.*0.8975)*(4.*np.pi*m_el*c/(3.*q))**(4./3.)
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

        # Q_Syn_el = [Q_syn_space(N_el/Volume(Radius),B,nu_syn[nu_ind],a_cr_el,C_syn_el,g_el) for nu_ind in range(len(nu_syn)-1)] 
        Q_Syn_el = [Q_syn_space(N_el/Volume(Radius),B,nu_syn[nu_ind],a_cr_el,g_el) for nu_ind in range(len(nu_syn)-1)] 

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