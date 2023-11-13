import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.modeling.models import BlackBody
import timeit
from astropy import constants as const 

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

# electron normalization for a power-law injection with sharp cut off
def Q_el_Lum(L_el,p_el,g_min_el,g_max_el):
    if p_el == 2.:
        return L_el/(m_el*c**2.*np.log(g_max_el/g_min_el)) 
    else:
        return L_el*(-p_el+2.)/(m_el*c**2.*(g_max_el**(-p_el+2.)-g_min_el**(-p_el+2.)))
    
# converts electron compactness to luminosity
def Lum_e_inj(comp,R):
    return 4.*np.pi*R*m_el*c**3.*comp/(sigmaT)

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

#Inverse Compton scattering emissivity  (from Blumenthal & Gould 1970, Rev. Mod. Phys. 42, 237)  
def Q_IC_space(Np, g_el, nu_ic_temp, photons, nu_targ, index):
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

#Computes the number of points per decade
def N_of_points_per_decade(lst, item1, item2):
    item_counter = 0
    for lst_item in lst:
        if lst_item > item1 and lst_item<item2:
            item_counter += 1
    return item_counter

#computes the total photon field for the tests
def photons_tot_test(nu_syn,photons_syn,nu_ic,photons_IC,nu_tot):
    return 10**(np.interp(np.log10(nu_tot),np.log10(nu_syn),np.log10(photons_syn)))+10**(np.interp(np.log10(nu_tot),np.log10(nu_ic),np.log10(photons_IC))) 

def photons_tot(nu_syn,nu_bb,photons_syn,photons_bb,nu_ic,photons_IC,nu_tot):
    return 10**(np.interp(np.log10(nu_tot),np.log10(nu_bb),np.log10(photons_bb)))+10**(np.interp(np.log10(nu_tot),np.log10(nu_syn),np.log10(photons_syn)))+10**(np.interp(np.log10(nu_tot),np.log10(nu_ic),np.log10(photons_IC))) 
    
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
 
#Correction to the synchrotron emissivity to account for the energy balance
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
        M_F = B0
        a_cr_el = 3.*q*M_F/(4.*np.pi*m_el*c)
        
        b_syn_el = const_el*M_F**2.
        dgdt_Syn_el_m = b_syn_el*np.divide(np.power(g_el_mp[0:-1],2.),dg_el)
        dgdt_Syn_el_p = b_syn_el*np.divide(np.power(g_el_mp[1:],2.),dg_el)

        V1 = np.zeros(len(g_el)-2)
        V2 = 1.+dt*(c/Radius+dgdt_Syn_el_m)
        V3 = -dt*(dgdt_Syn_el_p) 
        S_ij = N_el[1:-1]+el_inj[1:-1].copy()*dt
        N_el[1:-1] = thomas(V1, V2, V3, S_ij)

        Q_Syn_el = [Q_syn_space(N_el/Volume(Radius),M_F,nu_syn[nu_ind],a_cr_el,C_syn_el,g_el) for nu_ind in range(len(nu_syn)-1)] 

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

#Correction to the IC emissivity to account for the energy balance
def cor_factor_IC_el(g_el_space,R0,B0,p_el,Lum_e_injected):
    #Constants
    time_init = 0.
    time_end = 20.
    step_alg = 1.

    g_el = g_el_space
    g_el_mp = np.array([(g_el[im+1]+g_el[im-1])/2. for im in range(0,len(g_el)-1)])
    dg_el = np.array([((g_el[im+1])-(g_el[im-1]))/2. for im in range(1,len(g_el)-1)])

    #gamma-nu space
    nu_ic = np.logspace(12.,32.,200)
    temperature = 10**5.
    bb = BlackBody(temperature*u.K)
    nu_bb = np.array(np.logspace(np.log10(5.879*10**10*temperature)-6., np.log10(5.879*10**10*temperature)+1.)*u.Hz)
    photons_bb = np.array(4.*np.pi/c*bb(nu_bb)/(h*nu_bb))
    photons_bb = np.append(photons_bb,10**(-260.))
    nu_bb = np.append(nu_bb,10**(35.))
    day_counter=0.
    
    #Initialize values for particles and photons
    N_el = np.zeros(len(g_el))
    el_inj = np.zeros(len(g_el))
    el_inj = Q_el_Lum(Lum_e_injected,p_el,g_el[1],g_el[-2])*g_el**(-p_el)
    photons_IC = np.ones(len(nu_ic))*10**(-260.)
    N_el[0] = N_el[-1] = 10**(-160.)    
    
    El_lum = []
    Ph_lum = []
    t = []
    time_real = time_init

    while time_real <  time_end*R0/c:
        dt = 1.01*R0/c
        time_real += dt    
        t.append(time_real)
        Radius = R0
        U_ph = U_ph_f(g_el,nu_bb,photons_bb,Radius)
        b_Com_el = 4./3.*sigmaT*np.multiply(c,U_ph)/(m_el*c**2.)
        dgdt_IC_el_m = b_Com_el[1:-1]*np.divide(np.power(g_el_mp[0:-1],2.),dg_el)
        dgdt_IC_el_p = b_Com_el[2:]*np.divide(np.power(g_el_mp[1:],2.),dg_el)

        V1 = np.zeros(len(g_el)-2)
        V2 = 1.+dt*(c/R0+dgdt_IC_el_m)
        V3 = -dt*(dgdt_IC_el_p) 
        S_ij = N_el[1:-1]+np.multiply(el_inj[1:-1],dt)
        N_el[1:-1] = thomas(V1, V2, V3, S_ij)    
        Q_IC = [Q_IC(N_el/Volume(R0),g_el,nu_ic[nu_ind],np.array(photons_bb),nu_bb,len(nu_bb)-1) for nu_ind in range(0,len(nu_ic)-1)]

        V1 = np.zeros(len(nu_ic)-2)
        V2 = 1.+dt*(c/R0+np.zeros(len(nu_ic)-2))
        V3 = np.zeros(len(nu_ic)-2)
        S_ij = photons_IC[1:-1]+np.multiply(Q_IC,dt)[1:]*Volume(R0)
        photons_IC[1:-1] = thomas(V1, V2, V3, S_ij)

        if day_counter < time_real:
            day_counter = day_counter+step_alg*R0/c
            IC_temp_plot = np.multiply(photons_IC/Volume(R0),h*nu_ic**2.)*4.*np.pi/3.*R0**2.*c
            El_lum.append(np.trapz(el_inj*g_el**2.*m_el*c**2.,np.log(g_el)))
            Ph_lum.append(np.trapz(IC_temp_plot,np.log(nu_ic)))
    return np.divide(Ph_lum,El_lum)[-1]

def plotting_res1(x1,y1,x2,y2,axlabel_y1,axlabel_x1,xlim_l,xlim_u,ylim_l,ylim_u,label1,label2,filename):
    tot_fig,axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]},sharex=True, sharey=False,constrained_layout=True, figsize=(8,6))

    plt.subplots_adjust(wspace=1, hspace=0.1)

    ax = plt.subplot(211)
    ax.set_yscale("log", nonpositive='clip')
    ax.set_xscale("log", nonpositive='clip')
    plt.plot(x1,y1,c='orchid',alpha=1.,label=label1,linewidth=3)
    plt.plot(x2,y2,label=label2,linewidth=3,ls="-.",c="yellowgreen")
    ax.set_ylabel(axlabel_y1,fontsize=15)
    ax.tick_params(axis='x', which='major', labelsize=15)
    ax.tick_params(axis='y', which='major', labelsize=15) 
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
    ax.set_xlim(xlim_l,xlim_u)
    ax.set_ylim(ylim_l,ylim_u)
    plt.legend(fontsize=15)

    ax2 = plt.subplot(212)
    ax2.tick_params(axis='x', which='major', labelsize=15)
    ax2.tick_params(axis='y', which='major', labelsize=15) 
    ax2.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
    ax2.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
    ax2.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
    ax2.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
    ax2.plot(x2,np.divide(np.interp(x2,x1,y1),y2),color='black',linewidth=3)
    ax2.plot([(x2[0]-1),(x2[-1])+1],[1,1],'--', alpha=0.6,color='red')
    ax2.set_xscale("log", nonpositive='clip')
    ax2.set_yscale("log", nonpositive='clip')
    ax2.set_xlim(xlim_l,xlim_u)
    ax2.set_ylim(10**(-1.),10**(1.))
    plt.ylabel(r'$\chi$',fontsize=15)
    plt.xlabel(axlabel_x1,fontsize=15)
    plt.savefig(filename,dpi=300)

def plotting_res_2(x1,y1,x2,y2,x3,y3,axlabel_y1,axlabel_x1,xlim_l,xlim_u,ylim_l,ylim_u,label1,label2,label3,filename):
    tot_fig,axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]},sharex=True, sharey=False,constrained_layout=True, figsize=(8,6))

    plt.subplots_adjust(wspace=1, hspace=0.1)

    ax = plt.subplot(211)
    ax.set_yscale("log", nonpositive='clip')
    ax.set_xscale("log", nonpositive='clip')
    plt.plot(x1,y1,c='orchid',alpha=1.,label=label1,linewidth=3)
    plt.plot(x2,y2,label=label2,linewidth=3,ls="-.",c="yellowgreen")
    plt.plot(x3,y3,label=label3,linewidth=3,ls="dashed",c="grey")
    ax.set_ylabel(axlabel_y1,fontsize=15)
    ax.tick_params(axis='x', which='major', labelsize=15)
    ax.tick_params(axis='y', which='major', labelsize=15) 
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
    ax.set_xlim(xlim_l,xlim_u)
    ax.set_ylim(ylim_l,ylim_u)
    plt.legend(fontsize=15)

    ax2 = plt.subplot(212)
    ax2.tick_params(axis='x', which='major', labelsize=15)
    ax2.tick_params(axis='y', which='major', labelsize=15) 
    ax2.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
    ax2.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
    ax2.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
    ax2.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
    ax2.plot(x2,np.divide(np.interp(x2,x1,y1),y2),color='yellowgreen',ls="-.",linewidth=3)
    ax2.plot(x3,np.divide(np.interp(x3,x1,y1),y3),color='grey',ls="dashed",linewidth=3)

    ax2.plot([(x2[0]-1),(x2[-1])+1],[1,1],'--', alpha=0.6,color='red')
    ax2.set_xscale("log", nonpositive='clip')
    ax2.set_yscale("log", nonpositive='clip')
    ax2.set_xlim(xlim_l,xlim_u)
    ax2.set_ylim(10**(-1.),10**(1.))
    plt.ylabel(r'$\chi$',fontsize=15)
    plt.xlabel(axlabel_x1,fontsize=15)
    
    plt.savefig(filename,dpi=300)
    
def plotting_res_3(x1,y1,x2,y2,x3,y3,x4,y4,axlabel_y1,axlabel_x1,xlim_l,xlim_u,ylim_l,ylim_u,label1,label2,label3,label4,filename):
    tot_fig,axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]},sharex=True, sharey=False,constrained_layout=True, figsize=(8,6))

    plt.subplots_adjust(wspace=1, hspace=0.1)

    ax = plt.subplot(211)
    ax.set_yscale("log", nonpositive='clip')
    ax.set_xscale("log", nonpositive='clip')
    plt.plot(x1,y1,c='k',alpha=1.,label=label1,linewidth=3,ls="dashed")
    plt.plot(x2,y2,label=label2,lw=3,ls="solid",alpha=0.5,c="b")
    plt.plot(x3,y3,label=label3,lw=3,ls="solid",alpha=0.5,c="r")
    plt.plot(x4,y4,label=label4,lw=3,ls="solid",alpha=0.5,c="g")
    ax.set_ylabel(axlabel_y1,fontsize=15)
    ax.tick_params(axis='x', which='major', labelsize=15)
    ax.tick_params(axis='y', which='major', labelsize=15) 
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
    ax.set_xlim(xlim_l,xlim_u)
    ax.set_ylim(ylim_l,ylim_u)
    plt.legend(fontsize=15)

    ax2 = plt.subplot(212)
    ax2.tick_params(axis='x', which='major', labelsize=15)
    ax2.tick_params(axis='y', which='major', labelsize=15) 
    ax2.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
    ax2.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
    ax2.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
    ax2.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
    ax2.plot(x2,np.divide(np.interp(x2,x1,y1),y2),color='b',lw=3,ls="solid",alpha=0.5)
    ax2.plot(x3,np.divide(np.interp(x3,x1,y1),y3),color='r',lw=3,ls="solid",alpha=0.5)
    ax2.plot(x4,np.divide(np.interp(x4,x1,y1),y4),color='g',lw=3,ls="solid",alpha=0.5)

    ax2.plot([(x2[0]-1),(x2[-1])+1],[1,1],'--', alpha=0.6,color='k')
    ax2.set_xscale("log", nonpositive='clip')
    ax2.set_yscale("log", nonpositive='clip')
    ax2.set_xlim(xlim_l,xlim_u)
    ax2.set_ylim(10**(-1.),10**(1.))
    plt.ylabel(r'$\chi$',fontsize=15)
    plt.xlabel(axlabel_x1,fontsize=15)
    
    plt.savefig(filename,dpi=300)
    
def LeMoC_noBB_no_exp(params):
    
    time_init = day_counter = 0.
    time_end = 10.
    time_real = time_init
    step_alg = 1.
    
    R0 = 10**params[0]
    B0 = 10**params[1]
    g_min_el = params[2]
    g_max_el = params[3]
    comp_el = params[4]
    p_el = params[5]

    Syn_l_flag = 1.
    Syn_emis_flag = 1.
    IC_l_flag = 0.
    IC_emis_flag = 1.
    SSA_l_flag = 1.
    
    #gamma-nu space
    grid_size = params[7]
    g_el = np.logspace(0.,g_max_el,int(grid_size))
    g_el_mp = np.array([(g_el[im+1]+g_el[im-1])/2. for im in range(0,len(g_el)-1)])
    dg_el = np.array([((g_el[im+1])-(g_el[im-1]))/2. for im in range(1,len(g_el)-1)])
    dg_l_el = np.log(g_el[1])-np.log(g_el[0])
    nu_syn = np.logspace(7.5,np.log10(7.*nu_c(g_el[-1],B0))+1.4,100)
    nu_ic = np.logspace(14.,32.,100)
    nu_tot = np.logspace(np.log10(nu_syn[0]),np.log10(nu_ic[-1]),200)
    index_PL_max = -1
    index_PL_min = max(max(np.where(g_el < 10**g_min_el)))
    #Initialize values for particles and photons
    N_el = np.zeros(len(g_el))

    photons_syn = np.ones(len(nu_syn))*10**(-260.)
    photons_IC = np.ones(len(nu_ic))*10**(-260.)
    
    el_inj = np.zeros(len(g_el))
    el_inj[index_PL_min:index_PL_max] = Q_el_Lum(Lum_e_inj(10**comp_el,R0),p_el,10**g_min_el,10**g_max_el)*g_el[index_PL_min:index_PL_max]**(-p_el)
    nu_syn = np.append(nu_syn,nu_tot[-1])
    photons_syn = np.append(photons_syn,10**(-200.
                                            ))
    dg_l_el = np.log10(g_el[1])-np.log10(g_el[0])

    N_el[0] = N_el[-1] = 10**(-160.)
    g2N = []
    nu_L_nu = []
    #Constants
    C_syn_el = sigmaT*c/(h*24.*np.pi**2.*0.8975)*(4.*np.pi*m_el*c/(3.*q))**(4./3.)/cor_factor_syn_el(g_el,R0,10**4.,p_el,Lum_e_inj(10**comp_el,R0))
    const_el = 4./3.*sigmaT/(8.*np.pi*m_el*c)
    while time_real <  time_end*R0/c:
        dt = 1.*R0/c
        time_real += dt    
        Radius = R0
        M_F = B0
        a_cr_el = 3.*q*M_F/(4.*np.pi*m_el*c)
        
        # Calculate total dN/dVdν
        photons = photons_tot_test(nu_syn,photons_syn,nu_ic,photons_IC,nu_tot)/Volume(Radius)

        if Syn_l_flag == 1.:
            b_syn_el = const_el*M_F**2.
            dgdt_Syn_el_m = b_syn_el*np.divide(np.power(g_el_mp[0:-1],2.),dg_el)
            dgdt_Syn_el_p = b_syn_el*np.divide(np.power(g_el_mp[1:],2.),dg_el)
        else :
            dgdt_Syn_el_m = np.zeros(len(g_el)-2)      
            dgdt_Syn_el_p = np.zeros(len(g_el)-2)  

        if IC_l_flag == 1.:
            U_ph = U_ph_f(g_el,nu_tot,photons,Radius)
            b_Com_el = 4./3.*sigmaT*np.multiply(c,U_ph)/(m_el*c**2.)
            dgdt_IC_el_m = b_Com_el[1:-1]*np.divide(np.power(g_el_mp[0:-1],2.),dg_el)
            dgdt_IC_el_p = b_Com_el[2:]*np.divide(np.power(g_el_mp[1:],2.),dg_el)
        else:
            dgdt_IC_el_m = np.zeros(len(g_el)-2)    
            dgdt_IC_el_p = np.zeros(len(g_el)-2) 

        V1 = np.zeros(len(g_el)-2)
        V2 = 1.+dt*(c/Radius+dgdt_Syn_el_m+dgdt_IC_el_m)
        V3 = -dt*(dgdt_Syn_el_p+dgdt_IC_el_p) 
        S_ij = N_el[1:-1]+np.multiply(el_inj[1:-1],dt)
        N_el[1:-1] = thomas(V1, V2, V3, S_ij)    

        if Syn_emis_flag == 1.:
            Q_Syn_el = [Q_syn_space(N_el/Volume(Radius),M_F,nu_syn[nu_ind],a_cr_el,C_syn_el,g_el) for nu_ind in range(len(nu_syn)-1)] 
        else: 
            Q_Syn_el = np.zeros(len(nu_syn)-1)

        if IC_emis_flag == 1.:
            Q_IC = [Q_IC_space(N_el/Volume(Radius),g_el,nu_ic[nu_ind],photons,nu_tot,len(nu_tot)-1) for nu_ind in range(0,len(nu_ic)-1)]
        else:
            Q_IC = np.zeros(len(nu_ic)-1)  

        if SSA_l_flag == 1.:
            aSSA_space = [-np.absolute(aSSA(N_el/Volume(Radius),M_F,nu_syn[nu_ind],g_el,dg_l_el)) for nu_ind in range(0,len(nu_syn-1))]
        else:
            aSSA_space = np.zeros(len(nu_syn-1))  

        V1 = np.zeros(len(nu_syn)-2)
        V2 = 1.+dt*(c/Radius-np.multiply(aSSA_space[1:-1],1)*c)
        V3 = np.zeros(len(nu_syn)-2)
        S_ij = photons_syn[1:-1]+4.*np.pi*np.multiply(Q_Syn_el,dt)[1:]*Volume(Radius)
        photons_syn[1:-1] = thomas(V1, V2, V3, S_ij)  

        V1 = np.zeros(len(nu_ic)-2)
        V2 = 1.+dt*(c/Radius+np.zeros(len(nu_ic)-2))
        V3 = np.zeros(len(nu_ic)-2)
        S_ij = photons_IC[1:-1]+np.multiply(Q_IC,dt)[1:]*Volume(Radius)
        photons_IC[1:-1] = thomas(V1, V2, V3, S_ij )
        
        if day_counter<time_real:
            photons = photons_tot_test(nu_syn,photons_syn,nu_ic,photons_IC,nu_tot)/Volume(Radius)
            Spec_temp_tot = np.multiply(photons,h*nu_tot**2.)*4.*np.pi/3.*Radius**2.*c  
            day_counter=day_counter+step_alg*R0/c
            g2N.append((g_el**2.*N_el).copy())
            nu_L_nu.append(Spec_temp_tot.copy())
    #returns in the comoving frame
    return (nu_tot,nu_L_nu,g_el,g2N,cor_factor_syn_el(g_el,R0,10**4.,p_el,Lum_e_inj(10**comp_el,R0)))
        
def LeMoC_syn_test(params):
    start_time = timeit.default_timer()
    time_init = 0. #in R0/c
    time_end = 10. #in R0/c
    time_real = time_init
    step_alg = 2.
    day_counter = 0.
    
    R0 = 10**params[0]
    B0 = 10**params[1]
    g_min_el = params[2]
    g_max_el = params[3]
    comp_el = params[4]
    p_el = params[5]
    dt = step_alg*R0/c

    #gamma-nu space
    grid_size = int(params[7])
    g_el = np.logspace(g_min_el,g_max_el,grid_size)
    g_el_mp = np.array([(g_el[im+1]+g_el[im-1])/2. for im in range(0,len(g_el)-1)])
    dg_el = np.array([((g_el[im+1])-(g_el[im-1]))/2. for im in range(1,len(g_el)-1)])
    nu_syn = np.logspace(7.5,np.log10(7.*nu_c(g_el[-1],B0))+1.4,int(grid_size/2))
    
    #Initialize values for particles and photons
    N_el = np.zeros(len(g_el))
    N_el[0] = N_el[-1] = 10**(-160.)
    el_inj = Q_el_Lum(Lum_e_inj(10**comp_el,R0),p_el,10**g_min_el,10**g_max_el)*g_el**(-p_el)

    photons_syn = np.ones(len(nu_syn))*10**(-260.)

    El_lum = []
    Ph_lum = []
    #Constants
    C_syn_el = sigmaT*c/(h*24.*np.pi**2.*0.8975)*(4.*np.pi*m_el*c/(3.*q))**(4./3.)
    const_el = 4./3.*sigmaT/(8.*np.pi*m_el*c)
    
    while time_real < time_end*R0/c:
        time_real += dt    
        Radius = R0
        M_F = B0
        a_cr_el = 3.*q*M_F/(4.*np.pi*m_el*c)
        
        b_syn_el = const_el*M_F**2.
        dgdt_Syn_el_m = b_syn_el*np.divide(np.power(g_el_mp[0:-1],2.),dg_el)
        dgdt_Syn_el_p = b_syn_el*np.divide(np.power(g_el_mp[1:],2.),dg_el)

        V1 = np.zeros(len(g_el)-2)
        V2 = 1.+dt*(c/R0+dgdt_Syn_el_m)
        V3 = -dt*(dgdt_Syn_el_p) 
        S_ij = N_el[1:-1]+np.multiply(el_inj.copy()[1:-1],dt)
        N_el[1:-1] = thomas(V1, V2, V3, S_ij)    

        Q_Syn_el = [Q_syn_space(N_el/Volume(Radius),M_F,nu_syn[nu_ind],a_cr_el,C_syn_el,g_el) for nu_ind in range(len(nu_syn)-1)] 
        
        V1 = np.zeros(len(nu_syn)-2)
        V2 = 1.+dt*(c/R0+np.zeros(len(nu_syn)-2))
        V3 = np.zeros(len(nu_syn)-2)
        S_ij = photons_syn[1:-1]+4.*np.pi*np.multiply(Q_Syn_el,dt)[1:]*Volume(Radius)
        photons_syn[1:-1] = thomas(V1, V2, V3, S_ij)  
        if day_counter < time_real:
            day_counter=day_counter+step_alg*R0/c
            Syn_temp_plot = np.multiply(photons_syn/Volume(Radius),h*nu_syn**2.)*4.*np.pi/3.*Radius**2.*c
            El_lum.append(np.trapz(el_inj*g_el**2.*m_el*c**2.,np.log(g_el)))
            Ph_lum.append(np.trapz(Syn_temp_plot,np.log(nu_syn)))
    elapsed = timeit.default_timer() - start_time
    return (El_lum[-1],Ph_lum[-1],elapsed)


def LeMoC_IC_test(params):
    start_time = timeit.default_timer()
    #Constants
    time_init = 0.
    time_end = 10.
    step_alg = 2.

    R0 = 10**params[0]
    g_min_el = params[2]
    g_max_el = params[3]
    comp_el = params[4]
    p_el = params[5]
    dt = step_alg*R0/c

    #gamma_nu space
    grid_size = int(params[7])
    g_el = np.logspace(g_min_el,g_max_el,grid_size)
    g_el_mp = np.array([(g_el[im+1]+g_el[im-1])/2. for im in range(0,len(g_el)-1)])
    dg_el = np.array([((g_el[im+1])-(g_el[im-1]))/2. for im in range(1,len(g_el)-1)])

    nu_ic = np.logspace(12.,27.,int(grid_size/2.))
    temperature = 10**5.
    bb = BlackBody(temperature*u.K)
    nu_bb = np.array(np.logspace(np.log10(5.879*10**10*temperature)-6., np.log10(5.879*10**10*temperature)+1.,100)*u.Hz)
    photons_bb = np.array(4.*np.pi/c*bb(nu_bb)/(h*nu_bb))
    photons_bb = np.append(photons_bb,10**(-260.))
    nu_bb = np.append(nu_bb,10**(35.))
    day_counter=0.

    N_el = np.zeros(len(g_el))
    el_inj = np.zeros(len(g_el))
    photons_IC = np.ones(len(nu_ic))*10**(-260.)
    N_el[0] = N_el[-1] = 10**(-160.)
        
    El_lum = []
    Ph_lum = []
    t = []
    time_real = time_init
    el_inj = Q_el_Lum(Lum_e_inj(10**comp_el,R0),p_el,10**g_min_el,10**g_max_el)*g_el**(-p_el)

    while time_real <  time_end*R0/c:
        time_real += dt    
        t.append(time_real)
        U_ph = U_ph_f(g_el,nu_bb,photons_bb,R0)
        b_Com_el = 4./3.*sigmaT*np.multiply(c,U_ph)/(m_el*c**2.)
        dgdt_IC_el_m = b_Com_el[1:-1]*np.divide(np.power(g_el_mp[0:-1],2.),dg_el)
        dgdt_IC_el_p = b_Com_el[2:]*np.divide(np.power(g_el_mp[1:],2.),dg_el)

        V1 = np.zeros(len(g_el)-2)
        V2 = 1.+dt*(c/R0+dgdt_IC_el_m)
        V3 = -dt*(dgdt_IC_el_p) 
        S_ij = N_el[1:-1]+np.multiply(el_inj[1:-1],dt)
        N_el[1:-1] = thomas(V1, V2, V3, S_ij)    
        Q_IC = [Q_IC_space(N_el/Volume(R0),g_el,nu_ic[nu_ind],np.array(photons_bb),nu_bb,len(nu_bb)-1) for nu_ind in range(0,len(nu_ic)-1)]

        V1 = np.zeros(len(nu_ic)-2)
        V2 = 1.+dt*(c/R0+np.zeros(len(nu_ic)-2))
        V3 = np.zeros(len(nu_ic)-2)
        S_ij = photons_IC[1:-1]+np.multiply(Q_IC,dt)[1:]*Volume(R0)
        photons_IC[1:-1] = thomas(V1, V2, V3, S_ij)

        if day_counter < time_real:
            day_counter = day_counter+step_alg*R0/c
            IC_temp_plot = np.multiply(photons_IC/Volume(R0),h*nu_ic**2.)*4.*np.pi/3.*R0**2.*c
            El_lum.append(np.trapz(el_inj*g_el**2.*m_el*c**2.,np.log(g_el)))
            Ph_lum.append(np.trapz(IC_temp_plot,np.log(nu_ic)))
    elapsed = timeit.default_timer() - start_time
    return (El_lum[-1],Ph_lum[-1],elapsed)

def LeMoC_SSC_test(params):
    start_time = timeit.default_timer()
    time_init = 0. #in R0/c
    time_end = 10. #in R0/c
    time_real = time_init
    step_alg = 2.
    day_counter = 0.
    
    R0 = 10**params[0]
    B0 = 10**params[1]
    g_min_el = params[2]
    g_max_el = params[3]
    comp_el = params[4]
    p_el = params[5]
    #gamma-nu space
    grid_size = int(params[7])
    g_el = np.logspace(g_min_el,g_max_el,grid_size)
    g_el_mp = np.array([(g_el[im+1]+g_el[im-1])/2. for im in range(0,len(g_el)-1)])
    dg_el = np.array([((g_el[im+1])-(g_el[im-1]))/2. for im in range(1,len(g_el)-1)])

    nu_syn = np.logspace(7.5,np.log10(7.*nu_c(g_el[-1],B0))+1.4,int(grid_size/2))
    nu_ic = np.logspace(12.,32.,int(grid_size/2.))
    nu_tot = np.logspace(np.log10(nu_syn[0]),np.log10(nu_ic[-1]),200)
    
    #Initialize values for particles and photons
    N_el = np.zeros(len(g_el))
    N_el[0] = N_el[-1] = 10**(-160.)
    el_inj = Q_el_Lum(Lum_e_inj(10**comp_el,R0),p_el,10**g_min_el,10**g_max_el)*g_el**(-p_el)
    photons_IC = np.ones(len(nu_ic))*10**(-260.)
    photons_syn = np.ones(len(nu_syn))*10**(-260.)

    El_lum = []
    Ph_lum = []
    #Constants
    C_syn_el = sigmaT*c/(h*24.*np.pi**2.*0.8975)*(4.*np.pi*m_el*c/(3.*q))**(4./3.)
    const_el = 4./3.*sigmaT/(8.*np.pi*m_el*c)
    
    while time_real < time_end*R0/c:
        dt = R0/c
        time_real += dt    
        Radius = R0
        M_F = B0
        a_cr_el = 3.*q*M_F/(4.*np.pi*m_el*c)
        photons = photons_tot_test(nu_syn,photons_syn,nu_ic,photons_IC,nu_tot)/Volume(Radius)
        b_syn_el = const_el*M_F**2.
        dgdt_Syn_el_m = b_syn_el*np.divide(np.power(g_el_mp[0:-1],2.),dg_el)
        dgdt_Syn_el_p = b_syn_el*np.divide(np.power(g_el_mp[1:],2.),dg_el)

        U_ph = U_ph_f(g_el,nu_tot,photons,R0)
        b_Com_el = 4./3.*sigmaT*np.multiply(c,U_ph)/(m_el*c**2.)
        dgdt_IC_el_m = b_Com_el[1:-1]*np.divide(np.power(g_el_mp[0:-1],2.),dg_el)
        dgdt_IC_el_p = b_Com_el[2:]*np.divide(np.power(g_el_mp[1:],2.),dg_el)

        V1 = np.zeros(len(g_el)-2)
        V2 = 1.+dt*(c/R0+dgdt_IC_el_m+dgdt_Syn_el_m)
        V3 = -dt*(dgdt_IC_el_p+dgdt_Syn_el_p) 
        S_ij = N_el[1:-1]+np.multiply(el_inj[1:-1],dt)
        N_el[1:-1] = thomas(V1, V2, V3, S_ij)       

        Q_Syn_el = [Q_syn_space(N_el/Volume(Radius),M_F,nu_syn[nu_ind],a_cr_el,C_syn_el,g_el) for nu_ind in range(len(nu_syn)-1)] 
        Q_IC = [Q_IC_space(N_el/Volume(R0),g_el,nu_ic[nu_ind],np.array(photons),nu_tot,len(nu_tot)-1) for nu_ind in range(0,len(nu_ic)-1)]
        
        V1 = np.zeros(len(nu_syn)-2)
        V2 = 1.+dt*(c/R0+np.zeros(len(nu_syn)-2))
        V3 = np.zeros(len(nu_syn)-2)
        S_ij = photons_syn[1:-1]+4.*np.pi*np.multiply(Q_Syn_el,dt)[1:]*Volume(Radius)
        photons_syn[1:-1] = thomas(V1, V2, V3, S_ij)  

        V1 = np.zeros(len(nu_ic)-2)
        V2 = 1.+dt*(c/R0+np.zeros(len(nu_ic)-2))
        V3 = np.zeros(len(nu_ic)-2)
        S_ij = photons_IC[1:-1]+np.multiply(Q_IC,dt)[1:]*Volume(R0)
        photons_IC[1:-1] = thomas(V1, V2, V3, S_ij)
        
        if day_counter < time_real:
            day_counter=day_counter+step_alg*R0/c
            photons = photons_tot_test(nu_syn,photons_syn,nu_ic,photons_IC,nu_tot)/Volume(Radius)
            Spec_temp_tot = np.multiply(photons,h*nu_tot**2.)*4.*np.pi/3.*Radius**2.*c             
            El_lum.append(np.trapz(el_inj*g_el**2.*m_el*c**2.,np.log(g_el)))
            Ph_lum.append(np.trapz(Spec_temp_tot,np.log(nu_tot)))
    elapsed = timeit.default_timer() - start_time
    return (El_lum[-1],Ph_lum[-1],elapsed)

def gamma_b(M_F,Radius,U_ph):
    Ub = M_F**2./(8.*np.pi)
    return 3*m_el*c**2./(4.*(Ub+U_ph)*sigmaT*Radius)

def N_el_theory(g_el_space,gamma,M_F,Radius,Q_el,U_ph):
    gamma_space = np.logspace(np.log10(gamma),np.log10(g_el_space[-1]),100000)
    Q_el_interp = 10**np.interp(np.log10(gamma_space),np.log10(g_el_space),np.log10(Q_el))
    return np.exp(-gamma_b(M_F,Radius,U_ph)/gamma)*gamma_b(M_F,Radius,U_ph)*Radius/(c*gamma**2.)*np.trapz(Q_el_interp*gamma_space*np.exp(gamma_b(M_F,Radius,U_ph)/gamma_space),np.log(gamma_space))