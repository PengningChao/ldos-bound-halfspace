"""
Compute LDOS bounds for a 2D halfspace design region
given in-plane electric field polarization (TE)
with a dipolar source perpendicular to the surface of the half space
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize_scalar



def positiveRe_Laplace_poles(chi, phase, kyr, kyi, Br, Bi):
    mat_fac = np.imag(phase/np.conj(chi))
    BK = (Br*kyi + Bi*kyr)
    Delta = BK**2 + (8*mat_fac) * (Bi*kyr*kyi**2 - Br*kyr**2*kyi - mat_fac*kyr**2*kyi**2)
    #print('BK', BK, 'Delta', Delta)
    if Delta<0:
        sqrt_Delta = 1j*np.sqrt(-Delta)
    else:
        sqrt_Delta = np.sqrt(Delta)
    sqr_plus = kyi**2 - kyr**2 + (0.5/mat_fac) * (BK + sqrt_Delta)
    sqr_minus =kyi**2 - kyr**2 + (0.5/mat_fac) * (BK - sqrt_Delta)
    
    #print('sqr_plus', sqr_plus, 'sqr_minus', sqr_minus)
    pole_plus = np.sqrt(sqr_plus)
    pole_minus = np.sqrt(sqr_minus)
    
    if np.isnan(pole_plus) or np.isnan(pole_minus):
        raise ValueError('L2 invertibility lost, possible duality violation')
    
    pole_minus *= np.sign(np.real(pole_minus))
    pole_plus *= np.sign(np.real(pole_plus)) #take the Re>0 root
    
    return pole_plus, pole_minus



def Laplace_free_params(kyr, kyi, Br, Bi, pole_plus, pole_minus):
    Amat = np.zeros((2,2), dtype=np.complex)
    Amat[0,0] = (Br+1j*Bi)*0.25 * (pole_plus + (kyi-1j*kyr)) * (pole_plus**2 - (kyi+1j*kyr)**2)
    Amat[0,1] = (Br-1j*Bi)*0.25 * (pole_plus + (kyi+1j*kyr)) * (pole_plus**2 - (kyi-1j*kyr)**2)
    
    Amat[1,0] = (Br+1j*Bi)*0.25 * (pole_minus + (kyi-1j*kyr)) * (pole_minus**2 - (kyi+1j*kyr)**2)
    Amat[1,1] = (Br-1j*Bi)*0.25 * (pole_minus + (kyi+1j*kyr)) * (pole_minus**2 - (kyi-1j*kyr)**2)

    bvec = np.zeros(2, dtype=np.complex)
    bvec[0] = (pole_plus - (kyi-1j*kyr)) * (pole_plus**2 - (kyi+1j*kyr)**2)
    bvec[1] = (pole_minus - (kyi-1j*kyr)) * (pole_minus**2 - (kyi+1j*kyr)**2)

    gamma_list = la.solve(Amat, bvec)
    gamma_plus = gamma_list[1]
    gamma_minus = gamma_list[0]
    
    detA = la.det(Amat)
    
    return gamma_plus, gamma_minus, detA



def Laplace_residue_coeffs(chi, phase, kyr, kyi, Br, Bi, pole_plus, pole_minus, gamma_plus, gamma_minus):
    mat_fac = np.imag(phase / np.conj(chi))
    denom_plus = - mat_fac * 2 * pole_plus * (pole_plus**2 - pole_minus**2)
    num_plus = ( (-pole_plus-(kyi-1j*kyr)) * (pole_plus**2 - (kyi+1j*kyr)**2)
                - 0.25*(Br+1j*Bi) * (-pole_plus+(kyi-1j*kyr))*(pole_plus**2-(kyi+1j*kyr)**2)*gamma_minus
                - 0.25*(Br-1j*Bi) * (-pole_plus+(kyi+1j*kyr))*(pole_plus**2-(kyi-1j*kyr)**2)*gamma_plus
                )
    
    R_plus = num_plus / denom_plus
    
    denom_minus = - mat_fac * 2 * pole_minus * (pole_minus**2 - pole_plus**2)
    num_minus = ( (-pole_minus-(kyi-1j*kyr))*(pole_minus**2-(kyi+1j*kyr)**2) 
                 - 0.25*(Br+1j*Bi) * (-pole_minus+(kyi-1j*kyr))*(pole_minus**2-(kyi+1j*kyr)**2)*gamma_minus
                 - 0.25*(Br-1j*Bi) * (-pole_minus+(kyi+1j*kyr))*(pole_minus**2-(kyi-1j*kyr)**2)*gamma_plus
                )
    
    R_minus = num_minus / denom_minus
    
    return R_plus, R_minus



def bound_integrand_Cauchy_relaxed(d, chi, phase, k0, kx):
    """
    calculates the kx integrand 
    for the finite bandwidth TM LDOS bound near a half space with the
    Cauchy relaxation |<S2| AsymUP^-1 |S1>| <= |<S1| AsymUP^-1 |S1>| applied
    """
    k0r = np.real(k0); k0i = np.imag(k0)
    PA = np.sqrt(np.real(phase*np.conj(phase)))
    
    ky = np.sqrt(k0**2 - kx**2)
    kyr = np.real(ky); kyi = np.imag(ky)

    B = np.conj(phase)*k0**2 / ky
    Br = np.real(B); Bi = np.imag(B)
    
    pole_plus, pole_minus = positiveRe_Laplace_poles(chi, phase, kyr, kyi, Br, Bi)
    
    gamma_plus, gamma_minus, detA = Laplace_free_params(kyr, kyi, Br, Bi, pole_plus, pole_minus)
        
    R_plus, R_minus = Laplace_residue_coeffs(chi, phase, kyr, kyi, Br, Bi, pole_plus, pole_minus, gamma_plus, gamma_minus)
    
    return PA * (k0r**2+k0i**2)**1.5 * np.exp(-2*kyi*d)/(kyr**2+kyi**2) * np.real(R_plus/(pole_plus+(kyi+1j*kyr)) + R_minus/(pole_minus+(kyi+1j*kyr)))



def bound_integrand(d, chi, phase, k0, kx):
    """
    calculates the kx integrand 
    for the finite bandwidth TM LDOS bound near a half space
    """
    
    k0r = np.real(k0); k0i = np.imag(k0)
    PA = np.sqrt(np.real(phase*np.conj(phase)))
    
    ky = np.sqrt(k0**2 - kx**2)
    kyr = np.real(ky); kyi = np.imag(ky)

    B = np.conj(phase)*k0**2 / ky
    Br = np.real(B); Bi = np.imag(B)
    
    pole_plus, pole_minus = positiveRe_Laplace_poles(chi, phase, kyr, kyi, Br, Bi)
    
    gamma_plus, gamma_minus, detA = Laplace_free_params(kyr, kyi, Br, Bi, pole_plus, pole_minus)
        
    R_plus, R_minus = Laplace_residue_coeffs(chi, phase, kyr, kyi, Br, Bi, pole_plus, pole_minus, gamma_plus, gamma_minus)
    
    S2AinvS1_integrand = -0.5*np.real(np.conj(phase) * (k0**3/ky**2) * np.exp(2j*ky*d) * (R_plus/(pole_plus+kyi-1j*kyr) + R_minus/(pole_minus+kyi-1j*kyr))  )
    S1AinvS1_integrand = 0.5*PA * (k0r**2+k0i**2)**1.5/(kyr**2+kyi**2) * np.exp(-2*kyi*d) * np.real( R_plus/(pole_plus+(kyi+1j*kyr)) + R_minus/(pole_minus+(kyi+1j*kyr)) )
    
    return S2AinvS1_integrand + S1AinvS1_integrand



def check_AsymUP_full_space_mineigs(chi, k0, p):
    """
    returns the minimum transverse eigenvalue and the longitudinal eigenvalue for
    AsymUP + Im(p/chi^*) + Asym(p^* G) over the entire space
    for checking purposes
    """
    p_re = np.real(p); p_im = np.imag(p)
    rhoUP_l = np.imag(p/np.conj(chi)) + p_im
    
    #to find min(rho_t), look at the 4 critical points k^2=0, k^2->infty and the two internal stationary points
    #calculate the eigenvalues at the two finite k stationary points
    A_re = np.real(k0**2); A_im = np.real(k0**2)
    A_norm = np.sqrt(A_re**2 + A_im**2)
    p_norm = np.sqrt(p_re**2 + p_im**2)
    usqr_coeff = p_im*A_re - p_re*A_im
    
    if usqr_coeff==0.0:
        #u = k^2 = A_re is only finite k stationary point
        rhoG_t = p_i * (A_norm/A_im)**2
    else:
        u_plus = (p_im*(A_norm**2) + A_im*A_norm*p_norm) / (p_im*A_re - p_re*A_im)
        u_minus = (p_im*(A_norm**2) - A_im*A_norm*p_norm) / (p_im*A_re - p_re*A_im)
        if u_plus>=0:
            rhoG_t_plus = (p_re*A_im*u_plus + p_im*(A_norm**2 - A_re*u_plus)) / ((u_plus-A_re)**2 + A_im**2)
        else:
            rhoG_t_plus = np.inf #this stationary point outside definition for k^2
        if u_minus>=0:
            rhoG_t_minus = (p_re*A_im*u_minus + p_im*(A_norm**2 - A_re*u_minus)) / ((u_minus-A_re)**2 + A_im**2)
        else:
            rhoG_t_minus = np.inf #this stationary point outside definition for k^2
        rhoG_t = min(rhoG_t_plus, rhoG_t_minus)
        
    rhoG_t = min(p_im, 0, rhoG_t) #rhoG_t(k=0) = p_im, rhoG_t(k->\infty)=0
    rhoUP_t = np.imag(p/np.conj(chi)) + rhoG_t
    return rhoUP_t, rhoUP_l



def TM_halfspace_fixed_phase_bound(d, chi, phase, k0, tol=1e-4):
    """
    evaluate the dual bound given a specified constraint phase rotation
    as an integral over kx
    """
    k0r = np.real(k0); k0i = np.imag(k0)
    integrand = lambda kx: bound_integrand(d, chi, phase, k0, kx)
    
    end_kx = k0r
    integral, err = quad(integrand, 0, end_kx, epsrel=tol)

    delta_kx = 10*k0r
    while True: #keep on integrating the evanescent tail until desired accuracy is reached
        delta_integral, err = quad(integrand, end_kx, end_kx+delta_kx, epsrel=tol)
        integral += delta_integral
        if abs(delta_integral / integral) < tol:
            break
        end_kx += delta_kx
        
    #extra factor of 2 for the symmetric integral kx from 0 to -infinity
    integral *= 2
    
    #prefactor
    wvlgth0 = 2*np.pi / k0r
    return (wvlgth0/4/np.pi**2) * integral #extra factor so the end result is in units of the single frequency vacuum LDOS



def TM_halfspace_bound(d, chi, k0):
    """
    calculate the tightest possible dual bound for a TM dipole near a half-space design domain
    given the complex global energy conservation constraints
    for dielectrics chi should have a little loss to avoid division by 0 in the numerics
    use lossless dielectrics code for real positive chi
    """
    theta_boundfunc = lambda angle: TM_halfspace_fixed_phase_bound(d, chi, np.exp(1j*angle), k0)
    
    #phase angle -pi < theta < pi; find upper and lower limits on phase angle theta
    theta_r = 0.0
    delta_theta = np.imag(k0) / np.real(k0) / 2
    probe_bound = theta_boundfunc(theta_r)
    
    while True: #find upper bound on optimal phase angle theta
        reduced_stepsize = False
        while True:
            if theta_r+delta_theta > np.pi:
                delta_theta = 2*(np.pi-theta_r)/3.0
                reduced_stepsize = True
                continue
            try:
                t = theta_boundfunc(theta_r+delta_theta)
                break
            except ValueError: #inverse of AsymUP acting on S1 is not in L2
                delta_theta /= 2
                reduced_stepsize = True
            
        theta_r += delta_theta
        if not reduced_stepsize:
            delta_theta *= 2
        if t>probe_bound:
            break
        probe_bound = t
    
    theta_l = 2*theta_r/3
    probe_bound = theta_boundfunc(theta_l)
    delta_theta = theta_r / 2.0
    while True: #find lower bound on optimal phase angle theta
        reduced_stepsize = False
        while True:
            if theta_l - delta_theta < -np.pi:
                delta_theta = 2*(theta_l+np.pi)/3.0
                reduced_stepsize = True
                continue
            try:
                t = theta_boundfunc(theta_l-delta_theta)
                break
            except ValueError: #inverse of AsymUP acting on S1 is not in L2
                delta_theta /= 2
                reduced_stepsize = True
        
        theta_l -= delta_theta
        #print('theta_l', theta_l, 'probe_bound', probe_bound, 't', t)
        if not reduced_stepsize:
            delta_theta *= 2
        if t>probe_bound:
            break
        if reduced_stepsize and delta_theta < max(1e-10, abs(theta_l)*1e-4):
            break
        probe_bound = t
        
    opt = minimize_scalar(theta_boundfunc, bounds=(theta_l, theta_r), method='bounded', options={'xatol':min(1e-3, (theta_r-theta_l)/100)})

    theta_opt = opt.x
    bound = opt.fun
    """
    #the below are print-outs of the boundaries of the constraint rotation angle theta determined for checking purposes
    #print('theta_l', theta_l, 'theta_opt', theta_opt, 'theta_r', theta_r)
    rhoUP_t_theta_r, rhoUP_l_theta_r = check_AsymUP_full_space_mineigs(chi, k0, np.exp(1j*theta_r))
    rhoUP_t_theta_l, rhoUP_l_theta_l = check_AsymUP_full_space_mineigs(chi, k0, np.exp(1j*theta_l))
    rhoUP_t_theta_opt, rhoUP_l_theta_opt = check_AsymUP_full_space_mineigs(chi, k0, np.exp(1j*theta_opt))
    #print('checking the eigenvalues for AsymUP of the whole space')
    #print('at theta_r, the mineig for transverse and longitudinal', rhoUP_t_theta_r, rhoUP_l_theta_r)
    #print('at theta_l, the mineig for transverse and longitudinal', rhoUP_t_theta_l, rhoUP_l_theta_l)
    #print('at theta_opt, the mineig for transverse and longitudinal', rhoUP_t_theta_opt, rhoUP_l_theta_opt)
    """
    return bound, theta_opt
