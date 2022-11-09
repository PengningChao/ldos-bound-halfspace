"""
Compute LDOS bounds for a 2D halfspace design region
given in-plane electric field polarization (TE)
with a dipolar source perpendicular to the surface of the half space
uses the arbitrary precision package mpmath at certain points where
high precision is necessary / useful for testing
"""

import numpy as np
import scipy.linalg as la
from scipy.integrate import quad as np_quad
from scipy.optimize import minimize_scalar

import matplotlib.pyplot as plt

import sympy as sym
from sympy.parsing.sympy_parser import parse_expr
import mpmath
from mpmath import mp



"""
at startup, load expressions for evaluation (AsymUP)_kx^-1 dot S1
expressions are computed using Mathematica; see auxiliary files
"""
kx, kyr, kyi = sym.symbols('kx kyr kyi', real=True)
ky = kyr + sym.I*kyi
kyc = kyr - sym.I*kyi

Pr, Pi = sym.symbols('Pr Pi', real=True)
P = Pr + sym.I*Pi
Pc = Pr - sym.I*Pi

mf = sym.symbols('mf', real=True) #material factor, Im(P/chi^*)

Lx, Ly, Fxm, Fxc, Fym, Fyc = sym.symbols('Lx Ly Fxm Fxc Fym Fyc')

s = sym.symbols('s') #the Laplace transform variable

symbol_dict = {'kx':kx, 'kyr':kyr, 'kyi':kyi, 'pr':Pr, 'pi':Pi, 'mf':mf,
               'Lx':Lx, 'Ly':Ly, 'Fxm':Fxm, 'Fxc':Fxc, 'Fym':Fym, 'Fyc':Fyc, 
               's':s, 'I':sym.I}


f = open('../auxiliary/TEy_halfspace_denom.txt', 'r')
str_denom = f.readline()
f.close()

f = open('../auxiliary/TEy_halfspace_Fxs_num.txt', 'r')
str_Fxs_num = f.readline()
f.close()

f = open('../auxiliary/TEy_halfspace_Fys_num.txt', 'r')
str_Fys_num = f.readline()
f.close()

s_denom = parse_expr(str_denom, local_dict=symbol_dict)
Fxs_num = parse_expr(str_Fxs_num, local_dict=symbol_dict)
Fys_num = parse_expr(str_Fys_num, local_dict=symbol_dict)

math_module = ["mpmath"]

denom_s4_coeff_func = sym.lambdify([kx, kyr, kyi, Pr, Pi, mf], s_denom.coeff(s, n=4), modules=math_module)
denom_s2_coeff_func = sym.lambdify([kx, kyr, kyi, Pr, Pi, mf], s_denom.coeff(s, n=2), modules=math_module)
denom_s0_coeff_func = sym.lambdify([kx, kyr, kyi, Pr, Pi, mf], s_denom.coeff(s, n=0), modules=math_module)

Fxs_num_func = sym.lambdify([kx, kyr, kyi, Pr, Pi, mf, Fxm, Fxc, Fym, Fyc, s], Fxs_num, modules=math_module)
Fys_num_func = sym.lambdify([kx, kyr, kyi, Pr, Pi, mf, Fxm, Fxc, Fym, Fyc, s], Fys_num, modules=math_module)

param_list = [Fxm, Fxc, Fym, Fyc]
Fxs_num_param_coeff_func_list = []
Fys_num_param_coeff_func_list = []
Fxs_num_const_expr = Fxs_num.copy()
Fys_num_const_expr = Fys_num.copy()

for param in param_list:
    Fxs_num_param_coeff_func_list.append(sym.lambdify([kx, kyr, kyi, Pr, Pi, mf, s], Fxs_num.coeff(param, n=1), modules=math_module))
    Fys_num_param_coeff_func_list.append(sym.lambdify([kx, kyr, kyi, Pr, Pi, mf, s], Fys_num.coeff(param, n=1), modules=math_module))
    Fxs_num_const_expr -= param * Fxs_num.coeff(param, n=1)
    Fys_num_const_expr -= param * Fys_num.coeff(param, n=1)

#append the parts of the numerator not associated with any of the free parameters Fxm, Fxc, Fym, Fyc
Fxs_num_param_coeff_func_list.append(sym.lambdify([kx, kyr, kyi, Pr, Pi, mf, s], Fxs_num_const_expr, modules=math_module))
Fys_num_param_coeff_func_list.append(sym.lambdify([kx, kyr, kyi, Pr, Pi, mf, s], Fys_num_const_expr, modules=math_module))



def get_mp_TEy_AsymUPinv_expiky_y(chi, phase, k0, kx):
    ky = mp.sqrt(k0**2-kx**2)
    kyr = mp.re(ky); kyi = mp.im(ky)
    
    #print('kyr', kyr, 'kyi', kyi)
    Pr = mp.re(phase); Pi = mp.im(phase)
    mf = mp.im(phase/np.conj(chi))
    #print('material factor', mf)
    
    #find poles of Laplace transform
    quad_a = denom_s4_coeff_func(kx, kyr, kyi, Pr, Pi, mf)
    quad_b = denom_s2_coeff_func(kx, kyr, kyi, Pr, Pi, mf)
    quad_c = denom_s0_coeff_func(kx, kyr, kyi, Pr, Pi, mf)
    
    Delta = quad_b**2 - 4*quad_a*quad_c
    if mp.im(Delta)==0 and mp.re(Delta)<0:
        sqrtDelta = 1j*mp.sqrt(-Delta)
    else:
        sqrtDelta = mp.sqrt(Delta)
    
    sqr_plus = (-quad_b + sqrtDelta) / 2 / quad_a
    sqr_minus = (-quad_b - sqrtDelta) / 2 / quad_a
    
    pole_plus = mp.sqrt(sqr_plus)
    pole_minus = mp.sqrt(sqr_minus)
    
    #print('pole_plus', pole_plus)
    #print('pole_minus', pole_minus)
    if mp.fabs(mp.re(pole_plus))<1e3*mp.eps or mp.fabs(mp.re(pole_minus))<1e3*mp.eps:
        raise ValueError('nan encountered in poles, L2 invertibility lost')

    pole_list = [pole_plus, -pole_plus, pole_minus, -pole_minus]
    #now go through the poles to set up the linear system for solving for the free params
    #free params are set such that exponentially growing poles as y->\infty have residue 0

    negRe_pole_flag = np.zeros(len(pole_list), dtype=np.bool) #flag[i]=True when the pole[i] has negative Re part and is actually involved in the L2 inverse
    posRe_pole_count = 0
    
    param_mat = mp.matrix(4)
    param_b = mp.matrix(4,1)
    for pole_ind, pole in enumerate(pole_list):
        if mp.re(pole)<0:
            negRe_pole_flag[pole_ind] = True
        else:
            if posRe_pole_count == 2:
                raise ValueError('number of positive poles larger than 2, insufficient # of free parameters')
            
            for i in range(4):
                param_mat[2*posRe_pole_count,i] = Fxs_num_param_coeff_func_list[i](kx, kyr, kyi, Pr, Pi, mf, pole)
                param_mat[2*posRe_pole_count+1,i] = Fys_num_param_coeff_func_list[i](kx, kyr, kyi, Pr, Pi, mf, pole)

            param_b[2*posRe_pole_count] = -Fxs_num_param_coeff_func_list[-1](kx, kyr, kyi, Pr, Pi, mf, pole)
            param_b[2*posRe_pole_count+1] = -Fys_num_param_coeff_func_list[-1](kx, kyr, kyi, Pr, Pi, mf, pole)
            
            posRe_pole_count += 1
    
    if posRe_pole_count<2:
        raise ValueError('number of positive poles less than 2, too much freedom from free params')

    # solve for parameter values that will cancel out positive Re pole residues
    try:
        L2_param_vec = mp.lu_solve(param_mat, param_b)
    except ZeroDivisionError:
        """
        upon numerical testing using arbitrary precision arithmetic,
        it was discovered that param_mat is a 4x4 matrix with a rank of only 2
        resulting in an infinite number of possible L2_param_vec's
        however further testing indicated that all possible L2_param_vecs
        ultimately lead to the same Laplace transform of the unique inverse image
        so when mpmath has a problem with a direct lu_solve, use pseudoinverse instead
        """
        fp_param_mat = np.array(mpmath.fp.matrix(param_mat).tolist())
        fp_param_b = np.array(mpmath.fp.matrix(param_b).tolist()).flatten()
        L2_param_vec = la.pinv(fp_param_mat) @ fp_param_b
        
    Fxm = L2_param_vec[0]; Fxc = L2_param_vec[1]; Fym = L2_param_vec[2]; Fyc = L2_param_vec[3]

    # calculate all the residues of the negative Re poles
    negRe_pole_list = []
    Fxs_res_list = []
    Fys_res_list = []
    
    for pole_ind, pole in enumerate(pole_list):
        if not negRe_pole_flag[pole_ind]:
            continue
        
        negRe_pole_list.append(pole)
        
        res_denom = quad_a
        for i in range(len(pole_list)):
            if i==pole_ind:
                continue
            res_denom *= (pole-pole_list[i])
        
        Fxs_res_list.append(Fxs_num_func(kx, kyr, kyi, Pr, Pi, mf, Fxm, Fxc, Fym, Fyc, pole) / res_denom)
        Fys_res_list.append(Fys_num_func(kx, kyr, kyi, Pr, Pi, mf, Fxm, Fxc, Fym, Fyc, pole) / res_denom)
    
    return negRe_pole_list, Fxs_res_list, Fys_res_list



def mp_TEy_bound_integrand(d, chi, phase, k0, kx, toDouble=False):
    """
    calculates the kx integrand 
    for the finite bandwidth TM LDOS bound near a half space
    """
    
    PA = mp.sqrt(mp.re(phase*mp.conj(phase)))
    ky = mp.sqrt(k0**2-kx**2)
    kyi = mp.im(ky)
    
    r_list, Rx_list, Ry_list = get_mp_TEy_AsymUPinv_expiky_y(chi, phase, k0, kx)
    
    S2AsymUPinvS1 = mp.mpc(0)
    S1AsymUPinvS1 = mp.mpc(0)
    for i in range(len(r_list)):
        r = r_list[i]
        Rx = Rx_list[i]
        Ry = Ry_list[i]
        S2AsymUPinvS1 += -(kx*Rx + (kx**2/ky)*Ry) / (r + 1j*ky)
        S1AsymUPinvS1 += (kx*Rx - (kx**2/mp.conj(ky))*Ry) / (r - 1j*mp.conj(ky))
    
    S2AsymUPinvS1 = -(mp.conj(phase)/k0)*mp.exp(2j*ky*d)*S2AsymUPinvS1
    S1AsymUPinvS1 = (PA/mp.fabs(k0)) * mp.exp(-2*kyi*d) * S1AsymUPinvS1

    integrand = mp.re(S2AsymUPinvS1 + S1AsymUPinvS1)

    if toDouble:
        return np.double(integrand)
    else:
        return integrand



def TEy_halfspace_fixed_phase_bound(d, chi, phase, k0, tol=1e-4):
    """
    evaluate the dual bound given a specified constraint phase rotation
    as an integral over kx
    """
    k0r = np.real(k0); k0i = np.imag(k0)
    integrand = lambda kx: mp_TEy_bound_integrand(d, chi, phase, k0, kx, toDouble=True)
    
    end_kx = k0r
    integral, err = np_quad(integrand, 0, end_kx, epsrel=tol)

    delta_kx = 10*k0r
    while True: #keep on integrating the evanescent tail until desired accuracy is reached
        delta_integral, err = np_quad(integrand, end_kx, end_kx+delta_kx, epsrel=tol)
        integral += delta_integral
        if mp.fabs(delta_integral / integral) < tol:
            break
        end_kx += delta_kx
        
    #extra factor of 2 for the symmetric integral kx from 0 to -infinity
    integral *= 2
    
    wvlgth0 = 2*np.pi / k0r
    return (wvlgth0/8/np.pi**2) * integral #extra factor so the end result is in units of the single frequency vacuum LDOS



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



def TEy_halfspace_bound(d, chi, k0):
    """
    calculate the tightest possible dual bound for a TE y-polarized dipole near a half-space design domain
    given the complex global energy conservation constraints
    """
    theta_boundfunc = lambda angle: TEy_halfspace_fixed_phase_bound(d, chi, np.exp(1j*angle), k0)
    
    #phase angle -pi < theta < pi; find upper and lower limits on phase angle theta
    delta_theta = np.imag(k0) / np.real(k0) / 2
    theta_r = 1.3 * delta_theta
    while True:
        try:
            theta_boundfunc(theta_r)
            break
        except:
            theta_r /= 2.0
    
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
    print('theta_l', theta_l, 'theta_opt', theta_opt, 'theta_r', theta_r)
    rhoUP_t_theta_r, rhoUP_l_theta_r = check_AsymUP_full_space_mineigs(np.complex(chi), np.complex(k0), np.exp(1j*theta_r))
    rhoUP_t_theta_l, rhoUP_l_theta_l = check_AsymUP_full_space_mineigs(np.complex(chi), np.complex(k0), np.exp(1j*theta_l))
    rhoUP_t_theta_opt, rhoUP_l_theta_opt = check_AsymUP_full_space_mineigs(np.complex(chi), np.complex(k0), np.exp(1j*theta_opt))
    print('checking the eigenvalues for AsymUP of the whole space')
    print('at theta_r, the mineig for transverse and longitudinal', rhoUP_t_theta_r, rhoUP_l_theta_r)
    print('at theta_l, the mineig for transverse and longitudinal', rhoUP_t_theta_l, rhoUP_l_theta_l)
    print('at theta_opt, the mineig for transverse and longitudinal', rhoUP_t_theta_opt, rhoUP_l_theta_opt)
    """
    return bound, theta_opt

