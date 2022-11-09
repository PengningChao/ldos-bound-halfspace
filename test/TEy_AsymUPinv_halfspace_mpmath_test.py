"""
auxiliary function that TE_AsymUPinv_halfspace_mpmath_test
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



def get_modified_TEy_AsymUPinv_expiky_y(chi, phase, k0, kx, use_test_params=False):
    """
    The main purpose of this method is to check numerically what happens
    when we use different TEy Laplace transform parameters but maintaining that
    linear system leading to no poles with positive real part is satisfied
    """
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
        #first check eigenvalues of param_mat
        param_eigw, param_eigv = mp.eig(param_mat)
        print('param_eigw', param_eigw) #observe that param_mat is rank 2
        #print('eigenvalues of param_mat', param_eigw)
        L2_param_vec = mp.lu_solve(param_mat, param_b)
        
        #now test that adding a singular eigenvector to L2_param_vec still solves system
        mineignorm = mp.inf
        mineigind = -1
        for i in range(3):
            eignorm = mp.fabs(param_eigw[i])
            if eignorm < mineignorm:
                mineignorm = eignorm
                mineigind = i
        test_L2_param_vec = L2_param_vec + 3*param_eigv[:,mineigind]
        print('check adding a singular eigenvector to L2_param_vec', mp.mnorm(param_b - param_mat*test_L2_param_vec), mp.mnorm(param_b))
        if use_test_params == True:
            print('switching to modified test_params')
            L2_param_vec = test_L2_param_vec
    except ZeroDivisionError:
        print('Encountered singular matrix at kx', kx, 'use numpy pseudoinverse instead')

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
