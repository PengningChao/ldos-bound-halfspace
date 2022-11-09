"""
Numerically verify that the computed (AsymUP)_kx^{-1} images are correct 
for the TM polarization LDOS bounds
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.integrate import quad



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



def check_AsymUPinv(chi, phase, k0, kx, ymin=-10):
    """
    check the accuracy of the inverse image of AsymUPinv for TM by comparing
    AsymUP acting on (AsymUPinv S1) and S1
    with S1 here being e^{-ik_y y} = e^{(k_yi - ik_yr)y}
    """
    k0r = np.real(k0); k0i = np.imag(k0)
    PA = np.sqrt(np.real(phase*np.conj(phase)))
    
    ky = np.sqrt(k0**2 - kx**2)
    kyr = np.real(ky); kyi = np.imag(ky)

    B = np.conj(phase)*k0**2 / ky
    Br = np.real(B); Bi = np.imag(B)
    
    r_plus, r_minus = positiveRe_Laplace_poles(chi, phase, kyr, kyi, Br, Bi)
    
    gamma_plus, gamma_minus, detA = Laplace_free_params(kyr, kyi, Br, Bi, r_plus, r_minus)
    
    R_plus, R_minus = Laplace_residue_coeffs(chi, phase, kyr, kyi, Br, Bi, r_plus, r_minus, gamma_plus, gamma_minus)
    
    integrand = R_plus/(r_plus+(kyi+1j*kyr)) + R_minus/(r_minus+(kyi+1j*kyr))

    ylist = np.linspace(ymin, 0, 1000)
    
    S1list = np.exp(-1j*ky*ylist)
    
    AsymUPinvS1list = R_plus*np.exp(r_plus*ylist) + R_minus*np.exp(r_minus*ylist)
    #now calculate AsymUP acting on (AsymUPinv S1) with AsymUPinvS1 given by R_plus e^{r_plus*y} + R_minus e^{r_minus*y}
    mat_fac = np.imag(phase/np.conj(chi))
    
    plus_field = np.exp(r_plus*ylist) * (mat_fac + 0.25*B*(1.0/(r_plus+kyi-1j*kyr) - 1.0/(r_plus-kyi+1j*kyr)) + 0.25*np.conj(B)*(1.0/(r_plus+kyi+1j*kyr) - 1.0/(r_plus-kyi-1j*kyr)))
    plus_field += 0.25*B*np.exp((kyi-1j*kyr)*ylist) / (r_plus-kyi+1j*kyr) + 0.25*np.conj(B)*np.exp((kyi+1j*kyr)*ylist) / (r_plus-kyi-1j*kyr)
    plus_field *= R_plus
    
    minus_field = np.exp(r_minus*ylist) * (mat_fac + 0.25*B*(1.0/(r_minus+kyi-1j*kyr) - 1.0/(r_minus-kyi+1j*kyr)) + 0.25*np.conj(B)*(1.0/(r_minus+kyi+1j*kyr) - 1.0/(r_minus-kyi-1j*kyr)))
    minus_field += 0.25*B*np.exp((kyi-1j*kyr)*ylist) / (r_minus-kyi+1j*kyr) + 0.25*np.conj(B)*np.exp((kyi+1j*kyr)*ylist) / (r_minus-kyi-1j*kyr)
    minus_field *= R_minus
    
    tot_field = plus_field + minus_field
    
    fig, (ax1,ax2) = plt.subplots(ncols=2)
    ax1.plot(ylist, np.real(AsymUPinvS1list))
    ax2.plot(ylist, np.imag(AsymUPinvS1list))
    plt.suptitle('AsymUPinv @ S1')
    plt.show()
    
    fig, (ax1,ax2) = plt.subplots(ncols=2)
    ax1.plot(ylist, np.real(S1list))
    ax1.plot(ylist, np.real(tot_field), '--')
    ax2.plot(ylist, np.imag(S1list))
    ax2.plot(ylist, np.imag(tot_field), '--')
    plt.suptitle('S1 and AsymUPinv @ AsymUP @ S1')
    plt.show()
    
