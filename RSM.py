import numpy as np
import matplotlib.pyplot as plt
from dataAnalysis.fileIO import getPanalyticalXRD
from dataAnalysis.plotter import plotRC, FWHM
from dataAnalysis.functions import singlePeakFit, gauss_fit_2d, gauss_tilt_fit_2d, lorentz_tilt_fit_2d, vegard
import itertools
import matplotlib.colors as colors
from matplotlib import ticker
from scipy.interpolate import griddata
from scipy.optimize import least_squares, root
from scipy import linalg
from uncertainties import ufloat
from dataAnalysis.matDB import *
from dataAnalysis.tol_colors import tol_cmap

lam = 1.5405974  # angstroms
K = 2*np.pi/lam  # CuKalpha wavelength in angstroms


'''
lattice equations: https://duffy.princeton.edu/links
'''


def d_hex(a, c, h, k, l):
    temp = (4/3)*((h**2 + h*k + k**2)/a**2)+l**2/c**2
    if temp == 0:
        temp = np.nan
    return np.sqrt(1/temp)


def d_cubic(a, h, k, l):
    temp = (h**2+k**2+l**2)/a**2
    if temp == 0:
        temp = np.nan
    return np.sqrt(1/temp)


def d_tet(a, c, h, k, l):
    temp = ((h**2+k**2)/a**2) + l**2/c**2
    return np.sqrt(1/temp)


def d_rhomb(a, alpha, h, k, l):
    alpha = np.pi*alpha/180
    temp = ((h**2+k**2+l**2)*np.sin(alpha)**2 + 2*(h*k + k*l + l*h)*(np.cos(alpha)
            ** 2 - np.cos(alpha)))/(a**2*(1-3*np.cos(alpha)**2+2*np.cos(alpha)**3))
    return np.sqrt(1/temp)


def d_ortho(a, b, c, h, k, l):
    temp = h**2/a**2 + k**2/b**2 + l**2/c**2
    return np.sqrt(1/temp)


def d_mono(a, b, c, beta, h, k, l):
    beta = beta*np.pi/180
    temp = (1/np.sin(beta)**2)*(h**2/a**2 + k**2*np.sin(beta)
                                ** 2/b**2 + l**2/c**2 - 2*h*l*np.cos(beta)/(a*c))
    return np.sqrt(1/temp)


def d_tri(a, b, c, alpha, beta, gamma, h, k, l):
    '''
    Something not quite right about this eqn. not providing correct results currently
    '''
    alpha = alpha*np.pi/180
    beta = beta*np.pi/180
    gamma = gamma*np.pi/180
    V = a*b*c*np.sqrt(1-np.cos(alpha)**2-np.cos(beta)**2 -
                      np.cos(gamma)**2+2*np.cos(alpha)*np.cos(beta)*np.cos(gamma))
    S11 = b**2*c**2*np.sin(alpha)**2
    S22 = a**2*c**2*np.sin(beta)**2
    S33 = a**2*b**2*np.sin(gamma)**2
    S12 = a*b*c**2*(np.cos(alpha)*np.cos(beta)-np.cos(gamma))
    S23 = a**2*b*c*(np.cos(beta)*np.cos(gamma)-np.cos(alpha))
    S13 = a*b**2*c*(np.cos(gamma)*np.cos(alpha) - np.cos(beta))
    temp = (S11*h**2+S22*k**2+S33*l**2+2*S12*h*k+2*S23*k*l+2*S13*h*l)/V**2
    return np.sqrt(1/temp)


# calc lattice constants from d spacing

def lattice_hex(d_hk, d_l, h, k, l):
    c = np.sqrt(l**2*d_l**2)
    a = np.sqrt((4/3)*((h**2 + h*k + k**2)*d_hk**2))
    return a, c


def lattice_cubic(d, h, k, l):

    return np.sqrt((h**2+k**2+l**2)*d**2)


# least squares calculation module

def correct_symmetric(angle, n0, step=0.001):
    '''
    correction factor for refraction for symmetric reflections
    n0 = electrons per cubic angstrom
    '''
    theta = np.pi*angle/360
    delta = 4.48E-6*n0*lam**2
    err = np.sqrt(((cos(theta.n)*lam*step)/(2*sin(theta.n)**2))
                  ** 2 + (3.7E-4/(2*sin(theta.n)))**2)
    val = (lam/(2*sin(theta)))*(1 + delta/sin(2*theta))
    newErr = val.n*np.sqrt((val.s/val.n)**2 + (err/val.n)**2)
    return ufloat(val.n, newErr)


def correct_asymmetric(twotheta, omega, n0, step=0.001):
    '''
    correction factor for refraction for asymmetric reflections
    n0 = electrons per cubic angstrom
    '''
    theta = np.pi*twotheta/360
    omega = np.pi*omega/180
    err = np.sqrt(((cos(theta.n)*lam*step)/(2*sin(theta.n)**2))
                  ** 2 + (3.7E-4/(2*sin(theta.n)))**2)
    delta = 4.48E-6*n0*lam**2
    val = (lam/(2*sin(theta))) * \
        (1+(delta*cos(theta-omega))/(sin(2*theta-omega)*sin(omega)))
    newErr = val.n*np.sqrt((val.s/val.n)**2 + (err/val.n)**2)
    return ufloat(val.n, newErr)


def d_hex_lsq(x, h, k, l):
    '''
    function used for least squares regression through residuals_hex, residuals_hex_weight, and lattix_hex_lsq

    x is a tuple (a,c) of a hexagonal cell which is minimized. 
    h,k,l are miller indicies
    '''
    return 1/np.sqrt(4/3*(h**2+h*k+k**2)/(x[0]**2) + l**2/(x[1]**2))


def d_tet_lsq(x, h, k, l):
    '''
    function used for least squares regression through residuals_tet, residuals_tet_weight, and lattice_tet_lsq

    x is a tuple (a,c) of a tetragonal cell which is minimized. 
    h,k,l are miller indicies
    '''

    return 1/np.sqrt(((h**2+k**2)/x[0]**2) + l**2/x[1]**2)


def d_ortho_lsq(x, h, k, l):
    '''
    function used for least squares regression through residuals_ortho, residuals_ortho_weight, and lattice_ortho_lsq

    x is a tuple (a,b,c) of an orthorhombic cell which is minimized. 
    h,k,l are miller indicies
    '''

    return 1/np.sqrt(h**2/x[0]**2 + k**2/x[1]**2 + l**2/x[2]**2)


def residuals_hex_weight(x, h, k, l, d_meas, errors):
    '''
    inverse weighting by measurement error
    '''
    return (d_hex_lsq(x, h, k, l) - d_meas)/errors


def residuals_tet_weight(x, h, k, l, d_meas, errors):
    '''
    inverse weighting by measurement error
    '''
    return (d_tet_lsq(x, h, k, l) - d_meas)/errors


def residuals_ortho_weight(x, h, k, l, d_meas, errors):
    '''
    inverse weighting by measurement error
    '''
    return (d_ortho_lsq(x, h, k, l) - d_meas)/errors


def residuals_hex(x, h, k, l, d_meas):
    return (d_hex_lsq(x, h, k, l) - d_meas)


def residuals_tet(x, h, k, l, d_meas):
    return (d_tet_lsq(x, h, k, l) - d_meas)


def residuals_ortho(x, h, k, l, d_meas):
    return (d_ortho_lsq(x, h, k, l) - d_meas)


def mse_hex(x, h, k, l, d_meas):
    return (np.sum(d_hex_lsq(x, h, k, l) - d_meas)**2)/(d_meas.shape[0] - 2)


def lsq_err(res, args):
    '''
    calculate covariance matrix from least squares (scipy) jacobian
    return sqrt of covariance matrix diagonal
    '''
    J = res.jac
    hes = np.linalg.inv(J.T.dot(J))
    mse = mse_hex(res.x, *args)
    cov = hes*mse
    return np.sqrt(np.diag(cov))


def lsq_err_weight(res):
    '''
    calculate covariance matrix from least squares (scipy) jacobian
    return sqrt of covariance matrix diagonal

    for a least squares mininimization function that incorporates inverse weighting errors, this is the way to calc the covariance
    '''
    U, S, Vh = linalg.svd(res.jac, full_matrices=False)
    tol = np.finfo(float).eps*S[0]*max(res.jac.shape)
    w = S > tol
    cov = (Vh[w].T/S[w]**2) @ Vh[w]
    return np.sqrt(np.diag(cov))


def lattice_hex_lsq(d_array, h_array, k_array, l_array, x0, err=None):
    '''
    least sq minimization to find a and c given an array of d values and corresponding hkl indicies
    d_array - numpy array of d values
    h array - numpy array of h indicies
    k array - numpy array of k indicies
    l array - numpy array of l indicies

    x0 - initial guess as a numpy array - [a,c]
    '''

    if not all(s.shape == d_array.shape for s in [d_array, h_array, k_array, l_array]):
        raise ValueError('not all lists have same length!')
    if err is not None:
        args = (h_array, k_array, l_array, d_array, err)
        lsq = least_squares(residuals_hex_weight, x0, args=args)
        perr = lsq_err_weight(lsq)
    else:
        args = (h_array, k_array, l_array, d_array)
        lsq = least_squares(residuals_hex, x0, args=args)
        perr = lsq_err(lsq, args)

    a = ufloat(lsq.x[0], perr[0])
    c = ufloat(lsq.x[1], perr[1])
    print('a = {0:.4f}, c = {1:.4f}, c/a = {2:.4f}'.format(a, c, c/a))
    return a, c


def lattice_tet_lsq(d_array, h_array, k_array, l_array, x0, err=None):
    '''
    least sq minimization to find a and c given an array of d values and corresponding hkl indicies
    d_array - numpy array of d values
    h array - numpy array of h indicies
    k array - numpy array of k indicies
    l array - numpy array of l indicies

    x0 - initial guess as a numpy array - [a,c]
    '''

    if not all(s.shape == d_array.shape for s in [d_array, h_array, k_array, l_array]):
        raise ValueError('not all lists have same length!')
    if err is not None:
        args = (h_array, k_array, l_array, d_array, err)
        lsq = least_squares(residuals_tet_weight, x0, args=args)
        perr = lsq_err_weight(lsq)
    else:
        args = (h_array, k_array, l_array, d_array)
        lsq = least_squares(residuals_tet, x0, args=args)
        perr = lsq_err(lsq, args)

    a = ufloat(lsq.x[0], perr[0])
    c = ufloat(lsq.x[1], perr[1])
    print('a = {0:.4f}, c = {1:.4f}, c/a = {2:.4f}'.format(a, c, c/a))
    return a, c


def lattice_ortho_lsq(d_array, h_array, k_array, l_array, x0, err=None):
    '''
    least sq minimization to find a, b and c given an array of d values and corresponding hkl indicies
    d_array - numpy array of d values
    h array - numpy array of h indicies
    k array - numpy array of k indicies
    l array - numpy array of l indicies

    x0 - initial guess as a numpy array - [a,b,c]
    '''

    if not all(s.shape == d_array.shape for s in [d_array, h_array, k_array, l_array]):
        raise ValueError('not all lists have same length!')
    if err is not None:
        args = (h_array, k_array, l_array, d_array, err)
        lsq = least_squares(residuals_ortho_weight, x0, args=args)
        perr = lsq_err_weight(lsq)
    else:
        args = (h_array, k_array, l_array, d_array)
        lsq = least_squares(residuals_ortho, x0, args=args)
        perr = lsq_err(lsq, args)

    a = ufloat(lsq.x[0], perr[0])
    b = ufloat(lsq.x[1], perr[1])
    c = ufloat(lsq.x[2], perr[2])
    print('a = {0:.4f}, b = {1:.4f}, c = {2:.4f}'.format(a, b, c))
    return a, c


def chiAngle_hex(a, c, h1, k1, l1, h2, k2, l2):
    '''
    Calculate angle between hexagonal planes

    '''
    if (h1, k1, l1) == (h2, k2, l2):
        return 0
    else:
        num = h1*h2 + k1*k2 + 0.5*(h1*k2 + h2*k1) + ((3*a**2)/(4*c**2))*l1*l2
        denom = np.sqrt((h1**2 + k1**2 + h1*k1 + ((3*a**2)/(4*c**2))*l1**2)
                        * (h2**2 + k2**2 + h2*k2 + ((3*a**2)/(4*c**2))*l2**2))

        return np.arccos(num/denom)*180/np.pi


def chiAngle_cubic(h1, k1, l1, h2, k2, l2):
    '''
    Calculate angle between cubic planes

    '''
    if (h1, k1, l1) == (h2, k2, l2):
        return 0
    else:
        num = h1*h2 + k1*k2 + l1*l2
        denom = np.sqrt((h1**2 + k1**2 + l1**2)*(h2**2 + k2**2 + l2**2))
        return np.arccos(num/denom)*180/np.pi


def chiAngle_tet(a, c, h1, k1, l1, h2, k2, l2):
    if (h1, k1, l1) == (h2, k2, l2):
        return 0
    else:
        num = ((h1*h2+k1*k2)/a**2)+(l1*l2)/c**2
        denom = np.sqrt(((h1**2+k1**2)/a**2 + l1**2/c**2)
                        * ((h2**2+k2**2)/a**2 + l2**2/c**2))
        return np.arccos(num/denom)*180/np.pi


def chiAngle_rhomb(a, alpha, h1, k1, l1, h2, k2, l2):
    if (h1, k1, l1) == (h2, k2, l2):
        return 0
    else:
        d1 = d_rhomb(a, alpha, h1, k1, l1)
        d2 = d_rhomb(a, alpha, h2, k2, l2)
        alpha = alpha*np.pi/180
        V = a**3*np.sqrt(1-3*np.cos(alpha)**2 + 2*np.cos(alpha)**3)
        temp = a**4*d1*d2*(np.sin(alpha)**2*(h1*h2+k1*k2+l1*l2)+(np.cos(alpha) **
                           2-np.cos(alpha))*(k1*l2 + k2*l1 + l1*h2 + l2*h1 + h1*k2 + h2*k1))/V**2
        return np.arccos(temp)*180/np.pi


def chiAngle_ortho(a, b, c, h1, k1, l1, h2, k2, l2):
    if (h1, k1, l1) == (h2, k2, l2):
        return 0
    else:
        num = h1*h2/a**2 + k1*k2/b**2 + l1*l2/c**2
        denom = np.sqrt((h1**2/a**2 + k1**2/b**2 + l1**2/c**2)
                        * (h2**2/a**2+k2**2/b**2 + l2**2/c**2))
        return np.arccos(num/denom)*180/np.pi


def chiAngle_mono(a, b, c, beta, h1, k1, l1, h2, k2, l2):
    if (h1, k1, l1) == (h2, k2, l2):
        return 0
    else:
        d1 = d_mono(a, b, c, beta, h1, k1, l1)
        d2 = d_mono(a, b, c, beta, h2, k2, l2)
        beta = beta*np.pi/180
        temp = d1*d2*(h1*h2/a**2 + k1*k2*np.sin(beta)**2/b**2 + l1 *
                      l2/c**2 - (l1*h2+l2*h1)*np.cos(beta)/(a*c))/np.sin(beta)**2
        return np.arccos(temp)*180/np.pi


def chiAngle_tri(a, b, c, alpha, beta, gamma, h1, k1, l1, h2, k2, l2):
    '''
    Something not quite right about this eqn.
    '''
    if (h1, k1, l1) == (h2, k2, l2):
        return 0
    else:
        d1 = d_tri(a, b, c, alpha, beta, gamma, h1, k1, l1)
        d2 = d_tri(a, b, c, alpha, beta, gamma, h2, k2, l2)
        alpha = alpha*np.pi/180
        beta = beta*np.pi/180
        gamma = gamma*np.pi/180
        V = a*b*c*np.sqrt(1-np.cos(alpha)**2-np.cos(beta)**2 -
                          np.cos(gamma)**2+2*np.cos(alpha)*np.cos(beta)*np.cos(gamma))
        S11 = b**2*c**2*np.sin(alpha)**2
        S22 = a**2*c**2*np.sin(beta)**2
        S33 = a**2*b**2*np.sin(gamma)**2
        S12 = a*b*c**2*(np.cos(alpha)*np.cos(beta)-np.cos(gamma))
        S23 = a**2*b*c*(np.cos(beta)*np.cos(gamma)-np.cos(alpha))
        S13 = a*b**2*c*(np.cos(gamma)*np.cos(alpha) - np.cos(beta))
        temp = d1*d2*(S11*h1*h2 + S22*k1*k2 + S33*l1*l2+S23 *
                      (k1*l2+k2*l1)+S13*(l1*h2+l2*h1)+S12*(h1*k2+h2*k1))/V**2
        return np.arccos(temp)*180/np.pi


def calcQ(mat, h, k, l, norm=None, chi=0):
    '''
    Calculate (Qx,Qy) for a given plane hkl
    Defaults to asymmetric case (chi=0)


    '''
    sys = mat['sys']
    if norm == None:
        norm = mat['norm']

    if sys == 'hex':
        a = mat['a']
        c = mat['c']
        if norm == (0, 0, 1):
            if h == 0 and k == 0:
                qx = 0
            else:
                d_hk = d_hex(a, c, h, k, 0)
                qx = 2*np.pi/d_hk

            if l == 0:
                qz = 0
            else:
                d_l = d_hex(a, c, 0, 0, l)
                qz = 2*np.pi/d_l
        else:
            offset = chiAngle_hex(a, c, h, k, l, norm[0], norm[1], norm[2])
            resultant = 2*np.pi/d_hex(a, c, h, k, l)
            qx = resultant*np.sin(offset*np.pi/180)
            qz = resultant*np.cos(offset*np.pi/180)

        resultant = np.sqrt(qx**2+qz**2)
        offset = chiAngle_hex(a, c, h, k, l, norm[0], norm[1], norm[2])

    elif sys == 'cubic':
        offset = chiAngle_cubic(norm[0], norm[1], norm[2], h, k, l)
        resultant = 2*np.pi/d_cubic(mat['a'], h, k, l)
        qx = resultant*np.sin(offset*np.pi/180)
        qz = resultant*np.cos(offset*np.pi/180)

    elif sys == 'pc':
        a = mat['a_pc']
        d_h = d_cubic(a, 0, 0, l)
        d_kl = d_cubic(a, h, k, 0)
        qx = 2*np.pi/d_kl
        qz = 2*np.pi/d_h

        resultant = np.sqrt(qx**2+qz**2)
        offset = chiAngle_cubic(h, k, l, norm[0], norm[1], norm[2])
    elif sys == 'tet':
        a = mat['a']
        c = mat['c']
        d_hk = d_tet(a, c, h, k, 0)
        d_l = d_tet(a, c, 0, 0, l)
        qx = 2*np.pi/d_hk
        qz = 2*np.pi/d_l

    elif sys == 'mono':
        a = mat['a']
        b = mat['b']
        c = mat['c']
        beta = mat['beta']
        if norm == (0, 1, 0):
            if h == 0 and l == 0:
                qx = 0
            else:
                d_hl = d_mono(a, b, c, beta, h, 0, l)
                qx = 2*np.pi/d_hl

            d_k = d_mono(a, b, c, beta, 0, k, 0)
            qz = 2*np.pi/d_k
        elif norm == (1,0,0):
            if k == 0 and l == 0:
                qx = 0
            else:
                d_kl = d_mono(a,b,c,beta,0,k,l)
                qx = 2*(np.pi/d_kl)
            d_h = d_mono(a,b,c,beta,h,0,0)
            qz = 2*np.pi/d_h
        else:
            raise ValueError(
                'only (010) and (100) monoclinic crystals are so-far supported')
    elif sys == 'rhomb':
        offset = chiAngle_rhomb(
            mat['a'], mat['alpha'], h, k, l, norm[0], norm[1], norm[2])
        resultant = 2*np.pi/d_rhomb(mat['a'], mat['alpha'], h, k, l)
        qx = resultant*np.sin(offset*np.pi/180)
        qz = resultant*np.cos(offset*np.pi/180)
    elif sys == 'ortho':
        a = mat['a']
        b = mat['b']
        c = mat['c']
        if norm == (0, 0, 1):
            d_hk = d_ortho(a, b, c, h, k, 0)
            d_l = d_ortho(a, b, c, 0, 0, l)
            qx = 2*np.pi/d_hk
            qz = 2*np.pi/d_l
        elif norm == (1, 1, 0):
            raise ValueError('(110) calcQ not coded in yet')

    if chi != 0:
        angle = (chi+offset)*np.pi/180
        qx = resultant*np.sin(angle)
        qz = resultant*np.cos(angle)
    if np.isnan(qx):
        qx = 0
    if np.isnan(qz):
        qz = 0
    return (qx, qz)


def nitride_alloy_cubic(x, P, Q, R, S):
    return P*x**3 + Q*x**2 + R*x + S


def comp_and_relax(x, y, unit='angstroms', alloy_type='InGaN', substrate='GaN', hkl=None, a_hex=None):
    '''
    https://doi.org/10.1088/0022-3727/32/10A/312
    '''
    if alloy_type == 'InGaN':
        v_A = 2*InN['C13']/InN['C33']
        v_B = 2*GaN['C13']/GaN['C33']
        a0_A = InN['a']
        a0_B = GaN['a']
        c0_A = InN['c']
        c0_B = GaN['c']
    elif alloy_type == 'AlGaN':
        v_A = 2*AlN['C13']/AlN['C33']
        v_B = 2*GaN['C13']/GaN['C33']
        a0_A = AlN['a']
        a0_B = GaN['a']
        c0_A = AlN['c']
        c0_B = GaN['c']
    if unit == 'angstroms':
        a_L = x
        c_L = y
    if unit == 'Q':
        if hkl[0] + hkl[1] == 1:
            a_L = (4/np.sqrt(3))*np.pi/x
        elif hkl[0] + hkl[1] == 2:
            a_L = 2*np.pi/x
        c_L = hkl[2]*2*np.pi/y
    P = (v_A - v_B)*(a0_A - a0_B)*(c0_A - c0_B)
    Q = (1 + v_B)*(a0_A - a0_B)*(c0_A - c0_B) + (v_A - v_B) * \
        ((a0_A - a0_B)*c0_B + (a0_B-a_L)*(c0_A - c0_B))
    R = (a0_A - a0_B)*((1 + v_B)*c0_B - c_L) + (c0_A - c0_B) * \
        ((1 + v_B)*a0_B - v_B*a_L) + (v_A - v_B)*(a0_B - a_L)*c0_B
    S = (1 + v_B)*a0_B*c0_B - v_B*a_L*c0_B-a0_B*c_L

    guess = (c_L - c0_B)/(c0_A - c0_B)

    res = root(nitride_alloy_cubic, guess, args=(P, Q, R, S))

    comp = res.x

    if alloy_type == 'InGaN':
        InGaN_relaxed = alloy(comp, GaN, InN)
        a0_L = InGaN_relaxed['a']
    elif alloy_type == 'AlGaN':
        AlGaN_relaxed = alloy(comp, GaN, AlN)
        a0_L = AlGaN_relaxed['a']

    if substrate == 'GaN':
        a0_S = GaN['a']
    elif substrate == 'AlN':
        a0_S = AlN['a']
    elif substrate == 'TaC':
        a0_S = a_hex
    else:
        print('substrate not recognized')

    relax = (a_L - a0_S)/(a0_L - a0_S)

    print('InGaN properties\na : {0:0.2f} A\nc : {1:0.2f} A\ncomposition : {2:0.1f}%\nrelaxation : {3:0.1f}%'.format(
        a_L, c_L, float(comp*100), float(relax*100)))

    return comp, relax


def tet_strain(mat, sub):
    del_a = (sub['a'] - mat['a'])/mat['a']
    c11 = mat['C11']
    c12 = mat['C12']
    c13 = mat['C13']
    c33 = mat['C33']
    C1 = c33*(c11+c12) - 2*c13**2
    s11 = (c11*c33-c13**2)/((c11-c12)*C1)
    s13 = -c13/C1
    s33 = (c11+c12)/C1
    v13 = -s13/s33
    e_zz = (-2*v13/(1-v13))*(del_a)
    c = e_zz*mat['c'] + mat['c']
    return (sub['a'], c)


class RSM:
    '''
    Class for dealing with 2D RSM data. Will contain an importer, plotter, and analysis functions.
    When initiallizing, please define substrate and film as strings, and (hkl) as a tuple (h, k, l). Ex RA001 = RSM('AlN','GaN', 1,0,5))

    **NOTE: Currently set up for 1 film (layer) only
    '''

    def __init__(self, sub, film, hkl, chi=0, subNorm=None, filmNorm=None):
        if isinstance(sub, str):
            self.sub = matDB[sub]
        else:
            self.sub = sub
        if isinstance(film, str):
            self.film = matDB[film]
        else:
            self.film = film
        self.hkl = hkl
        self.chi = chi
        self.subQ = calcQ(
            self.sub, self.hkl[0], self.hkl[1], self.hkl[2], norm=subNorm, chi=chi)
        self.filmQ = calcQ(
            self.film, self.hkl[0], self.hkl[1], self.hkl[2], norm=filmNorm, chi=chi)
        self.subQ_r = None
        self.filmQ_r = None

       

    def importData(self, source, format=None):

        supportedFormats = ['csv', 'ras', 'xrdml', 'txt-SL', 'txt-rel', 'txt']

        try:
            self.intensity
        except AttributeError:
            var_exists = False
        else:
            var_exists = True
        if var_exists:
            print('Data already loaded')
        else:
            if format == None:
                format = source.split('.')[-1]
            if format not in supportedFormats:
                raise ValueError('Unsupported Format')

            if format == 'ras':
                f = open(source, errors='ignore')
                string = '*RAS_HEADER_START'
                subFile = []
                genList = []
                header = []
                headerList = []
                qx = []
                qz = []
                tth = []
                w = []
                intensity = []
                meta = []
                for line in f:
                    if line.find(string) == -1:
                        if line.find('*') == 0:
                            header.append(line)
                        else:
                            subFile.append(line)
                    else:
                        genList.append(itertools.chain(subFile))
                        headerList.append(header)
                        header = []
                        subFile = []
                for i, gen in enumerate(genList[1:]):
                    data = np.genfromtxt(gen, comments='*')
                    h = headerList[i+1]
                    axis = list(
                        filter(lambda x: '*MEAS_COND_AXIS_NAME-' in x, h))
                    positions = list(
                        filter(lambda x: '*MEAS_COND_AXIS_POSITION-' in x, h))
                    offsets = list(
                        filter(lambda x: '*MEAS_COND_AXIS_OFFSET-' in x, h))
                    RSM_meta = list(filter(lambda x: '*MEAS_3DE_' in x, h))
                    scan_meta = list(filter(lambda x: '*MEAS_SCAN_' in x, h))
                    offset_dict = {}
                    position_dict = {}
                    RSM_origin = {}
                    RSM_scan = {}
                    RSM_step = {}
                    scan_speed = {}

                    for ax, pos, off in zip(axis, positions, offsets):
                        n = ax.split('"')[-2]
                        try:
                            v = float(pos.split('"')[-2])
                        except ValueError:
                            v = pos.split('"')[-2]
                        try:
                            o = float(off.split('"')[-2])
                        except ValueError:
                            o = np.nan
                        offset_dict[n] = o
                        position_dict[n] = v

                    for line in RSM_meta:
                        if line.find('ORIGIN') != -1:
                            key = line.split('_')[-2]
                            value = float(line.split('_')[-1].split('"')[-2])
                            RSM_origin[key] = value
                        if line.find('SCAN') != -1:
                            key = line.split('_')[-1].split(' ')[0]
                            value = float(line.split('_')[-1].split('"')[-2])
                            RSM_scan[key] = value
                        if line.find('STEP') != -1:
                            key = line.split('_')[-1].split(' ')[0]
                            try:
                                value = float(line.split(
                                    '_')[-1].split('"')[-2])
                            except ValueError:
                                value = line.split('_')[-1].split('"')[-2]
                            RSM_step[key] = value

                    for line in scan_meta:
                        if line.find('SPEED') != -1:
                            try:
                                key = line.split('_')[-2]
                                value = value = float(
                                    line.split('_')[-1].split('"')[-2])
                            except ValueError:
                                pass
                            scan_speed[key] = value

                    scanAxis = list(
                        filter(lambda x: '*MEAS_SCAN_AXIS_X ' in x, h))[0].split('"')[-2]
                    stepAxis = RSM_step['INTERNAL']
                    if scanAxis == '2-Theta/Omega' and stepAxis == 'Omega':
                        offset = (
                            position_dict['Theta/2-Theta']/2-position_dict['Omega'])*np.pi/180
                        twotheta = data[:, 0]*np.pi/180
                        omega = twotheta/2 - offset
                        w.append(omega)
                        tth.append(twotheta)
                    if scanAxis == '2-Theta' and stepAxis == 'Omega':
                        twotheta = data[:, 0]*np.pi/180
                        omega = position_dict['Omega']*np.pi/180
                        w.append(np.full_like(twotheta, omega))
                        tth.append(twotheta)
                    if scanAxis == 'Omega' and stepAxis == 'TwoThetaOmega':
                        omega = data[:, 0]*np.pi/180
                        twotheta = position_dict['2-Theta']*np.pi/180
                        w.append(omega)
                        tth.append(np.full_like(omega, twotheta))

                    qx.append(K*(np.cos(omega) - np.cos(twotheta - omega)))
                    qz.append(K*(np.sin(omega) + np.sin(twotheta - omega)))
                    intensity.append(data[:, 1]/scan_speed['SPEED'])
                    metaDict = {'positions': position_dict,
                                'offsets': offset_dict,
                                'RSM_origin': RSM_origin,
                                'RSM_scan': RSM_scan,
                                'RSM_step': RSM_step}
                    meta.append(metaDict)

                qx = np.array(qx)
                qz = np.array(qz)
                intensity = np.array(intensity)
                tth = np.array(tth)
                w = np.array(w)

            if format == 'xrdml':
                tth, w, intensity = getPanalyticalXRD(source)
                tth = tth*np.pi/180
                w = w*np.pi/180
                qx = K*(np.cos(w) - np.cos(tth - w))
                qz = K*(np.sin(w) + np.sin(tth - w))

            if format == 'txt':
                # assuming csv data is symmetric with col1=relative omega and col2=2theta. Offset is in header line 1
                header = np.genfromtxt(source, comments=None, max_rows=1)
                omegaI = header[-1]
                data = np.genfromtxt(source, delimiter=',')
                omega = (data[:, 0] + omegaI)*np.pi/180
                twotheta = data[:, 1]*np.pi/180
                intensity = data[:, 2]

                qx = K*(np.cos(omega) - np.cos(twotheta - omega))
                qz = K*(np.sin(omega) + np.sin(twotheta - omega))

            if format == 'txt-SL':
                origin = np.genfromtxt(
                    source, skip_header=8, max_rows=1, comments=None)
                size = np.genfromtxt(source, skip_header=13,
                                     max_rows=1, comments=None)
                omegaI = origin[-4]
                twothetaI = origin[-3]
                rows = int(size[-2])
                cols = int(size[-1])
                chi = origin[-2]
                phi = origin[-1]
                file = open(source)
                x = None
                while x != 'axis':
                    line = file.readline()
                    x = line.split()[1]
                axis1 = line.split()[-2].strip('"')
                axis2 = line.split()[-1].strip('"')
                file.close()
                if axis1 == 'Omega' and axis2 == '2Theta':
                    data = np.genfromtxt(source)
                    omega = (data[:, 0] + omegaI)*np.pi/180
                    twotheta = data[:, 1]*np.pi/180
                    intensity = data[:, 2]

                elif axis1 == 'Omega' and axis2 == '2Theta/Omega':
                    print('Sorry! ' + axis1 + '-' +
                          axis2 + ' is not supported yet')
                elif axis1 == 'Qx' and axis2 == 'Qz':
                    print('Sorry! ' + axis1 + '-' +
                          axis2 + ' is not supported yet')
                else:
                    print('Axis not recognized')

                qx = K*(np.cos(omega) - np.cos(twotheta - omega))
                qz = K*(np.sin(omega) + np.sin(twotheta - omega))

                qx = qx.reshape((rows, cols))
                qz = qz.reshape((rows, cols))
                intensity = intensity.reshape((rows, cols))

            if format == 'txt-rel':
                origin = np.genfromtxt(
                    source, skip_header=8, max_rows=1, comments=None)
                size = np.genfromtxt(source, skip_header=13,
                                     max_rows=1, comments=None)
                omegaI = origin[-4]*np.pi/180
                twothetaI = origin[-3]*np.pi/180
                rows = int(size[-2])
                cols = int(size[-1])
                chi = origin[-2]
                phi = origin[-1]
                file = open(source)
                x = None
                while x != 'axis':
                    line = file.readline()
                    x = line.split()[1]
                axis1 = line.split()[-2].strip('"')
                axis2 = line.split()[-1].strip('"')
                file.close()
                if axis1 == 'Omega' and axis2 == '2Theta':
                    data = np.genfromtxt(source)
                    omega = (data[:, 0])*np.pi/180
                    twotheta = data[:, 1]*np.pi/180
                    intensity = data[:, 2]
                    omega = omega.reshape((rows, cols))
                    twotheta = twotheta.reshape((rows, cols))
                    for i in range(rows):
                        omega_0 = twotheta[i, :]/2
                        omega[i, :] = omega[i, :]+omega_0
                    qx = K*(np.cos(omega) - np.cos(twotheta - omega))
                    qz = K*(np.sin(omega) + np.sin(twotheta - omega))
                elif axis1 == 'Omega' and axis2 == '2Theta/Omega':
                    data = np.genfromtxt(source)
                    omega = (data[:, 0])*np.pi/180
                    twotheta_w = data[:, 1]*np.pi/180
                    intensity = data[:, 2]
                    omega = omega.reshape((rows, cols))
                    twotheta_w = twotheta_w.reshape((rows, cols))
                    for i in range(rows):
                        omega_0 = twotheta[i, :]/2
                        omega[i, :] = omega[i, :]+omega_0
                    qx = K*(np.cos(omega) - np.cos(twotheta - omega))
                    qz = K*(np.sin(omega) + np.sin(twotheta - omega))
                elif axis1 == 'Qx' and axis2 == 'Qz':
                    data = np.genfromtxt(source)
                    qx = (data[:, 0])*2*np.pi
                    qz = data[:, 1]*2*np.pi
                    intensity = data[:, 2]
                    qx = qx.reshape((rows, cols))
                    qz = qz.reshape((rows, cols))

                else:
                    print('Axis not recognized')

                intensity = intensity.reshape((rows, cols))

            intensity[intensity <= 0] = intensity[intensity > 0].min()

            self.qx = qx
            self.qz = qz
            self.w = w
            self.tth = tth
            self.intensity = intensity

    def plot(self, threshImin=None, threshImax=None, fig=None, ax=None, xmin=None, xmax=None, ymin=None, ymax=None, cb=False, optimize=False, peakSubRange=(5, 1), peakFilmRange=(5, 1), plotstyle='scatter', units='Q', colormap=tol_cmap('rainbow_discrete_white_transparent'), levels=None, show=True, routine='dot', showMasks=False):
        '''
        plotter function to plot RSM data provided by the getRSMdata function. Note - hi res data works best with scatter, low res use contours

        threshImin - minimum intensity threshold for plotting, default None

        fig: matplotlib figure object in which to plot the data. if given, it is assumed an axes object is also provided. default None
        ax: matplotlib axes object on which to plot the data. if given, it is assumed a figure object has also been provided. default None

        xmin: lower qx limit for plotting, default None
        xmax: upper qx limit for plotting, default None
        ymin: lower qz limit for plotting, default None
        ymax: upper qz limit for plotting, default None

        cb: boolean, include intensity color bar on plot. default True

        optimize: if true, will run the peak optimization routine to determine precise peak locations. default False - boolean

        subQ: tuple (Qx, Qz) of substrate peak location. default None - necessary (and only used) for optimization routine. 

        filmQ: tuple (Qx, Qz) of film peak location. default None - necessary (and only used) for optimization routine. 

        peakSubRange: tuple of ranges (%Qx, %Qz) in percentage to do a weighted average near the substrate peak for optimization routine. default None

        peakFilmRange: tuple of ranges (%Qx, %Qz) in percentage to do a weighted average near the film peak for optimization routine. default None

        plotstyle: sets style of plot. options are 'scatter', 'contour', and 'contourf'. default is scatter

        units: Q (2pi/lambda) or S(1/lambda)

        colormap - specify colormap to use for plotting. default plt.cm.gist_heat

        levels - specify contour levels if using plotstyle='contour', default None

        show - show plot, default True

        '''

        tempQx = self.qx
        tempQz = self.qz

        if units == 'S':
            self.qx = self.qx/(2*np.pi)
            self.qz = self.qz/(2*np.pi)
        elif units == 'd':
            self.qx = (2*np.pi)/self.qx
            self.qz = (2*np.pi)/self.qz

        
        if fig == None:
            fig = plt.figure()
        if ax == None:
            ax = fig.gca()

        if threshImin is not None:
            try:
                zmin = np.log10(threshImin).round()
            except AttributeError:
                print('RSM data has not been imported')
        else:
            try:
                zmin = np.log10(self.intensity.min()).round()
            except AttributeError:
                print('RSM data has not been imported')
        zmax = np.log10(self.intensity.max()).round()

        if plotstyle == 'scatter':
            if units == 'd':
                h = np.abs(self.qx.flatten()[1] - self.qx.flatten()[0])
                v = np.abs(self.qz.flatten()[0] - self.qz.flatten()[1])
            else:
                h = np.abs(self.qx.flatten()[0] - self.qx.flatten()[1])
                v = np.abs(self.qz.flatten()[1] - self.qz.flatten()[0])
            bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            width, height = bbox.width, bbox.height
            width *= fig.dpi
            height *= fig.dpi
            delW = h*width/(self.qx.max()-self.qx.min())/10
            delH = v*height/(self.qz.max()-self.qz.min())
            verts = list(zip([-delH, delH, delH, -delH],
                         [-delW, -delW, delW, delW]))
            plot = ax.scatter(self.qx.flatten(), self.qz.flatten(), c=self.intensity.flatten(), norm=colors.LogNorm(
                vmin=threshImin if threshImin else np.nanmin(self.intensity), vmax=threshImax if threshImax else np.nanmax(self.intensity)), cmap=colormap, marker=verts)
        elif plotstyle == 'contourf':
            if levels == None:
                levels = 50
            plot = ax.contourf(self.qx, self.qz, self.intensity, norm=colors.LogNorm(vmin=threshImin if threshImin else np.nanmin(self.intensity), vmax=threshImax if threshImax else np.nanmax(self.intensity)), cmap=colormap, levels=np.logspace(zmin, zmax, num=levels))
        elif plotstyle == 'contour':
            up = np.floor(np.log10(self.intensity.max()))
            down = np.ceil(np.log10(self.intensity.min()))
            points = self.qx[0, :].shape[0]*2
            xi = np.linspace(self.qx.min(), self.qx.max(), points)
            yi = np.linspace(self.qz.min(), self.qz.max(), points)
            zi = griddata((self.qx.flatten(), self.qz.flatten()), self.intensity.flatten(
            ), (xi[None, :], yi[:, None]), method='cubic', rescale=True)
            if levels == None:
                levels = up-down+1
            contours = np.logspace(down, up, levels)
            plot = ax.contour(
                xi, yi, zi, locator=ticker.LogLocator(), levels=contours, cmap=colormap)

        if xmin is not None:
            ax.set_xlim(left=xmin)
        if xmax is not None:
            ax.set_xlim(right=xmax)
        if ymin is not None:
            ax.set_ylim(bottom=ymin)
        if ymax is not None:
            ax.set_ylim(top=ymax)
        if cb:
            cb = plt.colorbar(plot)
            cb.set_ticks(np.logspace(zmin, zmax, num=zmax-zmin+1))
            cb.set_label('Intensity (cps)')

        if optimize:
            self.optimize(peakSubRange=peakSubRange, peakFilmRange=peakFilmRange,
                          routine=routine, threshImin=threshImin, showMasks=showMasks)
            ax.plot(self.subQ[0], self.subQ[1], '*r', markersize=10)
            ax.plot(self.filmQ[0], self.filmQ[1], '*r', markersize=10)
            ax.plot(self.subQ_r[0], self.subQ_r[1], '*k', markersize=10)
            ax.plot(self.filmQ_r[0], self.filmQ_r[1], '*k', markersize=10)
        if units == 'Q':
            ax.set_xlabel('Q$_{x}$ ($\AA^{-1}$)')
            ax.set_ylabel('Q$_{z}$ ($\AA^{-1}$)')
        elif units == 'S':
            ax.set_xlabel('S$_{x}$ ($\AA^{-1}$)')
            ax.set_ylabel('S$_{z}$ ($\AA^{-1}$)')
        elif units == 'd':
            ax.set_xlabel('d$_{hk}$ ($\AA$)')
            ax.set_ylabel('d$_{l}$ ($\AA$)')
        fig.tight_layout()

        if show:
            fig.show()

        self.qx = tempQx
        self.qz = tempQz

        return fig, ax

    def optimize(self, peakSubRange=(5, 1), peakFilmRange=(5, 1), routine='dot', threshImin=None, showMasks=False, p0_sub=None, p0_film=None, plot=False):
        '''
        peak optimization routine for the plotRSM function

        qx: 2D array of Qx values corresponding to each data point in reciprocal space - numpy array
        qz: 2D array of Qz values corresponding to each data point in reciprocal space, must be the same size as Qx - numpy array
        intensity: 2D array of intensity values corresponding to each qx and qz data point, must all be the same size - numpy array

        SubQ: tuple (Qx, Qz) of substrate peak location. 
        FilmQ: tuple (Qx, Qz) of film peak location. 
        peakSubRange: tuple of ranges (%Qx, %Qz) in percentage to do a weighted average near the substrate peak
        peakFilmRange: tuple of ranges (%Qx, %Qz) in percentage to do a weighted average near the film peak

        routine - routine to be used. Standard is dot product, also can use 2D for the 2D peak fitting algorithm which fits a voigt to each row of qx and qz, then fits the center and intensity of each fit (qx and qz) to interpolate the max. Note - 2D can take a VERY long time to run. Gauss fit and dot product are fairly quick. lorentz fit is medium speed.

        '''

        if p0_sub == None:
            SubQx = self.subQ[0] if self.subQ[0] != 0 else 0.01
            SubQz = self.subQ[1]
        else:
            SubQx, SubQz = p0_sub

        if p0_film == None:
            FilmQx = self.filmQ[0] if self.filmQ[0] != 0 else 0.01
            FilmQz = self.filmQ[1]
        else:
            FilmQx, FilmQz = p0_film
        peaksubxrange = peakSubRange[0]
        peaksubzrange = peakSubRange[1]
        peakfilmxrange = peakFilmRange[0]
        peakfilmzrange = peakFilmRange[1]

        if SubQx > 0:
            SubQx_min = SubQx*(1-peaksubxrange/100)
            SubQx_max = SubQx*(1+peaksubxrange/100)
        else:
            SubQx_min = SubQx*(1+peaksubxrange/100)
            SubQx_max = SubQx*(1-peaksubxrange/100)
        SubQz_min = SubQz*(1-peaksubzrange/100)
        SubQz_max = SubQz*(1+peaksubzrange/100)
        if FilmQx > 0:
            FilmQx_min = FilmQx*(1-peakfilmxrange/100)
            FilmQx_max = FilmQx*(1+peakfilmxrange/100)
        else:
            FilmQx_min = FilmQx*(1+peakfilmxrange/100)
            FilmQx_max = FilmQx*(1-peakfilmxrange/100)
        FilmQz_min = FilmQz*(1-peakfilmzrange/100)
        FilmQz_max = FilmQz*(1+peakfilmzrange/100)
        SubPeakRows = []
        SubPeakCols = []
        FilmPeakRows = []
        FilmPeakCols = []
        SubQx_num = 0
        SubQz_num = 0
        SubQ_denom = 0
        FilmQx_num = 0
        FilmQz_num = 0
        FilmQ_denom = 0

        qxIndex_sub = np.where((self.qx > SubQx_min) & (
            self.qx < SubQx_max), True, False)
        qzIndex_sub = np.where((self.qz > SubQz_min) & (
            self.qz < SubQz_max), True, False)
        subMask = np.logical_and(qxIndex_sub, qzIndex_sub)
        subI = np.where(subMask, self.intensity, self.intensity.min())

        qxIndex_film = np.where((self.qx > FilmQx_min) & (
            self.qx < FilmQx_max), True, False)
        qzIndex_film = np.where((self.qz > FilmQz_min) & (
            self.qz < FilmQz_max), True, False)
        filmMask = np.logical_and(qxIndex_film, qzIndex_film)
        filmI = np.where(filmMask, self.intensity, self.intensity.min())

        if showMasks:
            plt.figure()
            plt.scatter(self.qx.flatten(), self.qz.flatten(), c=subI.flatten(), norm=colors.LogNorm(
                vmin=self.intensity.min(), vmax=self.intensity.max()), cmap=plt.cm.gist_heat_r)
            plt.text(.95, .95, 'sub mask', transform=plt.gca().transAxes)
            plt.show()
            plt.figure()
            plt.scatter(self.qx.flatten(), self.qz.flatten(), c=filmI.flatten(), norm=colors.LogNorm(
                vmin=self.intensity.min(), vmax=self.intensity.max()), cmap=plt.cm.gist_heat_r)
            plt.text(.95, .95, 'film mask', transform=plt.gca().transAxes)
            plt.show()

        if routine == 'dot':
            for r in range(self.qx.shape[0]):
                for c in range(self.qx.shape[1]):
                    if ((SubQx_min < self.qx[r, c]) and (self.qx[r, c] < SubQx_max) and (SubQz_min < self.qz[r, c]) and (self.qz[r, c] < SubQz_max)):
                        SubPeakRows.append(r)
                        SubPeakCols.append(c)
                        SubQx_num = SubQx_num + \
                            np.dot(self.qx[r, c], subI[r, c])
                        SubQz_num = SubQz_num + \
                            np.dot(self.qz[r, c], subI[r, c])
                        SubQ_denom = SubQ_denom + subI[r, c]
                    if ((FilmQx_min < self.qx[r, c]) and (self.qx[r, c] < FilmQx_max) and (FilmQz_min < self.qz[r, c]) and (self.qz[r, c] < FilmQz_max)):
                        FilmPeakRows.append(r)
                        FilmPeakCols.append(c)
                        FilmQx_num = FilmQx_num + \
                            np.dot(self.qx[r, c], filmI[r, c])
                        FilmQz_num = FilmQz_num + \
                            np.dot(self.qz[r, c], filmI[r, c])
                        FilmQ_denom = FilmQ_denom + filmI[r, c]

            SubQx = SubQx_num / SubQ_denom
            SubQz = SubQz_num / SubQ_denom
            FilmQx = FilmQx_num / FilmQ_denom
            FilmQz = FilmQz_num / FilmQ_denom
            self.subQ_r = (SubQx, SubQz)
            self.filmQ_r = (FilmQx, FilmQz)

        elif routine == 'gauss_fit':
            x, y, A = gauss_fit_2d(
                self.qx, self.qz, subI, threshImin=threshImin)
            self.subQ_r = (x, y)

            x, y, A = gauss_fit_2d(
                self.qx, self.qz, filmI, threshImin=threshImin)
            self.filmQ_r = (x, y)

        elif routine == 'gauss_tilt_fit':
            x, y, A = gauss_tilt_fit_2d(
                self.qx, self.qz, subI, threshImin=threshImin)
            self.subQ_r = (x, y)

            x, y, A = gauss_tilt_fit_2d(
                self.qx, self.qz, filmI, threshImin=threshImin)
            self.filmQ_r = (x, y)

        elif routine == 'lorentz_tilt_fit':
            x, y, A = lorentz_tilt_fit_2d(
                self.qx, self.qz, subI, threshImin=threshImin)
            self.subQ_r = (x, y)

            x, y, A = lorentz_tilt_fit_2d(
                self.qx, self.qz, filmI, threshImin=threshImin)
            self.filmQ_r = (x, y)

        elif routine == '2D' or routine == '2D-rlu':
            self.subQ_r = peak_find_2D(self.qx, self.qz, subI)
            self.filmQ_r = peak_find_2D(self.qx, self.qz, filmI)

        elif routine == '2D-w-tth':
            omega, twotheta = peak_find_2D(self.w, self.tth, subI)
            self.subQ_r = (K*(np.cos(omega) - np.cos(twotheta - omega)),
                           K*(np.sin(omega) + np.sin(twotheta - omega)))

            omega, twotheta = peak_find_2D(self.w, self.tth, filmI)
            self.filmQ_r = (K*(np.cos(omega) - np.cos(twotheta - omega)),
                            K*(np.sin(omega) + np.sin(twotheta - omega)))

        if plot:
            self.RSM_ax.scatter(*self.subQ_r, marker='X', color='k')
            self.RSM_ax.scatter(*self.filmQ_r, marker='X', color='tab:green')
            self.redraw()

    def plotRC(self, peak='sub', theta=None, fig=None, ax=None, RSM_ax = None, shape='voigt', numpeaks=1, plotLine=False):
        if fig == None:
            fig = plt.figure()
        if ax == None:
            ax = fig.gca()
        if self.subQ_r == None and theta == None:
            self.optimize()
        if peak == 'sub':
            if theta == None:
                theta = calc_theta(self.subQ_r[0], self.subQ_r[1])

            # lines of constant theta or constant omega?
            if self.tth[0, 0] == self.tth[0, 1]:
                index = np.searchsorted(self.tth[:, 0], 2*theta)
                x = self.w[index, :]*180/np.pi
                y = self.intensity[index, :]
            else:
                index = np.searchsorted(self.tth[0, :], 2*theta)
                x = self.w[:, index]*180/np.pi
                y = self.intensity[:, index]

            fwhmVal = FWHM((x, y), source='data', report=False,
                           plot=True, fig=fig, ax=ax, shape=shape, peaks=numpeaks)
        elif peak == 'film':
            if theta == None:
                resultant = np.sqrt(self.filmQ_r[0]**2 + self.filmQ_r[1]**2)
                theta = np.arcsin(lam*resultant/(4*np.pi))
            index = np.searchsorted(self.tth[0, :], 2*theta)
            x = self.w[:, index]*180/np.pi
            y = self.intensity[:, index]
            fwhmVal = FWHM((x, y), source='data', report=False,
                           plot=True, fig=fig, ax=ax, shape=shape, peaks=numpeaks)

        if plotLine:
            w = x*np.pi/180
            tth = theta*2
            qx = K*(np.cos(w) - np.cos(tth - w))
            qz = K*(np.sin(w) + np.sin(tth - w))
            if RSM_ax == None:
                RSM_fig, RSM_ax = plt.subplots(1,1)
                
            RSM_ax.plot(qx, qz, 'k')

        return fwhmVal, fig, ax

    def latticeConstant(self, layer='film'):
        '''
        Calculate lattice constants from optimized peak position

        '''
        if layer == 'film':
            mat = self.film
            try:
                qx, qz = self.filmQ_r
            except:
                self.optimize()
                qx, qz = self.filmQ_r
        elif layer == 'sub':
            mat = self.sub
            try:
                qx, qz = self.subQ_r
            except:
                self.optimize()
                qx, qz = self.subQ_r

        norm = mat['norm']
        sys = mat['sys']
        h, k, l = self.hkl
        chi = self.chi

        if sys == 'hex':
            if chi != 0:
                resultant = np.sqrt(qx**2+qz**2)
                offset = chiAngle_hex(a, c, h, k, l, norm[0], norm[1], norm[2])
                angle = (chi+offset)*np.pi/180
                qx = resultant*np.sin(angle)
                qz = resultant*np.cos(angle)
            d_hk = 2*np.pi/qx
            d_l = 2*np.pi/qz
            a, c = lattice_hex(d_hk, d_l, h, k, l)
            print('a = {0:0.4f} angstroms \n c = {1:0.4f} angstroms'.format(a, c))
            return a, c

        elif sys == 'cubic':
            if chi != 0:
                resultant = np.sqrt(qx**2+qz**2)
                offset = chiAngle_cubic(h, k, l, norm[0], norm[1], norm[2])
                angle = (chi+offset)*np.pi/180
                qx = resultant*np.sin(angle)
                qz = resultant*np.cos(angle)

            d_inPlane = 2*np.pi/qx
            d_outOfPlane = 2*np.pi/qz
            a_inPlane = lattice_cubic(d_inPlane, h, k, 0)
            a_outOfPlane = lattice_cubic(d_outOfPlane, 0, 0, l)
            d_hkl = 2*np.pi/(np.sqrt(qx**2 + qz**2))

            print('a (in plane) = {0:0.4f} angstroms \n a (out of plane) = {1:0.4f} angstroms \n d spacing = {2:0.4f} angstroms'.format(
                a_inPlane, a_outOfPlane, d_hkl))
            return a_inPlane, a_outOfPlane

        elif sys == 'pc':
            if chi != 0:
                resultant = np.sqrt(qx**2+qz**2)
                offset = chiAngle_cubic(h, k, l, norm[0], norm[1], norm[2])
                angle = (chi+offset)*np.pi/180
                qx = resultant*np.sin(angle)
                qz = resultant*np.cos(angle)

            d_inPlane = 2*np.pi/qx
            d_outOfPlane = 2*np.pi/qz
            a_inPlane = lattice_cubic(d_inPlane, h, k, l)
            a_outOfPlane = lattice_cubic(d_outOfPlane, h, k, l)
            print('a (in plane) = {0:0.4f} angstroms \n a (out of plane) = {0:0.4f} angstroms'.format(
                a_inPlane, a_outOfPlane))
            return a_inPlane, a_outOfPlane
        
        

    def plot_Ideal_Points(self, RSM_fig, RSM_ax, peak='both'):

        if self.qx[-1, -1] < 0:
            xy_sub = tuple((self.subQ[0]*-1, self.subQ[1]))
            xy_film = tuple((self.filmQ[0]*-1, self.filmQ[1]))

        else:
            xy_sub = self.subQ
            xy_film = self.filmQ

        if peak == 'both':
            RSM_ax.scatter(*xy_sub, marker='*', color='k')
            RSM_ax.scatter(*xy_film, marker='*',
                                color='white', edgecolor='k')

        elif peak == 'sub':
            RSM_ax.scatter(*xy_sub, marker='*', color='k')
        elif peak == 'film':
            RSM_ax.scatter(*xy_film, marker='*',
                                color='white', edgecolor='k')
        RSM_fig.canvas.draw()
        
    def plot_Refined_Points(self, RSM_fig, RSM_ax, peak='both'):

        if self.qx[-1, -1] < 0:
            xy_sub = tuple((self.subQ_r[0]*-1, self.subQ_r[1]))
            xy_film = tuple((self.filmQ_r[0]*-1, self.filmQ_r[1]))

        else:
            xy_sub = self.subQ_r
            xy_film = self.filmQ_r

        if peak == 'both':
            RSM_ax.scatter(*xy_sub, marker='o', color='k')
            RSM_ax.scatter(*xy_film, marker='o',
                                color='white', edgecolor='k')

        elif peak == 'sub':
            RSM_ax.scatter(*xy_sub, marker='o', color='k')
        elif peak == 'film':
            RSM_ax.scatter(*xy_film, marker='o',
                                color='white', edgecolor='k')
        RSM_fig.canvas.draw()
        
    def annotatePlot(self, RSM_ax, keywords = 'all', lam=1.54059, point=None):
        if point == None:
            point=self.subQ
        acceptedKW = ['all', 'sample', 'ewald', 'rocking', 'analyzer', 'monochromator']
        xlim = RSM_ax.get_xlim()
        ylim = RSM_ax.get_ylim()
        r = 2*np.pi/lam
        xa = point[0]/2
        ya = point[1]/2
        a = np.sqrt(xa**2+ya**2)
        b = np.sqrt(r**2-a**2)
        x_c1 = xa + b*ya/a
        y_c1 = ya - b*xa/a
        x_c2 = xa - b*ya/a
        y_c2 = ya + b*xa/a
        if (x_c1 < 0) or (y_c1 < 0):
            center = [x_c2, y_c2]
        elif (x_c2 < 0) or (y_c2 < 0):
            center = [x_c1, y_c1]
        else:
            if xlim[0] > 0: #negative offset
                if x_c1 > xlim[0]:
                    center = [x_c1, y_c1]
                else:
                    center = [x_c2, y_c2]
            elif xlim[0] < 0: #positive offset
                if x_c1 < xlim[0]:
                    center = [x_c1, y_c1]
                else:
                    center = [x_c2, y_c2]
        for kw in keywords:
            if kw not in acceptedKW:
                raise ValueError('Inappropriate keyword. Accepted keywords are {0:s}'.format(acceptedKW))
            if kw == 'sample':
                ytop = (point[1]/point[0])*(xlim[1]-point[0])+point[1]
                RSM_ax.plot([0,xlim[1]], [0, ytop], '--', color='gray')
            elif (kw == 'ewald') or (kw == 'analyzer'):
                x_ewald = np.arange(xlim[0], xlim[1], .001)
                y_ewald = np.sqrt(r**2-(x_ewald - center[0])**2) + center[1]
                RSM_ax.plot(x_ewald, y_ewald, '--', color='gray')
            
            elif kw == 'rocking':
                x_rocking = np.arange(xlim[0], xlim[1], .001)
                r = np.sqrt(point[0]**2+point[1]**2)
                y_rocking = np.sqrt(r**2-(x_rocking)**2)
                RSM_ax.plot(x_rocking, y_rocking, '--', color='gray')
            elif kw == 'monochromator':
                x_mono = np.arange(xlim[0], xlim[1], .001)
                slope = (point[1]-center[1])/(point[0]-center[0])
                intercept = point[1]-slope*point[0]
                y_mono = x_mono*slope + intercept
                RSM_ax.plot(x_mono,y_mono, '--', color='gray')
            elif kw == 'all':
                self.annotatePlot(['sample', 'rocking', 'analyzer', 'monochromator'], lam=lam, point=point)
            RSM_ax.set_xlim(xlim)
            RSM_ax.set_ylim(ylim)
            
    def lineProfile(self, center, width, dimension='qz', fig=None, ax=None):
        if fig == None:
            fig = plt.figure()
        if ax == None:
            ax = fig.gca()
        
        if dimension == 'qz':
            mask1 = self.qx > center-width/2
            mask2 = self.qx < center+width/2
            mask = mask1 & mask2
            x = self.qz[mask]
            y = self.intensity[mask]
            ndx = x.argsort()
            ax.semilogy(x[ndx], y[ndx])
            ax.set_xlabel('Q$_{z}$ ($\AA^{-1}$)')
            ax.set_ylabel('Intensity (cps)')
        elif dimension == 'qx':
            print('Qx not yet implemented')
        return fig, ax, np.array([x[ndx], y[ndx]])
            


def calc_theta(Qx, Qz):
    resultant = np.sqrt(Qx**2 + Qz**2)
    theta = np.arcsin(lam*resultant/(4*np.pi))
    return theta


def calc_d(Qx, Qz):
    resultant = np.sqrt(Qx**2 + Qz**2)
    return 2*np.pi/resultant


def peak_find_2D(qx, qz, intensity, plot=False):
    tempCenter = []
    tempI = []
    for x, Ix in zip(qx, intensity):
        a, b = singlePeakFit(x, Ix)
        tempCenter.append(a)
        tempI.append(b)
    qx_max, Ix_max = singlePeakFit(
        np.array(tempCenter), np.array(tempI), plot=plot, report=False)

    tempCenter = []
    tempI = []
    for z, Iz in zip(qz, intensity):
        a, b = singlePeakFit(z, Iz)
        tempCenter.append(a)
        tempI.append(b)
    qz_max, Iz_max = singlePeakFit(
        np.array(tempCenter), np.array(tempI), plot=plot, report=False)

    return qx_max, qz_max


def calc_2theta(mat, h, k, l, relax=1, sub=None):
    if relax == 1:
        return (360/np.pi) * calc_theta(*calcQ(mat, h, k, l, mat['norm']))
    else:
        if sub == None:
            raise ValueError(
                'Cannot calculate relaxation without a substrate value')
        else:
            strainMat = mat.copy()
            if mat['sys'] == 'hex':
                if sub['sys'] == 'cubic':
                    sub_a = d_cubic(sub['a'], 1, 1, 0)
                elif sub['sys'] == 'rhomb':
                    sub_a = d_rhomb(sub['a'], sub['alpha'], 1, 1, 0)
                elif sub['sys'] == 'hex':
                    sub_a = sub['a']

                strainMat['a'] = sub_a*(1-relax) + mat['a']*relax
                try:
                    v = (mat['C13']/mat['C33'])/(1+mat['C13']/mat['C33'])
                except (ValueError):
                    print('elastic stiffness tensor values not defined in matDB')
                c_temp = -(2*v/(1-v))*((strainMat['a']-mat['a'])/mat['a'])
                strainMat['c'] = c_temp*mat['c'] + mat['c']
            return (360/np.pi) * calc_theta(*calcQ(strainMat, h, k, l, strainMat['norm']))


def symmetricCompNitride(tth, hkl, mat1, mat2, relax=1, sub=None):
    theta = tth*np.pi/360
    n = max(hkl)
    c = n*lam/(2*np.sin(theta))
    if relax == 1:
        comp = (mat1['c']-c)/(mat1['c'] - mat2['c'])
    elif relax == 0:
        if sub == None:
            raise ValueError(
                'Cannot calculate relaxation without a substrate value')

        guess_C13 = np.average([mat1['C13'], mat2['C13']])
        guess_C33 = np.average([mat1['C33'], mat2['C33']])
        residual = 1
        while residual > .0001:

            v = (guess_C13/guess_C33)/(1+guess_C13/guess_C33)
            if sub == AlN:
                comp = 1-((1-v)/(1+v))*((c - AlN['c'])/(mat1['c'] - AlN['c']))
            elif sub == GaN:
                comp = ((1-v)/(1+v))*((c - GaN['c'])/(mat2['c'] - GaN['c']))

            new_C13 = vegard(comp, mat1['C13'], mat2['C13'])
            new_C33 = vegard(comp, mat1['C33'], mat2['C33'])
            new_v = (new_C13/new_C33)/(1+new_C13/new_C33)
            if sub == AlN:
                new_comp = 1-((1-new_v)/(1+new_v)) * \
                    ((c - AlN['c'])/(mat1['c'] - AlN['c']))
            elif sub == GaN:
                new_comp = ((1-new_v)/(1+new_v)) * \
                    ((c - GaN['c'])/(mat2['c'] - GaN['c']))
            residual = (new_comp - comp)/comp
            guess_C13 = new_C13
            guess_C33 = new_C33
        comp = new_comp

    return comp


def symmetricCompNitrideCubic(tth, hkl, mat1, mat2, relax=1, sub=None):
    theta = tth*np.pi/360
    n = max(hkl)
    c = n*lam/(2*np.sin(theta))
    if relax == 1:
        comp = (mat1['c']-c)/(mat1['c'] - mat2['c'])
    elif relax == 0:
        if sub == None:
            raise ValueError(
                'Cannot calculate relaxation without a substrate value')

        guess_C13 = np.average([mat1['C13'], mat2['C13']])
        guess_C33 = np.average([mat1['C33'], mat2['C33']])
        residual = 1
        while residual > .0001:

            v = (guess_C13/guess_C33)/(1+guess_C13/guess_C33)
            if sub == AlN:
                comp = 1-((1-v)/(1+v))*((c - AlN['c'])/(mat1['c'] - AlN['c']))
            elif sub == GaN:
                comp = ((1-v)/(1+v))*((c - GaN['c'])/(mat2['c'] - GaN['c']))
            new_C13 = vegard(comp, mat1['C13'], mat2['C13'])
            new_C33 = vegard(comp, mat1['C33'], mat2['C33'])
            new_v = (new_C13/new_C33)/(1+new_C13/new_C33)
            if sub == AlN:
                new_comp = 1-((1-new_v)/(1+new_v)) * \
                    ((c - AlN['c'])/(mat1['c'] - AlN['c']))
            elif sub == GaN:
                new_comp = ((1-new_v)/(1+new_v)) * \
                    ((c - GaN['c'])/(mat2['c'] - GaN['c']))
            residual = (new_comp - comp)/comp
            guess_C13 = new_C13
            guess_C33 = new_C33
        comp = new_comp

    return comp

# from scipy.ndimage.filters import maximum_filter
# from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

# def detect_peaks(image, bg=0):
#     """
#     Takes an image and detect the peaks usingthe local maximum filter.
#     Returns a boolean mask of the peaks (i.e. 1 when
#     the pixel's value is the neighborhood maximum, 0 otherwise)
#
#
#     try fitting 1D voigt peaks, taking the center and fitting the center vs intensity
#     """
#
#     # define an 8-connected neighborhood
#     neighborhood = generate_binary_structure(2,2)
#
#     #apply the local maximum filter; all pixel of maximal value
#     #in their neighborhood are set to 1
#     local_max = maximum_filter(image, footprint=neighborhood)==image
#     #local_max is a mask that contains the peaks we are
#     #looking for, but also the background.
#     #In order to isolate the peaks we must remove the background from the mask.
#
#     #we create the mask of the background
#     background = (image==bg)
#
#     #a little technicality: we must erode the background in order to
#     #successfully subtract it form local_max, otherwise a line will
#     #appear along the background border (artifact of the local maximum filter)
#     eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
#
#     #we obtain the final mask, containing only peaks,
#     #by removing the background from the local_max mask (xor operation)
#     detected_peaks = local_max ^ eroded_background
#
#     return detected_peaks


def reciprocalSpacePlot(hklList, mat, fig=None, ax=None):
    qx = []
    qz = []
    for hkl in hklList:
        x, z = calcQ(mat, hkl[0], hkl[1], hkl[2])
        qx.append(x)
        qz.append(z)
    if fig == None:
        fig = plt.figure()
    if ax == None:
        ax = fig.gca()
    ax.scatter(qx, qz)
    ax.set_xlabel('Qx')
    ax.set_ylabel('Qz')
    fig.show()
    fig.tight_layout()


def calcThickFringe(tth_array, lam=1.54059):

    d = []
    w_array = tth_array/2
    for i, w in enumerate(w_array):
        if i < len(w_array)-1:
            d.append(
                lam/(2*(np.sin(w_array[i+1]*np.pi/180) - np.sin(w_array[i]*np.pi/180))))
    sep = np.diff(w_array)
    print('Peak Separation (w) \t Thickness(A)')
    for s, thick in zip(sep, d):
        print('{0:0.4f} \t\t\t\t {1:0.1f}'.format(s, thick))


def mergeRSM(RSM1, RSM2, side='right', sort=True, sortOn = 'tth'):
    '''
    function to merge two RSM parts into one 

    Parameters
    ----------
    RSM1 : RSM class with imported data
        parent for merging, the daughter will contain the metadata from this RSM
    RSM2 : RSM class with imported data
        data to merge into the first RSM class
    side : which side to append to. default right
    sort : boolean
        sort merged data on a certain value - default True
    sortOn: string
        value to sort on. default is 'tth' (two theta). Options are 'intensity', 'tth', 'w', 'qx', 'qz'
        
    Returns
    -------
    newRSM = merged dataset containing all metadata from first RSM

    '''
    sub = RSM1.sub
    film = RSM1.film
    hkl = RSM1.hkl
    newRSM = RSM(sub, film, hkl)

    arr1 = RSM1.tth.shape
    arr2 = RSM2.tth.shape
    height = arr1[0]
    width = arr1[1]+arr2[1]

    tth = np.full((height, width), np.nan)
    w = np.full_like(tth, np.nan)
    qx = np.full_like(tth, np.nan)
    qz = np.full_like(tth, np.nan)
    intensity = np.full_like(tth, np.nan)
    
    intensity[:arr1[0], :arr1[1]] = RSM1.intensity
    intensity[:arr2[0], arr1[1]:] = RSM2.intensity
    
    tth[:arr1[0], :arr1[1]] = RSM1.tth
    tth[:arr2[0], arr1[1]:] = RSM2.tth
    
    w[:arr1[0], :arr1[1]] = RSM1.w
    w[:arr2[0], arr1[1]:] = RSM2.w
    
    qx[:arr1[0], :arr1[1]] = RSM1.qx
    qx[:arr2[0], arr1[1]:] = RSM2.qx
    
    qz[:arr1[0], :arr1[1]] = RSM1.qz
    qz[:arr2[0], arr1[1]:] = RSM2.qz
    
    if sort:
        sortValueDict = {'intensity':intensity, 'tth':tth, 'w':w, 'qx':qx, 'qz':qz}
        sortValue = sortValueDict[sortOn]
        ind = np.unravel_index(np.argsort(sortValue, axis=None), sortValue.shape)
        intensity = intensity[ind].reshape(height, width)
        tth = tth[ind].reshape(height, width)
        w = w[ind].reshape(height, width)
        qx = qx[ind].reshape(height, width)
        qz = qz[ind].reshape(height, width)
        
            
                
    newRSM.intensity = intensity
    newRSM.tth = tth
    newRSM.w = w
    newRSM.qx = qx
    newRSM.qz = qz
    
    return newRSM

def mergeRSM_flatten(RSM1, RSM2, intensity_threshold=None, sort=True, sortOn='tth'):
    '''
    Function to merge two RSM parts into one by flattening, concatenating, sorting, and reshaping

    Parameters
    ----------
    RSM1 : RSM class with imported data
        Parent for merging, the daughter will contain the metadata from this RSM
    RSM2 : RSM class with imported data
        Data to merge into the first RSM class
    intensity_threshold: float or None
        Minimum intensity threshold. If specified, only data points above this threshold will be considered.
        If None, all data points will be considered. Default is None.
    sort : boolean
        Sort merged data on a certain value - default True
    sortOn: string
        Value to sort on. Default is 'tth' (two theta). Options are 'intensity', 'tth', 'w', 'qx', 'qz'

    Returns
    -------
    newRSM : Merged dataset containing all metadata from the first RSM
    '''

    # Extract metadata from RSM1
    sub = RSM1.sub
    film = RSM1.film
    hkl = RSM1.hkl
    newRSM = RSM(sub, film, hkl)

    # Flatten arrays
    flat_tth1 = RSM1.tth.flatten()
    flat_w1 = RSM1.w.flatten()
    flat_intensity1 = RSM1.intensity.flatten()

    flat_tth2 = RSM2.tth.flatten()
    flat_w2 = RSM2.w.flatten()
    flat_intensity2 = RSM2.intensity.flatten()

    # Concatenate flattened arrays
    flat_tth = np.concatenate([flat_tth1, flat_tth2])
    flat_w = np.concatenate([flat_w1, flat_w2])
    merged_intensity = np.concatenate([flat_intensity1, flat_intensity2])

    # Apply intensity threshold if specified
    if intensity_threshold is not None:
        below_threshold = merged_intensity <= intensity_threshold
        merged_intensity[below_threshold] = np.nan

    # Sort flattened data
    if sort:
        sortValueDict = {'intensity': merged_intensity, 'tth': flat_tth, 'w': flat_w}
        sortValue = sortValueDict[sortOn]
        ind = np.argsort(sortValue)
        flat_tth = flat_tth[ind]
        flat_w = flat_w[ind]
        merged_intensity = merged_intensity[ind]

    # Reshape the arrays
    rows = RSM1.w.shape[0] + RSM2.w.shape[0]
    tth = flat_tth.reshape((rows, -1))
    w = flat_w.reshape(tth.shape)
    intensity = merged_intensity.reshape(tth.shape)
    
    qx = K * (np.cos(w) - np.cos(tth - w))
    qz = K * (np.sin(w) + np.sin(tth - w))
    
    newRSM.tth = tth
    newRSM.w = w
    newRSM.intensity = intensity
    newRSM.qx = qx
    newRSM.qz = qz

    return newRSM
    
def latticeConstantMono_010(Qxz_hk, Qxz_kl, hkl1, hkl2, mat):
    
    h1, k1, l1 = hkl1
    h2, k2, l2 = hkl2
    
    d_hk_inPlane = 2*np.pi/Qxz_hk[0]
    d_hk_outOfPlane = 2*np.pi/Qxz_hk[1]
    
    d_kl_inPlane = 2*np.pi/Qxz_kl[0]
    d_kl_outOfPlane = 2*np.pi/Qxz_kl[1]
    
    beta = mat['beta']*np.pi/180
    
    b1 = np.sqrt(d_hk_outOfPlane**2 * k1**2)
    b2 = np.sqrt(d_kl_outOfPlane**2 * k2**2)
    
    if np.abs(b1-b2)/b1 > .002:
        print('b parameter discrepancy b1 = {0:0.4f} A \tb2 = {1:0.4f} A'.format(b1, b2))
    
    b = b1
    
    a = np.sqrt((d_hk_inPlane**2 * h1**2)/np.sin(beta)**2)
    c = np.sqrt((d_kl_inPlane**2 * l2**2)/np.sin(beta)**2)
    
    return a,b,c,beta*180/np.pi

