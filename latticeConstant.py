import numpy as np
import matplotlib.pyplot as plt
import uncertainties
from uncertainties import ufloat
from uncertainties.umath import sin, sqrt, tan, cos
from scipy.optimize import least_squares
from scipy import linalg
from dataAnalysis.fileIO import getPanalyticalXRD, rigakuXRD, slacXRD
from findpeaks import findpeaks
from lmfit.models import VoigtModel
import pandas as pd
from dataAnalysis.plotter import plotXRD

n0_ZGN = 1.659 #electrons per cubic angstrom
n0_AlN = 0.9583
lam = 1.54059 #angstroms CuKalpha



def lattice_hex(d_hk,d_l,h,k,l):
    c = np.sqrt(l**2*d_l**2)
    a = np.sqrt((4/3)*((h**2 + h*k + k**2)*d_hk**2))
    return a,c

    
def lattice_cubic(d,h,k,l):
    
    return np.sqrt((h**2+k**2+l**2)*d**2)
    
def d_tet_err(a,c,h,k,l):
    temp = ((h**2+k**2)/a**2) + l**2/c**2
    return np.sqrt(1/temp)   

def lattice_c_tet(angle, l, step=0.004):
    '''
    determine lattice parameter c, of a wurtzite unit cell, from a symmetric 00l scan. Higher angles (l) better
    step is step size in 2theta. See Moram and Vickers 2009
    '''
    n=1
    
    theta = angle*np.pi/360
    
    d_n = (n*1.54059)/(2*sin(theta))
    d_err = (d_n/tan(theta))*step/2
    d = ufloat(d_n.n, d_err.n+d_n.s)
    return l*d
    
def lattice_a_tet(angle, c, index, step=0.004):
    '''
    determine lattice parameter a, of a wurtzite unit cell, from an off-axis reflection after obtaining c from a symmetric reflection. Error from differentiating braggs law. Step is step size in 2theta. See Moram and Vickers 2009
    '''
    n=1
    
    theta = angle*np.pi/360
    
    d_n = (n*1.54056)/(2*sin(theta))
    d_err = (d_n/tan(theta))*step/2
    d = ufloat(d_n.n, d_err.n+d_n.s)
    a = sqrt((index[0]**2+index[1]**2)/((1/d**2) - (index[2]**2/c**2)))
    return a

def d_hex_ac(x, h, k, l):
    return 1/np.sqrt(4/3*(h**2+h*k+k**2)/(x[0]**2) + l**2/(x[1]**2))
    
def lattice_c_hex(angle, l, step=0.004):
    '''
    determine lattice parameter c, of a wurtzite unit cell, from a symmetric 00l scan. Higher angles (l) better
    step is step size in 2theta. See Moram and Vickers 2009
    '''
    n=1
    
    theta = angle*np.pi/360
    
    d_n = (n*1.54059)/(2*sin(theta))
    d_err = (d_n/tan(theta))*step/2
    try:
        d = ufloat(d_n.n, d_err.n+d_n.s)
    except(AttributeError):
        d = ufloat(d_n, d_err)
    return l*d
    
def lattice_a_hex(angle, c, index, step=0.004):
    '''
    determine lattice parameter a, of a wurtzite unit cell, from an off-axis reflection after obtaining c from a symmetric reflection. Error from differentiating braggs law. Step is step size in 2theta. See Moram and Vickers 2009
    '''
    n=1
    
    theta = angle*np.pi/360
    
    d_n = (n*1.54056)/(2*sin(theta))
    d_err = (d_n/tan(theta))*step/2
    try:
        d = ufloat(d_n.n, d_err.n+d_n.s)
    except(AttributeError):
        d = ufloat(d_n, d_err)
    a = sqrt((4/3)*(index[0]**2+index[1]**2+index[0]*index[1])/((1/d**2)-(index[2]**2/c**2)))
    return a
    
def residuals(x, h,k,l, d_meas, errors):
    return (d_hex_ac(x, h,k,l) - d_meas)/errors
    
def correct_symmetric(angle, n0, step=0.001):
    theta = np.pi*angle/360
    delta = 4.48E-6*n0*lam**2
    err = np.sqrt(((cos(theta.n)*lam*(step/2))/(2*sin(theta.n)**2))**2 + (3.7E-4/(2*sin(theta.n)))**2)  #https://doi.org/10.1063/1.2753122
    val = (lam/(2*sin(theta)))*(1 + delta/sin(2*theta))
    newErr = val.n*np.sqrt((val.s/val.n)**2 + (err/val.n)**2)
    return ufloat(val.n, newErr)
    
def correct_asymmetric(twotheta, omega, n0, step=0.001):
    theta = np.pi*twotheta/360
    omega = np.pi*omega/180
    err = np.sqrt(((cos(theta.n)*lam*(step/2))/(2*sin(theta.n)**2))**2 + (3.7E-4/(2*sin(theta.n)))**2)  #https://doi.org/10.1063/1.2753122
    delta = 4.48E-6*n0*lam**2
    val = (lam/(2*sin(theta)))*(1+(delta*cos(theta-omega))/(sin(2*theta-omega)*sin(omega)))
    newErr = val.n*np.sqrt((val.s/val.n)**2 + (err/val.n)**2)
    return ufloat(val.n, newErr)

def d_err(twotheta, step, n=1):
        theta = np.pi*twotheta/360
        d = n*lam/(2*sin(theta))
        err = np.sqrt(((cos(theta.n)*lam*(step/2))/(2*sin(theta.n)**2))**2 + (3.7E-4/(2*sin(theta.n)))**2)
        newErr = d.n*np.sqrt((d.s/d.n)**2 + (err/d.n)**2)
        return ufloat(d.n, newErr)
        
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
    
    
def d_cubic_lsq(x, h, k, l):
    '''
    function used for least squares regression through residuals_cubic, residuals_cubic_weight, and lattice_cubic_lsq
    
    x is a of the cubic cell which is minimized. 
    h,k,l are miller indicies
    '''
    
    return 1/np.sqrt((h**2+k**2+l**2)/x[0]**2)
    
def d_ortho_lsq(x, h, k, l):
    '''
    function used for least squares regression through residuals_ortho, residuals_ortho_weight, and lattice_ortho_lsq
    
    x is a tuple (a,b,c) of an orthorhombic cell which is minimized. 
    h,k,l are miller indicies
    '''
    
    return 1/np.sqrt(h**2/x[0]**2 + k**2/x[1]**2 + l**2/x[2]**2)
    
def d_rhomb_lsq(x, h, k, l):
    '''
    function used for least squares regression through residuals_rhomb, residuals_rhomb_weight, and lattice_rhomb_lsq
    
    x is a tuple (a,alpha) of a rhombohedral cell which is minimized. 
    h,k,l are miller indicies
    '''
    alpha = x[1]*np.pi/180
    return 1/np.sqrt(((h**2+k**2+l**2)*np.sin(alpha)**2 + 2*(h*k + k*l + l*h)*(np.cos(alpha)**2 - np.cos(alpha)))/(x[0]**2*(1-3*np.cos(alpha)**2+2*np.cos(alpha)**3)))    
    
def residuals_hex_weight(x, h,k,l, d_meas, errors):
    '''
    inverse weighting by measurement error
    '''
    return (d_hex_lsq(x, h,k,l) - d_meas)/errors
    
def residuals_tet_weight(x, h,k,l, d_meas, errors):
    '''
    inverse weighting by measurement error
    '''
    return (d_tet_lsq(x, h,k,l) - d_meas)/errors
    
def residuals_cubic_weight(x, h,k,l, d_meas, errors):
    '''
    inverse weighting by measurement error
    '''
    return (d_cubic_lsq(x, h,k,l) - d_meas)/errors
    
def residuals_ortho_weight(x, h,k,l, d_meas, errors):
    '''
    inverse weighting by measurement error
    '''
    return (d_ortho_lsq(x, h,k,l) - d_meas)/errors
    
def residuals_rhomb_weight(x, h,k,l, d_meas, errors):
    '''
    inverse weighting by measurement error
    '''
    return (d_rhomb_lsq(x, h,k,l) - d_meas)/errors

def residuals_hex(x, h,k,l, d_meas):
        return (d_hex_lsq(x, h,k,l) - d_meas)
        
def residuals_tet(x, h,k,l, d_meas):
        return (d_tet_lsq(x, h,k,l) - d_meas)
        
def residuals_cubic(x, h,k,l, d_meas):
        return (d_cubic_lsq(x, h,k,l) - d_meas)
        
def residuals_ortho(x, h,k,l, d_meas):
        return (d_ortho_lsq(x, h,k,l) - d_meas)
        
def residuals_rhomb(x, h,k,l, d_meas):
        return (d_rhomb_lsq(x, h,k,l) - d_meas)
        
#Why is the demoninator shape - 2? Shape should be number of datapoints?
def mse_hex(x, h,k,l,d_meas):
    return (np.sum(d_hex_lsq(x, h,k,l) - d_meas)**2)/(d_meas.shape[0] - 2)
    
def mse_tet(x, h,k,l,d_meas):
    return (np.sum(d_tet_lsq(x, h,k,l) - d_meas)**2)/(d_meas.shape[0])
    
def mse_ortho(x, h,k,l,d_meas):
    return (np.sum(d_ortho_lsq(x, h,k,l) - d_meas)**2)/(d_meas.shape[0] - 2)
    
def mse_rhomb(x, h,k,l,d_meas):
    return (np.sum(d_rhomb_lsq(x, h,k,l) - d_meas)**2)/(d_meas.shape[0])
    
def mse_cubic(x, h,k,l,d_meas):
    return (np.sum(d_cubic_lsq(x, h,k,l) - d_meas)**2)/(d_meas.shape[0])
    
def lsq_err(res, args, basis='hex'):
    '''
    calculate covariance matrix from least squares (scipy) jacobian
    return sqrt of covariance matrix diagonal
    '''
    J = res.jac
    hes = np.linalg.inv(J.T.dot(J))
    if basis == 'hex':
        mse = mse_hex(res.x, *args)
    elif basis == 'tet':
        mse = mse_tet(res.x, *args)
    elif basis == 'ortho':
        mse = mse_ortho(res.x, *args)
    elif basis == 'rhomb':
        mse = mse_rhomb(res.x, *args)
    elif basis == 'cubic':
        mse = mse_cubic(res.x, *args)
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
    w = S>tol
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
        args=(h_array,k_array,l_array, d_array, err)
        lsq = least_squares(residuals_hex_weight, x0, args=args)
        perr = lsq_err_weight(lsq)
    else:
        args=(h_array,k_array,l_array, d_array)
        lsq = least_squares(residuals_hex, x0, args=args)
        perr = lsq_err(lsq, args, basis='hex')
    
    a = ufloat(lsq.x[0], perr[0])
    c = ufloat(lsq.x[1], perr[1])
    print('a = {0:.4f}, c = {1:.4f}, c/a = {2:.4f}'.format(a, c, c/a))
    return a,c
    
def lattice_cubic_lsq(d_array, h_array, k_array, l_array, x0, err=None, report=True):
    '''
    least sq minimization to find a given an array of d values and corresponding hkl indicies
    d_array - numpy array of d values
    h array - numpy array of h indicies
    k array - numpy array of k indicies
    l array - numpy array of l indicies
    
    x0 - initial guess - a
    '''
    
    if not all(s.shape == d_array.shape for s in [d_array, h_array, k_array, l_array]):
        raise ValueError('not all lists have same length!')
    if err is not None:
        args=(h_array,k_array,l_array, d_array, err)
        lsq = least_squares(residuals_cubic_weight, x0, args=args)
        perr = lsq_err_weight(lsq)
    else:
        args=(h_array,k_array,l_array, d_array)
        lsq = least_squares(residuals_cubic, x0, args=args)
        perr = lsq_err(lsq, args, basis='cubic')
    
    a = ufloat(lsq.x[0], perr[0])
    if report:
        print('a = {0:.4f}'.format(a))
    return a

def lattice_tet_lsq(d_array, h_array, k_array, l_array, x0, err=None, report=True):
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
        args=(h_array,k_array,l_array, d_array, err)
        lsq = least_squares(residuals_tet_weight, x0, args=args)
        perr = lsq_err_weight(lsq)
    else:
        args=(h_array,k_array,l_array, d_array)
        lsq = least_squares(residuals_tet, x0, args=args)
        perr = lsq_err(lsq, args, basis='tet')
    
    a = ufloat(lsq.x[0], perr[0])
    c = ufloat(lsq.x[1], perr[1])
    if report:
        print('a = {0:.4f}, c = {1:.4f}, c/a = {2:.4f}'.format(a, c, c/a))
    return a,c
    
def lattice_ortho_lsq(d_array, h_array, k_array, l_array, x0, err=None, report=True):
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
        args=(h_array,k_array,l_array, d_array, err)
        lsq = least_squares(residuals_ortho_weight, x0, args=args)
        perr = lsq_err_weight(lsq)
    else:
        args=(h_array,k_array,l_array, d_array)
        lsq = least_squares(residuals_ortho, x0, args=args)
        perr = lsq_err(lsq, args, basis='ortho')
    
    a = ufloat(lsq.x[0], perr[0])
    b = ufloat(lsq.x[1], perr[1])
    c = ufloat(lsq.x[2], perr[2])
    if report:
        print('a = {0:.4f}, b = {1:.4f}, c = {2:.4f}'.format(a, b, c))
    return a,b,c
    
    
def lattice_rhomb_lsq(d_array, h_array, k_array, l_array, x0, err=None, bounds=None, report=True):
    '''
    least sq minimization to find a and c given an array of d values and corresponding hkl indicies
    d_array - numpy array of d values
    h array - numpy array of h indicies
    k array - numpy array of k indicies
    l array - numpy array of l indicies
    
    x0 - initial guess as a numpy array - [a,alpha]
    '''
    
    if not all(s.shape == d_array.shape for s in [d_array, h_array, k_array, l_array]):
        raise ValueError('not all lists have same length!')
    if err is not None:
        args=(h_array,k_array,l_array, d_array, err)
        if bounds:
            lsq = least_squares(residuals_rhomb_weight, x0, args=args, bounds = bounds)
        else:
            lsq = least_squares(residuals_rhomb_weight, x0, args=args)
        perr = lsq_err_weight(lsq)
    else:
        args=(h_array,k_array,l_array, d_array)
        if bounds:
            lsq = least_squares(residuals_rhomb, x0, args=args, bounds = bounds)
        else:
            lsq = least_squares(residuals_rhomb, x0, args=args)
        perr = lsq_err(lsq, args, basis='rhomb')
    
    a = ufloat(lsq.x[0], perr[0])
    alpha = ufloat(lsq.x[1], perr[1])
    if report:
        print('a = {0:.4f}, alpha = {1:.4f}'.format(a, alpha))
    return a,alpha
    
def d_bragg(tth, n=1, step = 0.01):
    if isinstance(tth, uncertainties.core.Variable):
        d_n = n*lam/(2*sin(tth*np.pi/360))
        d_err = (d_n.n/tan(tth.n/2))*step/2
        d_hkl = ufloat(d_n.n, sqrt(d_n.s**2 + d_err**2))
    else:
        d_n = n*lam/(2*np.sin(tth*np.pi/360))
        d_err = (d_n/tan(np.pi*tth/260))*step/2
        d_hkl = ufloat(d_n, d_err)
    return d_hkl

def pymatgenXRD(cif):
    '''
    

    Parameters
    ----------
    cif : string
        path to cif file.

    Returns
    -------
    pattern: pymatgen.analysis.diffraction.core.DiffractionPattern class
    material: pymatgen Structure class

    '''
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    from pymatgen.core import Structure
    calculator = XRDCalculator()
    material = Structure.from_file(cif)
    return calculator.get_pattern(material, two_theta_range=(0,120)), material

def planeMatcher(cif1, cif2, primary_hkl1, primary_hkl2, d_limit = 0.06, chi_limit = 4):
    '''
    

    Parameters
    ----------
    cif1 : string
        path to cif file for first crystal
    cif2 : string
        path to cif file for second crystal
    primary_hkl1 : tuple
        (h,k,l) tuple representing the surface normal (upward) orientation of the first crystal
    primary_hkl2 : tuple
        (h,k,l) tuple representing the surface normal (upward) orientation of the second crystal
    d_limit : float
        max separation in d. default 0.06
    chi_limit : float
        max separation in chi. default 4

    Returns
    -------
    None.

    '''
   
    from brooks.functions import flatten_comprehension
    from pymatgen.analysis.diffraction.tem import TEMCalculator
    import pandas as pd
    
    diffract1, struct1 = pymatgenXRD(cif1)
    diffract2, struct2 = pymatgenXRD(cif2)
    
    hkl1 = diffract1.hkls
    hkl2 = diffract2.hkls
    d1 = diffract1.d_hkls
    d2 = diffract2.d_hkls
    
    for a, b in zip(hkl1, d1):
        if len(a) == 1:
            a[0]['d_hkl'] = b
        if len(a) > 1:
            for a_sub in a:
                a_sub['d_hkl'] = b
    for a, b in zip(hkl2, d2):
        if len(a) == 1:
            a[0]['d_hkl'] = b
        if len(a) > 1:
            for a_sub in a:
                a_sub['d_hkl'] = b

    hkl1 = flatten_comprehension(hkl1)
    hkl2 = flatten_comprehension(hkl2)
    
    for i in [hkl1, hkl2]:
        if len(i[0]['hkl']) == 3:
            pass
        elif len(i[0]['hkl']) == 4:
            #reduce miller bravais to miller
            for el in i:
                mb_index = el['hkl']
                m_index = (mb_index[0], mb_index[1], mb_index[3])
                el['hkl'] = m_index

    

    
    calculator = TEMCalculator()
    for a in hkl1:
        a['chi'] = calculator.get_interplanar_angle(struct1, primary_hkl1, a['hkl'])
        
    for a in hkl2:
        a['chi'] = calculator.get_interplanar_angle(struct2, primary_hkl2, a['hkl'])
    
    
    
    
    d_hkl_1 = np.array([a['d_hkl'] for a in hkl1])
    d_hkl_2 = np.array([a['d_hkl'] for a in hkl2])
    chi1 = np.array([a['chi'] for a in hkl1])
    chi2 = np.array([a['chi'] for a in hkl2])
    
    diff_d = np.abs(np.subtract.outer(d_hkl_1, d_hkl_2))
    diff_chi = np.abs(np.subtract.outer(chi1, chi2))
    
    bool1 = diff_d < d_limit
    bool2 = diff_chi < chi_limit
    
    outputIndex = np.nonzero((bool1 == True) & (bool2 == True))
    hkl1_out = [hkl1[x]['hkl'] for x in outputIndex[0]]
    hkl2_out = [hkl2[x]['hkl'] for x in outputIndex[1]]
    d1_out = [hkl1[x]['d_hkl'] for x in outputIndex[0]]
    d2_out = [hkl2[x]['d_hkl'] for x in outputIndex[1]]
    chi1_out = [hkl1[x]['chi'] for x in outputIndex[0]]
    chi2_out = [hkl2[x]['chi'] for x in outputIndex[1]]
    d_diff_out = diff_d[outputIndex]
    d_chi_out = diff_chi[outputIndex]
    df = pd.DataFrame({'hkl_1':hkl1_out, 'd_hkl_1':d1_out, 'chi_1':chi1_out, 'hkl_2':hkl2_out, 'd_hkl_2':d2_out, 'chi_2':chi2_out, 'delta_d':d_diff_out, 'delta_chi':d_chi_out})
    df.sort_values(by=['delta_d', 'delta_chi'], inplace=True)
    print(df.to_string(index=False))
    return df
    
def fringeThicknessManual(peakList, lam=1.54059):
    if type(peakList) == list:
        peakList = np.array(peakList)
    peakRad = peakList*np.pi/360
    thickness = [lam/(2*(np.sin(peakRad[a])-np.sin(peakRad[a-1]))) for a in np.arange(1,len(peakList),1)]
    print('thickness (\u212B) = {0:0.2f} \u00B1 {1:0.2f}'.format(np.average(thickness), np.std(thickness)))
    print('\n\u00B02\u03B8-\u03C9\t\tFringe Period\tThickness (\u212B)')
    for a in np.arange(1,len(peakList),1):
        print('{0:0.4f} - {1:0.4f}\t\t{2:0.4f}\t{3:0.2f}'.format(peakList[a], peakList[a-1], peakList[a]-peakList[a-1], thickness[a-1]))
    
        
    return [lam/(2*(np.sin(peakRad[a])-np.sin(peakRad[a-1]))) for a in np.arange(1,len(peakList),1)]

def fringeThickness(dat, source=None, delimiter = ',', lam=1.54059, w=4, fitpeaks=False, plotFit=False, plot = False, lookahead=5):
    """
    Calculate the thickness of a thin film from X-ray diffraction data.

    Args:
        dat (str or array): Input data. If `source` is not provided, this should be a filepath 
                            to the data file. If `source` is provided, it should be an array 
                            containing X-ray diffraction data.
        source (str, optional): Source of the data. If provided, it should be one of the 
                                supported formats: 'ras', 'xrdml', 'data', 'slac', 'txt'. 
                                If not provided, the function will attempt to determine the 
                                source from the file extension.
        delimiter (str, optional): Delimiter used in the data file if `source` is 'txt'. 
                                   Default is ','.
        lam (float, optional): Wavelength of the X-rays used for the diffraction. 
                                Default is 1.54059 Å (copper Kα1 radiation).
        w (int, optional): Number of peaks on each side of the film peak to consider 
                           for calculating fringe thickness. Default is 4.
        fitpeaks (bool, optional): Whether to fit peaks using a Voigt model to determine 
                                    precise peak positions. Default is False.
        plotFit (bool, optional): Whether to plot the fits if fitting peaks. Default is False.
        plot (bool, optional): Whether to plot the X-ray diffraction data with the calculated 
                                fringe positions. Default is False.
        lookahead (int, optional): Number of points to look ahead for peak finding. 
                                    Default is 5.

    Returns:
        tuple: A tuple containing:
            - `values` (array): Fringe positions.
            - `avg_thickness` (float): Average thickness of the film in Å.
            - `std_thickness` (float): Standard deviation of the thickness measurements in Å.
    """
    
    #Note. For AlGaO on GaO (010) the composition is 0.4727*peak sub separation in omega

    # Supported file formats for XRD data
    supportedFormats = ['ras', 'xrdml', 'data', 'slac', 'txt']
    
    # Determine source if not provided based on file extension
    if source == None:
        source = dat.split('.')[-1]
    # Translate legacy source names to supported formats
    elif source == 'rigaku':
        source = 'ras'
    elif source == 'panalytical':
        source = 'xrdml'
    
    # Check if the provided source is supported
    if source not in supportedFormats:
        raise ValueError('Unsupported Format')
    
    # Load data based on the source format
    if source == 'ras':
        data = rigakuXRD(dat)
        x = data[:,0]
        y = data[:,1]
    elif source == 'slac':
        data = slacXRD(dat)
        x = data[:,1]
        y = data[:,2]
    elif source == 'xrdml':
        data = getPanalyticalXRD(dat)
        x = data[0]
        y = data[1]
    elif source == 'data':
        x = np.array(dat[0])
        y = np.array(dat[1])
    elif source == 'txt':
        data = np.genfromtxt(dat, delimiter=delimiter)
        x = data[:,0]
        y = data[:,1]

    # Ensure no zero values in y data
    minData = y[y>0].min()
    y[y==0] = minData
    
    # Find peaks in the data
    fp = findpeaks(lookahead=lookahead)
    result = fp.fit(y)
    
    # Separate peaks and valleys
    df = result['df'][result['df']['peak']==True]
    valley = result['df'][result['df']['valley']==True]
    peakValley = pd.concat((df, valley)).sort_index()
    
    # Identify substrate and film peaks
    subIDX = df.idxmax()['y']
    df_noSub = df.drop(axis=0, index=subIDX)
    filmIDX = df_noSub.idxmax()['y']
    idx = df_noSub.index.values
    filmIDX_num = np.argwhere(idx==filmIDX)[0][0]
    fringeIDX = np.concatenate((idx[filmIDX_num-w:filmIDX_num], idx[filmIDX_num+1:filmIDX_num+w+1]))
    
    # Fit peaks if required and determine fringe positions
    if fitpeaks:
        x_fringe = []
        for peak in fringeIDX:
            left = peakValley[peakValley['x']<peak].max()['x']
            right = peakValley[peakValley['x']>peak].min()['x']
            x_fit = x[left:right]
            y_fit = y[left:right]
            model = VoigtModel()
            pars=model.guess(y_fit, x_fit)
            out = model.fit(y_fit, pars, x=x_fit)
            x_fringe.append(out.params['center'].value)
            if plotFit:
                out.plot()
        x_fringe = np.array(x_fringe)
    else:
        x_fringe = x[fringeIDX]
    
    # Calculate thickness
    left = x_fringe[:w]
    right = x_fringe[w:]
    left_r = left*np.pi/360
    right_r = right*np.pi/360
    left_thickness = [lam/(2*(np.sin(left_r[a])-np.sin(left_r[a-1]))) for a in np.arange(1,w,1)]
    right_thickness = [lam/(2*(np.sin(right_r[a])-np.sin(right_r[a-1]))) for a in np.arange(1,w,1)]
    
    # Combine left and right thicknesses and positions
    values = np.concatenate((left, right))
    thickness = np.concatenate((left_thickness, right_thickness))
    
    # Print results
    print('\nSubstrate peak: {0:0.4f} \u00B02\u03B8-\u03C9'.format(x[subIDX]))
    print('Layer peak: {0:0.4f} \u00B02\u03B8-\u03C9'.format(x[filmIDX]))
    print('peak layer separation: {0:0.4f} \u00B02\u03C9'.format((x[subIDX]-x[filmIDX])/2))
    print('\n')
    print('thickness (\u212B) = {0:0.2f} \u00B1 {1:0.2f}'.format(np.average(thickness), np.std(thickness)))
    print('\n\u00B02\u03B8-\u03C9\t\tFringe Period\tThickness (\u212B)')
    for a in np.arange(1,w,1):
        print('{0:0.4f} - {1:0.4f}\t\t{2:0.4f}\t{3:0.2f}'.format(left[a], left[a-1], left[a]-left[a-1], left_thickness[a-1]))
    for a in np.arange(1,w,1):
        print('{0:0.4f} - {1:0.4f}\t\t{2:0.4f}\t{3:0.2f}'.format(right[a], right[a-1], right[a]-right[a-1], right_thickness[a-1]))
    
    # Plot if required
    if plot:
        fig, ax = plotXRD(dat)
        y_0 = y[np.searchsorted(x, values)]
        for xval, yval in zip(values, y_0):
            ax.plot([xval, xval], [minData, yval], '--', color='gray')
    
    return values, np.average(thickness), np.std(thickness)
    
    
