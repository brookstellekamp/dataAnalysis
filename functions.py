import numpy as np
from scipy.signal import gaussian, general_gaussian
from lmfit.models import VoigtModel, Gaussian2dModel, Model
from lmfit.lineshapes import lorentzian, gaussian
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def lorentzian2d(x, y, amplitude=1., centerx=0., centery=0., sigmax=1., sigmay=1., rotation=0):
    """Return a two dimensional lorentzian.

    The maximum of the peak occurs at ``centerx`` and ``centery``
    with widths ``sigmax`` and ``sigmay`` in the x and y directions
    respectively. The peak can be rotated by choosing the value of ``rotation``
    in radians.
    """
    xp = (x - centerx)*np.cos(rotation) - (y - centery)*np.sin(rotation)
    yp = (x - centerx)*np.sin(rotation) + (y - centery)*np.cos(rotation)
    R = (xp/sigmax)**2 + (yp/sigmay)**2

    return 2*amplitude*lorentzian(R)/(np.pi*sigmax*sigmay)
    
def gauss2d_tilt(x, y, amplitude=1., centerx=0., centery=0., sigmax=1., sigmay=1., rotation=0):
    """Return a two dimensional lorentzian.

    The maximum of the peak occurs at ``centerx`` and ``centery``
    with widths ``sigmax`` and ``sigmay`` in the x and y directions
    respectively. The peak can be rotated by choosing the value of ``rotation``
    in radians.
    """
    xp = (x - centerx)*np.cos(rotation) - (y - centery)*np.sin(rotation)
    yp = (x - centerx)*np.sin(rotation) + (y - centery)*np.cos(rotation)
    R = (xp/sigmax)**2 + (yp/sigmay)**2

    return 2*amplitude*gaussian(R)/(np.pi*sigmax*sigmay)

def singlePeakFit(x,y,shape='voigt', plot=False, report=False):
    '''
    Function to perform lmfit peak fit on a set of x,y data
    default shape - voigt
    returns - center, amplitude
    '''
    if shape == 'voigt':
        model = VoigtModel()
    else:
        raise ValueError('shape not recognized')
    params = model.guess(y, x=x)
    out = model.fit(y, params=params, x=x)
    if plot:
        out.plot()
        plt.show()
    if report:
        print(out.fit_report())
    return out.params['center'].value, out.params['amplitude'].value

def gauss_fit_2d(x,y,z, threshImin=None, plot=False):
    model = Gaussian2dModel()
    
    if x.ndim > 1:
        X = np.copy(x)
        x = X.flatten()
        Y = np.copy(y)
        y = Y.flatten()
        Z = np.copy(z)
        if threshImin:
            Z[Z<threshImin] = threshImin
        z = Z.flatten()
        
    params=model.guess(z,x,y)
    out = model.fit(z, x=x, y=y, params=params)
    if plot:
        fig = plt.figure()
        ax = fig.subplots(1,2)
        fit = model.func(X,Y,**out.best_values)
        ax[1].pcolor(X,Y,fit, norm=LogNorm(vmin=Z.min(), vmax=Z.max()), cmap=plt.cm.gist_heat_r)
        ax[0].pcolor(X,Y,Z, norm=LogNorm(vmin=Z.min(), vmax=Z.max()), cmap=plt.cm.gist_heat_r)
        ax[0].scatter(out.best_values['centerx'], out.best_values['centery'])
        fig.show()
    
    return out.best_values['centerx'], out.best_values['centery'], out.best_values['amplitude']
    
    
def gauss_tilt_fit_2d(x,y,z, plot=False, threshImin = None, p0={}):
    
    model = Model(gauss2d_tilt, independent_vars=['x', 'y'])
    
    if x.ndim > 1:
        X = np.copy(x)
        x = X.flatten()
        Y = np.copy(y)
        y = Y.flatten()
        if threshImin:
            z[z<threshImin] = threshImin
        Z = np.copy(z)
        z = Z.flatten()
    
    defaults = {    'amplitude': z.max()/1E3,
                    'centerx': x[np.argmax(z)],
                    'centery':y[np.argmax(z)],
                    'sigmax':.001,
                    'sigmay':.001,
                    'rotation':.1}
    
    for key in defaults.keys():
        if key in p0.keys():
            pass
        else:
            p0[key] = defaults[key]
        
  
    
    params = model.make_params(**p0)
    params['rotation'].set(min=0, max=np.pi/2)
    params['sigmax'].set(min=0)
    params['sigmay'].set( min=0)
    params['amplitude'].set(min=0)
    params['centerx'].set(min=x.min(), max=x.max())
    params['centery'].set(min=y.min(), max=y.max())
    
    out = model.fit(z, x=x, y=y, params=params)
    if plot:
        fig = plt.figure()
        ax = fig.subplots(1,2)
        fit = model.func(X,Y,**out.best_values)
        ax[1].pcolor(X,Y,fit, norm=LogNorm(vmin=Z.min(), vmax=Z.max()), cmap=plt.cm.gist_heat_r)
        ax[0].pcolor(X,Y,Z, norm=LogNorm(vmin=Z.min(), vmax=Z.max()), cmap=plt.cm.gist_heat_r)
        ax[0].scatter(out.best_values['centerx'], out.best_values['centery'])
        fig.show()
    
    return out.best_values['centerx'], out.best_values['centery'], out.best_values['amplitude']  
    
def lorentz_tilt_fit_2d(x,y,z, plot=False, threshImin = None, p0={}, fullReturn=False):
    model = Model(lorentzian2d, independent_vars=['x', 'y'])
   
    if x.ndim > 1:
        X = np.copy(x)
        x = X.flatten()
        Y = np.copy(y)
        y = Y.flatten()
        if threshImin:
            z[z<threshImin] = threshImin
        Z = np.copy(z)
        z = Z.flatten()
        
    defaults = {    'amplitude': z.max()/1E3,
                    'centerx': x[np.argmax(z)],
                    'centery':y[np.argmax(z)],
                    'sigmax':.001,
                    'sigmay':.001,
                    'rotation':.1}
        
    for key in defaults.keys():
        if key in p0.keys():
            pass
        else:
            p0[key] = defaults[key]
    
    params = model.make_params(**p0)
    params['rotation'].set(min=0, max=np.pi/2)
    params['sigmax'].set(min=0)
    params['sigmay'].set(min=0)
    params['amplitude'].set(min=0)
    params['centerx'].set(min=x.min(), max=x.max())
    params['centery'].set(min=y.min(), max=y.max())
    
    out = model.fit(z, x=x, y=y, params=params)
    if plot:
        fig = plt.figure()
        ax = fig.subplots(1,2)
        fit = model.func(X,Y,**out.best_values)
        ax[1].pcolor(X,Y,fit, norm=LogNorm(vmin=Z.min(), vmax=Z.max()), cmap=plt.cm.gist_heat_r)
        ax[0].pcolor(X,Y,Z, norm=LogNorm(vmin=Z.min(), vmax=Z.max()), cmap=plt.cm.gist_heat_r)
        ax[0].scatter(out.best_values['centerx'], out.best_values['centery'])
        fig.show()
    
    if fullReturn:
        print(out.fit_report())
        return out
    else:
        return out.best_values['centerx'], out.best_values['centery'], out.best_values['amplitude']

def mse(*args):
    return np.sqrt(np.sum([x**2 for x in args]))
    
def mse_m(errors, values):
    if len(errors) != len(values):
        raise ValueError('errors and values must be the same length')
    return np.sqrt(np.sum([(x/y)**2 for x, y in zip(errors, values)]))
    
def sheet_resistance(Rs, Ra, Rb):
    '''
    Numerical solver for van der pauw sheet resistance.
    Usage: scipy.optimize.fsolve(sheet_resistance, initial guess, args=(Ra, Rb))
    '''
    return np.exp(-np.pi*(Ra/Rs)) + np.exp(-np.pi*(Rb/Rs))-1
    
def gauss_amp(x, amplitude, center, sigma):
    return (amplitude/(sigma*np.sqrt(2*np.pi)))*np.exp(-.5*((x-center)/(sigma))**2)

    
def gauss(x, center, sigma):
    return (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-.5*((x-center)/(sigma))**2)

def offsetSine(x, f, phi, off):
    return np.sin(f*x + phi) + off

def fringes(x, c_f, F_f, c_s, F_s, N, delta, I0, sigma, sigma_n):
    
    broadening = 7 #gaussian width, in # of data points, to simulate instrumental broadening through convolution
    
    roughness=False
    
    use_thickness_variation = False
    
    q = 4*np.pi*np.sin(x*np.pi/180)/1.54059 #convert theta to q
    
    if roughness:
    
        roughness = np.exp(-0.5*sigma**2*(x-4*np.pi/c_f)**2) #Gaussian term for simulating roughness-based intensity decay. For Ga2O3 (020) n=2 so the multiplier for converting d to q is 2*2
    else:
        roughness=1
    
    if use_thickness_variation:
        n = 100 #number of points in the thickness gaussian envelope
        n_list = np.arange(N-n/2, N+n/2, 1)
        envelope = gauss(n_list, N, sigma_n)
        
        E = np.empty_like(q)
        for n_i, n_0 in zip(n_list, envelope):
            E = E + n_0*(F_f*((1 - np.exp(-1j*q*c_f*n_i))/(1-np.exp(-1j*q*c_f)))*roughness + F_s*((np.exp(-1j*q*(c_f*n_i + delta)))/(1-np.exp(-1j*q*c_s)))*roughness)
        E = E/n
    else:
        E = F_f*((1 - np.exp(-1j*q*c_f*N))/(1-np.exp(-1j*q*c_f)))*roughness + F_s*((np.exp(-1j*q*(c_f*N + delta)))/(1-np.exp(-1j*q*c_s)))*roughness #Base equation
        
    
    I = I0*abs(E)**2 #XRD Intensity is proportional to abs(E)**2
    
    broadened = np.convolve(I, gaussian(x.shape[0], broadening), mode='same') #convolve signal with broadening gaussian
    
    return broadened
    
def fringes_standard(x, c_f, F_f, c_s, F_s, N, I0):
    
    q = 4*np.pi*np.sin(x*np.pi/180)/1.54059 #convert theta to q
    

    E = F_f*((1 - np.exp(-1j*q*c_f*N))/(1-np.exp(-1j*q*c_f))) + F_s*((np.exp(-1j*q*(c_f*N)))/(1-np.exp(-1j*q*c_s))) #Base equation
        
    
    I = I0*abs(E)**2 #XRD Intensity is proportional to abs(E)**2
    
    
    return I

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y
    
def normalize(x, y, xmin=None, xmax=None):
    '''
    Function to normalize y-data. xmin and xmax values use lookups to define a normalization range based off of the x-data values and not indicies
    
    x and y are n-valued numpy arrays
    xmin and xmax are data values within the range of x
    
    returns: normalized y
    '''
    #determine if data is left-to-right or right-to-left
    if x[0] > x[1]:
        x_t = np.flip(x)
        y_t = np.flip(y)
    else:
        x_t = x
        y_t = y
    if xmin:
        x1 = np.searchsorted(x_t,xmin)
    else:
        x1=0
    if xmax:
        x2 = np.searchsorted(x_t,xmax)
    else:
        x2=-1
        
    nVal = max(y_t[x1:x2])
    return y/nVal
    
def vegard(x,a,b):
    return (1-x)*a+x*b

def vegard_quaternary(x,y,a,b,c):
    '''
    

    Parameters
    ----------
    x : fraction of a
        fraction of  a, where quaternary = A_x + B_y + C_1-x-y
    y : fraction of b
        fraction of b, where quaternary is given above
    a : compound A
        
    b : compound B
        
    c : compound C
        

    Returns
    -------
    a*x + b*y + c*(1-x-y)

    '''
    return a*x + b*y + c*(1-x-y)

def fftSmooth(X, sigma):
    XX = np.hstack((X,np.flip(X)))
    m = 1
    win = np.roll(general_gaussian(XX.shape[0], m, sigma), XX.shape[0]//2)
    fXX = np.fft.fft(XX)
    return np.real(np.fft.ifft(fXX*win))[:X.shape[0]]
    
def flatten_comprehension(matrix):
     return [item for row in matrix for item in row]