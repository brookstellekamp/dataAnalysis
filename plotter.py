#various plotting functions

from brooks.fileIO import rigakuXRD, slacXRD, bandEng, getPanalyticalXRD
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from lmfit.models import LorentzianModel, VoigtModel, GaussianModel, Pearson7Model
from matplotlib_scalebar.scalebar import ScaleBar
from io import BytesIO
from PIL import Image
from uncertainties import ufloat
from scipy import constants as c
from scipy import stats


AlN = { 'a':3.112,
        'c':4.98,
        'C13':99,
        'C33':389}
GaN = { 'a':3.189, 
        'c':5.185,
        'C13':106,
        'C33':398}
ZGN = { 'a':3.193, 
        'c':5.187,
        'C13':103,
        'C33':401}
        
global hc 
hc = (c.h/c.e)*c.c/1E-9

def plotXRD(*inputs, fig=None, ax=None, xlim=None, source=None, offset=1, normalize=False, delimiter = ','):
    '''
    plotter function that takes in an arbitrary number of xrd data files and plots them
    
    inputs: path to data file - string 
                if source = 'data', then input is a tuple or list of tuples
    
    fig: matplotlib figure object in which to plot the data. If provided, it is assumed an axes object will also be provided. Default None
    
    ax: matplotlib axes object in which to plot the data. If provided, it is assumed a figure object has also been supplied. Default None
    
    source - tool providing data, which points to different data importers. Default is none. input files are parsed by file extension. Currently supports 'ras' and 'xrdml' for rigaku and panalytical inputs. Also accepts 'data' for a numpy array of [2theta, counts]
    '''
    
    supportedFormats = ['ras', 'xrdml', 'data', 'slac', 'txt']
    source_input = source
    plt.style.use('publication')
    delta = 1
    i = 0
    z = 1
    if fig == None:
        fig = plt.figure()
    if ax == None:
        ax = fig.gca()
    for dat in inputs:
        if source == None:
            source = dat.split('.')[-1]
        elif source == 'rigaku':
            source = 'ras'
        elif source == 'panalytical':
            source = 'xrdml'
        if source not in supportedFormats:
            raise ValueError('Unsupported Format')
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
        minData = y[y>0].min()
        y[y==0]=minData
        if normalize:
            y = y/np.max(y)
        ax.semilogy(x, y*delta, zorder = z)
        
        
        i+=1
        z = z - 0.1
        delta = delta + offset**i
        source = source_input
    ax.set_xlim(left=x[0], right=x[-1])
    if source == 'slac':
        ax.set_xlabel('q ($\AA^{-1}$)')
    else:
        ax.set_xlabel(r'$2\theta-\omega$ (degrees)')
    ax.set_ylabel('Intensity (arb.)')
    if xlim:
        ax.set_xlim(left=xlim[0], right = xlim[1])
    plt.show()
    return fig, ax
    
#%%
def plotPF(sourceFile, fig=None, ax = None, colormap = plt.cm.gist_heat_r, threshImin = None, threshImax = None, show=True):
    
    data = rigakuXRD(sourceFile, scanType='poleFigure')
    dmin = data[2][data[2]>0].min()
    data[2][data[2]<=0] = dmin
    if fig == None:
        fig = plt.figure()
    if ax == None:
        ax = fig.add_subplot(projection='polar')
    if threshImin is not None:
        zmin = np.log10(threshImin).round()
        vmin = threshImin
    else:
        zmin = np.log10(data[2].min()).round()
        vmin = data[2].min()
        if vmin <= 0:
            vmin = .00001
    if threshImax is not None:
        zmax = np.log10(threshImax).round()
        vmax = threshImax
    else:
        zmax = np.log10(data[2].max()).round()
        vmax = data[2].max()
    
    
    cbar = ax.contourf(data[1], data[0], data[2], norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap=colormap, levels=np.logspace(zmin,zmax, num=50))
    
    #plt.colorbar(cbar)
    if show:
        fig.show()
    
    return fig, ax
        
    
#%%
def plotBand(*filename, offset=0):
    '''
    plotter function that takes in an arbitrary number of bandEng output files and plots them
    
    filename: path to bandEng file. string
    
    offset: Energy offset for plotting successive data files, waterfall style. Defualt 0
    '''
    plt.style.use('publication')
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    fig, ax = plt.subplots()
    delta = 0
    for i, file in enumerate(filename):
        data = bandEng(file)
        x = data[:,0]
        Ev = data[:,1]
        Ec = data[:,2]
        ax.plot(x/10,Ec + delta, color=colors[np.mod(i,10)])
        ax.plot(x/10,Ev + delta, color=colors[np.mod(i,10)], label='_nolegend_')
        delta = delta + offset
    ax.set_xlabel('Position (nm)')
    ax.set_ylabel('Energy (eV)')
    plt.show()

def XRD_peak_fit(data, plot=False, input='2Theta', source=None, report=True, annotate=True, peaks=1, fig=None, ax=None, fullReturn=None, plotComponents=True, shape='voigt', x0=None):
    '''
    function to fit a peak and return the center position. Can optionally plot the data and fit
    
    data - can be a tuple of 1D arrays (x, y) or a data file - indicated by the source kwarg which defaults to data (x,y)
    
    input - x val units - can be 'degrees' (default), 'arcsec', 'eV', or '2theta'
    
    source - tool providing data, which points to different data importers. Default is none. input files are parsed by file extension. Currently supports 'ras' and 'xrdml' for rigaku and panalytical inputs. Also accepts 'slac' as a source for SLAC style csv, and 'data' for a numpy array of [omega, counts]
    
    report  - boolean - print out fit report. Default True
    
    annotate - boolean - annotate plot with FWHM and Chi^2, default True
    
    plot - kwarg to plot the fit with the data - boolean - default False
    
    peaks - optional kwarg to fit multiple peaks
    
    shape - how to fit the curve. options are lorentzian (default), gaussian, voigt, and pearson (which represents the pearsonVII dist.)
    
    x0 - initial params for center positions, if desired. default None
    
    returns - peak center (ufloat)
    ''' 
    
    supportedFormats = ['ras', 'xrdml', 'data']
    
    
    if source == None:
        source = data.split('.')[-1]
    elif source == 'rigaku':
        source = 'ras'
    elif source == 'panalytical':
        source = 'xrdml'
    if source not in supportedFormats:
        raise ValueError('Unsupported Format')
    
    if source == 'data':
        x = data[0]
        y = data[1]
    elif source == 'ras':
        dat = rigakuXRD(data)
        x = dat[:,0]
        y = dat[:,1]
    elif source == 'xrdml': 
        dat = getPanalyticalXRD(data)
        x = dat[0]
        y = dat[1]
    if peaks == 1:
        if shape == 'lorentzian' or shape == 'Lorentzian' or shape == 'lorentz':
            mod = LorentzianModel()
        elif shape == 'voigt':
            mod = VoigtModel()
        elif shape == 'gauss' or shape == 'gaussian':
            mod = GaussianModel()
        elif shape == 'pearson' or shape == 'pearson7' or shape == 'pearsonVII':
            mod = Pearson7Model()
        else:
            raise ValueError('Shape type not understood')
        pars = mod.guess(y, x=x)
        if x0:
            if len(x0) != 1:
                raise ValueError('initial parameter size x0 not equal to number of peaks')
            else:
                pars['center'].set(x0[0])
        out = mod.fit(y, pars, x=x)
        
        if report:
            print(out.fit_report())
        
        try:
            center = ufloat(out.params['center'].value, out.params['center'].stderr)
        except(AttributeError):
            center = ufloat(out.params['center'].value, np.nan)
        
        if plot:
            if fig == None:
                fig = plt.figure()
            if ax == None:
                ax = fig.gca()
            plt.style.use('publication')
            
            ax.plot(x,y, 'o', markeredgecolor='grey', alpha = 0.5, markerfacecolor='none', zorder=0.3)
            ax.plot(x,out.best_fit)
            if input == '2Theta':
                ax.set_xlabel('2$\theta$-$\omega$ (degrees)')
            else:
                print('Input type not recognized')
            ax.set_ylabel('Intensity (arb.)')
            
            if annotate:
                if input == '2theta':
                    ax.text(.95, .95, 'center = {0:.2f} degrees\nChi squared = {1:.2f}'.format(center, out.chisqr), ha='right', va='top', transform = ax.transAxes)
               
            fig.show()
        
        if fullReturn:
            return center, out
        else:
            return center
            
            
    if peaks == 2:
        if shape == 'lorentzian' or shape == 'Lorentzian' or shape == 'lorentz':
            mod = LorentzianModel(prefix='L1_') + LorentzianModel(prefix='L2_')
        elif shape == 'voigt':
            mod = VoigtModel(prefix='L1_') + VoigtModel(prefix='L2_')
        elif shape == 'gauss' or shape == 'gaussian':
            mod = GaussianModel(prefix='L1_') + GaussianModel(prefix='L2_')
        elif shape == 'pearson' or shape == 'pearson7' or shape == 'pearsonVII':
            mod = Pearson7Model(prefix='L1_') + Pearson7Model(prefix='L2_')
            y[y<=0] = y[y>0].min()
        else:
            raise ValueError('Shape type not understood')
        pars = mod.make_params()
        if x0:
            if len(x0) != 2:
                raise ValueError('initial parameter size x0 not equal to number of peaks')
            else:
                pars['L1_center'].set(x0[0])
                pars['L2_center'].set(x0[1])
        pars['L1_amplitude'].set(min=0)
        pars['L2_amplitude'].set(min=0)
        out = mod.fit(y, pars, x=x)
        
        if report:
            print(out.fit_report())
        
        try:
            center1 = ufloat(out.params['L1_center'].value, out.params['L1_center'].stderr)
        except(AttributeError):
            center1 = ufloat(out.params['L1_center'].value, np.nan)
        try:
            center2 = ufloat(out.params['L2_center'].value, out.params['L2_center'].stderr)
        except(AttributeError):
            center2 = ufloat(out.params['L2_center'].value, np.nan)
        
        if plot:
            if fig == None:
                fig = plt.figure()
            if ax == None:
                ax = fig.gca()
            plt.style.use('publication')
            
            ax.plot(x,y, 'o', markeredgecolor='grey', alpha = 0.5, markerfacecolor='none', zorder=0.3)
            ax.plot(x,out.best_fit)
            if input == '2Theta':
                ax.set_xlabel('2$\theta$-$\omega$ (degrees)')
            else:
                print('Input type not recognized')
            ax.set_ylabel('Intensity (arb.)')
            if plotComponents:
                comps = out.eval_components(x=x)
                ax.plot(x, comps['L1_'])
                ax.plot(x, comps['L2_'])
            
            if annotate:
                if input == '2theta':
                    ax.text(.95, .95, 'center 1 = {0:.2f} degrees \ncenter 2 = {0:.2f} \nChi squared = {1:.2f}'.format(center1, center2, out.chisqr), ha='right', va='top', transform = ax.transAxes)
               
            fig.show()
            
        if fullReturn:
            return center1, center2, out
        else:
            return center1, center2
        

def FWHM(data, plot=False, input='degrees', source = None, report=True, annotate = True, peaks=1, fig=None, ax=None, fullReturn=False, plotComponents=True, shape='voigt', x0 = None):
    '''
    function to fit a peak to a xrd rocking curve and return the full width at half maximum (FWHM). Can optionally plot the data and fit
    
    data - can be a tuple of 1D arrays (x, y) or a data file - indicated by the source kwarg which defaults to data (x,y)
    
    input - x val units - can be 'degrees' (default), 'arcsec', 'eV', or '2theta'
    
    source - tool providing data, which points to different data importers. Default is none. input files are parsed by file extension. Currently supports 'ras' and 'xrdml' for rigaku and panalytical inputs. Also accepts 'slac' as a source for SLAC style csv, and 'data' for a numpy array of [omega, counts]
    
    report  - boolean - print out fit report. Default True
    
    annotate - boolean - annotate plot with FWHM and Chi^2, default True
    
    plot - kwarg to plot the fit with the data - boolean - default False
    
    peaks - optional kwarg to fit multiple peaks
    
    shape - how to fit the curve. options are lorentzian (default), gaussian, voigt, and pearson (which represents the pearsonVII dist.)
    
    returns - fwhm value (ufloat)
    '''
    
    
    supportedFormats = ['ras', 'xrdml', 'data']
    
    if x0:
        if len(x0) == 0:
            x0=None
    
    if source == None:
        source = data.split('.')[-1]
    elif source == 'rigaku':
        source = 'ras'
    elif source == 'panalytical':
        source = 'xrdml'
    if source not in supportedFormats:
        raise ValueError('Unsupported Format')
        
    
    if source == 'data':
        x = data[0]
        y = data[1]
    elif source == 'ras':
        dat = rigakuXRD(data)
        x = dat[:,0]
        y = dat[:,1]
    elif source == 'xrdml': 
        dat = getPanalyticalXRD(data)
        x = dat[0]
        y = dat[1]
    if peaks == 1:
        if shape == 'lorentzian' or shape == 'Lorentzian' or shape == 'lorentz':
            mod = LorentzianModel()
        elif shape == 'voigt':
            mod = VoigtModel()
        elif shape == 'gauss' or shape == 'gaussian':
            mod = GaussianModel()
        elif shape == 'pearson' or shape == 'pearson7' or shape == 'pearsonVII':
            mod = Pearson7Model()
        else:
            raise ValueError('Shape type not understood')
        pars = mod.guess(y, x=x)
        
        if x0:
            for key, value in x0.items():
                pars[key].set(value)
        
        out = mod.fit(y, pars, x=x)
        if report:
            print(out.fit_report())
        
        if input == 'degrees':
            try:
                fwhm = ufloat(out.params['fwhm'].value, out.params['fwhm'].stderr)*3600
            except(AttributeError):
                fwhm = ufloat(out.params['fwhm'].value, np.nan)*3600
        else:
            try:
                fwhm = ufloat(out.params['fwhm'].value, out.params['fwhm'].stderr)
            except(AttributeError):
                fwhm = ufloat(out.params['fwhm'].value, np.nan)
        
        
            
        if plot:
            if fig == None:
                fig = plt.figure()
            if ax == None:
                ax = fig.gca()
            plt.style.use('publication')
            
            ax.plot(x,y, 'o', markeredgecolor='grey', alpha = 0.5, markerfacecolor='none', zorder=0.3)
            ax.plot(x,out.best_fit)
            if input == 'eV':
                ax.set_xlabel('Energy (eV')
            elif input == '2theta':
                ax.set_xlabel('2$\theta$-$\omega$ (degrees)')
            else:
                ax.set_xlabel('$\omega$ ('+input+')')
            ax.set_ylabel('Intensity (arb.)')
            
            if annotate:
                if input == 'degrees':
                    ax.text(.95, .95, 'FWHM = {0:.0f} arcsec\nChi squared = {1:.0f}'.format(fwhm, out.chisqr), ha='right', va='top', transform = ax.transAxes)
                elif input == '2theta':
                    ax.text(.95, .95, 'FWHM = {0:.2f} degrees\nChi squared = {1:.2f}'.format(fwhm, out.chisqr), ha='right', va='top', transform = ax.transAxes)
                elif input == 'eV':
                    ax.text(.95, .95, 'FWHM = {0:.0f} eV\nChi squared = {1:.0f}'.format(fwhm, out.chisqr), ha='right', va='top', transform = ax.transAxes)
            fig.show()
            
        if fullReturn:
            return fwhm, out
        else:
            return fwhm
    
    if peaks == 2:
        
        if shape == 'lorentzian' or shape == 'Lorentzian' or shape == 'lorentz':
            mod = LorentzianModel(prefix='L1_') + LorentzianModel(prefix='L2_')
        elif shape == 'voigt':
            mod = VoigtModel(prefix='L1_') + VoigtModel(prefix='L2_')
        elif shape == 'gauss' or shape == 'gaussian':
            mod = GaussianModel(prefix='L1_') + GaussianModel(prefix='L2_')
        elif shape == 'pearson' or shape == 'pearson7' or shape == 'pearsonVII':
            mod = Pearson7Model(prefix='L1_') + Pearson7Model(prefix='L2_')
            y[y<=0] = y[y>0].min()
        else:
            raise ValueError('Shape type not understood')
            
        pars = mod.make_params()
        
        if x0:
            for key, value in x0.items():
                if len(value) != peaks:
                    raise ValueError('number of initial conditions must match number of peaks')
                pars['L1_'+key].set(value[0])
                pars['L2_'+key].set(value[0])
        
        out = mod.fit(y, pars, x=x)
        if report:
            print(out.fit_report())
        
        if input == 'degrees':
            try:
                fwhm1 = ufloat(out.params['L1_fwhm'].value, out.params['L1_fwhm'].stderr)*3600
            except(AttributeError):
                fwhm1 = ufloat(out.params['L1_fwhm'].value, np.nan)*3600
            try:
                fwhm2 = ufloat(out.params['L2_fwhm'].value, out.params['L2_fwhm'].stderr)*3600
            except(AttributeError):
                fwhm2 = ufloat(out.params['L2_fwhm'].value, np.nan)*3600
        else:
            try:
                fwhm1 = ufloat(out.params['L1_fwhm'].value, out.params['L1_fwhm'].stderr)
            except(AttributeError):
                fwhm1 = ufloat(out.params['L1_fwhm'].value, np.nan)
            try:
                fwhm2 = ufloat(out.params['L2_fwhm'].value, out.params['L2_fwhm'].stderr)
            except(AttributeError):
                fwhm2 = ufloat(out.params['L2_fwhm'].value, np.nan)
        
            
        if plot:
            if fig == None:
                fig = plt.figure()
            if ax == None:
                ax = fig.gca()
            plt.style.use('publication')
            
            ax.plot(x,y,'o', markeredgecolor='grey', alpha = 0.5, markerfacecolor='none', zorder=0.3)
            ax.plot(x,out.best_fit)
            if input == 'eV':
                ax.set_xlabel('Energy (eV')
            elif input == '2theta':
                ax.set_xlabel('2$\theta$-$\omega$ (degrees)')
            else:
                ax.set_xlabel('$\omega$ ('+input+')')
            ax.set_ylabel('Intensity (arb.)')
            if plotComponents:
                comps = out.eval_components(x=x)
                ax.plot(x, comps['L1_'])
                ax.plot(x, comps['L2_'])
            
            if annotate:
                if input == 'degrees':
                    ax.text(.95, .95, 'FWHM_1 = {0:.0f} arcsec'.format(fwhm1)+'\nFWHM_2 = {0:.0f} arcsec'.format(fwhm2)+'\nChi squared = {0:.0f}'.format(out.chisqr), ha='right', va='top', transform = ax.transAxes)
                elif input == '2theta':
                    ax.text(.95, .95, 'FWHM_1 = {0:.0f} degrees'.format(fwhm1)+'\nFWHM_2 = {0:.0f} degrees'.format(fwhm2)+'\nChi squared = {0:.0f}'.format(out.chisqr), ha='right', va='top', transform = ax.transAxes)
                elif input == 'eV':
                    ax.text(.95, .95, 'FWHM_1 = {0:.0f} eV'.format(fwhm1)+'\nFWHM_2 = {0:.0f} eV'.format(fwhm2)+'\nChi squared = {0:.0f}'.format(out.chisqr), ha='right', va='top', transform = ax.transAxes)
            fig.show()
            
        if fullReturn:
            return np.array([fwhm1, fwhm2]) if out.values['L1_center'] < out.values['L2_center'] else np.array([fwhm2, fwhm1]), out
        else:
            return np.array([fwhm1, fwhm2]) if out.values['L1_center'] < out.values['L2_center'] else np.array([fwhm2, fwhm1])
        
    if peaks == 3:
        
        if shape == 'lorentzian' or shape == 'Lorentzian' or shape == 'lorentz':
            mod = LorentzianModel(prefix='L1_') + LorentzianModel(prefix='L2_') + LorentzianModel(prefix='L3_')
        elif shape == 'voigt':
            mod = VoigtModel(prefix='L1_') + VoigtModel(prefix='L2_') + VoigtModel(prefix='L3_')
        elif shape == 'gauss' or shape == 'gaussian':
            mod = GaussianModel(prefix='L1_') + GaussianModel(prefix='L2_') + GaussianModel(prefix='L3_')
        elif shape == 'pearson' or shape == 'pearson7' or shape == 'pearsonVII':
            mod = Pearson7Model(prefix='L1_') + Pearson7Model(prefix='L2_') + Pearson7Model(prefix='L3_')
            y[y<=0] = y[y>0].min()
        else:
            raise ValueError('Shape type not understood')
            
        pars = mod.make_params()
        out = mod.fit(y, pars, x=x)
        if report:
            print(out.fit_report())
        
        if input == 'degrees':
            fwhm1 = ufloat(out.params['L1_fwhm'].value, out.params['L1_fwhm'].stderr)*3600
            fwhm2 = ufloat(out.params['L2_fwhm'].value, out.params['L2_fwhm'].stderr)*3600
            fwhm3 = ufloat(out.params['L3_fwhm'].value, out.params['L3_fwhm'].stderr)*3600
        elif input == 'arcsec':
            fwhm1 = ufloat(out.params['L1_fwhm'].value, out.params['L1_fwhm'].stderr)
            fwhm2 = ufloat(out.params['L2_fwhm'].value, out.params['L2_fwhm'].stderr)
            fwhm3 = ufloat(out.params['L3_fwhm'].value, out.params['L3_fwhm'].stderr)
        elif input == 'eV':
            fwhm1 = ufloat(out.params['L1_fwhm'].value, out.params['L1_fwhm'].stderr)
            fwhm2 = ufloat(out.params['L2_fwhm'].value, out.params['L2_fwhm'].stderr)
            fwhm3 = ufloat(out.params['L3_fwhm'].value, out.params['L3_fwhm'].stderr)
            
        if plot:
            if fig == None:
                fig = plt.figure()
            if ax == None:
                ax = fig.gca()
            plt.style.use('publication')
            
            ax.plot(x,y,'o', markeredgecolor='grey', alpha = 0.5, markerfacecolor='none', zorder=0.3)
            ax.plot(x,out.best_fit)
            if input == 'eV':
                ax.set_xlabel('Energy (eV')
            else:
                ax.set_xlabel('$\omega$ ('+input+')')
            ax.set_ylabel('Intensity (arb.)')
            if plotComponents:
                comps = out.eval_components(x=x)
                ax.plot(x, comps['L1_'])
                ax.plot(x, comps['L2_'])
                ax.plot(x, comps['L3_'])
            
            if annotate:
                ax.text(.95, .95, 'FWHM_1 = {0:.0f} arcsec'.format(fwhm1)+'\nFWHM_2 = {0:.0f} arcsec'.format(fwhm2)+'\nFWHM_3 = {0:.0f} arcsec'.format(fwhm3)+'\nChi squared = {0:.0f}'.format(out.chisqr), ha='right', va='top', transform = ax.transAxes)
            fig.show()
            
        if fullReturn:
            #note needs to be fixed to set order from smallest to largest omega
            return np.array([fwhm1, fwhm2, fwhm3]), out
        else:
            return np.array([fwhm1, fwhm2, fwhm3])
        
def plotRC(*filename, fig = None, ax = None, xlim=None, annotate=False, source=None, offset = 1, units='degrees'):
    
    supportedFormats = ['ras', 'xrdml', 'data']
    
    
    
    
    
    if fig == None:
        fig = plt.figure()
    if ax == None:
        ax = fig.gca()

    plt.style.use('publication')
    
    delta = 1
    i = 0
    z = 1
    for file in filename:
        if source == None:
            source = file.split('.')[-1]
        elif source == 'rigaku':
            source = 'ras'
        elif source == 'panalytical':
            source = 'xrdml'
        if source not in supportedFormats:
            raise ValueError('Unsupported Format')
        if source == 'ras':
            data = rigakuXRD(file)
            x = data[:,0]
            y = data[:,1]
        elif source == 'xrdml':
            data = getPanalyticalXRD(file)
            x = data[0]
            y = data[1]
        elif source == 'data':
            x = data[0]
            y = data[1]
        else:
            print('unknown source')
            pass
        if units == 'degrees':
            ax.set_xlabel(r'$\omega$ (degrees)')
        if units == 'relative arcsec':
            x_0 = x[int(len(x)/2)]
            x = 3600*(x-x_0)
            ax.set_xlabel('tilt (arcsec)')
        ax.semilogy(x, y * delta, zorder = z)
        
        ax.set_ylabel('Intensity (arb.)')
        if xlim:
            ax.set_xlim(left=xlim[0], right = xlim[1])
        if annotate:
            rc = FWHM([x,y], source='data')
            string = 'FWHM - {0:0.2f} arcsec'.format(rc)
            ax.text(x=.95, y=.95*(1-0.1*i), s=string , ha='right', va='top', fontsize='medium', transform=ax.transAxes)
        i += 1
        delta = delta * offset**i
        z = z - 0.1
    fig.show()
    
    return fig, ax
        
def plotPL(*filenames, step=.2, units='nm', normalize=False, offset=0, source='csv'):
    '''
    plotter function that takes in an arbitrary number of PL files
    
    filenames: path to PL data file, in csv format, with 1 header row - string
    
    step: step size in eV between major ticks for top axis. default is 0.2
    
    units: eV or nm. default nm
    
    normalize: Default False. If True will normalize all data - boolean
    
    offset: intensity offset for successive plots, waterfall style. Default is 0
    
    returns - plfig, plax, eVax - figure object and two axes objects for the plot
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    from brooks.plotter import axis_eV_nm
    
    plt.style.use('publication')
    
    lam = []
    I = []
    if source=='csv':
        for file in filenames:
            data = np.genfromtxt(file, delimiter = ',', skip_header=1)
            lam.append(data[:,0])
            I.append(data[:,1])
    elif source=='data':
        for file in filenames:
            lam.append(file[:,0])
            I.append(file[:,1])
        
    plfig = plt.figure()
    plax = plfig.gca()
    
    if normalize:
        Inorm = []
        for i in range(len(I)):
            Inorm.append(I[i]/max(I[i]))
        del I
        I = Inorm
        plax.set_ylabel('Normalized Intensity')
    else:
        plax.set_ylabel('Intensity (arb.)')
    
    ndx = 0    
    for i in range(len(lam)):
        plax.plot(lam[i], I[i]+offset*ndx)
        ndx+=1
    
    if units == 'nm':
        eVax = axis_eV_nm(plax, step)
        plax.set_xlabel('Wavelength (nm)')
        
    elif units == 'eV':
        nmAx = axis_eV_nm(plax, step, base='eV')
        plax.set_xlabel('Energy (eV)')
    plfig.show()
    
    return plfig, plax, eVax
    
    
def formatSEM(filename, scale, instrument, show=True, saveName=None, length_fraction=0.2, font_properties=None, PIL=False):
    '''
    Takes in an SEM image, returns a matplotlib figure with the black bar on the bottom stripped off and a scale bar appended.
    
    filename - path of SEM image
    
    scale - magnification for scale bar - calibrated for instrument listed. Can be one of :100X, 200X, 500X, 1K, 2K, 5K, 10K, 50K, 100K
    
    instrument - 'nova' or 'S4800'
    
    show - show the image or not. default True
    
    saveName - if given, will save to the path and filename given. If no path is given the filename will save in the default directory
    
    length_fraction - length of scalebar on image. Default 0.2 (20%)
    
    font_properties - dict of matplotlib font properties, default None
    
    PIL - default False. If true the image will be returned as a PIL Image object ***Note this code does not close the memory object. It should be closed with img.close() after use
    
    returns:
    fig - matplotlib figure object containing the SEM image
    ax - matplotlib axes object on which the SEM image is plotted
    img - PIL Image object - only if PIL=True
    '''
    #length of scale bar in um
    scale_um_nova = {'100X':1000, '200X':500, '500X':200, '1K':100, '2K':50, '5K':20, '10K':10, '20K':5, '50K':2, '100K':1}
    scale_um_s4800 = {'100X':300, '200X':150, '500X':60, '1K':30, '2K':15, '5K':6, '10K':3, '20K':1.5, '50K':0.6, '100K':0.3}
    if scale not in scale_um_nova.keys():
        raise ValueError('scale must be 100X, 200X, 500X, 1K, 2K, 5K, 10K, 20K, 50K, or 100K')
    if instrument not in ['nova', 'NOVA', 'Nova', 'S4800', 's4800']:
        raise ValueError('instrument must be nova or s4800')
        
    #instrument specific calibration, pixels per um for the instrument scale bar 
    if instrument == 'nova' or instrument == 'NOVA' or instrument == 'Nova':
        pix = 400
        set_scale = scale_um_nova[scale]/pix
    elif instrument == 'S4800' or instrument == 's4800':
        pix = 300
        set_scale = scale_um_s4800[scale]/pix
    fig = plt.figure()
    ax = plt.gca()
    image = plt.imread(filename)
    
    #Find black bar
    r1 = image[:,1]
    length = len(r1)
    i = 0
    while r1[length-1-i] == 0:
        i+=1
    lim = length-i
    
    ax.imshow(image[:lim, :], cmap='gray')
    ax.add_artist(ScaleBar(set_scale, 'um', length_fraction=length_fraction, font_properties=font_properties))
    ax.axis('off')
    
    #remove white space/margins. pad_inches in save call is also necessary
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    
    if show:
        plt.show()
    if saveName != None:
        fig.savefig(saveName, dpi=300, bbox_inches = 'tight', pad_inches=0)
    if PIL:
        mem = BytesIO()
        fig.savefig(mem, format='png')
        img = Image.open(mem)
        return fig, ax, img
    else:
        return fig, ax
            

def axis_eV_nm(ax, step, axis='x', minstep=10, decimal=2, nice=True, base='nm'):
    '''
    depricated. please use brooks.PL.axis_eV_nm
    
    create second matplotlib axes object, converted from nm to eV. Specify x or y. minstep specifies minor ticks per major
    
    ax: matplotlib axes object to convert. Must be nm or eV
    
    step: specifies major tick spacing in eV
    
    axis: specifies the axis containing the nm or eV data - must be 'x' or 'y'. default is x. string
    
    minstep: specifies the number of minor ticks per major tick. default 10. integer
    
    decimal: specifies the number of decimal places to display in eV
    
    nice: round min and max values to nice numbers. default True - boolean
    
    base: specifies if the input axes object contains data in wavelength or energy. default nm - string
    
    returns - matplotlib axes object containing the converted values, twinned to the the specified axis in the matplotlib axes object provided
    
    '''
    
    hc = (c.h/c.e)*c.c/1E-9
    #Value Error Handling
    if axis not in ['x', 'y']:
        raise ValueError('axis must be x or y')
    if base not in ['nm', 'ev', 'eV']:
        raise ValueError('base must be nm or eV')
    

    min = hc/ax.get_xlim()[1] if axis == 'x' else hc/ax.get_ylim()[1]
    max = hc/ax.get_xlim()[0] if axis == 'x' else hc/ax.get_ylim()[0]
    if nice:
        if base == 'nm':
            min = np.floor(min)
            max = np.ceil(max)
        elif base == 'eV' or 'ev':
            min = np.around(min, -int(np.floor(np.log10(step))))
            max = np.around(max, -int(np.floor(np.log10(step))))
    minor = step/minstep
    convert = lambda x: hc/x
    label = np.around(np.arange(max, min, -step), decimal)
    minorlabel = np.around(np.arange(max,min,-minor), int(decimal+np.floor(np.log10(minstep))))
    if decimal == 0:
        label = label.astype(int)
    if axis == 'x':
        ax2 = plt.twiny(ax)
    
        ax2.set_xticks([convert(x) for x in label])
        ax2.set_xticks([convert(x) for x in minorlabel], minor=True)
        ax2.set_xticklabels(label)
        ax2.set_xlim(ax.get_xlim())
        ax.tick_params(which='both', top=False)
        ax2.tick_params(which='minor', bottom=False, top=True)
    if axis == 'y':
        ax2 = plt.twinx(ax)
    
        ax2.set_yticks([convert(x) for x in label])
        ax2.set_yticks([convert(x) for x in minorlabel], minor=True)
        ax2.set_yticklabels(label)
        ax2.set_ylim(ax.get_ylim())
        ax.tick_params(which='both', right=False)
        ax2.tick_params(which='minor', left=False, right=True)
    if base == 'nm':
        ax2.set_xlabel('Energy (eV)')
    elif base == 'eV' or 'ev':
        ax2.set_xlabel('Wavelength (nm)')
    return ax2
    

        
def scientificNotation(value):
    '''
    Formatter for better scientific notation on plots
    
    Useage:
    formatter = mpl.ticker.FuncFormatter(lambda x, p: scientificNotation(x))
    plt.gca().xaxis.set_major_formatter(formatter)

    '''
    if value == 0:
        return '0'
    else:
        e = np.log10(np.abs(value))
        m = np.sign(value) * 10 ** (e - int(e))
        if m >= 1:
            return r'${:.1f} \cdot 10^{{{:d}}}$'.format(m, int(e))
        else:
            return r'${:1.1f} \cdot 10^{{{:d}}}$'.format(m*10, int(e-1))
            
def manualFWHM(x, y):
    ymax = np.max(y)
    imax = np.argmax(y)
    y1 = y[:imax]
    y2 = y[imax:]
    i1 = np.abs(ymax/2-y1).argmin()
    yinv = y2[-1:0:-1]
    i2 = np.abs(ymax/2-yinv).argmin()
    i2 = yinv.shape[0]-i2+imax
    x1 = x[i1]
    x2 = x[i2]
    return (x2-x1)*3600
    
def waterfall(*data, source=None, fig=None, ax=None, colors=None, offset=100, labels=None, location = 'left', x = None, lblAvg = None, delta=1):
    '''
    XRD Waterfall plotter - wrapper function for plotXRD for pretty formatting
    
    inputs - data - arbitrary number of input files. Can be rigaku, panalytical, or SLAC datafiles, or can be tuples of (x,y) data
    
    source - tells plotter what type of data to expect. default is auto select (None)
    
    colors - single color or list of colors matching the data list - default None
    
    offset - multiplier for offset - should be log10 - default 100
    
    labels - list of annotation labels for waterfall - default None
    
    location - if labels are present, place them on the left or right side of the traces. default left
    
    x - custom x value for label placement in data coordinates - default None
    
    lblAvg - number of datapoints over which to take the maximum y value (left or right depending on location kwarg) for label placement. Label is placed at delta times the maximum over the range given. default is 20 datapoints
    
    delta - default = 3
    
    returns - fig, ax
    '''
    
    supportedFormats = ['ras', 'xrdml', 'data']
    
    if fig == None:
        fig = plt.figure()
    if ax == None:
        ax = fig.gca()
    
    source_input = source
        
    plotXRD(*data, source=source, fig=fig, ax = ax, offset=offset)
    ax.set_yticklabels([])
    
    if colors:
        if len(colors) is not len(data):
            for lines in ax.lines:
                lines.set_color(colors)
        else:
            for lines, color in zip(ax.lines, colors):
                lines.set_color(color)
                
    if labels:
        rawData = []
        for d in data:
            if source == None:
                source = d.split('.')[-1]
            if source not in supportedFormats:
                raise ValueError('Unsupported Format')
            if source == 'xrdml':
                rawData.append((getPanalyticalXRD(d)))
            elif source == 'ras':
                rawData.append((rigakuXRD(d)))
            elif source == 'data':
                rawData.append(d)
            source = source_input
        labelList = []
        delta = 3
        if location == 'left':
            if x == None:
                x = rawData[0][0][0]*1.0005
            ha = 'left'
            for i, d in enumerate(rawData):
                if lblAvg == None:
                    lblAvg = int(d[1].shape[0]*.1)
                y = d[1][:lblAvg].max()
                labelList.append((x, y*delta))
                delta = delta + offset**(i+1)
        elif location == 'right':
            if x == None:
                x = rawData[0][0][-1]*.9995
            ha = 'right'
            for i, dat in enumerate(rawData):
                if lblAvg == None:
                    lblAvg = int(dat[1].shape[0]*.1)
                y = dat[1][-lblAvg:].max()
                labelList.append((x, y*delta))
                delta = delta + offset**(i+1)
        for loc, l in zip(labelList, labels):
            ax.text(loc[0], loc[1], l, ha=ha, transform = ax.transData)
        
        
    return fig, ax
    
def removeOutlier(x,y, w=10, thresh=2.75):
    
    l = len(y)
    i_remove = []
    for i in range(l-w):
        z = stats.zscore(y[i:i+10])
        for ndx, dev in enumerate(z):
            if dev > thresh:
                i_outlier = ndx+i
                if i_outlier not in i_remove:
                    i_remove.append(i_outlier)
    i_remove = np.array(i_remove)
    newX = np.delete(x,i_remove)
    newY = np.delete(y,i_remove)
    print('Removed %d outliers'%(len(i_remove)))
    return np.array([newX, newY])

# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work their way either side from a prescribed midpoint value)
    
    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)
    
    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))