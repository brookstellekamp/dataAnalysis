from brooks.plotter import plotRC, plotXRD, FWHM
from brooks.fileIO import rigakuXRD
import matplotlib.pyplot as plt
import numpy as np
import glob
from os import chdir
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cycler import cycler


def getGIXD(folder, plot=True):
    
    chdir(folder)
    
    files = sorted(glob.glob('*_frc.ras'))
    
    for i in range(len(files)):
        files[i] = folder + '/' + files[i]
    
    omega_in = []
    
    for file in files:
        omega_in.append(file.split('_')[1] + '.' + file.split('_')[2])
        
    
    if omega_in[-1].split('.')[-1] == 'ras':
        omega_in = [float(a) for a in omega_in[:-1]]
        files = files[:-1]
    else:
        omega_in = [float(a) for a in omega_in]
    
    if plot:
        fig, ax = plot_FWHM_omega_in(files, omega_in)
    
    return files, omega_in


def plotOmega(files, omega_in, cmap=plt.cm.tab10, fig=None, ax=None):
    plt.style.use('publication')
    
    if fig==None:
        fig = plt.figure()
    if ax == None:
        ax = fig.gca()
    
    ax.set_prop_cycle('color', cmap(np.linspace(0,1,len(files))))
    
    for file in files:
        plotRC(file, fig=fig, ax=ax)
    
    ax.set_xlim(left=0.5, right=6.25)
    ax.set_ylim(top=5E3)
    
    ax.text(.97, .97, r'2$\theta$ = 64.7$\degree$' + '\n' + r'<71$\bar{2}$>', ha='right', va='top', transform=ax.transAxes)
    
    om1 = 0.25#float(omega_in[0])
    om2 = float(omega_in[-1])
    delta = np.around(float(omega_in[1])-float(omega_in[0]), decimals=2)
    
    ax.text(0, 1.02, '$\omega_{in}$ = %0.2f $\degree$'%om1, transform=ax.transAxes)
    ax.text(1, 1.02, '$\omega_{in}$ = %0.2f $\degree$'%om2, ha = 'right', transform=ax.transAxes)
    ax.text(.5, 1.02, '$\Delta\omega_{in}$ = %0.2f $\degree$'%delta, ha='center', transform=ax.transAxes)
    
    fig.canvas.draw()
    
    return fig, ax


def plot2theta(folder):
    chdir(folder)
    
    xrdFiles = sorted(glob.glob('*_xrd.ras'))
    
    xrdFig = plt.figure()
    xrdAx = xrdFig.gca()
    
    offset=5
    i=0
    
    for file in xrdFiles:
        plotXRD(file, fig=xrdFig, offset=i)
        i = i+offset
    
    return xrdFig, xrdAx


def plot_FWHM_omega_in(files, omega_in, fig=None, ax=None, inset=False, marker=None, label='__nolabel__', error=False):
    
    if fig==None:
        fig = plt.figure()
    if ax==None:
        ax=fig.gca()
    
    FWHM_array = np.empty((len(files), 1), dtype=object)
    for i, file in enumerate(files):
        FWHM_array[i] = FWHM(file, source='rigaku', report=False, shape='voigt')
        
    plt.style.use('publication')


    if error == False: 
        if marker:
            ax.scatter(np.array(omega_in), np.array([a.n for a in FWHM_array[:,0]]), marker=marker, label=label)
        else:
            ax.scatter(np.array(omega_in), np.array([a.n for a in FWHM_array[:,0]]), label=label)
        
        if inset:
            ax2 = inset_axes(ax, width='50%', height='50%')
            FWHM(files[4], source='rigaku', annotate=False, report=False, plot=True, fig=fig, ax=ax2, shape='voigt')
    
    elif error == True:
        if marker:
            ax.errorbar(np.array(omega_in), np.array([a.n for a in FWHM_array[:,0]]), yerr = [a.s for a in FWHM_array[:,0]], lw=0, elinewidth=1, marker=marker, label=label)
        else:
            ax.errorbar(np.array(omega_in), np.array([a.n for a in FWHM_array[:,0]]), yerr = [a.s for a in FWHM_array[:,0]], lw=0, elinewidth=1, label=label)
        
        if inset:
            ax2 = inset_axes(ax, width='50%', height='50%')
            FWHM(files[4], source='rigaku', annotate=False, report=False, plot=True, fig=fig, ax=ax2, shape='voigt')
        
    
    ax.set_xlabel('Incidence Angle ($\degree$)')
    ax.set_ylabel(r'<71$\bar{2}$> $\omega$ FWHM (arcsec)')
    
    fig.show()
    
    if inset:
        return fig, ax, ax2, FWHM_array 
    else:
        return fig, ax, FWHM_array
