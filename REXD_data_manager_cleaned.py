##Rekha SLAC data

#Import statements
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
from os import chdir
from scipy import constants as const
from scipy.interpolate import interp1d
from lmfit import report_fit
from lmfit.models import GaussianModel

def GetREXDData(dir):
    """
    Return numpy array energy x numpts x 6 (2t, I, Err, Q, bg, I-bg)
    """
    #Select directory
    chdir(dir)
    
    #Import files 
    #extract file names
    filenames = sorted(glob.glob('*.xye'))
    
    #Extract energy data from file -- format = <typeScan>_<Element>Edge_<index>_scan1_<energy>p*.xye
    E = []
    for file in filenames:
        E.append(float(file[file.find('scan1')+6:file.find('p', file.find('scan1'))]))
    E = np.array(E)
    
    #import data - 5 columns: 2theta, Intensity, Error, Energy, Q
    data = []
    for n, file in enumerate(filenames):
        data.append(np.genfromtxt(file))
        
    #Convert data to relevant values - numpy array energy x numpts x 6 (2t, I, Err, Q, bg, I-bg)
    for n, Energy, scan in zip(range(len(E)),E, data):
        theta = scan[:,0]/2
        I = scan[:,1]
        lam = (const.c*const.h/const.e)/Energy
        Q = 4*np.pi*np.sin(theta*np.pi/180)/lam
        bg = backgroundSubtract(Q,I)
        data[n] = np.c_[scan, Q, bg, I - bg]
    return data, E
    
##
def GetTopasFit(dir):
    """
    Return list of numpy arrays with columns 2theta, yobs, ycalc, diff
    """
    topas = []
    chdir(dir)
    filenames = sorted(glob.glob('*.txt'))
    for file in filenames:
        topas.append(np.genfromtxt(file))
    return topas
    
##
def backgroundSubtract(x, y, mode='linear', deg=100, v1=20, v2=6, x0=1.226E10):
    if mode == 'linear':        
        y0 = np.average(y[0:10])
        y1 = np.average(y[-10:-1])
        yarray = np.linspace(y0, y1, x.shape[0])
    elif mode == 'cheb':
        cheb = np.polynomial.Chebyshev.fit(x,y,deg)
        yarray = cheb(x)
    elif mode == 'step':
        yarray = step(x,v1,v2, x0)
    elif mode == 'cheb+step':
        s = backgroundSubtract(x,y,mode='step', v1=v1, v2=v2, x0=x0)
        yarray = backgroundSubtract(x,s,mode='cheb', deg=deg)
    return yarray

def step(xval, v1, v2, x0):
    out = np.empty_like(xval)
    for n, x in enumerate(xval):
        out[n] = v1 if x < x0 else v2
    return out

##
def gaussfit(x,y):
    mod=GaussianModel()
    params= mod.guess(y,x=x)
    out=mod.fit(y, params, x=x)
    # print(out.fit_report())
    # init = mod.eval(params, x=x)
    # 
    # plt.figure()
    # plt.plot(x,y, 'k+')
    # plt.plot(x,init, 'b--')
    # plt.plot(x, out.best_fit, 'r-')
    # plt.show()
    return out.params['amplitude'].value
 
##
def plotRawData(data, peak='101'):
    """
    Plots I vs. Q
    """
    testPlot = plt.figure()
    testAx = plt.gca()
    testAx.plot(data[1][:,3], data[1][:,5], label='scan 1')
    testAx.plot(data[50][:,3], data[49][:,5], label='scan 50')
    testAx.plot(data[100][:,3], data[99][:,5], label='scan 100')
    testAx.plot(data[150][:,3], data[149][:,5],label='scan 150')
    testAx.plot(data[200][:,3], data[199][:,5], label='scan 200')
    testAx.legend(loc='upper right')
    testAx.set_xlabel('Q')
    testAx.set_ylabel('Intensity (arb)')
    testAx.tick_params(direction='in', right=True, top=True)
    if peak == '101':
         testAx.text(0.05, 0.95, 'ZnGeP$_{2}$ (101)', transform = testAx.transAxes, ha='left', va='top')
    elif peak == '103':
         testAx.text(0.05, 0.95, 'ZnGeP$_{2}$ (103)', transform = testAx.transAxes, ha='left', va='top')
    elif peak == '112':
        testAx.text(0.05, 0.95, 'ZnGeP$_{2}$ (112)', transform = testAx.transAxes, ha='left', va='top')
    elif peak == '204':
        testAx.text(0.05, 0.95, 'ZnGeP$_{2}$ (204)', transform = testAx.transAxes, ha='left', va='top')
    plt.style.use('publication')
    plt.show()
    plt.tight_layout()
    
##       
def plotREXDmaxI(data, peak='101'):
    """
    Plots I vs E by extracting peak max
    """
    # Q for plotting
    if peak == '101':
        q = 1.291682E10
    elif peak == '103':
        q = 2.102822E10
    elif peak == '112':
        q = 2.006250E10
    elif peak == '204':
        q = 3.286932E10
   
   #Extract intensity at a particular q value
   
    I_q = []
    for scan in data:
        #x = scan[:,3] - Q
        #y = scan[:,5] - BG subtracted I
        f = interp1d(scan[:,3], scan[:,5],kind='cubic')
        I_q.append(f(q))
    
    I_q = np.array(I_q)    
    
  #I vs E  
    resPlot = plt.figure()
    resAx = plt.gca()
    resAx.plot(E, I_q)
    resAx.set_xlabel('Energy (eV)')
    resAx.set_ylabel('Intensity (arb)')
    if peak == '101':
         resAx.text(0.05, 0.95, 'ZnGeP$_{2}$ (101)', transform = resAx.transAxes, ha='left', va='top')
    elif peak == '103':
         resAx.text(0.05, 0.95, 'ZnGeP$_{2}$ (103)', transform = resAx.transAxes, ha='left', va='top')
    elif peak == '112':
        resAx.text(0.05, 0.95, 'ZnGeP$_{2}$ (112)', transform = resAx.transAxes, ha='left', va='top')
    elif peak == '204':
        resAx.text(0.05, 0.95, 'ZnGeP$_{2}$ (204)', transform = resAx.transAxes, ha='left', va='top')
    plt.style.use('publication')
    plt.show()
    #plt.style.use('publication')
    plt.tight_layout()
    
##
def plotHeatMaps(data, E, peak='101'):     
    Q_array = np.concatenate((np.array([data[x][:,3] for x in range(len(data))])))
    I_array = np.concatenate((np.array([data[x][:,5] for x in range(len(data))])))
    for n,el in enumerate(I_array):
        if el <= 0:
            I_array[n] = 0.001
    E_array = []
    for x in range(len(data)):
        for n in range(len(data[x][:,0])):
            E_array.append(E[x])
    E_array = np.array(E_array)

#heatmaps:
    testfig, testax = plt.subplots()
    scat = testax.scatter(Q_array, E_array, c=I_array, vmin = I_array.min(), vmax = I_array.max(), cmap = plt.cm.viridis, s = 2, marker = 's', norm=matplotlib.colors.LogNorm())
    testax.set_xlabel('Q (1/$\AA$)')
    testax.set_ylabel('Energy (eV)')
    cb = testfig.colorbar(scat)
    cb.set_label('Intensity (arb.)')
    plt.show()
    plt.style.use('poster')
    plt.tight_layout()
    
    lintestfig, lintestax = plt.subplots()
    scat = lintestax.scatter(Q_array, E_array, c=I_array, vmin = I_array.min(), vmax = I_array.max(), cmap = plt.cm.viridis, s = 2, marker = 's',)
    lintestax.set_xlabel('Q (1/$\AA$)')
    lintestax.set_ylabel('Energy (eV)')
    cblin = lintestfig.colorbar(scat)
    cblin.set_label('Intensity (arb.)')
    if peak == '101':
         lintestax.text(0.45, 0.95, 'ZnGeP$_{2}$ (101)', color='white',transform = lintestax.transAxes, ha='left', va='top')
    elif peak == '103':
         lintestax.text(0.05, 0.95, 'ZnGeP$_{2}$ (103)', transform = lintestax.transAxes, ha='left', va='top')
    elif peak == '112':
        lintestax.text(0.45, 0.95, 'ZnGeP$_{2}$ (112)', color='white', transform = lintestax.transAxes, ha='left', va='top')
    elif peak == '204':
       lintestax.text(0.05, 0.95, 'ZnGeP$_{2}$ (204)', transform = lintestax.transAxes, ha='left', va='top')
    #lintestfig.subplots_adjust(left=0.25, right=0.85, top=0.93, bottom=0.26, wspace=.20, hspace=.20)
    plt.show()
    plt.style.use('poster')
    plt.tight_layout()
        
 ##I vs E Peak area   
def plotREXDarea(data, peak='101', cutoff = 0):    
    area = []
    for scan in data:
        areaFit = gaussfit(scan[:,3], scan[:,5])
        if areaFit > cutoff:
            area.append(areaFit)
        else:
            area.append(cutoff)
        
    areaPlot = plt.figure()
    areaAx = areaPlot.gca()
    areaAx.plot(E, area)
    areaAx.set_xlabel('Energy (eV)')
    areaAx.set_ylabel('Intensity (arb. u.)')
    areaAx.tick_params(direction='in',right=True, top=True)
    if peak == '101':
         areaAx.text(0.05, 0.95, 'ZnGeP$_{2}$ (101)', transform = areaAx.transAxes, ha='left', va='top')
    elif peak == '103':
         areaAx.text(0.05, 0.95, 'ZnGeP$_{2}$ (103)', transform = areaAx.transAxes, ha='left', va='top')
    elif peak == '112':
        areaAx.text(0.05, 0.95, 'ZnGeP$_{2}$ (112)', transform = areaAx.transAxes, ha='left', va='top')
    elif peak == '204':
        areaAx.text(0.05, 0.95, 'ZnGeP$_{2}$ (204)', transform = areaAx.transAxes, ha='left', va='top')
    areaPlot.subplots_adjust(left=0.23, right=0.95, top=0.93, bottom=0.22, wspace=.20, hspace=.20)
    plt.show()
    plt.style.use('poster')
    plt.tight_layout()