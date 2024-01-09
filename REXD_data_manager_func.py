##Rekha SLAC data

#Import statements
import numpy as np
import matplotlib.pyplot as plt
import glob
from os import chdir
from scipy import constants as const
from scipy.interpolate import interp1d

def backgroundSubtract(x, y, mode='linear', deg=100):
    if mode == 'linear':
        fit = np.polynomial.Chebyshev.fit(x,y,deg)
        y0 = fit(x[0])
        y1 = fit(x[-1])
        m = (y1-y0)/(x[-1]-x[0])
        b = y0/(m*x[0])
        return m*x + b

def plotREXD(dir, peak='101'):
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
        
    #Data is a list of arrays, of length n. Energy is an array of energies, also of length n
    
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
        f = interp1d(scan[:,3], scan[:,5])
        I_q.append(f(q))
    
    I_q = np.array(I_q)
    
    
    testPlot = plt.figure()
    testAx = plt.gca()
    testAx.plot(data[1][:,3], data[1][:,5], label='scan 1')
    testAx.plot(data[50][:,3], data[50][:,5], label='scan 50')
    testAx.plot(data[100][:,3], data[100][:,5], label='scan 100')
    testAx.plot(data[150][:,3], data[150][:,5],label='scan 150')
    testAx.plot(data[200][:,3], data[200][:,5], label='scan 200')
    testAx.legend(loc='upper right', fontsize=5)
    testAx.set_xlabel('Q')
    testAx.set_ylabel('Intensity (arb)')
    testAx.tick_params(direction='in', right=True, top=True)
    if peak == '101':
         testAx.text(0.05, 0.95, 'ZnGeP$_{2}$ (101)', transform = testAx.transAxes, ha='left', va='top',fontsize=8)
    elif peak == '103':
         testAx.text(0.05, 0.95, 'ZnGeP$_{2}$ (103)', transform = testAx.transAxes, ha='left', va='top', fontsize=8)
    elif peak == '112':
        testAx.text(0.05, 0.95, 'ZnGeP$_{2}$ (112)', transform = testAx.transAxes, ha='left', va='top', fontsize=8)
    elif peak == '204':
        testAx.text(0.05, 0.95, 'ZnGeP$_{2}$ (204)', transform = testAx.transAxes, ha='left', va='top',fontsize=8)
    plt.show()
    
    #2 plotter functions, I vs E and heat map I vs E vs Q where I-BG is colorbar
    # plt.style.use('publication')
    
    resPlot = plt.figure()
    resAx = plt.gca()
    resAx.plot(E, I_q)
    resAx.set_xlabel('Energy (eV)')
    resAx.set_ylabel('Intensity (arb)')
    resAx.tick_params(direction='in',right=True, top=True)
    if peak == '101':
         resAx.text(0.05, 0.95, 'ZnGeP$_{2}$ (101)', transform = resAx.transAxes, ha='left', va='top')
    elif peak == '103':
         resAx.text(0.05, 0.95, 'ZnGeP$_{2}$ (103)', transform = resAx.transAxes, ha='left', va='top')
    elif peak == '112':
        resAx.text(0.05, 0.95, 'ZnGeP$_{2}$ (112)', transform = resAx.transAxes, ha='left', va='top')
    elif peak == '204':
        resAx.text(0.05, 0.95, 'ZnGeP$_{2}$ (204)', transform = resAx.transAxes, ha='left', va='top')
   
    plt.show()
    
    
    #hmapPlot = plt.figure()
    #hmapAx = plt.gca()
    
    Q_array = np.concatenate((np.array([data[x][:,3] for x in range(len(data))])))
    I_array = np.concatenate((np.array([data[x][:,5] for x in range(len(data))])))
    E_array = []
    for x in range(len(data)):
        for n in range(len(data[x][:,0])):
            E_array.append(E[x])
    E_array = np.array(E_array)
    
    #hist = hmapAx.hist2d(Q_array, E_array, bins=[70,201], weights=I_array, cmap=plt.cm.viridis)
    #hmapAx.set_xlabel(r'Q ($\AA^{-1}$)')
    #hmapAx.set_ylabel('Energy (eV)')
    #hmapAx.tick_params(direction='in')
    #cb = hmapPlot.colorbar(hist[3], ax = hmapAx)
    #cb.set_label('Intensity (arb.)')
    #plt.show()

    import matplotlib.tri as tri
    import matplotlib.colors as colors
    trimap = plt.figure()
    trimapAx = trimap.gca()
    trimapAx.set_xlabel('Q')
    trimapAx.set_ylabel('Energy (eV)')
    triang = tri.Triangulation(Q_array, E_array)
    tcf = trimapAx.tricontourf(triang, I_array, levels = [x for x in np.logspace(0,3,num=100)])
    trimap.colorbar(tcf)
    plt.show()
   
    #from matplotlib.mlab import griddata
    #test2 = plt.figure()
    #ax2 = test2.gca()
    #xi = np.linspace(Q_array.min(), Q_array.max(), 450)
    #yi = np.linspace(E_array.min(), E_array.max(), 200)
    #zi = griddata(Q_array, E_array, I_array, xi, yi, interp='linear')
    #ctr = ax2.contourf(xi, yi, zi)
    #plt.colorbar(ctr)
    #plt.show()
    
