from brooks.latticeConstant import *
from brooks.RSM import *
from brooks.plotter import plotXRD, FWHM, plotRC, XRD_peak_fit, MidpointNormalize
from brooks.fileIO import rigakuXRD
from os import chdir, listdir, walk
import glob
from dataclasses import dataclass
from collections import namedtuple
import seaborn as sns



def alpha_angle(phi):
    '''
    solve for alpha knowing the 113 - 331 interplanar angle
    plotting the chi angle (as y) between 113 and 331 vs alpha (as x) proves it is a linear relationship with slope =-1.1642 and intercept = 156.277
    calculating the chi angle for various a-spacings confirms the angle is independent of the a-spacing
    '''
    m = -1.1642
    b = 156.277
    return (phi - b)/m

@dataclass
class dataContainer:
    name: str
    h: int
    k: int
    l: int
    hkl: tuple
    scanType: str
    sourceFile: str
    numPeaks: int
    FWHM: float
    
    
@dataclass
class scan:
    XRD: dataContainer
    RC: dataContainer
    
class wafer(dict):
    def __init__(self, path, waferMapType='1-inch'):
        '''
        parent class for sample, contains multiple samples and organizes them by piece
        '''
        
        chdir(path)
        self.measuredPieces = sorted(next(walk('.'))[1], key=int)
        
        if waferMapType == '1-inch':
            allPieces = np.array(range(1,17), dtype=str)
        
        for piece in allPieces:
            name = 'p' + piece
            if piece in self.measuredPieces:
                chdir(piece)
                self[name] = sample(path + '/' + piece)
                chdir(path)
            else:
                self[name] = []
                
            
    
    def calc(self, pieces = 'all', report=True):
        if pieces == 'all':
            p = ['p' + x for x in self.measuredPieces]
        else:
            p = ['p' + x for x in pieces]
        for piece in p:
            self[piece].calculateLattice()
            if report:
                self[piece].report()

            
    def plotWaferMap(self, parameter):
            
        '''
        plot wafer map view of specific parameters. Acceptable parameters include:
        111 center
        111 FWHM sharp
        111 FWHM broad
        
        113 center
        113 FWHM
        
        331 center
        331 FWHM
        
        interplanar angle
        a
        alpha
        
        d_111
        d_113
        d_331
        
        hex spacing
        '''
        default_cmap = 'viridis'
        #xy mapping
        # xy_dict = { '1':(1,4),
        #             '2':(2,4),
        #             '3':(3,4),
        #             '4':(4,4),
        #             '5':(1,3),
        #             '6':(2,3),
        #             '7':(3,3),
        #             '8':(4,3),
        #             '9':(1,2),
        #             '10':(2,2),
        #             '11':(3,2),
        #             '12':(4,2),
        #             '13':(1,1),
        #             '14':(2,1),
        #             '15':(3,1),
        #             '16':(4,1)}
        
        measuredINT = [int(x) for x in self.measuredPieces]
        c = np.zeros(16)                
        if parameter == '111 center':
            mid = calc_2theta(TaC, 1,1,1)
            vmin = 34.5
            vmax = 36
            cmap='seismic'
            norm=True
            for x in np.arange(0,16):
                if x+1 in measuredINT:
                    c[x] = self['p' + str(x+1)].dataDict['111 center'].n
        elif parameter == '111 FWHM sharp':
            norm=False
            cmap=default_cmap
            for x in np.arange(0,16):
                if x+1 in measuredINT:
                    a = np.argmin(self['p' + str(x+1)].index['111'].RC.FWHM)
                    c[x] = self['p' + str(x+1)].index['111'].RC.FWHM[a].n
        elif parameter == '111 FWHM broad':
            norm=False
            cmap=default_cmap
            for x in np.arange(0,16):
                if x+1 in measuredINT:
                    a = np.argmax(self['p' + str(x+1)].index['111'].RC.FWHM)
                    c[x] = self['p' + str(x+1)].index['111'].RC.FWHM[a].n
            
        elif parameter == '113 center':
            mid = calc_2theta(TaC, 1,1,3)
            vmin = mid*0.98
            vmax = mid*1.02
            cmap='seismic'
            norm=True
            for x in np.arange(0,16):
                if x+1 in measuredINT:
                    c[x] = self['p' + str(x+1)].dataDict['113 center'].n
        elif parameter == '113 FWHM':
            norm=False
            cmap=default_cmap
            for x in np.arange(0,16):
                if x+1 in measuredINT:
                    c[x] = self['p' + str(x+1)].index['113'].RC.FWHM.n
            
        elif parameter == '331 center':
            mid = calc_2theta(TaC, 3,3,1)
            vmin = mid*0.98
            vmax = mid*1.02
            cmap='seismic'
            norm=True
            for x in np.arange(0,16):
                if x+1 in measuredINT:
                    c[x] = self['p' + str(x+1)].dataDict['331 center'].n
        elif parameter == '331 FWHM':
            norm=False
            cmap=default_cmap
            for x in np.arange(0,16):
                if x+1 in measuredINT:
                    c[x] = self['p' + str(x+1)].index['331'].RC.FWHM.n
        
        elif parameter == 'interplanar angle':
            mid = chiAngle_cubic(1,1,3,3,3,1)
            vmin = 50.5
            vmax = 52.5
            cmap='seismic'
            norm=True
            for x in np.arange(0,16):
                if x+1 in measuredINT:
                    c[x] = self['p' + str(x+1)].dataDict['interplanar angle'].n
        elif parameter == 'a':
            mid = TaC['a']
            vmin = 4.3
            vmax = TaC['a']
            cmap='seismic'
            norm=True
            for x in np.arange(0,16):
                if x+1 in measuredINT:
                    c[x] = self['p' + str(x+1)].dataDict['a_bound'].n
        elif parameter == 'alpha':
            mid = 90
            vmin = 89
            vmax = 91
            cmap='seismic'
            norm=True
            for x in np.arange(0,16):
                if x+1 in measuredINT:
                    c[x] = self['p' + str(x+1)].dataDict['alpha_bound'].n
            
        elif parameter == 'd_111':
            mid = d_cubic(TaC['a'], 1,1,1)
            vmin = mid*.98
            vmax = mid*1.02
            cmap='seismic'
            norm=True
            for x in np.arange(0,16):
                if x+1 in measuredINT:
                    c[x] = self['p' + str(x+1)].dataDict['d_111'].n
        elif parameter == 'd_113':
            mid = d_cubic(TaC['a'], 1,1,3)
            vmin = mid*.98
            vmax = mid*1.02
            cmap='seismic'
            norm=True
            for x in np.arange(0,16):
                if x+1 in measuredINT:
                    c[x] = self['p' + str(x+1)].dataDict['d_113'].n
        elif parameter == 'd_331':
            mid = d_cubic(TaC['a'], 3,3,1)
            vmin = mid*.98
            vmax = mid*1.02
            cmap='seismic'
            norm=True
            for x in np.arange(0,16):
                if x+1 in measuredINT:
                    c[x] = self['p' + str(x+1)].dataDict['d_331'].n
            
        elif parameter == 'hex spacing':
            mid = 3.1495 #50% AlGaN
            vmin = AlN['a']
            vmax = GaN['a']
            cmap='viridis'
            norm=True
            for x in np.arange(0,16):
                if x+1 in measuredINT:
                    c[x] = self['p' + str(x+1)].dataDict['hex spacing']
            
        if norm:
            sns.heatmap(c.reshape(4,4), linewidth=0.5, norm=MidpointNormalize(midpoint=mid, vmin=vmin, vmax=vmax), cmap=cmap)
        else:
            sns.heatmap(c.reshape(4,4), linewidth=0.5, cmap=cmap)
        plt.show()

class sample:
    def __init__(self, path):
        ''' 
        Note: only set up for rigaku smartlab ras files so far. 
        
        takes in path containing rocking curve and 2theta-omega data
        loads files
        
        
        '''
        chdir(path)
        self.path = path
        self.XRDFiles = sorted(glob.glob('*xrd.ras'))
        self.RCFiles = sorted(glob.glob('*frc.ras'))
        self.chamber = self.XRDFiles[0].split('_')[0]
        self.sampleNumber = self.XRDFiles[0].split('_')[1].split('-')[0]
        try:
            self.piece = self.XRDFiles[0].split('_')[1].split('-')[1]
        except IndexError:
            print('no piece number found')
            self.piece = None
        try:
            self.ID = self.chamber + '_' + self.sampleNumber + '-' + self.piece
        except:
            self.ID = self.chamber + '_' + self.sampleNumber
        self.index = {}
        self.dataDict = {}
        for x_f, r_f in zip(self.XRDFiles, self.RCFiles):
            s = x_f.split('_')
            hkl = s[2][:s[2].find('rc')]
            XRDPath = self.path + '/' + x_f
            RCPath = self.path + '/' + r_f
            self.index[hkl] = scan(dataContainer(hkl, int(hkl[0]), int(hkl[1]), int(hkl[2]), (int(hkl[0]), int(hkl[1]), int(hkl[2])), 'XRD', XRDPath, 1, 0), dataContainer(hkl, int(hkl[0]), int(hkl[1]), int(hkl[2]), (int(hkl[0]), int(hkl[1]), int(hkl[2])), 'RC', RCPath, 1, 0))
                        
                        
    def plotAll(self, hkl='all'):
        '''
        plot 2theta-omega scans and rocking curves for all reflections
        
        hkl: list of hkl indicies to plot *as strings*  if only certain reflections are desired, e.g. hkl = ['111', '113']
        '''
        chdir(self.path)
        if hkl == 'all':
            plotIndex = self.index.keys()
        else:
            plotIndex = hkl
            
        for i in plotIndex:
            label = self.ID + ' (' + i + ')'
            plotXRD(self.index[i].XRD.sourceFile)
            plt.title(label)
            plotRC(self.index[i].RC.sourceFile)
            plt.title(label)
            
    def rockingCurveFWHM(self, x0=None, hkl='all'):
        '''
        runs fits on the rocking curves
        x0 is a a dictionary of dictionaries containing initial conditions to be supplied to the fitting function
        
        hkl: list of hkl indicies to plot *as strings*  if only certain reflections are desired, e.g. hkl = ['111', '113']
        
        the first (outside) dictionary contains indicies hkl as strings. The second dictionary contains initiaal parameters.
        
        acceptable keys are:
        center
        sigma
        numPeaks
        peakShape
        
        example:
        
        x0 =    {'111': {'center':[17, 17.5], 'numPeaks':2},
                 '113': {'sigma':0.05, 'peakShape':'pearson'}
                }
        '''
        chdir(self.path)
        if hkl == 'all':
            plotIndex = self.index.keys()
        else:
            plotIndex = hkl
        
        peakShape='voigt'
        
        for i in plotIndex:
            if x0:
                if i in x0.keys():
                    if 'numPeaks' in x0[i].keys():
                        self.index[i].RC.numPeaks = x0[i]['numPeaks']
                        del x0[i]['numPeaks']
                    if 'peakShape' in x0[i].keys():
                        peakShape = x0[i]['peakShape']
                        del x0[i]['peakShape']
                    params = x0[i]
                else:
                    x0[i] = {}   
                params = x0[i]
            else:
                params=None
            self.index[i].RC.FWHM = FWHM(self.index[i].RC.sourceFile, plot=True, peaks = self.index[i].RC.numPeaks, shape=peakShape, x0=params)
        
    def calculateLattice(self):
        '''
        
        Bug notes: save calculated FWHM values in FWHM container
        
        
        '''
        chdir(self.path)
        self.dataDict['111 center'] = XRD_peak_fit(self.index['111'].XRD.sourceFile, report=False)
        self.dataDict['113 center'] = XRD_peak_fit(self.index['113'].XRD.sourceFile, report=False)
        self.dataDict['331 center'] = XRD_peak_fit(self.index['331'].XRD.sourceFile, report=False)
        
        self.dataDict['111 tth step'] = rigakuXRD(self.index['111'].XRD.sourceFile, fullReturn=True)[1]['scan']['*MEAS_SCAN_STEP ']
        self.dataDict['113 tth step'] = rigakuXRD(self.index['113'].XRD.sourceFile, fullReturn=True)[1]['scan']['*MEAS_SCAN_STEP ']
        self.dataDict['331 tth step'] = rigakuXRD(self.index['331'].XRD.sourceFile, fullReturn=True)[1]['scan']['*MEAS_SCAN_STEP ']
        
        
        self.dataDict['d_111'] = d_bragg(self.dataDict['111 center'], step=self.dataDict['111 tth step'])
        self.dataDict['d_113'] = d_bragg(self.dataDict['113 center'], step=self.dataDict['113 tth step'])
        self.dataDict['d_331'] = d_bragg(self.dataDict['331 center'], step=self.dataDict['331 tth step'])
        
       
        self.dataDict['113 FWHM'], self.dataDict['113 RC fitparams'] = FWHM(self.index['113'].RC.sourceFile, report=False, fullReturn=True)
        self.dataDict['113 omega center'] = ufloat(self.dataDict['113 RC fitparams'].params['center'].value, self.dataDict['113 RC fitparams'].params['center'].stderr)
        self.dataDict['331 FWHM'], self.dataDict['331 RC fitparams'] = FWHM(self.index['331'].RC.sourceFile, report=False, fullReturn=True)
        self.dataDict['331 omega center'] = ufloat(self.dataDict['331 RC fitparams'].params['center'].value, self.dataDict['331 RC fitparams'].params['center'].stderr)
        
        self.dataDict['113 offset'] = (self.dataDict['113 center']/2) - self.dataDict['113 omega center']
        self.dataDict['331 offset'] = (self.dataDict['331 center']/2) - self.dataDict['331 omega center']
        self.dataDict['interplanar angle'] = np.abs(self.dataDict['113 offset']) + np.abs(self.dataDict['331 offset'])
        self.dataDict['alpha'] = alpha_angle(self.dataDict['interplanar angle'])
        bounds=([4,self.dataDict['alpha'].n-self.dataDict['alpha'].s], [5, self.dataDict['alpha'].n+self.dataDict['alpha'].s])
        
        d_array = np.array([self.dataDict['d_111'].n, self.dataDict['d_113'].n, self.dataDict['d_331'].n])
        err_array = np.array([self.dataDict['d_111'].s, self.dataDict['d_113'].s, self.dataDict['d_331'].s])
        h_array = np.array([1,1,3])
        k_array = np.array([1,1,3])
        l_array = np.array([1,3,1])
        
        l_rhomb = lattice_rhomb_lsq(d_array, h_array, k_array, l_array, x0=[4.44, self.dataDict['alpha'].n], err=err_array, bounds=bounds, report=False)
        TaC_rhomb = TaC.copy()
        TaC_rhomb['a'] = l_rhomb[0].n
        TaC_rhomb['alpha'] = l_rhomb[1].n
        TaC_rhomb['sys'] = 'rhomb'
        self.dataDict['a'] = l_rhomb[0]
        self.dataDict['alpha'] = l_rhomb[1]
        self.dataDict['a_bound'] = l_rhomb[0]
        self.dataDict['alpha_bound'] = l_rhomb[1]
        
        error_rhomb = []
        for d, h, k, l in zip(d_array, h_array, k_array, l_array):
            error_rhomb.append(100*(d_rhomb(TaC_rhomb['a'], TaC_rhomb['alpha'], h,k,l) - self.dataDict['d_'+str(h)+str(k)+str(l)])/d_rhomb(TaC_rhomb['a'], TaC_rhomb['alpha'], h,k,l))
        
        
        l_cubic = lattice_cubic_lsq(d_array, h_array, k_array, l_array, x0=[4.44], err=err_array, report=False)
        TaC_c = TaC.copy()
        TaC_c['a'] = l_cubic.n
        TaC_c['sys'] = 'cubic'
        self.dataDict['a_cubic'] = l_cubic.n
        
        error_cubic = []
        for d, h, k, l in zip(d_array, h_array, k_array, l_array):
            error_cubic.append(100*(d_cubic(TaC_c['a'], h,k,l) - self.dataDict['d_'+str(h)+str(k)+str(l)])/d_cubic(TaC_c['a'], h,k,l))
        
        self.dataDict['rhomb_mse_bound'] = np.sqrt(np.sum(np.array([x.n for x in error_rhomb])**2))
        self.dataDict['cubic_mse'] = np.sqrt(np.sum(np.array([x.n for x in error_cubic])**2))
        
        l_rhomb = lattice_rhomb_lsq(d_array, h_array, k_array, l_array, x0=[4.44, self.dataDict['alpha'].n], err=err_array, report=False)
        TaC_rhomb = TaC.copy()
        TaC_rhomb['a'] = l_rhomb[0].n
        TaC_rhomb['alpha'] = l_rhomb[1].n
        TaC_rhomb['sys'] = 'rhomb'
        self.dataDict['a'] = l_rhomb[0]
        self.dataDict['alpha'] = l_rhomb[1]
        
        error_rhomb = []
        for d, h, k, l in zip(d_array, h_array, k_array, l_array):
            error_rhomb.append(100*(d_rhomb(TaC_rhomb['a'], TaC_rhomb['alpha'], h,k,l) - self.dataDict['d_'+str(h)+str(k)+str(l)])/d_rhomb(TaC_rhomb['a'], TaC_rhomb['alpha'], h,k,l))
        
        self.dataDict['rhomb_mse_unbound'] = np.sqrt(np.sum(np.array([x.n for x in error_rhomb])**2))
        
        self.dataDict['hex spacing'] = d_rhomb(self.dataDict['a_bound'].n, self.dataDict['alpha_bound'].n, 1,1,0)
        
        self.index['111'].RC.FWHM = FWHM(self.index['111'].RC.sourceFile, peaks=2, shape='pearson', report=False)
        self.index['113'].RC.FWHM = self.dataDict['113 FWHM']
        self.index['331'].RC.FWHM = self.dataDict['331 FWHM']
        
        
    def report(self):
        
        if len(self.dataDict) == 0:
            self.calculateLattice()
            
        print(self.ID + '\n \n')
        print('Unbound Rhomb:\na = {0:0.4f} angstrom, alpha = {1:0.4f} degrees\nMSE = {2:0.4f}'.format(self.dataDict['a'], self.dataDict['alpha'], self.dataDict['rhomb_mse_unbound']))
        print('\n')
        print('Bound Rhomb:\na = {0:0.4f} angstrom, alpha = {1:0.4f} degrees\nMSE = {2:0.4f}'.format(self.dataDict['a_bound'], self.dataDict['alpha_bound'], self.dataDict['rhomb_mse_bound']))
        print('\n')
        print('113 - 331 interplanar angle = {0:0.4f}'.format(self.dataDict['interplanar angle']))
        print('\n')
        print('Cubic:\na = {0:0.4f} angstrom\nMSE = {1:0.4f}'.format(self.dataDict['a_cubic'], self.dataDict['cubic_mse']))
        print('\n')
        try:
            print('d-spacings\n(111): {0:0.4f} angstrom\n(113): {1:0.4f} angstrom\n(331): {2:0.4f} angstrom'.format(self.dataDict['d_111'], self.dataDict['d_113'], self.dataDict['d_331']))
        except KeyError:
            print('d-spacings\n(111): {0:0.4f} angstrom\n(113): {1:0.4f} angstrom'.format(self.dataDict['d_111'], self.dataDict['d_113']))
        
        print('\n')
        print('Effective hexagonal lattice spacing = {0:0.4f}'.format(self.dataDict['hex spacing']))
        print('\n')
        print('AlGaN Al-content: {0:0.1f}'.format(100*(GaN['a']-self.dataDict['hex spacing'])/(GaN['a']-AlN['a'])))
        
        print('\n')
        print('Rocking Curve FWHM:\n')
        print('111: {0:0.2f} arcsec\n     {1:0.4f} arcsec\n'.format(self.index['111'].RC.FWHM[0], self.index['111'].RC.FWHM[1]))
        print('113: {0:0.1f}\n'.format(self.index['113'].RC.FWHM))
        print('331: {0:0.1f}\n'.format(self.index['331'].RC.FWHM))
            
        print('\n-----------------------\n')
        
        
