import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import scipy.constants as c
import pandas as pd
from dataAnalysis.functions import normalize

'''
PL data file definition:

pandas dataframe which takes in, on initialization, an XLS seed file with all metadata and filename. Requires directory for filename
Columns: 
Filename | Date | Sample ID |, Excitation Wavelength (nm) | Power (mW) | Temperature (K) | filter | slit size (um) | center wavelength (nm) | grating (grooves/mm)/blaze wavelength(nm) | acquisition time (s) | wavelength (np array of x data in nm) | energy (np array of x data in eV) | intensity (np array of y data in counts per second)
'''

global hc 
hc = (c.h/c.e)*c.c/1E-9

def importFromMetaData(xl, fileDirectory):
    '''
    PL metadata handler
    xl - spreadsheet with PL metadata and filenames
    fileDirectory - location (folder) of PL csv files
    
    
    returns - pandas dataframe of metadata
    '''
    dataFile = pd.read_excel(xl)
    dataFile = dataFile.assign()
    dataRead = pd.DataFrame(columns=['wavelength', 'energy', 'intensity'], index = dataFile.index)
    for index, row in dataFile.iterrows():
        try:
            data = np.genfromtxt(fileDirectory + '/' + row['Filename'] + '.csv', delimiter=',', skip_header=1)
            dataRead.iloc[index]['wavelength'] = data[:,0]
            dataRead.iloc[index]['energy'] = hc/data[:,0]
            dataRead.iloc[index]['intensity'] = data[:,1]/row['acquisition time (s)']
        except(IOError):
            print('File not found: %s' % row['Filename'])
            dataRead.iloc[index]['wavelength'] = np.nan
            dataRead.iloc[index]['energy'] = np.nan
            dataRead.iloc[index]['intensity'] = np.nan
    return dataFile.join(dataRead)

def FFcorrect(data, FF):
    '''
    apply flat-field correction to a single dataset as a numpy array
    data - intensity data with wavelength matching the flat field
    '''
    correction = np.genfromtxt(FF, delimiter = ',', skip_header=1)
    
    return data/correction[:,1]
    
    
    
    
def slicer(dataFile, sampleID=None, excitation = None, power=None, temp = None, filt = None, slit=None, center=None, grating=None, time=None, exclude=None):
    baseIndex = dataFile.index
    
    if sampleID:
        if type(sampleID) == list:
            sampleIndex = pd.Index([])
            for x in sampleID:
                sampleIndex = sampleIndex.union(dataFile[dataFile['Sample ID'] == x].index)
        else:
            sampleIndex = dataFile[dataFile['Sample ID'] == sampleID].index
    else:
        sampleIndex = baseIndex
        
    if excitation:
        if type(excitation) == list:
            excitationIndex = pd.Index([])
            for x in excitation:
                excitationIndex = excitationIndex.union(dataFile[dataFile['excitation wavelength (nm)'] == x].index)
        else:
            excitationIndex = dataFile[dataFile['excitation wavelength (nm)'] == excitation].index
    else:
        excitationIndex = baseIndex
        
    if power:
        if type(power) == list:
            powerIndex = pd.Index([])
            for x in power:
                powerIndex = powerIndex.union(dataFile[dataFile['Power (mW)'] == x].index)
        else:
            powerIndex = dataFile[dataFile['Power (mW)'] == power].index
    else:
        powerIndex = baseIndex
        
    if temp:
        if type(temp) == list:
            tempIndex = pd.Index([])
            for x in temp:
                tempIndex = tempIndex.union(dataFile[dataFile['temp (K)'] == x].index)
        else:
            tempIndex = dataFile[dataFile['temp (K)'] == temp].index
    else:
        tempIndex = baseIndex

    if filt:
        if type(filt) == list:
            filterIndex = pd.Index([])
            for x in filt:
                filterIndex = filterIndex.union(dataFile[dataFile['Filter'] == x].index)
        else:
            filterIndex = dataFile[dataFile['Filter'] == filt].index
    else:
        filterIndex = baseIndex
    
    if slit:
        if type(slit) == list:
            slitIndex = pd.Index([])
            for x in slit:
                slitIndex = slitIndex.union(dataFile[dataFile['Slit (um)'] == x].index)
        else:
            slitIndex = dataFile[dataFile['Slit (um)'] == slit].index
    else:
        slitIndex = baseIndex
    
    if center:
        if type(center) == list:
            centerIndex = pd.Index([])
            for x in center:
                centerIndex = centerIndex.union(dataFile[dataFile['center (nm)'] == x].index)
        else:
            centerIndex = dataFile[dataFile['center (nm)'] == center].index
    else:
        centerIndex = baseIndex
        
    if grating:
        if type(grating) == list:
            gratingIndex = pd.Index([])
            for x in grating:
                gratingIndex = gratingIndex.union(dataFile[dataFile['grating'] == x].index)
        else:
            gratingIndex = dataFile[dataFile['grating'] == grating].index
    else:
        gratingIndex = baseIndex
        
    if time:
        if type(time) == list:
            timeIndex = pd.Index([])
            for x in time:
                timeIndex = timeIndex.union(dataFile[np.around(dataFile['aquisition time (s)'], decimals=2) == np.around(x, decimals=2)].index)
        else:
            timeIndex = dataFile[dataFile['acquisition time (s)'] == time].index
    else:
        timeIndex = baseIndex
        
    if exclude:
        excludeIndex = dataFile[dataFile['exclude'] != 'x'].index
    else:
        excludeIndex = baseIndex
    
        
    return dataFile.loc[sampleIndex.intersection(tempIndex).intersection(centerIndex).intersection(slitIndex).intersection(gratingIndex).intersection(excitationIndex).intersection(powerIndex).intersection(filterIndex).intersection(timeIndex).intersection(excludeIndex)]

def sellmeier(E, a=4.73, b=0.0113, c = 4.0):
    '''
    function to calculate index of refraction based off of sellmeier coefficients a (alpha), b (beta), and c (gamma) as a function of photon energy (E) in eV according to n**2(E) - 1 = a*(1+b*(E**2/(E-c)**2))
    
    default coefficients are for GaN
    
    inputs - 
    E - photon energy at which to calculate the index of refraction
    a, b, c, - sellmeier coefficients
    
    returns - index of refraction (unitless)
    '''
    #n**2(E) - 1 = a*(1+b*(E**2/(E-c)**2))
    rhs = a*(1+b*(E**2/(E-c)**2))
    return np.sqrt(rhs-1)


def fabry_perot_thickness(E, a=4.73, b=0.0113, c = 4.0, plot=True):
    '''
    function to extract thickness from a set of fabry perot fringes - will also plot fringe position vs fringe number to determine linearity
    
    inputs - E - numpy array of fabry perot peak energies in increasing order
    
    return - thickness in um
    '''
    
    x = np.arange(1,len(E)+1, 1)
    
    y = 2*sellmeier(E)/((hc/E)/1000)
    reg = linregress(x,y)
    
    if plot:
        plt.scatter(x,y)
        plt.plot(x, x*reg.slope + reg.intercept,ls='--')
        plt.show()
    
   
    print('Thickness is %0.2f um'%(1/reg.slope))
    
    return 1/reg.slope
    

def axis_eV_nm(ax, step, axis='x', minstep=10, decimal=2, nice=True, base='nm'):
    '''
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


def csvPL(f, correctionFile=None):
    '''
    Given a csv file f with columns wavelength and intensity, returns an array of [wavelength, intensity]
    
    
    '''
    if correctionFile == None:
        return np.genfromtxt(f, delimiter=',', skip_header=1)
    else:
        xy = np.genfromtxt(f, delimiter=',', skip_header=1)
        return np.c_[xy[:,0], FFcorrect(xy[:,1], correctionFile)]

def csvPlot(*filenames, fig = None, ax = None, correctionFile=None, norm=False, labels=None, xmin=None, xmax=None):
    if fig==None:
        fig = plt.figure()
    if ax == None:
        ax = fig.gca()
    if labels == None:
        labels = [x.split('.')[0] for x in filenames] 
    for f, label in zip(filenames, labels):
        data = csvPL(f, correctionFile=correctionFile)
        x = [hc/x for x in data[:,0]]
        if norm:
            ax.plot(x, normalize(x, data[:,1], xmin=xmin, xmax=xmax), label=label)
        else:
            ax.plot(x, data[:,1], label=label)
    
    ax.set_xlabel('Energy (eV)')
    if norm == True:
        ax.set_ylabel('Normalized Counts')
    else:
        ax.set_ylabel('Counts per second (arb.)')
        
    ax.set_yscale('log')
    fig.show()
    fig.tight_layout()
    
    return fig, ax

def plotPL(*f, source='csv', fig=None, ax=None, step=0.2, offset=0, normalize=False, units='nm', left=None, right=None):
    '''
    plotter function that takes in an arbitrary number of PL files
    
    NOTE: secondary axis labels do not support resizing. To specify x-axis length use left and right kwargs in the function call to resize before drawing the secondary axis
    
    filenames: path to PL data file, in csv format, with 1 header row - string
    
    step: step size in eV between major ticks for top axis. default is 0.2
    
    units: eV or nm. default nm
    
    normalize: Default False. If True will normalize all data - boolean
    
    offset: intensity offset for successive plots, waterfall style. Default is 0
    
    left = left x-axis limit
    
    right = right x-axis limit
    
    returns - plfig, plax, eVax - figure object and two axes objects for the plot
    '''
    plt.style.use('publication')
    
    lam = []
    I = []
    
    if source == 'csv':
        dataList = []
        for filename in f:
            data = csvPL(filename)
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
    
    
    eV = [hc/x for x in lam]
    ndx = 0    
    
    
    if units == 'nm':
        for i in range(len(lam)):
            plax.plot(lam[i], I[i]+offset*ndx)
            ndx+=1
        plax.set_xlim(left=left, right=right)
        ax_top = axis_eV_nm(plax, step)
        plax.set_xlabel('Wavelength (nm)')
        
    elif units == 'eV':
        for i in range(len(eV)):
            plax.plot(eV[i], I[i]+offset*ndx)
            ndx+=1
        plax.set_xlim(left=left, right=right)
        ax_top = axis_eV_nm(plax, step, base='eV')
        plax.set_xlabel('Energy (eV)')
    plfig.show()
    
    return plfig, plax, ax_top
    
def plotTdep(dataFile, sampleID, center=None, slit=None, grating=None, power=None, excitation = None, time = None, filt = None, scale = 'log', save = False, savedir = None, Tmin = None, Tmax = None, obeyExclude=False):
    
    samples = slicer(dataFile, sampleID=sampleID, center=center, slit=slit, grating=grating, power=power, excitation=excitation, filt=filt, time=time, exclude=obeyExclude)
    samples.sort_values(by=['temp (K)'], inplace=True)
    if Tmin:
        samples = samples[samples['temp (K)'] >= Tmin]
    if Tmax:
        samples = samples[samples['temp (K)'] <= Tmax]
    fig = plt.figure()
    ax = fig.gca()
    ax.set_prop_cycle(color=[plt.cm.viridis(i) for i in np.linspace(0,1,samples.shape[0])])
    for index, sample in samples.iterrows():
        ax.plot(sample['energy'], sample['intensity'], label=str(sample['temp (K)']) + ' K')
    ax.legend()
    ax.set_yscale(scale)
    ax.text(.5,1.1,sampleID, ha='center', transform=ax.transAxes)
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Counts per second (arb.)')
    fig.show()
    fig.tight_layout()
        
    if save:
        if path.isdir(savedir) == False:
            mkdir(savedir)
        chdir(savedir)
        savename = savedir + ' ' + sample['label']
        fig.savefig(savename + '.png')
        fig.savefig(savename + '.svg')
    return fig, ax
    
def plotPdep(dataFile, sampleID, temp = None, center=None, slit=None, grating=None, excitation=None, time = None, scale='log', save=False, savedir=None, obeyExclude=False):
    samples= slicer(dataFile, sampleID=sampleID, temp = temp, center=center, slit=slit, grating=grating, excitation=excitation, time=time, exclude=obeyExclude)
    fig = plt.figure()
    ax = fig.gca()
    for index, sample in samples.iterrows():
        ax.plot(sample['energy'], sample['intensity'], label=str(sample['Power (mW)']) + ' mW')
    ax.legend()
    ax.set_yscale(scale)
    ax.text(.5,1.1,sampleID, ha='center', transform=ax.transAxes)
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Counts per second (arb.)')
    fig.show()
    fig.tight_layout()
    
    if save:
        if path.isdir(savedir) == False:
            mkdir(savedir)
        chdir(savedir)
        savename = savedir + ' ' + sample['label']
        fig.savefig(savename + '.png')
        fig.savefig(savename + '.svg')
    return fig, ax
    
def multiPlot(dataFile, sampleID = None, temp = None, center=None, slit=None, grating=None, power=None, excitation=None, filt=None, time = None, scale='log', save=False, savedir=None, norm=False, fig=None, ax=None, xmin=None, xmax=None, obeyExclude=False):
    samples= slicer(dataFile, sampleID=sampleID, temp = temp, center=center, slit=slit, grating=grating, excitation=excitation, power=power, filt=filt, time=time, exclude=obeyExclude)
    if fig==None:
        fig = plt.figure()
    if ax == None:
        ax = fig.gca()
    for index, sample in samples.iterrows():
        label = sample['Sample ID'] + ' ' + str(sample['temp (K)']) + ' K ' + str(sample['Power (mW)']) + ' mW ' + str(sample['excitation wavelength (nm)']) + ' nm'
        if norm:
            ax.plot(sample['energy'], normalize(sample['energy'], sample['intensity'], xmin=xmin, xmax=xmax), label=label)
        else:
            ax.plot(sample['energy'], sample['intensity'], label=label)
    ax.legend()
    ax.set_yscale(scale)
    ax.set_xlabel('Energy (eV)')
    if norm == True:
        ax.set_ylabel('Normalized Counts')
    else:
        ax.set_ylabel('Counts per second (arb.)')
    fig.show()
    fig.tight_layout()
    
    return fig, ax
