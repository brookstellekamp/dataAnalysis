import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from dataAnalysis.plotter import axis_eV_nm

def getData(filename, sampleID, substrate = None):
    '''
    script to get data from Cary UV Vis into a useful format.
    ideal measurement procedure: Use a bare substrate as the 100%T reference, scan substrate in for the reflectance correction, and scan the sample in both reflectance and transmission mode. Then you can subtract the substrate reflectance from the sample reflectance.
    
    filename - path to csv file exported from Cary
    sampleID - list of sample IDs as given in the csv file
    substrate - if given, the program looks for a reflection column from the substrate as listed (name must match) and subtracts the reflectance from the sample reflectance.
    
    output - dataframe with multi-index columns. the first level of column indicies is titled 'Sample' and contains the sample ID. the second index level is named 'Data', and contains wavelength, energy, %T, %R, %R_corrected, and Absorbance for each sample.
    
    If no substrate is given, then %R_corrected will not be calibrated
    
    
    '''
    
    data = pd.read_csv(filename, header=[0,1]) #read data
    for col in data.iteritems(): #convert strings to floats
        col = pd.to_numeric(col, errors='coerce')
    
    dataColumns = [c for c in data.columns if c[0].split('.')[0] in sampleID] #get the sample IDs present
    newIndexTuples = data.columns.values 
    for loc, col in enumerate(data.columns): #this loop applies the sample ID multi-index to the %R or %T column. As reported the sample ID only appears in the wavelength column
        if col in dataColumns:
            
            tempSeries = data.iloc[:,loc+1]
            newName = (col[0], tempSeries.name[1])
            newIndexTuples[loc+1] = newName
    newIndex = pd.MultiIndex.from_tuples(newIndexTuples) #new multi-index 
    data = pd.DataFrame(data.values, columns=newIndex) #make a new dataframe with the correct multi-index
    # #check for duplicate columns
    # if type(sampleID) == list:
    #     if len(dataColumns) > len(sampleID):
    #         w = ' '.join(str(x[0]) for x in dataColumns)
    #         raise ImportError('Sample duplicates exist: %s'%w)
    
    multi_index = pd.MultiIndex.from_product([sampleID, ['wavelength', 'energy', '%T', '%R', 'Absorbance']], names=['Sample', 'Data'])
    outFrame = pd.DataFrame(columns = multi_index) #Here make the output dataframe which is more nicely formatted
    dataColumns = [c for c in data.columns if c[0].split('.')[0] in sampleID] #recalculate the data columns now that the indexes have been fixed
    
    for sample in sampleID: #iterate through the list of samples given
        
        loc = data.columns.get_loc(sample) #mask of the data for the sample of interest
        subdata = data.iloc[:,loc] #apply mask to dataframe
        wavelength = subdata[sample, 'Wavelength (nm)']
        R = subdata[sample, '%R']
        wavelength = wavelength[wavelength.notna()] #get rid of nan values
        R = R[R.notna()]
        outFrame[sample, 'wavelength'] = wavelength.values #put data into the output dataframe
        outFrame[sample, 'energy'] = 1239.84/wavelength.values
        outFrame[sample, '%R'] = R.values/100
        try: #because the substrate will not typically be taken in transmission mode, it is possible to not have %T data. 
            T = subdata[sample, '%T']
            T = T[T.notna()]/100
            outFrame[sample, '%T'] = T.values
            outFrame[sample, 'Absorbance'] = -np.log10(T.astype(np.float64)) #Calculate absorbance from %T, mostly for comparison with other tools
            outFrame[sample, 'light absorbed'] = 1-R-T
        except:
            pass
        
    if substrate: #If a substrate is specified, recursively call getData on the substrate to calculate it's reflectance, subtract it out, and put it in %R_corrected
        substrateData = getData(filename, sampleID=[substrate])
        for sample in sampleID:
            outFrame[sample, '%R_corrected'] = outFrame[sample, '%R'].values - substrateData[substrate, '%R']
            outFrame[sample, 'light absorbed corrected'] = 1-T - outFrame[sample, '%R_corrected']
    
    
    return outFrame.astype(np.float64)

def calcAlpha(df, sampleID, thickness, correction=True):
    '''
    Function to calculate absorption coefficient. Expects a dataframe from getData. if Correction is true it will use the %R_corrected values, else just %R defaults to correction=True
    
    inputs:
    df - dataframe from getData
    sampleID - list of sampleIDs in the df to calculate alpha on 
    thickness - list of thicknesses for each sample ID given, must be the same length as sampleID. Units are in nm
    correction - default True. Use %R-substrate%R
    
    output - returns the dataframe. As written the function will modify the original frame and storing the return is not necessary
    '''
    if len(sampleID) != len(thickness):
        raise ValueError('sampleID and thickness must be lists of the same length')
    for sample, thickness in zip(sampleID, thickness):
        T = df[sample, '%T'].values
        if correction:
            R = df[sample, '%R_corrected'].values
        else:
            R = df[sample, '%R'].values
        t = thickness*1E-7 #nm to cm
        df[sample, 'alpha'] = (1/t)*np.log(((1-R)**2)/T)
        df[sample, 'mu'] = df[sample,'Absorbance'].values/t
    return df

def plotFromFrame(df, sampleID, mode='abs', n=1/2, fig=None, ax=None, ax2=False):
    '''
    plotter function to plot absorption coefficient or tauc for UV-Vis data which comes from the getData function. alpha must be calculated beforehand with thickness values.
    
    inputs
    df - dataframe from getData
    sampleID - list of sample IDs to be plotted on the same plot
    mode - plotting mode. default 'abs' which is absorption coefficient in cm^-2
            other options include:
                                    'abs-T' - absorption coefficient from transmission data only
                                    'tauc' - tauc style plot of (alpha*E)^1/n vs E
    n - tauc coefficient. default to 1/2 for direct allowed. 2 for indirect allowed, 3 for direct forbidden, 3/2 for indirect forbiddens
    fig - matplotlib figure
    ax - matplotlib axes
    
    returns - fig, ax
    '''
    
    plt.style.use('publication')
    if fig == None:
        fig = plt.figure()
    if ax == None:
        ax = fig.gca()
    for sample in sampleID:
        if mode=='abs':
            ax.semilogy(df[sample, 'energy'], df[sample, 'alpha'])
        if mode=='abs-T':
            ax.semilogy(df[sample, 'energy'], df[sample, 'mu'])
        if mode=='tauc':
            y = (df[sample, 'alpha']*df[sample, 'energy'])**(1/n)
            ax.plot(df[sample, 'energy'], y)
        if mode == 'Absorbance':
            y = df[sample, 'Absorbance']
            ax.plot(df[sample, 'energy'], y)
        if mode == 'light absorbed':
            y = df[sample, 'light absorbed corrected']
            ax.plot(df[sample, 'energy'], y)
    ax.set_xlabel('Energy')
    if mode=='abs' or mode == 'abs-T':
        ax.set_ylabel(r'$\alpha$ (cm$^{-2}$)')
    if mode=='tauc':
        if n == 1/2:
            ax.set_ylabel(r'$(\alpha$$h\nu)^{2} (\frac{eV}{cm})^{2}$')
    if mode == 'Absorbance':
        ax.set_ylabel('Absorbance')
    if mode == 'light absorbed':
        ax.set_ylabel('1-R-T')
    if ax2:
        ax2 = axis_eV_nm(ax, 200, base='eV')
    fig.show()
    return fig, ax
