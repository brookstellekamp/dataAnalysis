from xml.etree import ElementTree as ET
import numpy as np
from dataAnalysis.functions import mse, sheet_resistance
from scipy.optimize import fsolve, leastsq
from uncertainties import ufloat, wrap
from uncertainties.umath import log as ulog
import pandas as pd
from scipy.stats import linregress, zscore
import matplotlib.pyplot as plt
import scipy.constants as const
from os import chdir
    
'''
This script contains functions to load, calculate, and plot hall data from the Lakeshore Hall using hrdml format files, which are xml files.
Many calculations including the reduced hall formulation are from https://doi.org/10.1063/1.4990470

Does not yet include a magnetoresistance correction. In order to do so, the magnetic field dependent sheet resistivity (or sheet resistance) must be measured and fit, then used to weight the sheet resistance used in the calculation of mobility. To be entirely correct, the sheet resistance measurement must also correct for measured temperature, where tenths of percentage point variation can lead to measureable variation in sheet resistance.

Brooks Tellekamp 2019 
Updated 10/2020

'''

dummy = ufloat(1,1)
uType = type(dummy)

q = const.elementary_charge

ns = {'i':"http://www.w3.org/2001/XMLSchema-instance"}

linFunc = lambda p, x: p[0] + p[1]*x
linErrFunc = lambda p, x, y, err: (y - linFunc(p, x))/err

def linearLeastSq(x, yarray):
    '''
    Take in x and y where y is an array of type ufloat and x is an array of the same size.
    return - ufloat(slope), ufloat(intercept)
    '''
    x = x.values
    y = [a.n for a in yarray]
    yerr = [a.s for a in yarray]
    p = []
    slopeGuess = (y[-1]-y[0])/(x[-1]-x[0])
    p.append(slopeGuess)
    p.append(y[0] - slopeGuess*x[0])
    out = leastsq(linErrFunc, p, args=(x, y, yerr), full_output=1)
    slope = out[0][1]
    intercept = out[0][0]
    cov = out[1]
    slopeErr = np.sqrt(cov[1][1])
    interceptErr = np.sqrt(cov[0][0])
    
    return ufloat(slope, slopeErr), ufloat(intercept, interceptErr)
    
def sheet_resistance_wrapper(Ra, Rb, guess):
    '''
    wrapper function to solve for Van der Pauw resistances Ra and Rb while propogating error
    '''
    solution = fsolve(sheet_resistance, guess, args=(Ra, Rb))[0]
    return solution
    

def singleHallMeasurement(hall, VDP, field, thickness, returnMode='normal'):
    '''
    Extract Hall parameters and Van der Pauw resistivity from a single Lakeshore Hall measurement.
    
    hall = first xml element [0] of the SingleGeometryResult element. type-ElementTree.element
    VDP = second xml element [0] of the SingleGeometryResult element. type-ElementTree.element
    
    field - field in Tesla
    thickness - thickness in nm
    
    returnMode:
    normal - mu, carriers (geometry averaged single values)
    full - RHall, RHall_err, VDP, VDP_err, mu, carriers (2 element np arrays with each geometry)
    debug - VHallC, IHallC, VHallC_err, VHallD, IHallD, VHallD_err, VDP_A, VDP_A_err, VDP_B, VDP_B_err
    
    '''
    
    
    VHallMatrix = []
    IHallMatrix = []
    VHall_err = []
    
    VVDPMatrix = []
    IVDPMatrix = []
    VVDP_err = []
    
    for current, voltage in zip(hall.iter(tag='RawCurrent'), hall.iter(tag='RawVoltage')):
        
        Itemp = (np.mean([float(x.text) for x in current.iter(tag='double')]))
        Vtemp = (np.mean([float(x.text) for x in voltage.iter(tag='double')]))
        Itemp_std = (np.std([float(x.text) for x in current.iter(tag='double')]))
        Vtemp_std = (np.std([float(x.text) for x in voltage.iter(tag='double')]))
        VHallMatrix.append(ufloat(Vtemp, Vtemp_std))
        IHallMatrix.append(ufloat(Itemp, Itemp_std))
    
    VHallMatrix = np.array(VHallMatrix)
    IHallMatrix = np.array(IHallMatrix)
        
    for (current, voltage) in zip(VDP.iter(tag='RawCurrent'), VDP.iter(tag='RawVoltage')):
        Itemp = (np.mean([float(x.text) for x in current.iter(tag='double')]))
        Vtemp = (np.mean([float(x.text) for x in voltage.iter(tag='double')]))
        Itemp_std = (np.std([float(x.text) for x in current.iter(tag='double')]))
        Vtemp_std = (np.std([float(x.text) for x in voltage.iter(tag='double')]))
        VVDPMatrix.append(ufloat(Vtemp, Vtemp_std))
        IVDPMatrix.append(ufloat(Itemp, Itemp_std))
    
    VVDPMatrix = np.array(VVDPMatrix)
    IVDPMatrix = np.array(IVDPMatrix)
    
    IntermediateVHall = []
    IntermediateVHall_err = []
    IntermediateIHall = []
    IntermediateVVDP = []
    IntermediateVVDP_err = []
    IntermediateIVDP = []

    for j in np.arange(0,8,2):
        IntermediateVHall.append((VHallMatrix[j] - VHallMatrix[j+1])/2)
        IntermediateIHall.append((IHallMatrix[j] - IHallMatrix[j+1])/2)
        IntermediateVVDP.append((VVDPMatrix[j] - VVDPMatrix[j+1])/2)
        IntermediateIVDP.append((IVDPMatrix[j] - IVDPMatrix[j+1])/2)
    
    VHallC = (IntermediateVHall[0]-IntermediateVHall[1])/2
    IHallC = (IntermediateIHall[0]+IntermediateIHall[1])/2
    
    VHallD = (IntermediateVHall[2]-IntermediateVHall[3])/2
    IHallD = (IntermediateIHall[2]+IntermediateIHall[3])/2
    
    RVDP = np.divide(np.array(IntermediateVVDP), np.array(IntermediateIVDP))
    
    VDP_A = wrap(sheet_resistance_wrapper)(RVDP[0], RVDP[1], 1000)
    VDP_B = wrap(sheet_resistance_wrapper)(RVDP[2], RVDP[3], 1000)
    
    VDP = np.array([VDP_A, VDP_B])
    
    RHall = np.divide([VHallC, VHallD], [IHallC, IHallD])

    mu_1 = np.abs(100**2*np.divide(RHall[0], VDP_A*field))
    mu_2 = np.abs(100**2*np.divide(RHall[1], VDP_B*field))
    
    mu = np.array([mu_1, mu_2])
    
    carriers1 = 1/(-q*thickness*1E-7*np.multiply(VDP_A, mu_1))
    carriers2 = 1/(-q*thickness*1E-7*np.multiply(VDP_B, mu_2))
    
    carriers = np.array([carriers1, carriers2])
    
    if returnMode == 'full':
        return RHall, VDP, mu, carriers
    elif returnmode == 'debug':
        return VHallC, IHallC, VHallD, IHallD, VDP_A, VDP_B
    elif returnMode == 'normal':
        return np.mean((mu_1, mu_2)), np.mean((carriers1, carriers2))
        
def TDepHall(source, thickness, field, returnMode = 'full'):
    '''
    Function to extract temperature dependent hall data from a lakeshore hall temperature loop using the Hall Measurement part. A wrapper of singleHallMeasurement for multiple temperature points
    Source - hrdml file
    thickness - sample thickness in nm
    field - excitation field in Tesla - may be removed in the future by mining metadata
    
    return modes same as singleHallMeasurement
    
    '''
    
    tree = ET.parse(source)
    root = tree.getroot()
    
    results = root.findall('.//*ResultNode')
    
    T_Results = root.findall('.//*[@i:type="TemperatureLoopResultNode"]/ResultNodes/ResultNode', ns)
    startT = float(root.findall('.//*StartingTemperature')[0].text)
    endT = float(root.findall('.//*EndingTemperature')[0].text)
    Tstep = float(root.findall('.//*StepSize')[0].text)
    Tarray = np.linspace(startT, endT, 1+(endT-startT)/Tstep)
    
    RHall = np.empty((Tarray.shape[0], 2), dtype=uType)
    VDP = np.empty_like(RHall)
    mu = np.empty_like(RHall)
    carriers = np.empty_like(RHall)
    VHallC = np.empty_like(RHall)
    IHallC = np.empty_like(RHall)
    VHallD = np.empty_like(RHall)
    IHallD = np.empty_like(RHall)
    VDP_A = np.empty_like(RHall)
    VDP_B = np.empty_like(RHall)

    
    for i, element in enumerate(T_Results):
        
        results = element.findall('.//*SingleGeometryResult')
        hallElement = results[0]
        VDPElement = results[1]
        
        result = singleHallMeasurement(hallElement, VDPElement, field, thickness, returnMode=returnMode)
        if returnMode == 'full':
            RHall[i,:] = result[0]
            VDP[i,:] = result[1]
            mu[i,:] = result[2]
            carriers[i,:] = result[3]
        elif returnMode == 'debug':
            VHallC[i,:] = result[0]
            IHallC[i,:] = result[1]
            VHallD[i,:] = result[2]
            IHallD[i,:] = result[3]
            VDP_A[i,:] = result[4]
            VDP_B[i,:] = result[5]
    
    for T, u, n in zip(Tarray, mu, carriers):
        mobility = np.mean((u[0], u[1]), axis=0)
        conc = np.mean((n[0], n[1]), axis=0)
        print('{0:.0f}K \tmobility = {1:2.2f}\tCarrier Concentration = {2:.2E}'.format(T, mobility, conc))
    if returnMode == 'full':
        return Tarray, RHall, VDP, mu, carriers
    elif returnMode == 'debug':
        return Tarray, VHallC, IHallC, VHallD, IHallD, VDP_A, VDP_B

def calcHallData(df, Rc0, Rd0):
    '''
    Take in a dataframe from a single B-field value (a row with columns - Field, Temp, Ra, Rb, Rc, Rd) along with the zero field resistances Rc0 = Ra - Rb and Rd0 = Rb-Ra
    return reduced hall data Rs, Zc, Zd
    '''
    try:
        init_guess = np.abs(df['Ra'].values[0])
        guess = init_guess
        try:
            Rs = wrap(sheet_resistance_wrapper)(df['Ra'].values[0], df['Rb'].values[0], guess)
        except RuntimeWarning:
            try:
                guess = init_guess/100
                Rs = wrap(sheet_resistance_wrapper)(df['Ra'].values[0], df['Rb'].values[0], guess)
            except RuntimeWarning:
                try:
                    guess = init_guess*100
                    Rs = wrap(sheet_resistance_wrapper)(df['Ra'].values[0], df['Rb'].values[0], guess)
                except RuntimeWarning:
                    try:
                        guess = init_guess/10000
                        Rs = wrap(sheet_resistance_wrapper)(df['Ra'].values[0], df['Rb'].values[0], guess)
                    except RuntimeWarning:
                        try:
                            guess = init_guess*10000
                            Rs = wrap(sheet_resistance_wrapper)(df['Ra'].values[0], df['Rb'].values[0], guess)
                        except RuntimeWarning as rw:
                            print(rw)
    except IndexError:
        print('caught index error at %0.2f K and %0.2f T'%(df['Temperature'].iloc[0], df['Field'].iloc[0]))
    
    Zc = (df['Rc'].values[0] - Rc0)/Rs
    Zd = (df['Rd'].values[0] - Rd0)/Rs
    
    return Rs, Zc, Zd

def FieldDepHall(*sources, thickness, removeOutliers = False):
    '''
    source - Input data file which is a Lakeshore Hall hres file (xml). Data should be taken as a set of Field loops at (optionally) different temperatures, recording R1243, R2314, R3421, R4132, R2431, and R1324 at every field point.
    
    returns: dataframes of intermediate, reduced, and final data.
    
    Intermediate data:  Field, Temperature, Ra, Rb, Rc, Rd, Vhall - for each measurement field
    
    Reduced data: Field, Temperature, Rs, Zc, Zd - calculated sheet resistance and reduced hall coefficients for each measurement field
    
    Final Data: Temperature, Mobility C, Mobility D, Carriers C, Carriers D, Resistivity, Normalized Mobility, Normalized Carriers, Standard Mobility, Standard Carriers - calculated for each temperature point
    
    Mobility C and Mobility D are extracted from the slope of the reduced hall resistance (Zc and Zd) vs field, and should be consistent.
    Normalized Mobility is extracted from the slope of the geometrically averaged normalized hall resistance vs field
    Standard Mobility is extracted from the slope of the geometrically averaged hall resistance versus field
    For each 'carriers' field, the Carrier concentration is determined from 1/(q*rho*u) using the mobility determined by each method.
    
    more info
    
    Ra = (R1243 + R3421)/2
    Rb = (R2314 + R4132)/2
    Rc = R1324
    Rd = R2431
    Each contact resistance incorporates current reversal, where in a perfect geometry R1243 = -R2143 - averaged as (R1243 - R2143)/2, which is identical to writing (R1243 - R2134)/2 (switched voltage convention)
    These are individually extracted and could be tracked, if desired. 
    
    Rs is calculated using the scipy numerical solver incorporating the equation - np.exp(-np.pi*(Ra/Rs)) + np.exp(-np.pi*(Rb/Rs))-1
    The numerical solver requires a guess. If it returns an error, the code will report the error but try again with 2 orders of mag smaller, then 2 OOM larger, then 4 OOM smaller, then 4 OOM larger. this is incorporated into calcHallData
    
    Normalized Hall Resistance is the Hall resistance normalized by the field-specific sheet resistance, RHall = Vhall/I = Rs(mu*B + misalignment), so that RHall_norm = mu*B + misalignment, where the slope of the field vs normalized hall resisntace is the mobility. The value is unitless.
    
    Zc and Zd are reduced hall coefficients, calculated by calcHallData, using the zero field voltage (Rc and Rd at B=0) in the form (Rc - Rc,0)/Rs where Rs is the sheet resistance. Thus Zc and Zd are unitless. Identity used --> Rc,0 = Ra-Rb and Rd,0 = -(Ra-Rb)
    
    '''
    
    thickness = thickness / 10**7
    
    intermediate = pd.DataFrame(columns = ['Field', 'Temperature', 'Ra', 'Rb', 'Rc', 'Rd', 'Hall Voltage'])
    reduced = pd.DataFrame(columns = ['Field', 'Temperature', 'Rs', 'Zc', 'Zd'])
    final = pd.DataFrame(columns = ['Temperature', 'Mobility C', 'Mobility D', 'Carriers C', 'Carriers D', 'Resistivity', 'Normalized Mobility', 'Normalized Carriers', 'Standard Mobility', 'Standard Carriers'])
    
    for source in sources:
        tree = ET.parse(source)
        root = tree.getroot()
        
        VMatrix = []
        IMatrix = []
        FieldMatrix = []
        TemperatureMatrix = []
        SetTemperatureMatrix = []
        ContactMatrix = []
        
        byContact = root.findall('.//*[@i:type="ResistanceMeasurementNode"]', ns)
        
        for c in byContact:
            
            negSource = c.find('.//NegativeSourceContact').text
            negMeas = c.find('.//NegativeMeasureContact').text
            posMeas = c.find('.//PositiveMeasureContact').text
            posSource = c.find('.//PositiveSourceContact').text
            
            contact = 'R' + posSource[-1] + negSource[-1] + posMeas[-1] + negMeas[-1]
        
            resistanceMeasurements = c.findall('.//*ResistanceMeasurementResult')
            
            for element in resistanceMeasurements:
                Ipos, Ineg = element.findall('.//*RawCurrent')
                Vpos, Vneg = element.findall('.//*RawVoltage')
                
                VposTemp = (np.mean([float(x.text) for x in Vpos.iter(tag='double')]))
                Vpos_std = (np.std([float(x.text) for x in Vpos.iter(tag='double')]))
                
                VnegTemp = (np.mean([float(x.text) for x in Vneg.iter(tag='double')]))
                Vneg_std = (np.std([float(x.text) for x in Vpos.iter(tag='double')]))
                
                IposTemp = (np.mean([float(x.text) for x in Ipos.iter(tag='double')]))
                Ipos_std = (np.std([float(x.text) for x in Ipos.iter(tag='double')]))
                
                InegTemp = (np.mean([float(x.text) for x in Ineg.iter(tag='double')]))
                Ineg_std = (np.std([float(x.text) for x in Ineg.iter(tag='double')]))
                
                Vpos = ufloat(VposTemp, Vpos_std)
                Vneg = ufloat(VnegTemp, Vneg_std)
                Ipos = ufloat(IposTemp, Ipos_std)
                Ineg = ufloat(InegTemp, Ineg_std)
                
                VMatrix.append(.5*(Vpos-Vneg))
                IMatrix.append(.5*(Ipos-Ineg))
                FieldMatrix.append(np.around(float(element.find('.//*Field').text), decimals=2))
                Temperature = float(element.find('.//*Temperature').text)
                TemperatureMatrix.append(Temperature)
                SetTemperatureMatrix.append(np.round(Temperature))
                ContactMatrix.append(contact)
        
        data = pd.DataFrame({'SetTemperature':SetTemperatureMatrix, 'Temperature':TemperatureMatrix, 'Field':FieldMatrix, 'Contact':ContactMatrix, 'Voltage':VMatrix, 'Current':IMatrix, 'Resistance':np.divide(np.array(VMatrix), np.array(IMatrix))})
        
        sortedData =  data.sort_values(by=['SetTemperature', 'Field', 'Contact'])
        
        B = []
        T = []
        Ra = []
        Rb = []
        Rc = []
        Rd = []
        Vhall = []
        
        Temps = sortedData.SetTemperature.unique()
        Temps.sort()
        Fields = sortedData.Field.unique()
        Fields.sort()
        
        for temp in Temps:
            T_subData = sortedData[sortedData['SetTemperature'] == temp]
            for field in Fields:
                try:
                    B_subData = T_subData[T_subData['Field'] == field]
                    Ra.append((B_subData[B_subData['Contact'] == 'R1243']['Resistance'].values[0] + B_subData[B_subData['Contact'] == 'R3421']['Resistance'].values[0])/2)
                    Rb.append((B_subData[B_subData['Contact'] == 'R2314']['Resistance'].values[0] + B_subData[B_subData['Contact'] == 'R4132']['Resistance'].values[0])/2)
                    Rc.append(B_subData[B_subData['Contact'] == 'R1324']['Resistance'].values[0])
                    Rd.append(B_subData[B_subData['Contact'] == 'R2431']['Resistance'].values[0])
                    B.append(B_subData['Field'].iloc[0])
                    T.append(B_subData['SetTemperature'].iloc[0])
                    VhallC = B_subData[B_subData['Contact'] == 'R1324']['Voltage'].values[0]
                    VhallD = B_subData[B_subData['Contact'] == 'R2431']['Voltage'].values[0]
                    Vhall.append((VhallC+VhallD)/2)
                except IndexError:
                    print('Empty dataframe for %0.2f K and %0.2f T'%(temp, field))
        
        
        intermediateData = pd.DataFrame({'Field':np.array(B), 'Temperature':np.array(T), 'Ra':np.array(Ra), 'Rb':np.array(Rb), 'Rc':np.array(Rc), 'Rd':np.array(Rd), 'Hall Voltage':Vhall})
        intermediate = pd.concat([intermediate, intermediateData])
        
        # for each T --> B, Rs(B), Rc(B), Rd(B) Rc,0, Rd,0, Zc(B), Zd(B)
        B = []
        T = []
        Rs = []
        Rc = []
        Rd = []
        Zc = []
        Zd = []
        
        for t in Temps:
            T_subData = intermediateData[intermediateData['Temperature'] == t]
            ZeroField = T_subData[T_subData['Field'] == 0.01]
            Rc0_temp = ZeroField['Ra'].values[0] - ZeroField['Rb'].values[0]
            Rd0_temp = ZeroField['Rb'].values[0] - ZeroField['Ra'].values[0]
            for field in Fields:
                B_subData = T_subData[T_subData['Field'] == field]
                Rs_temp, Zc_temp, Zd_temp = calcHallData(B_subData, Rc0_temp, Rd0_temp)
                Rs.append(Rs_temp)
                Zc.append(Zc_temp)
                Zd.append(Zd_temp)
                T.append(t)
                B.append(field)
        
        reducedHall = pd.DataFrame({'Field':np.array(B), 'Temperature':np.array(T), 'Rs':np.array(Rs), 'Zc':np.array(Zc), 'Zd':np.array(Zd)})
        reduced = pd.concat([reduced, reducedHall])
        
        mu_1 = []
        mu_2 = []
        rho = []
        carriers1 = []
        carriers2 = []
        
        normMobility = []
        normCarriers = []
        
        standMobility = []
        standCarriers = []
        
        
        for t in Temps:
            
            T_subData = reducedHall[reducedHall['Temperature'] == t]
            x = T_subData['Field']
            #Reduced Hall Resistance
            Zc_vals = T_subData['Zc']
            Zd_vals = T_subData['Zd']
            
            Zc_line = linearLeastSq(x, Zc_vals)
            Zd_line = linearLeastSq(x, Zd_vals)
            u1 = Zc_line[0]*100**2
            mu_1.append(u1)
            u2 = Zd_line[0]*100**2
            mu_2.append(u2)
            ZeroFieldRs = T_subData[np.abs(T_subData['Field']) == 0.01]['Rs'].values[0]
            carriers1.append(1/(-q*ZeroFieldRs*thickness*u1))
            carriers2.append(1/(-q*ZeroFieldRs*thickness*u2))
            
            #Normalized Hall Resistance
            T_subData2 = intermediateData[intermediateData['Temperature'] == t]
            T_subData2_Reduced = reducedHall[reducedHall['Temperature'] == t]
            Rc = T_subData2['Rc']
            Rd = T_subData2['Rd']
            reducedHallR = (Rc + Rd)/(2*T_subData2_Reduced['Rs'])
            values = np.array([a.n for _,a in reducedHallR.iteritems()])
            x2 = x
            if removeOutliers:
                z = np.abs(zscore(values))
                reducedHallR = reducedHallR[z < 2]
                x2 = x[z<2]
                print('%d outliers removed from Normalized Hall'%(int(len(x) - len(x2))))
            lineR = linearLeastSq(x2, reducedHallR)
            mobilityTemp = lineR[0]*100**2
            normMobility.append(mobilityTemp)
            normCarriers.append(1/(-q*ZeroFieldRs*thickness*mobilityTemp))
            
            #Standard Hall Resistance
            standardHallR = (Rc+Rd)/2
            values = np.array([a.n for _,a in standardHallR.iteritems()])
            x3 = x
            if removeOutliers:
                z = np.abs(zscore(values))
                standardHallR = standardHallR[z < 2]
                x3 = x[z < 2]
                print('%d outliers removed from Standard Hall'%(int(len(x) - len(x3))))
            lineS = linearLeastSq(x3, standardHallR)
            mobilityTemp2 = lineS[0]*100**2/ZeroFieldRs
            standMobility.append(mobilityTemp2)
            standCarriers.append(1/(-q*ZeroFieldRs*thickness*mobilityTemp2))
            
            rho.append(ZeroFieldRs*thickness)
        
        finalData = pd.DataFrame({'Temperature':Temps, 'Mobility C':mu_1, 'Mobility D':mu_2, 'Carriers C':carriers1, 'Carriers D':carriers2, 'Resistivity':rho, 'Normalized Mobility':normMobility, 'Normalized Carriers':normCarriers, 'Standard Mobility':standMobility, 'Standard Carriers':standCarriers})
        final = pd.concat([final, finalData])
    
    
    return intermediate, reduced, final
    
def plotResistivity(*finalData, fig=None, ax=None, arrhenius=False, fitRange=None):
    '''
    plotter function to plot resistivity vs temperature for T-dep hall, takes in a 'final' dataframe
    
    The Arrhenius fit does not seem to be working correctly, must be fixed.
    
    returns fig, ax
    '''
    plt.style.use('publication')
    if fig==None:
        fig = plt.figure()
    if ax == None:
        ax = fig.gca()
    for data in finalData:
        
        if arrhenius:
            T = 1/data['Temperature']
            yvals = np.array([ulog(a) for _,a in data['Resistivity'].iteritems()])
            rho = np.array([a.n for a in yvals])
            rho_err = np.array([a.s for a in yvals])
        else:
            T = data['Temperature']
            yvals = data['Resistivity']
            rho = np.array([a.n for _,a in yvals.iteritems()])
            rho_err = np.array([a.s for _,a in yvals.iteritems()])
            
        points = ax.errorbar(T, rho, yerr=rho_err, lw=0, elinewidth=1, marker='o')
        
    if fitRange and arrhenius:
        fitmin = 1/fitRange[1]
        fitmax = 1/fitRange[0]
        
        indicies = np.argsort(np.array(T))
        
        x1 = int(np.searchsorted(T, fitmin, sorter = indicies))
        x2 = int(np.searchsorted(T, fitmax, side='right', sorter = indicies))

        x = T.iloc[indicies[x1:x2]]
        y = yvals[indicies[x1:x2][::-1]]
        fit = linearLeastSq(x, y)
        ax.plot(x[::-1], x*fit[0].n + fit[1].n, ls = '--', color=points.get_children()[0].get_color())
    
    if arrhenius:
        ax.set_xlabel('1/T (K$^{-1}$)')
        ax.set_ylabel(r'ln($\rho$) (ln($\Omega\cdot$cm))')
    else:
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Resistivity ($\Omega\cdot$cm)')
    
    if arrhenius == False:
        ax.set_yscale('log')
    fig.show()
    
    if arrhenius and fitRange:
        return fig, ax, fit[0]
    else:
        return fig, ax
        
def plotConductivity(*finalData, fig=None, ax=None, arrhenius=False, fitRange=None):
    '''
    plotter function to plot conductivity vs temperature for T-dep hall, takes in a 'final' dataframe
    
    returns fig, ax
    '''
    plt.style.use('publication')
    if fig==None:
        fig = plt.figure()
    if ax == None:
        ax = fig.gca()
    for data in finalData:
        
        if arrhenius:
            T = 1/data['Temperature']
            yvals = np.array([1/ulog(a) for _,a in data['Resistivity'].iteritems()])
            sigma = np.array([a.n for a in yvals])
            sigma_err = np.array([a.s for a in yvals])
        else:
            T = data['Temperature']
            yvals = data['Resistivity']
            sigma = np.array([a.n for _,a in yvals.iteritems()])
            sigma_err = np.array([a.s for _,a in yvals.iteritems()])
            
        points = ax.errorbar(T, sigma, yerr=sigma_err, lw=0, elinewidth=1, marker='o')
        
    if fitRange and arrhenius:
        fitmin = 1/fitRange[1]
        fitmax = 1/fitRange[0]
        
        indicies = np.argsort(np.array(T))
        
        x1 = int(np.searchsorted(T, fitmin, sorter = indicies))
        x2 = int(np.searchsorted(T, fitmax, side = 'right', sorter = indicies))
        
        x = T.iloc[indicies[x1:x2]]
        y = yvals[indicies[x2:x1:-1]]
        fit = linearLeastSq(x, y)
        ax.plot(x[::-1], x*fit[0].n + fit[1].n, ls = '--', color=points.get_children()[0].get_color())
    
    if arrhenius:
        ax.set_xlabel('1/T (K$^{-1}$)')
        ax.set_ylabel(r'ln($\sigma$) (ln($\frac{S}{cm}$))')
    else:
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Conductivity (\frac{S}{cm}$)')
    
    if arrhenius == False:
        ax.set_yscale('log')
    fig.show()
    
    if arrhenius and fitRange:
        return fig, ax, fit[0]
    else:
        return fig, ax
    
def plotActivationEnergy(*finalData, fig=None, ax=None):
    '''
    plotter function to plot geometrically averaged carrier concentration vs. 1/temperature
    takes a 'final data' dataframe
    returns fig, ax
    '''
    plt.style.use('publication')
    if fig==None:
        fig = plt.figure()
    if ax == None:
        ax = fig.gca()
    for data in finalData:
        T = data['Temperature']
        x = 1/T
        yvals = np.mean([data['Carriers C'], data['Carriers D']], axis=0)
        y = [a.n for a in yvals]
        y_err = [a.s for a in yvals]
        ax.errorbar(x, y, yerr=y_err, lw=0, elinewidth=1, marker='o')
    ax.set_xlabel('1/T (K$^{-1}$)')
    ax.set_ylabel('Carrier Concentration (cm$^{-3}$)')
    ax.set_yscale('log')
    fig.show()
    return fig, ax
        
    
def plotMobilityCarriers(*finalData, fig=None, method='reduced'):
    '''
    plotter function to plot geometrically averaged carrier concentration and mobility versus temperature on separate y-axes
    takes a 'final data' dataframe
    
    method - which mobility calculation to use. options are:
            'reduced' - reduced hall mobility (Rc-Rc,0)/Rsh
            'normalized' - geometrically averaged Rc,d/Rsh
            'standard' - geometrically averaged Rc,d
    returns fig, ax
    '''
    plt.style.use('publication')
    if fig==None:
        fig = plt.figure()
    ax1 = fig.gca()
    ax2 = ax1.twinx()
    for data in finalData:
        T = data['Temperature']
        
        if method == 'reduced':
            y1vals = np.mean([data['Carriers C'], data['Carriers D']], axis=0)
            y1 = [a.n for a in y1vals]
            y1_err = [a.s for a in y1vals] 
            y2vals = np.abs(np.mean([data['Mobility C'], data['Mobility D']], axis=0))
            y2 = [a.n for a in y2vals]
            y2_err = [a.s for a in y2vals] 
        elif method == 'normalized':
            y1vals = data['Normalized Carriers']
            y1 = [a.n for a in y1vals]
            y1_err = [a.s for a in y1vals] 
            y2vals = np.abs(data['Normalized Mobility'])
            y2 = [a.n for a in y2vals]
            y2_err = [a.s for a in y2vals] 
        elif method == 'standard':
            y1vals = data['Standard Carriers']
            y1 = [a.n for a in y1vals]
            y1_err = [a.s for a in y1vals] 
            y2vals = np.abs(data['Standard Mobility'])
            y2 = [a.n for a in y2vals]
            y2_err = [a.s for a in y2vals] 
        ax1.errorbar(T, y1, yerr=y1_err, lw=0, elinewidth=1, marker='o')
        ax2.errorbar(T, y2, yerr=y2_err, lw=0, elinewidth=1, marker='s')
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('Carrier Concentration (cm$^{-3}$)')
    ax2.set_ylabel(r'Mobility ($\frac{cm^{2}}{V \cdot s}$)')
    fig.show()
    
def plotVHall(intermediateData, fig=None, ax=None):
    '''
    plotter function to plot hall voltage as a function of applied field
    takes in an 'intermediate data' dataframe
    returns fig, ax
    '''
    plt.style.use('publication')
    if fig==None:
        fig = plt.figure()
    if ax == None:
        ax = fig.gca()
    B = intermediateData['Field']
    y = [a.n for a in intermediateData['Hall Voltage']]
    yerr = [a.s for a in intermediateData['Hall Voltage']]
    
    ax.errorbar(B, y, yerr=yerr, lw=0, elinewidth=1, marker='o')
    ax.set_xlabel('Field (T)')
    ax.set_ylabel('Hall Voltage (V)')
    
    fig.show()
    fig.tight_layout()
    return fig, ax
    
def plotSheetResistance(reducedData, fig=None, ax=None):
    '''
    plotter function to plot sheet resistance as a function of applied field
    inputs - 'reduced data' dataframe
    returns fig, ax
    '''
    plt.style.use('publication')
    if fig==None:
        fig = plt.figure()
    if ax == None:
        ax = fig.gca()
    B = reducedData['Field']
    y = [a.n for a in reducedData['Rs']]
    yerr = [a.s for a in reducedData['Rs']]
    
    ax.errorbar(B, y, yerr=yerr, lw=0, elinewidth=1, marker='o')
    ax.set_xlabel('Field (T)')
    ax.set_ylabel('R$_{sh}$ ($\Omega$/sq)')
    
    fig.show()
    fig.tight_layout()
    return fig, ax
    
def plotNormalizedHallResistance(intermediateData, reducedData, fig=None, ax=None, label=True, removeOutliers = False):
    '''
    plotter function to plot normalized hall resistance versus magnetic field
    
    Normalized hall resistance is the geometrically averaged hall resistance divided by the sheet resistance at each field
    inputs 'intermediate data' and 'reduced data' dataframes
    returns - fig, ax, and mobility which is the slope from a linear regression in cm*2/V*s 
    '''
    plt.style.use('publication')
    if fig==None:
        fig = plt.figure()
    if ax == None:
        ax = fig.gca()
    B = intermediateData['Field']
    Rc = intermediateData['Rc']
    Rd = intermediateData['Rd']
    yvals = (Rc+Rd)/(2*reducedData['Rs'])
    y = np.array([a.n for _,a in yvals.iteritems()])
    yerr = np.array([a.s for _,a in yvals.iteritems()])
    
    points = ax.errorbar(B, y, yerr=yerr, lw=0, elinewidth=1, marker='s', zorder=.5)
    
    if removeOutliers:
        z = np.abs(zscore(y))
        yvals = yvals[z < 1]
        B = B[z<1]
    
    line = linearLeastSq(B, yvals)
    ax.plot(B, B*line[0].n + line[1].n, ls='--', color='k', zorder=1)
    
    if label:
        ax.set_xlabel('Field (T)')
        ax.set_ylabel(r'Normalized Hall Resistance $\frac{R_{H}}{R_{SH}}$')
    
    str1 = '$\mu_{H}$ =' + ' {:2.2E}'.format(line[0]*100**2)
    ax.text(.5, .9, str1, color='tab:blue', fontsize = 'small', ha='center', transform = ax.transAxes)
    
    fig.show()
    return fig, ax, line[0]*100**2
    
def calcNormalizedHallResistanceGrid(intermediateData, reducedData, thickness, removeOutliers = False):
    '''
    
    '''
    
    thickness = thickness / 10**7
    
    plt.style.use('publication')
    
    temps = reducedData.Temperature.unique()
    temps.sort()
    oldtemps = temps
    
    while temps.shape[0] is not int(np.around(np.sqrt(temps.shape))[0])**2:
        temps = np.append(temps, np.nan)
    
    temps = temps.reshape((int(np.around(np.sqrt(temps.shape))), -1))
    
    fig, axes = plt.subplots(temps.shape[0], temps.shape[1], sharex=True, squeeze=True, figsize=(11,7))
    
    it = np.nditer(temps, flags=['f_index', 'multi_index'])
    slopes = []
    carriers = []
    
    while not it.finished:
        if np.isnan(it.value.item()) == False:
            subI = intermediateData[intermediateData['Temperature'] == it[0]]
            subR = reducedData[reducedData['Temperature'] == it[0]]
            _, _, slope = plotNormalizedHallResistance(subI, subR, fig=fig, ax=axes[it.multi_index], label=None, removeOutliers = removeOutliers)
            slopes.append(slope)
            ZeroFieldRs = subR[subR['Field'] == 0.01]['Rs'].values[0]
            carriers.append(1/(-q*slope*ZeroFieldRs*thickness))
            axes[it.multi_index].text(.5, 1.05, '{:.0f} K'.format(it[0]), transform = axes[it.multi_index].transAxes)
        it.iternext()
        
    for ax in axes.flatten():
        ax.set_xticklabels([])
        ax.set_yticklabels([])
            
    fig.show()
    fig.tight_layout()
    fig.subplots_adjust(hspace=.19, wspace=0)
    
    slopes = np.array(slopes)
    return fig, axes, pd.DataFrame({'Temperature':oldtemps, 'Mobility':slopes, 'Carriers':carriers})
    
def plotNormalizedHallMobilityCarriers(*normData, fig = None, ax = None):
    plt.style.use('publication')
    if fig==None:
        fig = plt.figure()
    ax1 = fig.gca()
    ax2 = ax1.twinx()
    for data in normData:
        T = data['Temperature']
        y1vals = data['Carriers']
        y2vals = data['Mobility']
        y1 = [a.n for a in y1vals]
        y1_err = [a.s for a in y1vals] 
        y2 = [np.abs(a.n) for a in y2vals]
        y2_err = [a.s for a in y2vals] 
        ax1.errorbar(T, y1, yerr=y1_err, lw=0, elinewidth=1, marker='o')
        ax2.errorbar(T, y2, yerr=y2_err, lw=0, elinewidth=1, marker='o', markerfacecolor='None')
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('Carrier Concentration (cm$^{-3}$)')
    ax2.set_ylabel(r'Mobility ($\frac{cm^{2}}{V \cdot s}$)')
    fig.show()
    
    return fig, ax1, ax2

def plotReducedHall(reducedData, fig = None, ax = None, doubleAxis = False, ax2 = None, label=None):
    '''
    plotter function to plot the reduced hall resistance versus applied magnetic field.
    
    reduced hall resistance, Zc and Zd, are the zero-field corrected hall resistances (Rc - Rc,0), (Rd - Rd,0) normalized against sheet resistance at each field.
    
    input - 'reduced data' dataframe
    returns fig, ax
    '''
    plt.style.use('publication')
    if fig==None:
        fig = plt.figure()
    if ax == None:
        if doubleAxis:
            axes = fig.subplots(2,1,sharex=True)
        else:
            ax = fig.gca()
    elif doubleAxis:
        if ax2 == None:
            ax2 = fig.add_subplot(211, sharex=True)
        axes = (ax, ax2)
                
    B = reducedData['Field']
    y1_vals = reducedData['Zc']
    y2_vals = reducedData['Zd']
    y1 = [a.n for _,a in y1_vals.iteritems()]
    y1_err = [a.s for _,a in y1_vals.iteritems()]
    y2 = [b.n for _,b in y2_vals.iteritems()]
    y2_err = [b.s for _,b in y2_vals.iteritems()]
    
    y1_line = linearLeastSq(B, y1_vals)
    y2_line = linearLeastSq(B, y2_vals)
    
    if doubleAxis:
        axes[0].errorbar(B, y1, yerr=y1_err, lw=0, elinewidth=1, marker='o', color='tab:blue', zorder=0.5)
        axes[0].plot(B, y1_line[0].n*B + y1_line[1].n, ls = '--', color='k', zorder = 1)
        axes[1].errorbar(B, y2, yerr=y2_err, lw=0, elinewidth=1, marker='o', color='tab:orange', zorder = 0.5)
        axes[1].plot(B, y2_line[0].n*B + y2_line[1].n, ls = '--', color='k', zorder = 1)
    else:
        ax.errorbar(B, y1, yerr=y1_err, lw=0, elinewidth=1, marker='o', color='tab:blue', zorder = 0.5)
        ax.plot(B, y1_line[0].n*B + y1_line[1].n, ls = '--', color='k', zorder = 1)
        ax.errorbar(B, y2, yerr=y2_err, lw=0, elinewidth=1, marker='o', color='tab:orange', zorder = 0.5)
        ax.plot(B, y2_line[0].n*B + y2_line[1].n, ls = '--', color='k', zorder = 1)
    
    if y1_line[0] < .1:
        str1 = '$\mu_{H}$ =' + ' {:2.2E}'.format(y1_line[0]*100**2)
        str2 = '$\mu_{H}$ =' + ' {:2.2E}'.format(y2_line[0]*100**2)
    else:
        str1 = '$\mu_{H}$ =' + ' {:2.2f}'.format(y1_line[0]*100**2)
        str2 = '$\mu_{H}$ =' + ' {:2.2f}'.format(y2_line[0]*100**2)
        
    if doubleAxis:
        axes[0].text(.5, .9, str1, color='tab:blue', fontsize = 'small', ha='center', transform = axes[0].transAxes)
        axes[1].text(.5, .9, str2, color='tab:orange', fontsize = 'small', ha='center', transform=axes[1].transAxes)
    else:
        ax.text(.05, .4, str1, color='tab:blue', fontsize = 'small', transform = ax.transAxes)
        ax.text(.05, .6, str2, color='tab:orange', fontsize = 'small', transform=ax.transAxes)
    
    if label:
        ax.set_xlabel('Field (T)')
        ax.set_ylabel('Reduced Hall Resistance (unitless)')
    
    
    return fig, ax
    
def plotReducedHallGrid(reducedData):
    '''
    plot reduced hall resistance (hall resistance normalized by sheet resistance at each field point) versus field for each temperature point on an individual subplot
    input - 'reduced data' dataframe
    returns fig, axes array
    '''
    
    plt.style.use('publication')
    
    temps = reducedData.Temperature.unique()
    temps.sort()
    
    while temps.shape[0] is not int(np.around(np.sqrt(temps.shape))[0])**2:
        temps = np.append(temps, np.nan)
    
    temps = temps.reshape((int(np.around(np.sqrt(temps.shape))), -1))
    
    fig, axes = plt.subplots(temps.shape[0], temps.shape[1], sharex=True, squeeze=True, figsize=(11,7))
    
    it = np.nditer(temps, flags=['f_index', 'multi_index'])
    
    while not it.finished:
        if np.isnan(it.value.item()) == False:
            subData = reducedData[reducedData['Temperature'] == it[0]]
            plotReducedHall(subData, fig=fig, ax=axes[it.multi_index])
            axes[it.multi_index].text(.5, 1.05, '{:.0f} K'.format(it[0]), transform = axes[it.multi_index].transAxes)
        it.iternext()
        
    for ax in axes.flatten():
        ax.set_xticklabels([])
        ax.set_yticklabels([])
            
    fig.show()
    fig.tight_layout()
    fig.subplots_adjust(hspace=.19, wspace=0)
    return fig, axes
    
def calculateCondTensor(intermediateData, reducedData, finalData):
    '''
    Calculate the conductivity tensor using reduced hall resistanes and sheet resistance
    inputs - 'intermediate data', 'reduced data', 'final data' dataframes
    returns - calculated Tensor dataframe with columns from reduced data along with Temperature, Gxx, Gxy, Sigma xx, and Sigma xy
    '''
    
    Rxx = reducedData['Rs']
    Rxy = (intermediateData['Rc']+intermediateData['Rd'])/2
    
    #Longitudinal and transverse components of the conductivity tensor - tensor is formed as [xx, xy; -xy, xx]
    Gxx = Rxx/(Rxx**2+Rxy**2)
    Gxy = Rxy/(Rxx**2+Rxy**2)
    
    calcTens = pd.concat([reducedData, Gxx.rename('Gxx'), Gxy.rename('Gxy')], axis=1)
    
    #Zero Field component for tensor normalization
    for temp in calcTens.Temperature.unique():
        T_subData = calcTens[calcTens['Temperature'] == temp]
        G0 = T_subData[T_subData['Field'] == np.abs(0.01)]['Gxx']
    
        calcTens.loc[calcTens['Temperature'] == temp, 'Xm'] = calcTens[calcTens['Temperature'] == temp]['Gxx']/G0.values
        calcTens.loc[calcTens['Temperature'] == temp, 'Ym'] = 2*calcTens[calcTens['Temperature'] == temp]['Gxy']/G0.values
    
    #Calculated from extracted data
        T_subData_2 = finalData[finalData['Temperature'] == temp]
        n = (T_subData_2['Carriers C'] + T_subData_2['Carriers D'])/2
        mobility = (T_subData_2['Mobility C'] + T_subData_2['Mobility D'])/2
        
        sigma_0 = n*q*mobility
        gamma = mobility.values*T_subData['Field']/100**2
        
        calcTens.loc[calcTens['Temperature'] == temp, 'Sigma xx'] = sigma_0.values/(1+gamma**2)
        calcTens.loc[calcTens['Temperature'] == temp, 'Sigma xy'] = gamma*(sigma_0.values/(1+gamma**2))
    
    
    return calcTens
    
def plotCompare(intermediateData,reducedData, removeOutliers = False):
    '''
    plots hall resistance, normalized hall resistance, and reduced hall resistance as a function of field
    inputs - 'intermediate data' and 'reduced data' dataframes
    '''
    figs = []
    ax = []
    for i, temperature in enumerate(reducedData['Temperature'].unique()):
        plt.style.use('publication')
        fig, axes = plt.subplots(4, 1, sharex=True, figsize = (7.15, 13.3))
        i = intermediateData[intermediateData['Temperature'] == temperature]
        r = reducedData[reducedData['Temperature'] == temperature]
        Rc = i['Rc'].values
        Rd = i['Rd'].values
        Rcd = (Rc+Rd)/2
        y1 = np.array([a.n for a in Rcd])
        y1err = np.array([a.s for a in Rcd])
        B = r['Field']
        points1 = axes[0].errorbar(B, y1, yerr = y1err, lw=0, elinewidth=1, marker='o', zorder=0.5)
        
        if removeOutliers:
            z = np.abs(zscore(y1))
            Rcd = Rcd[z<2]
            B = B[z<2]
        
        line1 = linearLeastSq(B, Rcd)
        axes[0].plot(B, B*line1[0].n + line1[1].n, ls = '--', color='k', zorder=1)
        
        ZeroFieldResistance = r[r['Field'] == 0.01]['Rs'].values[0]
        str1 = '$\mu_{H}$ =' + ' {:2.2E}'.format(line1[0]*100**2/ZeroFieldResistance)
        axes[0].text(.5, .9, str1, ha='center', transform = axes[0].transAxes, fontsize='small')
        
        plotNormalizedHallResistance(i, r, fig=fig, ax=axes[1], label=None, removeOutliers = removeOutliers)
        
        plotReducedHall(r, fig=fig, ax=axes[2], doubleAxis=True, ax2=axes[3])
        
        axes[3].set_xlabel('Field (T)')
        axes[0].set_ylabel('R$_{H}$ ($\Omega$)')
        axes[1].set_ylabel(r'$\frac{R_{H}}{R_{sh}}$')
        axes[2].set_ylabel(r'$\xi_{c}$')
        axes[3].set_ylabel(r'$\xi_{d}$')
        
        fig.canvas.draw()
        fig.tight_layout()
        figs.append(fig)
        ax.append(axes)
    figs = np.array(figs)
    ax = np.array(ax)
    if len(figs) < 2:
        figs = figs[0]
        ax = ax[0]
    else:
        figs = np.array(figs)
        ax = np.array(ax)
    
    return figs, ax
    
def report(f):
    for i, temp in enumerate(f['Temperature'].unique()):
        print('Temperature: {0:.1f}'.format(temp))
        print('Reduced Hall Resistance Data:')
        print('Carrier Concentration (Geometry C): {0:.2E} +/- {1:.2E} cm^-3'.format(f['Carriers C'].iloc[i].n, f['Carriers C'].iloc[i].s))
        print('Carrier Concentration (Geometry D): {0:.2E} +/- {1:.2E} cm^-3'.format(f['Carriers D'].iloc[i].n, f['Carriers D'].iloc[i].s))
        print('Mobility (Geometry C): {0:.2f} +/- {1:.2f} cm^2/V-s'.format(f['Mobility C'].iloc[i].n, f['Mobility C'].iloc[i].s))
        print('Mobility (Geometry D): {0:.2f} +/- {1:.2f} cm^2/V-s'.format(f['Mobility D'].iloc[i].n, f['Mobility D'].iloc[i].s))
        print('\n')
        print('Normalized Hall Resistance Data:')
        print('Carrier Concentration: {0:.2E} +/- {1:.2E} cm^-3'.format(f['Normalized Carriers'].iloc[i].n, f['Normalized Carriers'].iloc[i].s))
        print('Mobility: {0:.2f} +/- {1:.2f} cm^2/V-s'.format(f['Normalized Mobility'].iloc[i].n, f['Normalized Mobility'].iloc[i].s))
        print('\n')
        print('Standard Hall Resistance Data:')
        print('Carrier Concentration: {0:.2E} +/- {1:.2E} cm^-3'.format(f['Standard Carriers'].iloc[i].n, f['Standard Carriers'].iloc[i].s))
        print('Mobility: {0:.2f} +/- {1:.2f} cm^2/V-s'.format(f['Standard Mobility'].iloc[i].n, f['Standard Mobility'].iloc[i].s))
        print('---------------------------------\n\n\n')
        
        
def removeTemp(filename, temperature, saveDirectory, newFilename):
    tree = ET.parse(filename)
    root = tree.getroot()
    
    elementList = []
    
    contactParent = root.findall('.//*[@i:type="ResistanceMeasurementNode"]...', ns)
    for parent in contactParent:
        for c in parent.getchildren():
            try:
                T = np.around(float(c.find('.//*Temperature').text), decimals=0)
            except(AttributeError):
                break
            if T == temperature:
                parent.remove(c)
    chdir(saveDirectory)
    tree.write(open(newFilename, 'w'), encoding='unicode')
    return(saveDirectory+'/'+newFilename)
    
