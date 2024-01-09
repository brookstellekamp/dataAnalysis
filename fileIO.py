import numpy as np
from os import chdir, mkdir, path, getcwd
import glob
from xml.etree import ElementTree as ET
import re
from io import BytesIO
from PIL import Image
import pandas as pd
import itertools

def rigakuXRD(source, scanType = 'standard', fullReturn=False):
    '''
    Rigaku XRD file importer. Takes .ras or .txt, returns numpy array of [angle, intensity]
    
    input - file name (string) in .ras or .txt format
    
    scanType - default to standard (single axis scan), also kwargs for 2-axis scans RSM or poleFigure
    
    return - numpy array of [angle, intensity]
    '''
    lam = 1.5405974 #angstroms
    K = 2*np.pi/lam # CuKalpha wavelength in angstroms
    
    if scanType == 'standard': 
        subFile = []
        header = []
        string = '*RAS_HEADER_START'
        f = open(source, errors='ignore')
        for line in f:
            if line.find(string) == -1:
                if line.find('*') == 0:
                    header.append(line)
                else:
                    subFile.append(line)
            
        data = np.genfromtxt(subFile, comments='*')
        h = header
        axis = list(filter(lambda x: '*MEAS_COND_AXIS_NAME-' in x, h))
        positions = list(filter(lambda x: '*MEAS_COND_AXIS_POSITION-' in x, h))
        offsets = list(filter(lambda x: '*MEAS_COND_AXIS_OFFSET-' in x, h))
        scan_meta = list(filter(lambda x: '*MEAS_SCAN_' in x, h))
        offset_dict = {}
        position_dict = {}
        scan_dict = {}
        for ax, pos, off in zip(axis, positions, offsets):
            n = ax.split('"')[-2]
            try:
                v = float(pos.split('"')[-2])
            except ValueError:
                v = pos.split('"')[-2]
            try:
                o = float(off.split('"')[-2])
            except ValueError:
                o = np.nan
            offset_dict[n] = o
            position_dict[n] = v
        for line in scan_meta:
            key = line.split('"')[0]
            try:
                value = float(line.split('"')[1])
            except ValueError:
                value = line.split('"')[1]
            if line.find('SPEED') != -1:
                try:
                    key = line.split('_')[-2]
                    value = float(line.split('_')[-1].split('"')[-2])
                except ValueError:
                    pass
            scan_dict[key] = value
        meta = {'positions':position_dict,
                    'offsets':offset_dict,
                    'scan':scan_dict}
        data[:,1] = data[:,1]/scan_dict['SPEED']
        
            
    elif scanType == 'RSM':
        f = open(source, errors='ignore')
        string = '*RAS_HEADER_START'
        subFile = []
        genList = []
        header = []
        headerList = []
        qx = []
        qz = []
        tth = []
        w = []
        intensity = []
        meta = []
        for line in f:
            if line.find(string) == -1:
                if line.find('*') == 0:
                    header.append(line)
                else:
                    subFile.append(line)
            else:
                genList.append(itertools.chain(subFile))
                headerList.append(header)
                header=[]
                subFile = []
        for i, gen in enumerate(genList[1:]):
            data = np.genfromtxt(gen, comments='*')
            h = headerList[i+1]
            axis = list(filter(lambda x: '*MEAS_COND_AXIS_NAME-' in x, h))
            positions = list(filter(lambda x: '*MEAS_COND_AXIS_POSITION-' in x, h))
            offsets = list(filter(lambda x: '*MEAS_COND_AXIS_OFFSET-' in x, h))
            RSM_meta = list(filter(lambda x: '*MEAS_3DE_' in x, h))
            scan_meta = list(filter(lambda x: '*MEAS_SCAN_' in x, h))
            offset_dict = {}
            position_dict = {}
            RSM_origin = {}
            RSM_scan = {}
            RSM_step = {}
            scan_speed = {}
            
            for ax, pos, off in zip(axis, positions, offsets):
                n = ax.split('"')[-2]
                try:
                    v = float(pos.split('"')[-2])
                except ValueError:
                    v = pos.split('"')[-2]
                try:
                    o = float(off.split('"')[-2])
                except ValueError:
                    o = np.nan
                offset_dict[n] = o
                position_dict[n] = v
                
            for line in RSM_meta:
                if line.find('ORIGIN') != -1:
                    key = line.split('_')[-2]
                    value = float(line.split('_')[-1].split('"')[-2])
                    RSM_origin[key] = value
                if line.find('SCAN') != -1:
                    key = line.split('_')[-1].split(' ')[0]
                    value = float(line.split('_')[-1].split('"')[-2])
                    RSM_scan[key] = value
                if line.find('STEP') != -1:
                    key = line.split('_')[-1].split(' ')[0]
                    try:
                        value = float(line.split('_')[-1].split('"')[-2])
                    except ValueError:
                        value = line.split('_')[-1].split('"')[-2]
                    RSM_step[key] = value
                    
            for line in scan_meta:
                if line.find('SPEED') != -1:
                    try:
                        key = line.split('_')[-2]
                        value = value = float(line.split('_')[-1].split('"')[-2])
                    except ValueError:
                        pass
                    scan_speed[key] = value
                    
            scanAxis = list(filter(lambda x: '*MEAS_SCAN_AXIS_X ' in x, h))[0].split('"')[-2]
            stepAxis = RSM_step['INTERNAL']
            if scanAxis == '2-Theta/Omega' and stepAxis == 'Omega':
                offset = (position_dict['Theta/2-Theta']/2-position_dict['Omega'])*np.pi/180
                twotheta = data[:,0]*np.pi/180
                omega = twotheta/2 - offset
                w.append(omega)
                tth.append(twotheta)
            if scanAxis == '2-Theta' and stepAxis == 'Omega':
                twotheta = data[:,0]*np.pi/180
                omega = position_dict['Omega']*np.pi/180
                w.append(np.full_like(twotheta, omega))
                tth.append(twotheta)
            if scanAxis == 'Omega' and stepAxis == 'TwoThetaOmega':
                omega = data[:,0]*np.pi/180
                twotheta = position_dict['2-Theta']*np.pi/180
                w.append(omega)
                tth.append(np.full_like(omega, twotheta))
                
            
        
            qx.append(K*(np.cos(omega) - np.cos(twotheta - omega)))
            qz.append(K*(np.sin(omega) + np.sin(twotheta - omega)))
            intensity.append(data[:,1]/scan_speed['SPEED'])
            metaDict = {'positions':position_dict,
                        'offsets':offset_dict,
                        'RSM_origin':RSM_origin,
                        'RSM_scan':RSM_scan,
                        'RSM_step':RSM_step}
            meta.append(metaDict)
            
        qx = np.array(qx)
        qz = np.array(qz)
        intensity = np.array(intensity)
        tth = np.array(tth)
        w = np.array(w)
        data = [qx, qz, intensity, tth, w]
    
    
    
    
    
    
    
    elif scanType == 'poleFigure':
        f = open(source, errors='ignore')
        string = '*RAS_HEADER_START'
        subFile = []
        genList = []
        header = []
        headerList = []
        phi = []
        chi = []
        intensity = []
        meta = []
        for line in f:
            if line.find(string) == -1:
                if line.find('*') == 0:
                    header.append(line)
                else:
                    subFile.append(line)
            else:
                genList.append(itertools.chain(subFile))
                headerList.append(header)
                header=[]
                subFile = []
        for i, gen in enumerate(genList[1:]):
            data = np.genfromtxt(gen, comments='*')
            h = headerList[i+1]
            axis = list(filter(lambda x: '*MEAS_COND_AXIS_NAME-' in x, h))
            positions = list(filter(lambda x: '*MEAS_COND_AXIS_POSITION-' in x, h))
            offsets = list(filter(lambda x: '*MEAS_COND_AXIS_OFFSET-' in x, h))
            PF_meta = list(filter(lambda x: '*MEAS_3DE_' in x, h))
            scan_meta = list(filter(lambda x: '*MEAS_SCAN_' in x, h))
            offset_dict = {}
            position_dict = {}
            PF_alpha = {}
            PF_BG = {}
            PF_tth = {}
            scan_speed = {}
            
            for ax, pos, off in zip(axis, positions, offsets):
                n = ax.split('"')[-2]
                try:
                    v = float(pos.split('"')[-2])
                except ValueError:
                    v = pos.split('"')[-2]
                try:
                    o = float(off.split('"')[-2])
                except ValueError:
                    o = np.nan
                offset_dict[n] = o
                position_dict[n] = v
                
            for line in PF_meta:
                if line.find('BG') != -1:
                    key = line.split('_')[-2]+'_'+line.split('_')[-1].split(' ')[0]
                    value = float(line.split('_')[-1].split('"')[-2])
                    PF_BG[key] = value
                if line.find('ALPHA') != -1:
                    key = line.split('_')[-1].split(' ')[0]
                    value = float(line.split('_')[-1].split('"')[-2])
                    PF_alpha[key] = value
                if line.find('TWOTHETA') != -1:
                    key = line.split('_')[-1].split(' ')[0]
                    value = float(line.split('_')[-1].split('"')[-2])
                    PF_tth[key] = value
                    
            for line in scan_meta:
                if line.find('SPEED') != -1:
                    try:
                        key = line.split('_')[-2]
                        value = value = float(line.split('_')[-1].split('"')[-2])
                    except ValueError:
                        pass
                    scan_speed[key] = value
                    
            scanAxis = list(filter(lambda x: '*MEAS_SCAN_AXIS_X ' in x, h))[0].split('"')[-2]
            if scanAxis == 'Phi':
                stepAxis = 'Chi'
            elif scanAxis == 'Chi':
                stepAxis == 'Phi'
            
            if scanAxis == 'Chi' and stepAxis == 'Phi':
                chi_temp = data[:,0]*np.pi/180
                chi.append(chi_temp)
                phi_temp = position_dict['Phi']
                phi.append(np.full_like(chi_temp, phi_temp))
            if scanAxis == 'Phi' and stepAxis == 'Chi':
                phi_temp = data[:,0]*np.pi/180
                phi.append(phi_temp)
                chi_temp = position_dict['Chi']
                chi.append(np.full_like(phi_temp, chi_temp))
                
            
        
            
            intensity.append(data[:,1]/scan_speed['SPEED'])
            metaDict = {'positions':position_dict,
                        'offsets':offset_dict,
                        'PF_alpha':PF_alpha,
                        'PF_BG':PF_BG,
                        'PF_tth':PF_tth}
            meta.append(metaDict)
            
        chi = np.array(chi)
        phi = np.array(phi)
        intensity = np.array(intensity)
        
        data = [chi,phi,intensity]
    if fullReturn:
        return data, meta
    else:
        return data
    
def slacXRD(filename, fullReturn=False):
    '''
    Data import function for slac data with "comments, first line lambda, and column format: X, Yobs, weight, Ycalc, Ybg, Q
    
    inputs:
        filename - filename (string) in .csv format
        fullReturn - all SLAC xrd data (X, Yobs, weight, Ycalc, Ybg, Q)
                     default = False
                     
    return - (rawdata, lambda) -  raw data in numpy array [2theta, Q, intensity] and wavelength used
            if fullReturn
                returns all SLAC xrd data (X, Yobs, weight, Ycalc, Ybg, Q)
            
    '''
    extension = filename.split('.')[-1]
    if extension == 'csv':
        lam = np.genfromtxt(filename, delimiter=',', max_rows=1)[-1]
        rawdata = np.genfromtxt(filename, delimiter=',', comments='"')
    if fullReturn:
        return rawdata, lam
    elif fullReturn == False:
        return np.c_[rawdata[:,0], rawdata[:,5], rawdata[:,1]]
    
def getPanalyticalXRD(filename, fullReturn=False):
    
    '''
    script to import panalytical xml data 
    
    filename - full path (or filename if the correct working directory is selected)
    
    source - default to data. when pulling 2D data it recursively calls itself
    
    Currently returns 2theta and intensity (CPS), but can be set up to return all axes as fixed or scanned
    '''

    tree = ET.parse(filename)
    
    root = tree.getroot()
    
    if filename.split('.')[-1] == 'xrdml':
        namespace = {'all': re.match(r'\{.*\}', root.tag).group(0)[1:-1]}
        
        measurementType = root.find('.//all:xrdMeasurement', namespace).attrib['measurementType']
        
        if root.findall('.//all:intensities', namespace):
            dataField = './/all:intensities'
        elif root.findall('.//all:counts', namespace):
            dataField = './/all:counts'
        else:
            raise ValueError('Panalytical has changed the name of the "counts" or "intensities" tag again...')
    
        
        if measurementType == 'Scan':
        
            scanAxis = root.find('.//*all:scan', namespace).attrib['scanAxis']
            
            #Get offset data
            offsets = {}
            for element in root.findall('.//all:sampleOffset/', namespace):
                key = element.attrib['axis']
                value = element.text
                offsets[key] = value
            
            #Get position data
            
            starts = {}
            ends = {}
            fixed = {}
            
            timePerStep = float(root.find('.//*all:commonCountingTime', namespace).text)
            
            #iterate through each 'positions' element inside dataPoints, determine which is scanned and which is fixed, put the data in appropriate dicitonaries
            for element in root.findall('.//all:dataPoints/all:positions', namespace):
                for subelement in element.findall('all:startPosition', namespace):
                    key = element.attrib['axis']
                    value = subelement.text
                    starts[key] = float(value)
                    
                for subelement in element.findall('all:endPosition', namespace):
                    key = element.attrib['axis']
                    value = subelement.text
                    ends[key] = float(value)
                
                for subelement in element.findall('all:commonPosition', namespace):
                    key = element.attrib['axis']
                    value = subelement.text
                    fixed[key] = float(value)
            
            
            scanData = root.findall(dataField, namespace)
        
            
            intensities = np.genfromtxt(x for x in scanData[0].text.split(' '))
            
            axisDict = {}
            
            if '2Theta' in starts:
                twoTheta_array = np.linspace(float(starts['2Theta']), float(ends['2Theta']), num=intensities.shape[0])
                if scanAxis == '2Theta-Omega':
                    axisDict['2Theta-Omega'] = twoTheta_array
                elif scanAxis == '2Theta':
                    axisDict['2Theta'] = twoTheta_array
            if 'Omega' in starts:
                omegaArray = np.linspace(float(starts['Omega']), float(ends['Omega']), num=intensities.shape[0])
                axisDict['Omega'] = omegaArray
            if 'Phi' in starts:
                phiArray = np.linspace(float(starts['Phi']), float(ends['Phi']), num=intensities.shape[0])
                axisDict['Phi'] = phiArray
            if 'Chi' in starts:
                chiArray = np.linspace(float(starts['Chi']), float(ends['Chi']), num=intensities.shape[0])
                axisDict['Chi'] = chiArray
            
            
            intensities[intensities==0] = intensities[intensities>0].min()
            
            if fullReturn == False:
                return np.array((axisDict[scanAxis], intensities/timePerStep))
            elif fullReturn == True:
                return np.array((axisDict[scanAxis], intensities/timePerStep)), starts, ends, fixed
            
        elif measurementType == 'Area measurement':
            stepAxis = root.find('.//all:xrdMeasurement', namespace).attrib['measurementStepAxis']
            scanAxis = root.find('.//all:scan', namespace).attrib['scanAxis']
            allData = root.findall('.//all:scan', namespace)
            timePerStep = float(root.find('.//*all:commonCountingTime', namespace).text)
            
            
            if scanAxis == '2Theta':
                #1D detector scan
                
                #get Step axis center
                stepCenter = root.find('.//all:measurementStepAxisCenter/all:position', namespace).text
                
                scans = len(allData)
                points = len(root.find(dataField, namespace).text.split(' '))
                intensity = np.empty((scans, points))
                tth = np.empty_like(intensity)
                w = np.empty_like(intensity)
                
                
                
                for n, element in enumerate(allData):
                    center = {}
                    starts = {}
                    ends = {}
                    fixed = {}
                    listed = {}
                    center['step'] = stepCenter
                    for el in element.findall('.//all:scanAxisCenter/all:position', namespace):
                        key = el.attrib['axis']
                        value = el.text
                        center[key] = value
                    
                    
                    for el in element.findall('.//all:dataPoints/all:positions', namespace):
                        for subelement in el.findall('all:startPosition', namespace):
                            key = el.attrib['axis']
                            value = subelement.text
                            starts[key] = float(value)
                            
                        for subelement in el.findall('all:endPosition', namespace):
                            key = el.attrib['axis']
                            value = subelement.text
                            ends[key] = float(value)
                        
                        for subelement in el.findall('all:commonPosition', namespace):
                            key = el.attrib['axis']
                            value = subelement.text
                            fixed[key] = float(value)
                        
                        for subelement in el.findall('all:listPositions', namespace):
                            key = el.attrib['axis']
                            value = subelement.text
                            listed[key] = np.genfromtxt(x for x in value.split(' '))
                    
                    scanData = element.findall(dataField, namespace)
                    intensity[n,:] = np.genfromtxt(x for x in scanData[0].text.split(' '))
                    if listed:
                        tth[n,:] = listed[scanAxis]
                        w[n,:] = np.full_like(tth[n,:], fixed[stepAxis])
                    else:
                        tth[n,:] = np.linspace(starts[scanAxis], ends[scanAxis], points)
                        w[n,:] = np.full_like(tth[n,:], fixed[stepAxis])
                        
                
                
            elif scanAxis == 'Omega-2Theta' or scanAxis == '2Theta-Omega':
                #0D detector scan
                #get Step axis center
                stepCenter = root.find('.//all:measurementStepAxisCenter/all:position', namespace).text
                
                scans = len(allData)
                points = len(root.find(dataField, namespace).text.split(' '))
                intensity = np.empty((scans, points))
                tth = np.empty_like(intensity)
                w = np.empty_like(intensity)
                
                
                
                for n, element in enumerate(allData):
                    center = {}
                    starts = {}
                    ends = {}
                    fixed = {}
                    center['step'] = stepCenter
                    for el in element.findall('.//all:scanAxisCenter/all:position', namespace):
                        key = el.attrib['axis']
                        value = el.text
                        center[key] = value
                    
                    
                    for el in element.findall('.//all:dataPoints/all:positions', namespace):
                        for subelement in el.findall('all:startPosition', namespace):
                            key = el.attrib['axis']
                            value = subelement.text
                            starts[key] = float(value)
                            
                        for subelement in el.findall('all:endPosition', namespace):
                            key = el.attrib['axis']
                            value = subelement.text
                            ends[key] = float(value)
                        
                        for subelement in el.findall('all:commonPosition', namespace):
                            key = el.attrib['axis']
                            value = subelement.text
                            fixed[key] = float(value)
                        
                    
                    scanData = element.findall(dataField, namespace)
                    intensity[n,:] = np.genfromtxt(x for x in scanData[0].text.split(' '))
                    tth[n,:] = np.linspace(starts['2Theta'], ends['2Theta'], points)
    
                    w[n,:] = np.linspace(starts['Omega'], ends['Omega'], points)
                    
            
            
            return tth, w, intensity/timePerStep
        
    elif filename.split('.')[-1] == 'ascnxl':
        center = {}
        centerData = root.findall('.//ScanCenter/')
        for element in centerData:
            key = element.tag
            value = element.text
            center[key] = float(value)
        
        scan = {}
        scanData = root.findall('.//ScanAxisDetails/')
        for element in scanData:
            key = element.tag
            value = element.text
            try:
                scan[key] = float(value)
            except(ValueError):
                scan[key] = value
        
        scanAxis = scan['ScanAxisName']
        x = np.linspace(scan['Start'], scan['End'], int(scan['NumberOfSteps']))
        timePerStep = scan['StepTime']
        intensityData = root.find('.//Intensities').text
        intensity = np.genfromtxt(x for x in intensityData.split(' '))
        
        return x, intensity/timePerStep
        
def ICSDimport(filename):
    return np.genfromtxt(filename, delimiter = ',')
    
def bandEng(filename):
    f = open(filename, mode='rb+')
    lines = (line for line in f)
    data = np.genfromtxt(lines, skip_header=1)
    f.close()
    return data
    

def getSIMS(fileName, header=2):
    '''
    return dataframe of SIMS data from IONTOF SIMS excel output
    
    inputs:
    
    fileName -  excel filename (string)
    
    header - number of header lines (default 2)
    
    output - list of dataframes corresponding to each sheet
    
    NOTE: Due to the way IONTOF (stupidly) formats their excel data, the element names are not the column names. Therefore for the script to work correctly (until it is coded in) the 'sputter time (s)' label should be manually moved up to row 3 (2 for 0-index) to line up with the element names.
    Double Note: Sometimes the data I get from Steve is right.
    
    '''
    
    xl = pd.ExcelFile(fileName)
    datalist = []
    for sheet in xl.sheet_names:
        data = pd.read_excel(xl, sheet_name=sheet, header=header)
        data = data.drop(data.index[[0,1]])
        data = data.infer_objects()
        datalist.append(data)
    
    return datalist