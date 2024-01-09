from brooks.functions import vegard
import numpy as np

    
def d_ortho(a,b,c,h,k,l):
    temp = h**2/a**2 + k**2/b**2 + l**2/c**2
    return np.sqrt(1/temp)

#hexagonal materials

def alloy(x, mat1, mat2):
        mat = {}
        if mat1['sys'] == 'hex':
            mat = {   'x':x,
                        'a':vegard(x, mat1['a'], mat2['a']), 
                        'c':vegard(x, mat1['c'], mat2['c']),
                        'C13':vegard(x, mat1['C13'], mat2['C13']),
                        'C33':vegard(x, mat1['C33'], mat2['C33']), 
                        'sys':mat1['sys'],
                        'norm':mat1['norm']}
        elif mat1['sys'] == 'cubic':
            try:
                mat = { 'x':x,
                        'a':vegard(x, mat1['a'], mat2['a']), 
                        'C11':vegard(x, mat1['C11'], mat2['C11']),
                        'C12':vegard(x, mat1['C12'], mat2['C12']),
                        'sys':mat1['sys'],
                        'norm':mat1['norm']}
            except:
                mat = {   'x':x,
                        'a':vegard(x, mat1['a'], mat2['a']), 
                        'sys':mat1['sys'],
                        'norm':mat1['norm']}
        if mat1['sys'] == 'mono':
            mat = {     'a':vegard(x, mat1['a'], mat2['a']), 
                        'b':vegard(x, mat1['b'], mat2['b']), 
                        'c':vegard(x, mat1['c'], mat2['c']), 
                        'beta':vegard(x, mat1['beta'], mat2['beta']), 
                        'sys':'mono',
                        'norm':(0,1,0)}
        else:
            print('elastic constants not defined for crystal system: %s'%mat1['sys'])
            
            if mat1['sys'] == 'pc':
                mat = {   'x':x,
                            'a_pc':vegard(x, mat1['a_pc'], mat2['a_pc']), 
                            'sys':mat1['sys'],
                            'norm':mat1['norm']}
                
        return mat

#%% wurtzite

AlN = { 'a':3.112,
        'c':4.98,
        'C13':99,
        'C33':389, 
        'sys':'hex',
        'norm':(0,0,1)}
GaN = { 'a':3.189, 
        'c':5.185,
        'C13':106,
        'C33':398, 
        'sys':'hex',
        'norm':(0,0,1)}
        
InN = { 'a':3.5378, 
        'c':5.7033,
        'C13':40,
        'C33':292, 
        'sys':'hex',
        'norm':(0,0,1)}
        
InGaN = alloy(.5, GaN, InN)
            



ZGN = { 'a':3.1962, 
        'c':5.2162,
        'C13':103,
        'C33':401, 
        'sys':'hex',
        'norm':(0,0,1)}

ZGN_stephan = { 'a':3.193, 
                'c':5.187,
                'C13':103,
                'C33':401, 
        'sys':'hex',
        'norm':(0,0,1)}    
                
ZGN_Du =  { 'a':3.157, 
            'c':5.137,
            'C13':103,
            'C33':401, 
            'sys':'hex',
            'norm':(0,0,1)}  
        
ZTN = { 'a':3.377,
        'c':5.501,
        'C13':103,
        'C33':401,
        'sys':'hex',
        'norm':(0,0,1)}
        
MTN = { 'a':3.42646, 
        'c': 5.47431,
        'C13':103,
        'C33':401, 
        'sys':'hex',
        'norm':(0,0,1)}
        
MGN = { 'a':3.299, 
        'c': 5.165,
        'C13':103,
        'C33':401, 
        'sys':'hex',
        'norm':(0,0,1)}
        
MGN_exp = { 'a':3.24, 
            'c': 5.18,
            'C13':103,
            'C33':401, 
            'sys':'hex',
            'norm':(0,0,1)}

ZTiN = { 'a':3.1245, 
        'c': 5.006,
        'C13':103,
        'C33':401, 
        'sys':'hex',
        'norm':(0,0,1)}
        
ZTiN_exp = { 'a':3.10, 
            'c': 5.089,
            'C13':103,
            'C33':401, 
            'sys':'hex',
            'norm':(0,0,1)}
        


#%% monoclinic

b_GaO = {   'a':12.229,
            'b':3.039,
            'c':5.8024,
            'beta':103.82,
            'sys':'mono',
            'norm':(0,1,0)}
            
        
b_AlO = {   'a':11.795,
            'b':2.91,
            'c':5.621,
            'beta':103.79,
            'sys':'mono',
            'norm':(0,1,0)}

        
b_InO = {   'a':12.301,
            'b':3.060,
            'c':5.858,
            'beta':103.83,
            'sys':'mono',
            'norm':(0,1,0)}


#%%cubic/pseudocubic materials

#https://doi.org/10.1002/crat.2170260420
cubic_GaN = {'a': 4.50597,
            'sys':'cubic',
            'C11':293,
            'C12':159,
            'norm':(1,1,1)}
            
cubic_AlN = {'a': 4.372,
            'sys':'cubic',
            'C11':304,
            'C12':160,
            'norm':(1,1,1)}
            
cubic_InN = {'a': 5.02,
            'sys':'cubic',
            'C11':187,
            'C12':125,
            'norm':(1,1,1)}


Si = {  'a':5.43,
        'sys':'cubic',
        'norm':(1,0,0)}
        
Ge = {  'a':5.65,
        'sys':'cubic',
        'norm':(1,0,0)}
        
GaP = { 'a':5.4505,
        'sys':'cubic',
        'norm':(1,0,0)}
        
ZGP_ZB = {  'a':5.433,
            'sys':'cubic',
            'norm':(1,0,0)}
            
ZGP = { 'a':5.46,
        'c':10.71,
        'a_pc':5.46,
        'sys':'tet',
        'C11':122.46,
        'C12':50.46,
        'C13':52.92,
        'C33':121.74,
        'C44':64.65,
        'C66':63.46,
        'norm':(0,0,1)}



LNO = { 'a':5.4573,
        'c':13.1462,
        'a_pc':3.83771, 
        'sys':'pc',
        'C11':266,
        'C12':162,
        'C13':151,
        'C33':342,
        'C44':100,
        'C66':52,#https://doi.org/10.1016/j.commatsci.2015.06.034
        'norm':(0,0,1)}

GNO = { 'a':5.2610,
        'b':5.4879,
        'c':7.5112,
        'a_pc':d_ortho(5.2610, 5.4879, 7.5112, 1,1,0),
        'sys':'pc',
        'norm':(0,0,1)}

ENO = { 'a':5.2917,
        'b':5.4635,
        'c':7.5364,
        'a_pc':d_ortho(5.2917, 5.4635, 7.5364, 1,1,0),
        'sys':'pc',
        'norm':(0,0,1)}

SNO = { 'a':5.3261,
        'b':5.4351,
        'c':7.5647,
        'a_pc':d_ortho(5.3261, 5.4351, 7.5647, 1,1,0),
        'sys':'pc',
        'norm':(0,0,1)}

NNO = { 'a':5.38957,
        'b':5.38063,
        'c':7.61110,
        'a_pc':d_ortho(5.38957, 5.38063, 7.61110, 1,1,0),
        'sys':'pc',
        'norm':(0,0,1)}




TaC = { 'a':4.453,
        'sys':'cubic',
        'norm':(1,1,1)}

TiN = { 'a':4.238,
        'sys':'cubic',
        'norm':(1,1,1)}

ZrN = { 'a':4.585,
        'sys':'cubic',
        'norm':(1,1,1)}




Mg2GeO4 = {'a':8.246,
           'b':8.246,
           'c':8.246,
           'sys':'cubic',
           'norm':(0,0,1)}
#%%substrates

Sapphire = {'a':4.785,
            'c':12.991,
            'sys':'hex',
            'norm':(0,0,1)}

LSAT = {'a':7.72,
        'a_pc':3.86, 
        'sys':'pc', #pc = pseudocubic
        'C11':308,
        'C12':125,
        'C44':139,
        'norm':(0,0,1)}
        
NGO = { 'a':5.4293,
        'b':5.4991,
        'c':7.708,
        'a_pc':3.86385,
        'sys':'pc',
        'norm':(0,0,1)}

MgAl2O4 = {'a': 8.085,
           'b': 8.085,
           'c': 8.085,
           'sys':'cubic',
           'norm':(0,0,1)}

NiGa2O4 = {'a': 8.261,
           'b': 8.261,
           'c': 8.261,
           'sys':'cubic',
           'norm':(0,0,1)}
  
#%%
      
matDB = {   'AlN':AlN,
            'GaN':GaN,
            'InN':InN,
            'ZGN':ZGN, 
            'MTN':MTN,
            'MGN':MGN,
            'ZTiN':ZTiN,
            'Sapphire':Sapphire,
            'LSAT':LSAT,
            'LNO': LNO,
            'GNO':GNO,
            'ENO':ENO,
            'SNO':SNO,
            'NNO':NNO,
            'NGO':NGO,
            'TaC':TaC,
            'ZrN':ZrN,
            'TiN':TiN,
            'Si':Si,
            'Ge':Ge,
            'ZGP_ZB':ZGP_ZB,
            'GaP':GaP, 
            'b_GaO':b_GaO,
            'b_AlO':b_AlO,
            'b_InO':b_InO,
            'MgAl2O4':MgAl2O4,
            'NiGa2O4':NiGa2O4,
            'Mg2GeO4':Mg2GeO4}