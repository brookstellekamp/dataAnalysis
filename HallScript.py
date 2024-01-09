##Hall script
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.stats import linregress
from lmfit import Model

def sheet_resistance(Rs, Ra, Rb):
    '''
    Numerical solver for van der pauw sheet resistance.
    Usage: scipy.optimize.root(sheet_resistance, x0=guess, args=(Ra, Rb))
    '''
    return np.exp(-np.pi*(Ra/Rs)) + np.exp(-np.pi*(Rb/Rs))-1

def condTensor(sigma, f, B):#reduced conductivity tensor for fitting
    X = np.empty_like(B)
    Y = np.empty_like(B)
    
    X = f/(1+(sigma*B)**2)
    Y = f*sigma/(1+(sigma*B)**2) #note, we do not sum the factor 2B for Ym, leave that for the fit call
    
    return X, Y

q = 1.602176634E-19

## Input data

Thickness = 200 #nm
d = Thickness*1E-7 #cm

#Current averaged voltages
Is = 10E-6 #A
Vc = np.array([(15.753E-6--16.9E-6)/2, (-3.7382E-6--3.5029E-6)/2])
Vd = np.array([(22.728E-6-23.034E-6)/2, (3.59E-6-34.880E-6)/2])


#Here goes current reversal averaged resistances
Ra_1 = np.array([13.507, 13.507])
Rb_1 = np.array([14.420, 14.420])
Rc = Vc/Is

Ra_2 = np.array([13.548, 13.548])
Rb_2 = np.array([14.365, 14.365])
Rd = Vd/Is

B = np.array([-1.99, 1.99])
Rs_1 = np.empty_like(B)
Rs_2 = np.empty_like(B)

for i in range(B.shape[0]):
    Rs_1[i] = root(sheet_resistance, x0=0, args=(Ra_1[i], Rb_1[i]))['x']
    Rs_2[i] = root(sheet_resistance, x0=0, args=(Ra_2[i], Rb_2[i]))['x']
    
#zero field values    
Rs_0 = (Rs_1[0]+Rs_2[0])/2

Zc = (Rc-(Ra_1-Rb_1))/Rs_1
Zd = (Rd+(Ra_2-Rb_2))/Rs_2

mobility_c_solved = linregress(B, Zc)
mobility_d_solved = linregress(B, Zd)

mobility_c = mobility_c_solved.slope*100**2
mobility_d = mobility_d_solved.slope*100**2

mobility = (mobility_c+mobility_d)/2

n_s_c = 1/(q*Rs_0*mobility_c_solved.slope*100**2)
n_s_d = 1/(q*Rs_0*mobility_d_solved.slope*100**2)

n_c = n_s_c/d
n_d = n_s_d/d
n = (n_c+n_d)/2


print('mobility = %0.2f'%(mobility))

print('carrier concentration 1 = {0:1.2E}'.format(n))

##Calculate conductivity tensor

#From raw data - Sheet magnetoresistivity tensor components
Rxx = (Rs_1 + Rs_2)/2
Rxy = (Rc+Rd)/2

#Longitudinal and transverse components of the conductivity tensor - tensor is formed as [xx, xy; -xy, xx]
Gxx = Rxx/(Rxx**2+Rxy**2)
Gxy = Rxy/(Rxx**2+Rxy**2)

#Zero Field component for tensor normalization
G0 = Gxx[np.searchsorted(B, 0)]

Xm = Gxx/G0
Ym = 2*Gxy/G0

#Calculated from extracted data
sigma_0 = n*q*mobility
gamma = mobility*B

sigma_xx = sigma_0/(1+gamma**2)
sigma_xy = gamma*sigma_xx

##Fit conductivity tensor
numCarriers = 1
report=True
plot=True
#Fitting Functions
def condTensor1(B, sigma):
    X, Y = condTensor(sigma, 1, B)
    
    return X1, 2*B*Y

def condTensor2(B, sigma1, sigma2, f1):
    f2 = 1-f1
    X1, Y1 = condTensor(sigma1, f1, B)
    X2, Y2 = condTensor(sigma2, f2, B)
    
    return X1+X2, 2*B*(Y1+Y2)

if numCarriers==1:
    mod = Model(condTensor1)
    params=mod.make_params()
    
    params['sigma'].set(1, min=0)
    
    init = mod.eval(params, x=B)
    
elif numCarriers==2:
    mod=Model(condTensor2)
    params=mod.make_params()
    
    params['sigma1'].set(1, min=0)
    params['sigma2'].set(1, min=0)
    params['f1'].set(.8, min=0, max=1)
    
    init = mod.eval(params,x=B)

out = mod.fit([Xm, Ym], params, x=B)
print(out.fit_report()) if report

if plot:
    tensorFitFig = plt.figure()
    tensorFitAx = tensorFitFig.gca()
    
    plt.plot(B, Xm, B, Ym)



##plotting

plt.style.use('publication')

mobilityFig = plt.figure()
mobilityAx = mobilityFig.gca()

mobilityAx.scatter(B, Zc)
mobilityAx.plot(B, mobility_c_solved.slope*B+mobility_c_solved.intercept)
mobilityAx.scatter(B, Zd)
mobilityAx.plot(B, mobility_d_solved.slope*B+mobility_d_solved.intercept)

mobilityAx.set_xlabel('B (T)')
mobilityAx.set_ylabel('Reduced Hall Resistance (unitless)')

mobilityFig.show()

#plot the conductivity tensor
tensorFig = plt.figure()
tensorAx = tensorFig.gca()

tensorAx.scatter(Gxx, Gxy, color='tab:blue', marker='s', label='experimental')
tensorAx.scatter(sigma_xx, sigma_xy, color='k', marker='s', label='ideal')

tensorAx.set_xlabel('Longitudinal conductivity tensor $\sigma_{xx}$ (S)')
tensorAx.set_ylabel('Transverse conductivity tensor $\sigma_{xy}$ (S)')

tensorFig.show()

#normalized conductivity tensor
tensorFig2 = plt.figure()
tensorAx2 = tensorFig2.gca()

tensorAx2.plot(B, Xm, B, Ym)

tensorAx2.set_xlabel('B(T)')
tensorAx2.set_ylabel('Reduced Conductivity Tensor (unitless)')

tensorFig2.show()

