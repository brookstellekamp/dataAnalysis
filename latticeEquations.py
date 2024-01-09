#Lattice Equations

'''
lattice equations: http://duffy.princeton.edu/sites/default/files/pdfs/links/xtalgeometry.pdf
'''

def vegard(x,a,b):
    return (1-x)*a+x*b
        
def d_hex(a,c,h,k,l):
    temp = (4/3)*((h**2 + h*k + k**2)/a**2)+l**2/c**2
    return np.sqrt(1/temp)
    
def lattice_hex(qx,qz,h,k,l):
    d_x = 2*np.pi/qx
    d_z = 2*np.pi/qz
    a = np.sqrt(d_x**2*(4/3)*h**2+h*k+k**2)
    c = np.sqrt(d_z**2*l**2)
    return a,c
    
def d_cubic(a,h,k,l):
    temp = (h**2+k**2+l**2)/a**2
    return np.sqrt(1/temp)

def d_tet(a,c,h,k,l):
    temp = ((h**2+k**2)/a**2) + l**2/c**2
    return np.sqrt(1/temp)
    
def d_rhomb(a,alpha,h,k,l):
    alpha = np.pi*alpha/180
    temp = ((h**2+k**2+l**2)*np.sin(alpha)**2 + 2*(h*k + k*l + l*h)*(np.cos(alpha)**2 - np.cos(alpha)))/(a**2*(1-3*np.cos(alpha)**2+2*np.cos(alpha)**3))
    return np.sqrt(1/temp)
    
def d_ortho(a,b,c,h,k,l):
    temp = h**2/a**2 + k**2/b**2 + l**2/c**2
    return np.sqrt(1/temp)

def d_mono(a,b,c,beta,h,k,l):
    beta = beta*np.pi/180
    temp = (1/np.sin(beta)**2)*(h**2/a**2 + k**2*np.sin(beta)**2/b**2 + l**2/c**2 - 2*h*l*np.cos(beta)/(a*c))
    return np.sqrt(1/temp)
    
def d_tri(a,b,c,alpha,beta,gamma,h,k,l):
    '''
    Something not quite right about this eqn.
    '''
    alpha = alpha*np.pi/180
    beta = beta*np.pi/180
    gamma = gamma*np.pi/180
    V = a*b*c*np.sqrt(1-np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2+2*np.cos(alpha)*np.cos(beta)*np.cos(gamma))
    S11 = b**2*c**2*np.sin(alpha)**2
    S22 = a**2*c**2*np.sin(beta)**2
    S33 = a**2*b**2*np.sin(gamma)**2
    S12 = a*b*c**2*(np.cos(alpha)*np.cos(beta)-np.cos(gamma))
    S23 = a**2*b*c*(np.cos(beta)*np.cos(gamma)-np.cos(alpha))
    S13 = a*b**2*c*(np.cos(gamma)*np.cos(alpha) - np.cos(beta))
    temp = (S11*h**2+S22*k**2+S33*l**2+2*S12*h*k+2*S23*k*l+2*S13*h*l)/V**2
    return np.sqrt(1/temp)
    
    
def chiAngle_hex(a,c,h1,k1,l1, h2, k2, l2):
    '''
    Calculate angle between hexagonal planes

    '''
    num = h1*h2 + k1*k2 + 0.5*(h1*k2 + h2*k1) + ((3*a**2)/(4*c**2))*l1*l2
    denom = np.sqrt((h1**2 + k1**2 + h1*k1 + ((3*a**2)/(4*c**2))*l1**2)*(h2**2 + k2**2 + h2*k2 + ((3*a**2)/(4*c**2))*l2**2))
    
    return np.arccos(num/denom)*180/np.pi
    
def chiAngle_cubic(h1,k1,l1, h2, k2, l2):
    
    '''
    Calculate angle between cubic planes
    
    '''
    num = h1*h2 + k1*k2 + l1*l2
    denom = np.sqrt((h1**2 + k1**2 + l1**2)*(h2**2 + k2**2 + l2**2))
    return np.arccos(num/denom)*180/np.pi

def chiAngle_tet(a,c,h1,k1,l1,h2,k2,l2):
    num = ((h1*h2+k1*k2)/a**2)+(l1*l2)/c**2
    denom = np.sqrt(((h1**2+k1**2)/a**2 + l1**2/c**2)*((h2**2+k2**2)/a**2 + l2**2/c**2))
    return np.arccos(num/denom)*180/np.pi
    
def chiAngle_rhomb(a, alpha, h1, k1, l1, h2, k2, l2):
    d1 = d_rhomb(a, alpha, h1,k1,l1)
    d2 = d_rhomb(a,alpha,h2,k2,l2)
    alpha = alpha*np.pi/180
    V = a**3*np.sqrt(1-3*np.cos(alpha)**2 + 2*np.cos(alpha)**3)
    temp = a**4*d1*d2*(np.sin(alpha)**2*(h1*h2+k1*k2+l1*l2)+(np.cos(alpha)**2-np.cos(alpha))*(k1*l2 + k2*l1 + l1*h2 + l2*h1 + h1*k2 + h2*k1))/V**2
    return np.arccos(temp)*180/np.pi
        
def chiAngle_ortho(a,b,c,h1,k1,l1,h2,k2,l2):
    num = h1*h2/a**2 + k1*k2/b**2 + l1*l2/c**2
    denom = np.sqrt((h1**2/a**2 + k1**2/b**2 + l1**2/c**2)*(h2**2/a**2+k2**2/b**2 + l2**2/c**2))
    return np.arccos(num/denom)*180/np.pi
    
def chiAngle_mono(a,b,c,beta,h1,k1,l1,h2,k2,l2):
    d1 = d_mono(a,b,c,beta,h1,k1,l1)
    d2 = d_mono(a,b,c,beta,h2,k2,l2)
    beta = beta*np.pi/180
    temp = d1*d2*(h1*h2/a**2 + k1*k2*np.sin(beta)**2/b**2 + l1*l2/c**2 - (l1*h2+l2*h1)*np.cos(beta)/(a*c))/np.sin(beta)**2
    return np.arccos(temp)*180/np.pi

def chiAngle_tri(a,b,c,alpha,beta,gamma,h1,k1,l1,h2,k2,l2):
    '''
    Something not quite right about this eqn.
    '''
    d1 = d_tri(a,b,c,alpha,beta,gamma,h1,k1,l1)
    d2 = d_tri(a,b,c,alpha,beta,gamma,h2,k2,l2)
    alpha = alpha*np.pi/180
    beta = beta*np.pi/180
    gamma = gamma*np.pi/180
    V = a*b*c*np.sqrt(1-np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2+2*np.cos(alpha)*np.cos(beta)*np.cos(gamma))
    S11 = b**2*c**2*np.sin(alpha)**2
    S22 = a**2*c**2*np.sin(beta)**2
    S33 = a**2*b**2*np.sin(gamma)**2
    S12 = a*b*c**2*(np.cos(alpha)*np.cos(beta)-np.cos(gamma))
    S23 = a**2*b*c*(np.cos(beta)*np.cos(gamma)-np.cos(alpha))
    S13 = a*b**2*c*(np.cos(gamma)*np.cos(alpha) - np.cos(beta))
    temp = d1*d2*(S11*h1*h2 + S22*k1*k2+ S33*l1*l2+S23*(k1*l2+k2*l1)+S13*(l1*h2+l2*h1)+S12*(h1*k2+h2*k1))/V**2
    return np.arccos(temp)*180/np.pi

def calcQ(mat,h,k,l,norm, chi=0, sys = 'hex'):
    '''
    Calculate (Qx,Qy) for a given plane hkl
    Defaults to asymmetric case (chi=0)
    

    '''
    if norm==None:
        norm = mat['norm']
    
    if sys == 'hex':
        a = mat['a']
        c = mat['c']
        d_hk = d_hex(a,c,h,k,0)
        d_l = d_hex(a,c,0,0,l)
        qx = 2*np.pi/d_hk
        qz = 2*np.pi/d_l

        resultant = np.sqrt(qx**2+qz**2)
        offset = chiAngle_hex(a,c,h,k,l, norm[0], norm[1], norm[2])
        
    elif sys == 'cubic':
        a = mat['a']
        d_h = d_cubic(a,h,0,0)
        d_kl = d_cubic(a,0,k,l)
        qx = 2*np.pi/d_kl
        qz = 2*np.pi/d_h
        
        resultant = np.sqrt(qx**2+qz**2)
        offset = chiAngle_cubic(h,k,l, norm[0], norm[1], norm[2])
        
    elif sys == 'pc':
        a = mat['a_pc']
        d_h = d_cubic(a,h,0,0)
        d_kl = d_cubic(a,0,k,l)
        qx = 2*np.pi/d_kl
        qz = 2*np.pi/d_h
        
        resultant = np.sqrt(qx**2+qz**2)
        offset = chiAngle_cubic(h,k,l, norm[0], norm[1], norm[2])
    
    if chi != 0:    
        angle = (chi+offset)*np.pi/180
        qx = resultant*np.sin(angle)
        qz = resultant*np.cos(angle)
    return (qx,qz)