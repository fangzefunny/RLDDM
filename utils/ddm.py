import numpy as np 

def wfpt(ts, v, a, w=.5, tol=1e-6):
    '''Frist passage time for Wiener diffusion model 
    
    Approximation based on Navarro & Fuss (2009).

    Args: 
        t: hit time, an array
        v: drift rate 
        a: threshold
        w: normalized starting bias
        tol: tolerence of convergence

    Outputs:
        p: probability density of the given t
    '''

    if np.isscalar(ts): ts = [ts] 

    pdfs = []
    for t in ts:
        
        # normalized time 
        t_norm = t / a**2 

        # calculate the number of terms needed for large time
        if np.pi*t_norm*tol<1:
            kl = np.sqrt(-2*np.log(np.pi*t_norm*tol) / (np.pi**2*t_norm))
            kl = np.max([kl, 1/(np.pi*np.sqrt(t_norm))])
        else:
            kl = 1/(np.pi*np.sqrt(t_norm))
        
        # calculate number of terms needed for small time 
        if 2*np.sqrt(2*np.pi*t_norm)*tol<1:
            ks = 2+np.sqrt(-2*t_norm*np.log(2*np.sqrt(2*np.pi*t_norm)*tol))
            ks = np.max([ks, np.sqrt(t_norm)+1])
        else:
            ks = 2 # minimal kappa for the case

        # compute f(t_norm|0,1,w)
        if ks < kl:
            K = np.ceil(ks)
            k = np.arange(-np.floor((K-1)/2), np.ceil((K-1)/2)+1)
            p = ((w+2*k)*np.exp(-((w+2*k)**2)/2/t_norm)).sum()/\
                np.sqrt(2*np.pi*t_norm**3)
        else:
            K = np.ceil(kl)
            k = np.arange(1, K+1)
            p = (k*np.exp(-(k**2)*(np.pi**2)*t_norm/2)*
                 np.sin(k*np.pi*w)).sum()*np.pi
        
        # convert to f(t|v,a,w)
        pdfs.append(p*np.exp(-v*a*w - (v**2)*t/2) / (a**2))

    output = pdfs[0] if len(pdfs) == 1 else np.array(pdfs)

    return output 
