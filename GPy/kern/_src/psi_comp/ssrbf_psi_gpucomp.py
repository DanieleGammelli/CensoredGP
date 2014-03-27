# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
The package for the psi statistics computation on GPU
"""

import numpy as np
from GPy.util.caching import Cache_this

try:
    import scikits.cuda.linalg as culinalg
    import pycuda.gpuarray as gpuarray
    from scikits.cuda import cublas
    import pycuda.autoinit
    from pycuda.reduction import ReductionKernel    
    from pycuda.elementwise import ElementwiseKernel
    from ....util import linalg_gpu
    
    # The kernel form computing psi1
    comp_psi1 = ElementwiseKernel(
        "double *psi1, double var, double l, double *Z, double *mu, double *S, double *logGamma, double *log1Gamma, double *logpsi1denom, int N, int M, int Q",
        "psi1[i] = comp_psi1_element(var,l, Z, mu, S, logGamma, log1Gamma, logpsi1denom, N, M, Q, i)",
        "comp_psi1",
        preamble="""
        #define IDX_NMQ(n,m,q) ((q*M+m)*N+n)
        #define IDX_NQ(n,q) (q*N+n)
        #define IDX_MQ(m,q) (q*M+m)
        #define LOGEXPSUM(a,b) (a>=b?a+log(1.0+exp(b-a)):b+log(1.0+exp(a-b)))
        
        __device__ double comp_psi1_element(double var, double l, double *Z, double *mu, double *S, double *logGamma, double *log1Gamma, double *logpsi1denom, int N, int M, int Q, int idx)
        {
            int n = idx%N;
            int m = idx/N;
            double psi1_exp=0;
            for(int q=0;q<Q;q++){
                double muZ = mu[IDX_NQ(n,q)]-Z[IDX_MQ(m,q)];
                double exp1 = logGamma[IDX_NQ(n,q)] - (logpsi1denom[IDX_NQ(n,q)] + muZ*muZ/(S[IDX_NQ(n,q)]+l) )/2.0;
                double exp2 = log1Gamma[IDX_NQ(n,q)] - Z[IDX_MQ(m,q)]*Z[IDX_MQ(m,q)]/(l*2.0);
                psi1_exp += LOGEXPSUM(exp1,exp2);
            }
            return var*exp(psi1_exp);
        }
        """)
    
    # The kernel form computing psi1 het_noise
    comp_psi1_het = ElementwiseKernel(
        "double *psi1, double var, double *l, double *Z, double *mu, double *S, double *logGamma, double *log1Gamma, double *logpsi1denom, int N, int M, int Q",
        "psi1[i] = comp_psi1_element_het(var,l, Z, mu, S, logGamma, log1Gamma, logpsi1denom, N, M, Q, i)",
        "comp_psi1_het",
        preamble="""
        #define IDX_NMQ(n,m,q) ((q*M+m)*N+n)
        #define IDX_NQ(n,q) (q*N+n)
        #define IDX_MQ(m,q) (q*M+m)
        #define LOGEXPSUM(a,b) (a>=b?a+log(1.0+exp(b-a)):b+log(1.0+exp(a-b)))
        
        __device__ double comp_psi1_element_het(double var, double *l, double *Z, double *mu, double *S, double *logGamma, double *log1Gamma, double *logpsi1denom, int N, int M, int Q, int idx)
        {
            int n = idx%N;
            int m = idx/N;
            double psi1_exp=0;
            for(int q=0;q<Q;q++){
                double muZ = mu[IDX_NQ(n,q)]-Z[IDX_MQ(m,q)];
                double exp1 = logGamma[IDX_NQ(n,q)] - (logpsi1denom[IDX_NQ(n,q)] + muZ*muZ/(S[IDX_NQ(n,q)]+l[q]) )/2.0;
                double exp2 = log1Gamma[IDX_NQ(n,q)] - Z[IDX_MQ(m,q)]*Z[IDX_MQ(m,q)]/(l[q]*2.0);
                psi1_exp += LOGEXPSUM(exp1,exp2);
            }
            return var*exp(psi1_exp);
        }
        """)
    
    # The kernel form computing psi2 het_noise
    comp_psi2_het = ElementwiseKernel(
        "double *psi2, double var, double *l, double *Z, double *mu, double *S, double *logGamma, double *log1Gamma, double *logpsi2denom, int N, int M, int Q",
        "psi2[i] = comp_psi2_element_het(var,l, Z, mu, S, logGamma, log1Gamma, logpsi2denom, N, M, Q, i)",
        "comp_psi2_het",
        preamble="""
        #define IDX_NMQ(n,m,q) ((q*M+m)*N+n)
        #define IDX_NQ(n,q) (q*N+n)
        #define IDX_MQ(m,q) (q*M+m)
        #define LOGEXPSUM(a,b) (a>=b?a+log(1.0+exp(b-a)):b+log(1.0+exp(a-b)))
        
        __device__ double comp_psi2_element_het(double var, double *l, double *Z, double *mu, double *S, double *logGamma, double *log1Gamma, double *logpsi2denom, int N, int M, int Q, int idx)
        {
            // psi2 (n,m1,m2)
            int m2 = idx/(M*N);
            int m1 = (idx%(M*N))/N;
            int n = idx%N;

            double psi2_exp=0;
            for(int q=0;q<Q;q++){ 
                double dZ = Z[IDX_MQ(m1,q)]-Z[IDX_MQ(m2,q)];
                double muZ = mu[IDX_NQ(n,q)] - (Z[IDX_MQ(m1,q)]+Z[IDX_MQ(m2,q)])/2.0;
                double exp1 = logGamma[IDX_NQ(n,q)] - (logpsi2denom[IDX_NQ(n,q)])/2.0 - dZ*dZ/(l[q]*4.0) - muZ*muZ/(2*S[IDX_NQ(n,q)]+l[q]);
                double exp2 = log1Gamma[IDX_NQ(n,q)] - (Z[IDX_MQ(m1,q)]*Z[IDX_MQ(m1,q)]+Z[IDX_MQ(m2,q)]*Z[IDX_MQ(m2,q)])/(l[q]*2.0);
                psi2_exp += LOGEXPSUM(exp1,exp2);
            }
            return var*var*exp(psi2_exp);
        }
        """)
    
    # The kernel form computing psi2 
    comp_psi2 = ElementwiseKernel(
        "double *psi2, double var, double l, double *Z, double *mu, double *S, double *logGamma, double *log1Gamma, double *logpsi2denom, int N, int M, int Q",
        "psi2[i] = comp_psi2_element(var,l, Z, mu, S, logGamma, log1Gamma, logpsi2denom, N, M, Q, i)",
        "comp_psi2",
        preamble="""
        #define IDX_NMQ(n,m,q) ((q*M+m)*N+n)
        #define IDX_NQ(n,q) (q*N+n)
        #define IDX_MQ(m,q) (q*M+m)
        #define LOGEXPSUM(a,b) (a>=b?a+log(1.0+exp(b-a)):b+log(1.0+exp(a-b)))
        
        __device__ double comp_psi2_element(double var, double l, double *Z, double *mu, double *S, double *logGamma, double *log1Gamma, double *logpsi2denom, int N, int M, int Q, int idx)
        {
            // psi2 (n,m1,m2)
            int m2 = idx/(M*N);
            int m1 = (idx%(M*N))/N;
            int n = idx%N;

            double psi2_exp=0;
            for(int q=0;q<Q;q++){ 
                double dZ = Z[IDX_MQ(m1,q)]-Z[IDX_MQ(m2,q)];
                double muZ = mu[IDX_NQ(n,q)] - (Z[IDX_MQ(m1,q)]+Z[IDX_MQ(m2,q)])/2.0;
                double exp1 = logGamma[IDX_NQ(n,q)] - (logpsi2denom[IDX_NQ(n,q)])/2.0 - dZ*dZ/(l*4.0) - muZ*muZ/(2*S[IDX_NQ(n,q)]+l);
                double exp2 = log1Gamma[IDX_NQ(n,q)] - (Z[IDX_MQ(m1,q)]*Z[IDX_MQ(m1,q)]+Z[IDX_MQ(m2,q)]*Z[IDX_MQ(m2,q)])/(l*2.0);
                psi2_exp += LOGEXPSUM(exp1,exp2);
            }
            return var*var*exp(psi2_exp);
        }
        """)
    
    
    # compute psidenom
    comp_logpsidenom_het = ElementwiseKernel(
        "double *out, double *S, double *l, double scale",
        "out[i] = comp_logpsidenom_het_element(S, l, scale, i)",
        "comp_logpsidenom_het",
        preamble="""        
        __device__ double comp_logpsidenom_het_element(double *S, double *l, double scale, int idx)
        {
            int q = idx/N;
            int n = idx%N;

            return scale*S[idx]/l[q]+1.0;
        }
        """)
    
    # compute psidenom
    comp_logpsidenom = ElementwiseKernel(
        "double *out, double *S, double l, double scale",
        "out[i] = comp_logpsidenom_element(S, l, scale, i)",
        "comp_logpsidenom",
        preamble="""        
        __device__ double comp_logpsidenom_element(double *S, double l, double scale, int idx)
        {
            int q = idx/N;
            int n = idx%N;

            return scale*S[idx]/l+1.0;
        }
        """)
    
except:
    pass

class PSICOMP_SSRBF(object):
    def __init__(self):
        self.gpuCache = None
    
    def _initGPUCache(self, N, M, Q):
        if self.gpuCache == None:
            self.gpuCache = {
                             'l_gpu'                :gpuarray.empty((Q,),np.float64,order='F'),
                             'Z_gpu'                :gpuarray.empty((M,Q),np.float64,order='F'),
                             'mu_gpu'               :gpuarray.empty((N,Q),np.float64,order='F'),
                             'S_gpu'                :gpuarray.empty((N,Q),np.float64,order='F'),
                             'gamma_gpu'            :gpuarray.empty((N,Q),np.float64,order='F'),
                             'logGamma_gpu'         :gpuarray.empty((N,Q),np.float64,order='F'),
                             'log1Gamma_gpu'        :gpuarray.empty((N,Q),np.float64,order='F'),
                             'logpsidenom_gpu'      :gpuarray.empty((N,Q),np.float64,order='F'),
                             'psi0_gpu'             :gpuarray.empty((N,),np.float64,order='F'),
                             'psi1_gpu'             :gpuarray.empty((N,M),np.float64,order='F'),
                             'psi2_gpu'             :gpuarray.empty((N,M,M),np.float64,order='F'),
                             }
    
    def psicomputations(self, variance, lengthscale, Z, mu, S, gamma):
        if isinstance(lengthscale, np.ndarray) and len(lengthscale)>1:
            het_noise = True
        else:
            het_noise = False
        
        N = mu.shape[0]
        M = Z.shape[0]
        Q = mu.shape[1]
        
        self._initGPUCache(N,M,Q)
        if het_noise:
            l_gpu = self.gpuCache['l_gpu']
            l_gpu.set(np.asfortranarray(lengthscale**2))
        else:
            lengthscale2 = lengthscale**2
        
        Z_gpu = self.gpuCache['Z_gpu']
        mu_gpu = self.gpuCache['mu_gpu']
        S_gpu = self.gpuCache['S_gpu']
        gamma_gpu = self.gpuCache['gamma_gpu']
        logGamma_gpu = self.gpuCache['logGamma_gpu']
        log1Gamma_gpu = self.gpuCache['log1Gamma_gpu']
        logpsidenom_gpu = self.gpuCache['logpsidenom_gpu']
        psi0_gpu = self.gpuCache['psi0_gpu']
        psi1_gpu = self.gpuCache['psi1_gpu']
        psi2_gpu = self.gpuCache['psi2_gpu']
        
        Z_gpu.set(np.asfortranarray(Z))
        mu_gpu.set(np.asfortranarray(mu))
        S_gpu.set(S)
        gamma_gpu.set(gamma)
        linalg_gpu.log(gamma_gpu,logGamma_gpu)
        linalg_gpu.logOne(gamma_gpu,log1Gamma_gpu)
        
        psi0_gpu.fill(variance)
        if het_noise:
            comp_logpsidenom_het(logpsidenom_gpu, S_gpu,l_gpu,1.0)
            comp_psi1_het(psi1_gpu, variance, lengthscale2, Z_gpu, mu_gpu, S_gpu, logGamma_gpu, log1Gamma_gpu, logpsidenom_gpu, N, M, Q)
            comp_logpsidenom_het(logpsidenom_gpu, S_gpu,l_gpu,2.0)
            comp_psi2_het(psi2_gpu, variance, lengthscale2, Z_gpu, mu_gpu, S_gpu, logGamma_gpu, log1Gamma_gpu, logpsidenom_gpu, N, M, Q)
        else:
            comp_logpsidenom(logpsidenom_gpu, S_gpu,lengthscale2,1.0)
            comp_psi1(psi1_gpu, variance, lengthscale2, Z_gpu, mu_gpu, S_gpu, logGamma_gpu, log1Gamma_gpu, logpsidenom_gpu, N, M, Q)
            comp_logpsidenom(logpsidenom_gpu, S_gpu,lengthscale2,2.0)
            comp_psi2(psi2_gpu, variance, lengthscale2, Z_gpu, mu_gpu, S_gpu, logGamma_gpu, log1Gamma_gpu, logpsidenom_gpu, N, M, Q)

        return psi0_gpu.get(), psi1_gpu.get(), psi2_gpu.get()
        

@Cache_this(limit=1)
def _Z_distances(Z):
    Zhat = 0.5 * (Z[:, None, :] + Z[None, :, :]) # M,M,Q
    Zdist = 0.5 * (Z[:, None, :] - Z[None, :, :]) # M,M,Q
    return Zhat, Zdist

def _psicomputations(variance, lengthscale, Z, mu, S, gamma):
    """
    """
    

@Cache_this(limit=1)
def _psi1computations(variance, lengthscale, Z, mu, S, gamma):
    """
    Z - MxQ
    mu - NxQ
    S - NxQ
    gamma - NxQ
    """
    # here are the "statistics" for psi1 and psi2
    # Produced intermediate results:
    # _psi1                NxM
    # _dpsi1_dvariance     NxM
    # _dpsi1_dlengthscale  NxMxQ
    # _dpsi1_dZ            NxMxQ
    # _dpsi1_dgamma        NxMxQ
    # _dpsi1_dmu           NxMxQ
    # _dpsi1_dS            NxMxQ
    
    lengthscale2 = np.square(lengthscale)

    # psi1
    _psi1_denom = S[:, None, :] / lengthscale2 + 1.  # Nx1xQ
    _psi1_denom_sqrt = np.sqrt(_psi1_denom) #Nx1xQ
    _psi1_dist = Z[None, :, :] - mu[:, None, :]  # NxMxQ
    _psi1_dist_sq = np.square(_psi1_dist) / (lengthscale2 * _psi1_denom) # NxMxQ
    _psi1_common = gamma[:,None,:] / (lengthscale2*_psi1_denom*_psi1_denom_sqrt) #Nx1xQ
    _psi1_exponent1 = np.log(gamma[:,None,:]) -0.5 * (_psi1_dist_sq + np.log(_psi1_denom)) # NxMxQ
    _psi1_exponent2 = np.log(1.-gamma[:,None,:]) -0.5 * (np.square(Z[None,:,:])/lengthscale2) # NxMxQ
    _psi1_exponent_max = np.maximum(_psi1_exponent1,_psi1_exponent2)
    _psi1_exponent = _psi1_exponent_max+np.log(np.exp(_psi1_exponent1-_psi1_exponent_max) + np.exp(_psi1_exponent2-_psi1_exponent_max)) #NxMxQ
    _psi1_exp_sum = _psi1_exponent.sum(axis=-1) #NxM
    _psi1_exp_dist_sq = np.exp(-0.5*_psi1_dist_sq) # NxMxQ
    _psi1_exp_Z = np.exp(-0.5*np.square(Z[None,:,:])/lengthscale2) # 1xMxQ
    _psi1_q = variance * np.exp(_psi1_exp_sum[:,:,None] - _psi1_exponent) # NxMxQ
    _psi1 = variance * np.exp(_psi1_exp_sum) # NxM
    _dpsi1_dvariance = _psi1 / variance # NxM
    _dpsi1_dgamma = _psi1_q * (_psi1_exp_dist_sq/_psi1_denom_sqrt-_psi1_exp_Z) # NxMxQ
    _dpsi1_dmu = _psi1_q * (_psi1_exp_dist_sq * _psi1_dist * _psi1_common) # NxMxQ
    _dpsi1_dS = _psi1_q * (_psi1_exp_dist_sq * _psi1_common * 0.5 * (_psi1_dist_sq - 1.)) # NxMxQ
    _dpsi1_dZ = _psi1_q * (- _psi1_common * _psi1_dist * _psi1_exp_dist_sq - (1-gamma[:,None,:])/lengthscale2*Z[None,:,:]*_psi1_exp_Z) # NxMxQ
    _dpsi1_dlengthscale = 2.*lengthscale*_psi1_q * (0.5*_psi1_common*(S[:,None,:]/lengthscale2+_psi1_dist_sq)*_psi1_exp_dist_sq + 0.5*(1-gamma[:,None,:])*np.square(Z[None,:,:]/lengthscale2)*_psi1_exp_Z) # NxMxQ

    N = mu.shape[0]
    M = Z.shape[0]
    Q = mu.shape[1]

    l_gpu = gpuarray.to_gpu(np.asfortranarray(lengthscale2))
    Z_gpu = gpuarray.to_gpu(np.asfortranarray(Z))
    mu_gpu = gpuarray.to_gpu(np.asfortranarray(mu))
    S_gpu = gpuarray.to_gpu(np.asfortranarray(S))
    #gamma_gpu = gpuarray.to_gpu(gamma)
    logGamma_gpu = gpuarray.to_gpu(np.asfortranarray(np.log(gamma)))
    log1Gamma_gpu = gpuarray.to_gpu(np.asfortranarray(np.log(1.-gamma)))
    logpsi1denom_gpu = gpuarray.to_gpu(np.asfortranarray(np.log(S/lengthscale2+1.)))
    psi1_gpu = gpuarray.empty((mu.shape[0],Z.shape[0]),np.float64, order='F')
    
    comp_psi1(psi1_gpu, variance, lengthscale2, Z_gpu, mu_gpu, S_gpu, logGamma_gpu, log1Gamma_gpu, logpsi1denom_gpu, N, M, Q)
    
    #print np.abs(psi1_gpu.get()-_psi1).max()

    return _psi1, _dpsi1_dvariance, _dpsi1_dgamma, _dpsi1_dmu, _dpsi1_dS, _dpsi1_dZ, _dpsi1_dlengthscale

@Cache_this(limit=1)
def _psi2computations(variance, lengthscale, Z, mu, S, gamma):
    """
    Z - MxQ
    mu - NxQ
    S - NxQ
    gamma - NxQ
    """
    # here are the "statistics" for psi1 and psi2
    # Produced intermediate results:
    # _psi2                NxMxM
    # _psi2_dvariance      NxMxM
    # _psi2_dlengthscale   NxMxMxQ
    # _psi2_dZ             NxMxMxQ
    # _psi2_dgamma         NxMxMxQ
    # _psi2_dmu            NxMxMxQ
    # _psi2_dS             NxMxMxQ
    
    lengthscale2 = np.square(lengthscale)
    
    _psi2_Zhat, _psi2_Zdist = _Z_distances(Z)
    _psi2_Zdist_sq = np.square(_psi2_Zdist / lengthscale) # M,M,Q
    _psi2_Z_sq_sum = (np.square(Z[:,None,:])+np.square(Z[None,:,:]))/lengthscale2 # MxMxQ

    # psi2
    _psi2_denom = 2.*S[:, None, None, :] / lengthscale2 + 1. # Nx1x1xQ
    _psi2_denom_sqrt = np.sqrt(_psi2_denom)
    _psi2_mudist = mu[:,None,None,:]-_psi2_Zhat #N,M,M,Q
    _psi2_mudist_sq = np.square(_psi2_mudist)/(lengthscale2*_psi2_denom)
    _psi2_common = gamma[:,None,None,:]/(lengthscale2 * _psi2_denom * _psi2_denom_sqrt) # Nx1x1xQ
    _psi2_exponent1 = -_psi2_Zdist_sq -_psi2_mudist_sq -0.5*np.log(_psi2_denom)+np.log(gamma[:,None,None,:]) #N,M,M,Q
    _psi2_exponent2 = np.log(1.-gamma[:,None,None,:]) - 0.5*(_psi2_Z_sq_sum) # NxMxMxQ
    _psi2_exponent_max = np.maximum(_psi2_exponent1, _psi2_exponent2)
    _psi2_exponent = _psi2_exponent_max+np.log(np.exp(_psi2_exponent1-_psi2_exponent_max) + np.exp(_psi2_exponent2-_psi2_exponent_max))
    _psi2_exp_sum = _psi2_exponent.sum(axis=-1) #NxM
    _psi2_q = np.square(variance) * np.exp(_psi2_exp_sum[:,:,:,None]-_psi2_exponent) # NxMxMxQ 
    _psi2_exp_dist_sq = np.exp(-_psi2_Zdist_sq -_psi2_mudist_sq) # NxMxMxQ
    _psi2_exp_Z = np.exp(-0.5*_psi2_Z_sq_sum) # MxMxQ
    _psi2 = np.square(variance) * np.exp(_psi2_exp_sum) # N,M,M
    _dpsi2_dvariance = 2. * _psi2/variance # NxMxM
    _dpsi2_dgamma = _psi2_q * (_psi2_exp_dist_sq/_psi2_denom_sqrt - _psi2_exp_Z) # NxMxMxQ
    _dpsi2_dmu = _psi2_q * (-2.*_psi2_common*_psi2_mudist * _psi2_exp_dist_sq) # NxMxMxQ
    _dpsi2_dS = _psi2_q * (_psi2_common * (2.*_psi2_mudist_sq - 1.) * _psi2_exp_dist_sq) # NxMxMxQ
    _dpsi2_dZ = 2.*_psi2_q * (_psi2_common*(-_psi2_Zdist*_psi2_denom+_psi2_mudist)*_psi2_exp_dist_sq - (1-gamma[:,None,None,:])*Z[:,None,:]/lengthscale2*_psi2_exp_Z) # NxMxMxQ
    _dpsi2_dlengthscale = 2.*lengthscale* _psi2_q * (_psi2_common*(S[:,None,None,:]/lengthscale2+_psi2_Zdist_sq*_psi2_denom+_psi2_mudist_sq)*_psi2_exp_dist_sq+(1-gamma[:,None,None,:])*_psi2_Z_sq_sum*0.5/lengthscale2*_psi2_exp_Z) # NxMxMxQ

    N = mu.shape[0]
    M = Z.shape[0]
    Q = mu.shape[1]

#    l_gpu = gpuarray.to_gpu(np.asfortranarray(lengthscale2))
    Z_gpu = gpuarray.to_gpu(np.asfortranarray(Z))
    mu_gpu = gpuarray.to_gpu(np.asfortranarray(mu))
    S_gpu = gpuarray.to_gpu(np.asfortranarray(S))
    #gamma_gpu = gpuarray.to_gpu(gamma)
    logGamma_gpu = gpuarray.to_gpu(np.asfortranarray(np.log(gamma)))
    log1Gamma_gpu = gpuarray.to_gpu(np.asfortranarray(np.log(1.-gamma)))
    logpsi2denom_gpu = gpuarray.to_gpu(np.asfortranarray(np.log(2.*S/lengthscale2+1.)))
    psi2_gpu = gpuarray.empty((mu.shape[0],Z.shape[0],Z.shape[0]),np.float64, order='F')
    
    comp_psi2(psi2_gpu, variance, lengthscale2, Z_gpu, mu_gpu, S_gpu, logGamma_gpu, log1Gamma_gpu, logpsi2denom_gpu, N, M, Q)
    
    print np.abs(psi2_gpu.get()-_psi2).max()

    return _psi2, _dpsi2_dvariance, _dpsi2_dgamma, _dpsi2_dmu, _dpsi2_dS, _dpsi2_dZ, _dpsi2_dlengthscale
