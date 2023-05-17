import numpy as np
from sklearn.datasets import make_spd_matrix
from numpy.linalg import det,inv
from tqdm import trange
from numpy import sqrt
from scipy.stats import special_ortho_group
from numpy.random import dirichlet
from scipy.stats import invwishart, matrix_normal


def sample(n_iterations : int,K : int,X_data, seed = 123):

    """ 
    n_iterations : samples to generate
    K : int, number of latent states
    X_data : array-like with shape (n_features,n_samples)
    """

    D = X_data.shape[0]
    T = X_data.shape[1]

    #preparing data
    X = X_data[:, 0:T-1]                                          #this X_{t-1} so times run between [0,T-1]
    Y = X_data[:, 1:T]                                            #this is X_t so times run between [1,T], dimensions D x (T-1)


    Y = Y.T
    X = X.T
    X = np.column_stack((X,np.ones(T-1)))             # dummy variable


    #instantiting everything
    z = np.zeros((n_iterations, T - 1), dtype = np.int8)
    N = np.zeros((K,K))

    Q = np.zeros((n_iterations, K, D, D))             #covariance matrix
    L = np.zeros((n_iterations, K, D + 1, D + 1))     #lambda parameter for matrix normal
    V = np.zeros((n_iterations, K, D, D))             #V parameter for the inverse Wishart
    nu = np.ones((n_iterations, K))*D                 #nu parameter for inverse wishart, already initialised
    A = np.zeros((n_iterations, K, D + 1, D)) 
    M = np.zeros((n_iterations, K, D + 1, D))         #mean matrix for matrix normal
    PI = np.zeros((n_iterations, K, K))               #transition matrix
    alp = np.ones(K)                                  #parameter for the dirichlet

    #defining dictionary of indices for X_k, Y_k
    keys = np.array([str(k) for k in np.arange(K)]) 
    k_indices = dict.fromkeys(keys)

    #ITERATION ZERO             
                
    #randomly sample z from uniform dist
    z[0] = np.random.choice(K, size = T - 1, p = None)

    #filling the matrix N
    for i in range(T - 2):
        N[z[0,i],z[0,i+1]] += 1

    for k in range(K): 
                                                                            #nu has already been initialised
        L[0,k] = make_spd_matrix(D + 1, random_state = seed)                  
        V[0,k] = make_spd_matrix(D, random_state = seed)
                                                                            #M has already been initialised
        k_indices[str(k)] = np.where(z[0,:] == k)[0]                          #indices where z = k
        PI[0,k] = dirichlet(alpha = alp, size = 1).flatten()                  #first approximation of all rows of transition matrix
        Q[0,k] = invwishart.rvs(df = nu[0,k], scale = V[0,k])                 #first approximation of variance matrix
        A[0,k] = matrix_normal.rvs(mean = M[0,k],\
                                    rowcov = inv(L[0,k]), colcov = Q[0,k])

    #ITERATION FROM ONE TO THE END

    for it in trange(1, n_iterations):
        for k in range(K):
            
            X_k = X[k_indices[str(k)],:]                                       #selecting X_k and Y_k
            Y_k = Y[k_indices[str(k)],:]
                
            PI[it,k] = dirichlet(alpha = alp + N[k,:], size = 1).flatten() 
                
            #updating L_k, nu_k, A_k, V_k, Q_k, Mn_k, A_k
            L[it,k] = X_k.T @ X_k +  L[0, k]
            nu[it,k] = nu[0,k] + Y_k.shape[0]
                
            M[it,k] = inv(L[it,k]) @ (X_k.T @ Y_k + L[0,k] @ M[0,k])

            V[it,k] = V[0,k] + (Y_k - X_k @ M[it,k]).T @ (Y_k - X_k @ M[it,k]) +\
            (M[it,k] - M[0,k]).T @ L[0,k] @ (M[it,k] - M[0,k])

            Q[it,k] = invwishart.rvs(df = nu[it,k], scale =  V[it,k])
            A[it,k] = matrix_normal.rvs(mean = M[it,k], rowcov = inv(L[it,k]), colcov = Q[it,k])
            
        for t in np.arange(T-1):
                
            p_prime = np.array([(PI[it,z[it,t-1],k]/sqrt(det(Q[it,k]))) * np.exp(-0.5 * (Y[t,:] - X[t,:] @ A[it,k]) @ inv(Q[it,k]) @ (Y[t,:] - X[t,:] @ A[it,k]).T) for k in np.arange(K)])
            p_prime = p_prime/np.sum(p_prime)
            z[it,t] = np.random.choice(K, size = 1, p = p_prime)
                
        N = np.zeros((K,K))

        for i in range(T - 2):
            N[z[it,i],z[it,i+1]] += 1 
                
        for k in range(K):
            k_indices[str(k)] = np.where(z[it,:] == k)[0]

    return {
              "z"  : z, 
              "N"  : N, 
              "Q"  : Q, 
              "L"  : L, 
              "V"  : V, 
              "nu"  : nu, 
              "A"  : A, 
              "M"  : M, 
              "PI"  : PI, 
              "alp"  : alp
    }
        