import numpy as np


def sampling_linear_model(n, d, a):
    X    = np.random.multivariate_normal(mean =np.zeros((d,)), 
                                      cov     =np.eye(d),
                                      size    =n
                                     )
    beta    = np.ones((d,1))/np.sqrt(d)
    cov     = np.eye(2) + a*(np.ones((2,2)) - np.eye(2))
    epsilon = np.random.multivariate_normal(mean=np.zeros((2,)), cov=cov, size=n)
    Y1      = np.dot(X, beta).ravel() + epsilon[:,0]
    Y2      = np.dot(X, beta).ravel() + epsilon[:,1]

    return X, Y1, Y2, beta

            
    
            

        