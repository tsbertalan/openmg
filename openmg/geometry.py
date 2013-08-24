'''
Functions specific to the geometric nature of our sample
problems. Specifically, it only (for now) contains functions for generating
coeffecient matrices for a central-difference approximation to the Poisson
operator.
 
@author: bertalan@princeton.edu
'''
import numpy as np


def poisson1D(n):
    if isinstance(n, tuple):
        n = n[0]
    x = -1
    y =  2
    z = -1
    x = x * np.ones(n - 1)
    z = z * np.ones(n - 1)
    y = y * np.ones(n)
    return np.diag(x, -1) + np.diag(y) + np.diag(z, 1)
    

def poisson2D((NX, NY)):
    '''Returns a dense square coefficient matrix for the 2D Poisson equation.
    '''
    N = NX * NY
    main = np.eye(N) * -4
    oneup = np.hstack((
                np.zeros((NX * NY, 1)),
                np.vstack((
                    np.eye(NX * NY - 1),
                    np.zeros((1, NX * NY - 1))
                ))
            ))
    twoup = np.hstack((
                np.zeros((NX * NY, 1 + NX)),
                np.vstack((
                    np.eye(NX * NY - 1 - NX),
                    np.zeros((1 + NX, NX * NY - 1 - NX))
                ))
            ))
    return main + oneup + twoup + oneup.T + twoup.T


def poisson3D((NX, NY, NZ)):
    '''Returns a dense square coefficient matrix, for the 3D Poisson equation.
    '''
    N = NX * NY * NZ
    A = np.zeros((N, N))
    for i in xrange(N):
        A[i,i] = -6
        for index in i+1, i+NX, i+NX*NY:
            if index < N:
                A[i, index] = 1
    A += A.T
    return A


def poissonnd(shape):
    '''Using a 1-, 2-, or 3-element tuple for the shape,
    return a dense square Poisson matrix for that question.
    # TODO These should use a stencils instead, like PyAMG's examples.
    '''
    if len(shape) == 1:
        return poisson1D(shape)
    elif len(shape) == 2:
        return poisson2D(shape)
    elif len(shape) == 3:
        return poisson3D(shape)
    else:
        raise ValueError('Only 1, 2 or 3 dimensions are allowed.')

poisson = poissonnd

if __name__ == '__main__':
    import doctest
    doctest.testmod()
