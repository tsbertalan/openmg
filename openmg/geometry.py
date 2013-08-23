'''
Created on Aug 21, 2013

@author: tsbertalan
'''
import numpy as np


def poisson(N):
    '''Returns a sparse square coefficient matrix for the 1D Poisson equation.
    '''
    import scipy.sparse as sparse
    #A = sparse.lil_matrix((N, N))
    #for i in range(N):
    #    A[i, i] = 4
    #    if i > 1:
    #        A[i, i-1] = -1
    #    if i < N-1:
    #        A[i, i+1] = -1
    #i = sparse.eye(n, n)
    #A = s + i * 2.0
    #return sparse.csr_matrix(A)
    i=0
    main = sparse.eye(N, N) * 4
    off = (sparse.eye(N-1, N-1) * -1).tocsr()
    longPad = np.zeros((N, 1))
    shortPad = np.zeros((N-1, 1))
    top = sparse.vstack((sparse.hstack((shortPad, off)), longPad.T))
    bottom = top.T
    A = main + top + bottom
    return A


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


if __name__ == '__main__':
    import doctest
    doctest.testmod()
