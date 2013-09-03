"""
Direct solvers and relaxation methods.

Currently, the direct solver (coarseSolve) uses either np.linalg.solve() or
scipy.sparse.linalg.spsolve(), and the relaxation method is a pure-Python
Gauss-Seidel implementation.

@author: bertalan@princeton.edu
"""
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

from tools import getresidual

def coarseSolve(A, b):
    """Uses scipy.sparse.numpy.linalg.solve or np.linalg.solve, depending on
    the sparsity of A.
    """
    if sparse.issparse(A):
        #toreturn = sparse.sputils.np.linalg.solve(A, b)
        #toreturn = sparse.linalg.dsolve.spsolve(A, b)
        toreturn = splinalg.spsolve(A, b)
    else:
        toreturn = np.linalg.solve(A, b)
    return np.ravel(toreturn)

def smooth(A, b, x, iterations, verbose=False):
    return gaussSeidel(A, b, x, iterations=iterations, verbose=verbose)

def smoothToThreshold(A, b, x, threshold, verbose=False):
    return gaussSeidel(A, b, x, threshold=threshold, verbose=verbose)

def gaussSeidel(A, b, x, iterations=None, threshold=None, verbose=False):
    """An implementation of Gauss-Seidel iterative solution.
    Will stop when the given number of iterations are exausted OR an absolute
    error norm is achieved.
    """
    if iterations is None and threshold is None:
        iterations = 1
    N = x.size

    def stop(iteration, x):
        iterStatus = threshStatus = False
        if iterations is not None:
            iterStatus = (iteration >= iterations)
        if threshold is not None:
            norm = np.linalg.norm(getresidual(b, A, x, N))
            threshStatus = (norm < threshold)
        return iterStatus or threshStatus
    
    iteration = 0
    stopping = stop(iteration, x)
    while not stopping:
        if sparse.issparse(A):
            for i in range(N):
                
                # Thank you, anonymous reviewer! 47% speedup (dependent on problem size)!
                # Aix = A[i,:] * x
                # .. translates to: >>>
                rowstart = A.indptr[i]
                rowend = A.indptr[i+1]
                Aix = np.dot(A.data[rowstart:rowend],
                            x[A.indices[rowstart:rowend]]
                            )
                # <<< 
                
                x[i] = x[i] + (b[i] - Aix) / A[i, i]
        else:
            for i in range(N):
                x[i] = x[i] + (np.array(b).ravel()[i] - np.dot(A[i, :], x.reshape((N, 1)))) / A[i, i]
        
        iteration += 1
        stopping = stop(iteration, x)
    return x


if __name__ == '__main__':
    import doctest
    doctest.testmod()
