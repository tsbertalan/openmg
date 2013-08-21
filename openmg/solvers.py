'''
Created on Aug 21, 2013

@author: tsbertalan
'''
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

from tools import flexible_mmult, getresidual

def coarse_solve(A, b):
    '''Uses scipy.sparse.numpy.linalg.solve or np.linalg.solve, depending on
    the sparsity of A.
    '''
    if sparse.issparse(A):
        #toreturn = sparse.sputils.np.linalg.solve(A, b)
        #toreturn = sparse.linalg.dsolve.spsolve(A, b)
        toreturn = splinalg.spsolve(A,b)
    else:
        toreturn = np.linalg.solve(A, b)
    return np.ravel(toreturn)


def iterative_solve(A, b, x, iterations, verbose=False):
    '''
    An implementation of Gauss-Seidel iterative solution.
    Will stop when the given number of iterations are exausted.
    '''
    # TODO: I should change this to (A, b, iterations, x='none', verbose=False)
    N = b.size
    if verbose: print "starting gs,  %i iterations." % iterations
    for iteration in range(iterations):
        if sparse.issparse(A):
            for i in range(N):
                Aix = A[i,:] * x  # scipy 0.10 insists on making a copy
                # for this slice, which is about 60% of our wasted time.
                # v0.11 seems to have lil_matrix.getrowview(). But, for
                # production, this whole code should be written in C, C++,
                # or Fortran. So I will make no further efforts to make
                # the sparse version of OpenMG fast.
                #for j in range(N):
                #    Aix += A[i, j] * x[j]  
                x[i] = x[i] + (b[i] - Aix) / A[i, i]
        else:
            for i in range(N):
                x[i] = x[i] + (b[i] - np.dot(A[i, :], x.reshape((N, 1)))                                      ) / A[i, i]
    return x


def iterative_solve_to_threshold(A, b, x, threshold, verbose=False):
    '''
    An implementation of Gauss-Seidel iterative solution.
    Will stop when the norm of the residual is below the given threshold.
    However,  Gauss-Seidel doesn't always converge,  so it might never return.
    '''
    iteration = 0
    N = len(b)
    if sparse.issparse(A):
        for i in range(b.size):  # [ 0 1 2 3 4 ... n-1 ]
            x[i] = x[i] + (b[i] - flexible_mmult(
                                    A.getrow(i), x.reshape((N, 1))
                                  )
                          ) / A[i, i]
    else:
        for i in range(b.size):  # [ 0 1 2 3 4 ... n-1 ]
            x[i] = x[i] + (b[i] - flexible_mmult(
                                    A[i, :].reshape((1, N)), x.reshape((N, 1))
                                  )
                          ) / A[i, i]
    norm = np.linalg.norm(getresidual(b, A, x, N))
    while norm > threshold:
        if verbose:
            if not iteration % 100:
                print 'iteration', iteration, ': norm is', norm
        iteration += 1
        if sparse.issparse(A):
            for i in range(b.size):
                x[i] = x[i] + (b[i] - flexible_mmult(
                                        A.getrow(i), x.reshape((N, 1))
                                      )
                              ) / A[i, i]
        else:
            for i in range(b.size):
                x[i] = x[i] + (b[i] - flexible_mmult(
                                        A[i, :].reshape((1, N)),
                                        x.reshape((N, 1))
                                      )
                              ) / A[i, i]
        norm = np.linalg.norm(getresidual(b, A, x, N))
    if verbose: print "Finished iterative solver in %i iterations." % iteration
    return (x, iteration)


def slow_iterative_solve(A, b, x, iterations, verbose=False):
    '''An implementation of Gauss-Seidel iterative solution.
    Will stop when the given number of iterations are exausted.
    Uses loops instead of flexible_mmult() for matrix multiplication.
    If it were a Jacobi implementation, it might be easy to parallelize.
    However, GS is inherently sequential (or so I hear).
    But changing this to a Jacobi implementation later might not be too hard.
    '''
    iteration = 0
    last = 0
    for iteration in range(iterations):
        currentpercent = np.round(100. * iteration / iterations, decimals=0)
        if (currentpercent % 10 == 0) and (currentpercent != 0) and (currentpercent != last):
            last = currentpercent
            print "mg_solve_tom: iterative progress: %i%%" % currentpercent
        for i in range(b.size):
            lowersum = 0
            for j in range(i):
                lowersum += A[(i, j)] * x[j]
            uppersum = 0
            for j in range(i + 1, b.size):
                uppersum += A[(i, j)] * x[j]
            x[i] = (b[i] - lowersum - uppersum) / A[(i, i)]
    return x


if __name__ == '__main__':
    import doctest
    doctest.testmod()
