'''
Created on Aug 21, 2013

@author: tsbertalan
'''
import numpy as np
import scipy.sparse as sparse



def getresidual(b, A, x, N):
    '''b and x are dense ndarrays; N is an int,  and A is a sparse matrix
    '''
    return b.reshape((N, 1)) - flexible_mmult(A, x.reshape((N, 1)))


def flexible_mmult(x, y):
    '''Dot two 2D arrays
    '''
    if (not sparse.issparse(x)) and (not sparse.issparse(y)):
        return np.dot(x, y)
    else:
        # we can do overloaded-asterisk multiplication
        # as long as at least one matrix is sparse
        return x * y


if __name__ == '__main__':
    import doctest
    doctest.testmod()
