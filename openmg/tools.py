'''
Helper/utility functions for doing a few repeated tasks that
are not of any real algorithmic importance.

@author: bertalan@princeton.edu
'''
import numpy as np
import scipy.sparse as sparse



def getresidual(b, A, x, N):
    '''b and x are dense ndarrays; N is an int,  and A is a sparse matrix
    '''
    return b.reshape((N, 1)) - flexibleMmult(A, x.reshape((N, 1)))


def flexibleMmult(x, y):
    '''Dot two 2D arrays
    '''
    if (not sparse.issparse(x)) and (not sparse.issparse(y)):
        return np.dot(x, y)
    else:
        # we can do overloaded-asterisk multiplication
        # as long as at least one matrix is sparse
        return x * y


def dictUpdateNoClobber(updateDict, targetDict):
    """Like dict.update, but will not clobber existing entries.
    >>> adict = {'a': 'A'}
    >>> 'b' in adict or 'c' in adict
    False
    >>> out = dictUpdateNoClobber({'b': 'B', 'c': 'C'}, adict)
    >>> 'b' in adict and 'c' in adict
    True
    """
    for key, value in updateDict.iteritems():
        dictAddNoClobber(targetDict, key, value)
    return targetDict


def dictAddNoClobber(dictionary, key, value):
    """
    Add entries to a dictionary only if they're not already there. 
    >>> adict = {"hello": 42} 
    >>> out = dictAddNoClobber(adict, "huh", "no")
    >>> "huh" in adict and "huh" in out
    True
    """
    if key not in dictionary:
        dictionary[key] = value
    return dictionary


def product(iterableThing):
    out = 1
    for thing in iterableThing:
        out *= thing
    return out


if __name__ == '__main__':
    import doctest
    doctest.testmod()
