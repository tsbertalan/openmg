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


def startProgressFile(cycle, norm):
    from time import strftime, localtime
    isolocaltime = localtime()
    isodate = strftime('%Y%m%d', isolocaltime)
    isotime = strftime('T%H%M%S', isolocaltime)
    cycles_progress_file_name = 'output/cyclesvsresid-%s%s.csv' % (isodate, isotime)
    print "saving progress to", cycles_progress_file_name
    cycles_progress_file = open(cycles_progress_file_name, 'a')
    cycles_progress_file.write('cycle, residual_norm\n')
    cycles_progress_file.write('%d, %.012f\n' % (cycle, norm))
    cycles_progress_file.flush()
    return cycles_progress_file


def dictUpdateNoClobber(updateDict, targetDict):
    for key, value in updateDict.iteritems():
        dictAddNoClobber(targetDict, key, value)
    return targetDict


def dictAddNoClobber(dictionary, key, value):
    """
    Can be used "in-place"
    >>> adict = {"hello": 42} 
    >>> out = dictAddNoClobber(adict, "huh", "no")
    >>> "huh" in adict and "huh" in out
    True
    """
    if key not in dictionary:
        dictionary[key] = value
    return dictionary


if __name__ == '__main__':
    import doctest
    doctest.testmod()
