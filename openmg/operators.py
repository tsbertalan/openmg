'''
Operator matrices:
 1. Restriction operators (via a geometric argument) for 1, 2, and 3 spatial dimensions
 2. A central-difference approximation to the Poisson operator
 3. The action of (1) on (2)
 
@author: bertalan@princeton.edu
'''
import numpy as np
import scipy.sparse

import tools


def restriction(shape, dense=False):
    """
    Create a nonsquare restriction matrix by taking every second point along
    each axis in a 1D, 2D, or 3D domain.
    
    Parameters
    ----------
    shape : tuple of ints
        The (rectangular) geometry of the problem.
        
    Optional Parameters
    -------------------
    dense=False : bool
        Whether to use and return a sparse (CSR) restriction matrix.
        
    Returns
    -------
    R : ndarray
        Array of shape (N/(2**alpha), N), where the int N is the number of
        unknowns (product of shape ints), and alpha=len(shape).
    
    Requiring the shape of the domain is bad for the black-boxibility.
    Implementing a separate Ruge Steuben restriction method would avoid this.
    Returns a CSR matrix by default, but will return dense if dense=True is given.

    It would be nice to make this rely only on N, not shape. But, really, the
    restriction operator is problem-dependent. There are certainly problem
    domains where "dimensionality" means something other than spatial dimensions. 
    """
    # alpha is the dimensionality of the problem.
    alpha = len(shape)
    NX = shape[0]
    if alpha >= 2:
        NY = shape[1]
        #NZ = shape[3] # don't need this
    N = tools.product(shape)

    n = N / (2 ** alpha)
    if n == 0:
        raise ValueError('New restriction matrix would have shape ' + str((n,N))
                         + '.' + 'Coarse set would have 0 points! ' + 
                        'Try a larger problem or fewer gridLevels.')
    if dense:
        R = np.zeros((n, N))
    else:
        R = scipy.sparse.lil_matrix((n, N))
    
    # columns in the restriction matrix:
    if alpha == 1:
        coarseColumns = np.array(range(N)).reshape(shape)[::2].ravel()
    elif alpha == 2:
        coarseColumns = np.array(range(N)).reshape(shape)[::2, ::2].ravel()
    elif alpha == 3:
        coarseColumns = np.array(range(N)).reshape(shape)[::2, ::2, ::2].ravel()
    else:
        raise ValueError("restriction(): Greater than 3 dimensions is not"
                         "implemented. (shape was" + str(shape) +" .)")
    
    each = 1.0 / (2 ** alpha)
    for r, c in zip(range(n), coarseColumns):
        R[r, c] = each
        R[r, c + 1] = each
        if alpha >= 2:
            R[r, c + NX] = each
            R[r, c + NX + 1] = each
            if alpha == 3:
                R[r, c + NX * NY] = each
                R[r, c + NX * NY + 1] = each
                R[r, c + NX * NY + NX] = each
                R[r, c + NX * NY + NX + 1] = each

    if dense:
        return R
    else:
        return R.tocsr()


def restrictionList(problemShape, coarsestLevel, dense=False, verbose=False):
    """
    Returns a list of restriction matrices (non-square) for each level.
    It should, therefore, be a list with as many elements
    as the output of coeffecientList().
    
    Parameters
    ----------
    problemShape : tuple of ints
        The problem geometry. Only rectangular geometries of less than 3
        spatial dimensions are dealt with here. The product of the values in
        this tuple should be the total number of unknowns, N.
    coarsestLevel : int
        With a numbering like [1, 2, ..., coarsest level], this defines
        how deep the coarsening hierarchy should go.
    
    Optional Parameters
    -------------------
    dense=False : bool
        Whether to not use CSR storage for the returned restriction operators.
    verbose=False: bool
        Whether to print progress information.
        
    Returns
    -------
    R : list of ndarrays
        A list of (generally nonsquare) restriction operators, one for each
        transition between levels of resolution.
    """
    if verbose: print "Generating restriction matrices; dense=%s" % dense
    levels = coarsestLevel + 1
    R = list(range(levels - 1))  # We don't need R at the coarsest level.
    for level in range(levels - 1):
        R[level] = restriction(tuple(np.array(problemShape) / (2 ** level)), dense=dense)
    return R


def coeffecientList(A_in, R, dense=False, verbose=False):
    """Returns a list of coeffecient matrices.
    
    Parameters
    ----------
    A_in : ndarray
        The (square) coeffecient matrix for the original problem.
    R : list of ndarrays    
        A list of (generally nonsquare) restriction matrices, as produced by
        restrictionList(N, problemShape, coarsestLevel)
        
    Optional Parameters
    -------------------
    dense=False : bool
        whether to not use sparse (CSR) storage for the coeffecient matrices
    verbose=False: bool
        Whether to print progress information.
        
    Returns
    ------
    A : list of ndarrays
        A list of square 2D ndarrays of decreasing size; one for each level
        level of resolution.
    """
    if verbose: print "Generating coefficient matrices; dense=%s ..." % dense,
    coarsestLevel = len(R)
    levels = coarsestLevel + 1
    A = list(range(levels))
    if dense:
        if scipy.sparse.issparse(A_in):
            A[0] = A_in.todense()
        else:
            A[0] = A_in
    else:
        A[0] = scipy.sparse.csr_matrix(A_in)
    for level in range(1, levels):
        # This is the Galerkin "RAP" coarse-grid operator
        # A_H = R * A_h * R^T
        # or
        # A_H = R * A_h * P :
        A[level] = tools.flexibleMmult(
                                  tools.flexibleMmult(R[level-1], A[level-1]),
                                                             R[level-1].T)
    if verbose: print 'made %i A matrices' % len(A)
    return A


def poisson1Dsparse(N):
    '''Returns a sparse square coefficient matrix for the 1D Poisson equation.
    '''
    import scipy.sparse as sparse
    i=0
    main = sparse.eye(N, N) * 4
    off = (sparse.eye(N-1, N-1) * -1).tocsr()
    longPad = np.zeros((N, 1))
    shortPad = np.zeros((N-1, 1))
    top = sparse.vstack((sparse.hstack((shortPad, off)), longPad.T))
    bottom = top.T
    A = main + top + bottom
    return A


def poisson1D(shape, sparse=False):
    N = shape[0]
    if sparse:
        return poisson1Dsparse(N)
    if isinstance(N, tuple):
        N = N[0]
    x = -1
    y =  2
    z = -1
    x = x * np.ones(N - 1)
    z = z * np.ones(N - 1)
    y = y * np.ones(N)
    return np.diag(x, -1) + np.diag(y) + np.diag(z, 1)
    

def poisson2D((NX, NY), sparse=False):
    '''Returns a dense square coefficient matrix for the 2D Poisson equation.
    '''
    if sparse: raise NotImplementedError("Sparse poisson for alpha>1 is not yet implemented.")
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


def poisson3D((NX, NY, NZ), sparse=False):
    '''Returns a dense square coefficient matrix, for the 3D Poisson equation.
    '''
    if sparse: raise NotImplementedError("Sparse poisson for alpha>1 is not yet implemented.")
    N = NX * NY * NZ
    A = np.zeros((N, N))
    for i in xrange(N):
        A[i,i] = -6
        for index in i+1, i+NX, i+NX*NY:
            if index < N:
                A[i, index] = 1
    A += A.T
    return A


def poissonnd(shape, sparse=False):
    '''Using a 1-, 2-, or 3-element tuple for the shape,
    return a dense square Poisson matrix for that question.
    # TODO These should use a stencils instead, like PyAMG's examples.
    '''
    if isinstance(shape, int):
        shape = (shape,)
    if len(shape) == 1:
        toReturn = poisson1D(shape, sparse)
    elif len(shape) == 2:
        toReturn = poisson2D(shape, sparse)
    elif len(shape) == 3:
        toReturn = poisson3D(shape, sparse)
    else:
        raise ValueError('Only 1, 2 or 3 dimensions are allowed.')
    if sparse:
        toReturn = scipy.sparse.csr_matrix(toReturn)
    return toReturn


poisson = poissonnd


if __name__ == '__main__':
    import doctest
    doctest.testmod()
