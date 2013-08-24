"""
Primary functions for a demonstration implementation
of standard multigrid.
 
@author: bertalan@princeton.edu
"""
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

from solvers import smooth, smoothToThreshold, coarseSolve
import tools
import geometry
flexibleMmult = tools.flexibleMmult


defaults = {
    'problemShape': (200,),
    'gridLevels': 2,
    'verbose': False,         
    'threshold': 0.1,
    'cycles': 0,
    'preIterations': 1,
    'postIterations': 0,
    'dense': False,
    'giveInfo': False,
}
def mgSolve(A_in, b, parameters):
    """
    The main externally-usable function.
    
    Parameters
    ----------
        A_in : ndarray
            The (square) coefficient matrix.
        b : ndarray
            The right-hand-side vector.
        parameters : dictionary
            Should include these fields:
            
            Required:
                problemShape : tuple of ints
                    The shape of the spatial domain.
                gridLevels : int
                    How many levels to generate in the restriction hierarchy.
                
            Optional:
                coarsestLevel=gridLevels - 1 : int
                    An int larger than 0 denoting the level at which
                    to use coarseSolve().
                verbose=False : bool
                    Whether to print progress information.
                threshold=.1 : float_like
                    Absolute tolerance for doing v-cycles, if an explicit number
                     of cycles to perform is not given (i.e., for cycles=0).
                cycles=0 : int
                    How many v-cycles to perform. 0 is interpreted as meaning
                    that v-cycles should be continued until the
                    residual norm falls below threshold.
                preIterations=1 : int
                    How many iterations to use in the pre-smoother.
                postIterations=0 : int
                    How many iterations to use in the post-smoother.
                giveInfo=False : bool
                    Whether to return the infoDict
              
            A sample parameters dictionary is available at openmg.defaults .
    
    Returns
    -------
        result : ndarray
            The solution vector; same size as b.
        infoDict : dictionary
            Only returned if giveInfo=True was supplied in the parameters dictionary.
            A dictionary of interesting information about how the solution was calculated.
    """
    problemShape = parameters['problemShape']
    gridLevels = parameters['gridLevels']
    defaults['coarsestLevel'] = gridLevels - 1
    tools.dictUpdateNoClobber(defaults, parameters)
    
    verbose = parameters['verbose']
    dense = parameters['dense']
    
    # Generate a list of restriction matrices; one for each level-transition.
    # Their transposes are prolongation matrices.
    R = restrictionList(problemShape, parameters['coarsestLevel'], dense=dense, verbose=verbose)

    # Using this list, generate a list of coefficient matrices; one for each level:
    A = coeffecientList(A_in, R, dense=dense, verbose=verbose)
    
    # do at least one cycle
    result, infoDict = mgCycle(A, b, 0, R, parameters)
    norm = infoDict['norm']
    cycle = 1
    if verbose: print "Residual norm from cycle %d is %f." % (cycle, norm)

    # set up stopping conditions for v-cycling    
    assert not (parameters['cycles'] <= 0 and 'threshold' not in parameters),\
            "The parameters dictionary must contain either cycles>0 or a threshold."
    def stop(cycle, norm):
        """Returns True if either stopping condition is met."""
        cycleStop = thresholdStop = False
        if 'cycles' in parameters and parameters['cycles'] > 0:
            if cycle >= parameters['cycles']:
                cycleStop = True
        if 'threshold' in parameters:
            if norm < parameters['threshold']:
                thresholdStop = True
        return cycleStop or thresholdStop
    
    stopping = stop(cycle, norm)
    while not stopping:
        if verbose: print 'cycle %i < cycles %i' % (cycle, parameters['cycles'])
        cycle += 1
        (result, infoDict) = mgCycle(A, b, 0, R, parameters, initial=result)
        norm = infoDict['norm']
        if verbose: print "Residual norm from cycle %d is %f." % (cycle, norm)
        stopping = stop(cycle, norm)
            

    infoDict['cycle'] = cycle
    infoDict['norm'] = norm
    if verbose: print 'Returning mgSolve after %i cycle(s) with norm %f' % (cycle, norm)
    if parameters["giveInfo"]:
        return result, infoDict
    else:
        return result


def mgCycle(A, b, level, R, parameters, initial=None):
    """
    Internally used function that shows the actual multi-level solution method,
    through a recursive call within the "level < coarsestLevel" block, below.
    
    Parameters
    ----------
    A : list of ndarrays
        A list of square arrays; one for each level of resolution. As returned
        by coeffecientList().
    b : ndarray
        top-level RHS vector
    level : int
        The current multigrid level. This value is 0 at the entry point for standard,
        recursive multigrid.
    R : list of ndarrays
        A list of (generally nonsquare) arrays; one for each transition between
        levels of resolution. As returned by restrictionList().
    parameters : dict 
        A dictionary of parameters. See documentation for mgSolve() for details.
        
    Optional Parameters
    -------------------
    initial=np.zeros((b.size, )) : ndarray
        Initial iterate. Defaults to the zero vector, but the last supplied
        solution should be used for chained v-cycles.
    
    Returns
    -------
    
    uOut : ndarray
        solution
    
    infoDict
        Dictionary of information about the solution process. Fields:
        norm
            The final norm of the residual
    
    """
    verbose = parameters['verbose']
    if initial is None:
        initial = np.zeros((b.size, ))
    N = b.size
    
    # The general case is a recursive call. It comprises (1) pre-smoothing,
    # (2) finding the coarse residual, (3) solving for the coarse correction
    # to account for that residual via recursive call, (4) adding that
    # correction to the pre-smoothed solution, (5) then possibly post-smoothing. 
    if level < parameters['coarsestLevel']:
        # (1) pre-smoothing
        uApx = smooth(A[level], b, initial,
                                parameters['preIterations'], verbose=verbose)
        
        # (2) coarse residual
        bCoarse = flexibleMmult(R[level], b.reshape((N, 1)))
        NH = len(bCoarse)
        
        if verbose: print level * " " + "calling mgCycle at level %i" % level
        residual = tools.getresidual(b, A[level], uApx, N)
        coarseResidual = flexibleMmult(R[level], residual.reshape((N, 1))).reshape((NH,))
        
        # (3) correction for residual via recursive call
        coarse_correction = mgCycle(A, coarseResidual, level+1, R, parameters)[0]
        correction = (flexibleMmult(R[level].transpose(), coarse_correction.reshape((NH, 1)))).reshape((N, ))
        
        if parameters['postIterations'] > 0:
            # (5) post-smoothing
            uOut = smooth(A[level],
                                    b,
                                    uApx + correction, # (4)
                                    parameters['postIterations'],
                                    verbose=verbose)
        else:
            uOut = uApx + correction # (4)
            
        # Save norm, to monitor for convergence at topmost level.
        norm = sparse.base.np.linalg.norm(tools.getresidual(b, A[level], uOut, N))
    
    # The recursive calls only end when we're at the coarsest level, where
    # we simply apply the chosen coarse solver.
    else:
        norm = 0
        if verbose: print level * " " + "direct solving at level %i" % level
        uOut = coarseSolve(A[level], b.reshape((N, 1)))
        
    return uOut, {'norm': norm}


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
        R = sparse.lil_matrix((n, N))
    
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
        if sparse.issparse(A_in):
            A[0] = A_in.todense()
        else:
            A[0] = A_in
    else:
        A[0] = sparse.csr_matrix(A_in)
    for level in range(1, levels):
        # This is the Galerkin "RAP" coarse-grid operator
        # A_H = R * A_h * R^T
        # or
        # A_H = R * A_h * P :
        A[level] = flexibleMmult(
                                  flexibleMmult(R[level-1], A[level-1]),
                                                             R[level-1].T)
    if verbose: print 'made %i A matrices' % len(A)
    return A
