import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

from solvers import smooth, smooth_to_threshold, coarse_solve
import tools
import geometry
flexible_mmult = tools.flexible_mmult


def dense_restriction(shape):
    '''
    It would be nice to make this rely only on N, not shape. But, really, the
    restriction operator is problem-dependent. There are certainly problem
    domains where "dimensionality" means something other than spatial dimensions. 
    '''
    N = tools.product(shape)
    alpha = len(shape)
    R = np.zeros((N / (2 ** alpha), N))
    r = 0  # row index in the resulting restriction matrix
    NX = shape[0]
    if alpha >= 2:
        NY = shape[1]
    each = 1.0 / (2 ** alpha)
    # columns in the restriction matrix:
    if alpha == 1:
        coarse_columns = np.array(range(N)).reshape(shape)[::2].ravel()
    elif alpha == 2:
        coarse_columns = np.array(range(N)).reshape(shape)[::2, ::2].ravel()
    elif alpha == 3:
        coarse_columns = np.array(range(N)).reshape(shape)[::2, ::2, ::2].ravel()
    else:
        raise ValueError("restriction(): Greater than 3 dimensions is not implemented. (shape was", shape, ".)")
    for c in coarse_columns:
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
        r += 1
    #return scipy.sparse.csc_matrix(R)
    return R


def restriction(shape, verbose=False, dense=False):
    '''
    shape is a tuple in the shape of the problem domain.
    Requiring the shape of the domain is bad for the black-boxibility.
    Implementing a separate Ruge Steuben restriction method would avoid this.
    Returns a CSR matrix by default, but will return dense if dense=True is given.
    '''
    if dense:
        return dense_restriction(shape) 
    N = tools.product(shape)
    # alpha is the dimensionality of the problem.
    alpha = len(shape)
    n = N / (2 ** alpha)
    if n == 0:
        raise ValueError('New restriction matrix would have shape ' + str((n,N))
                         + '.' + 'Coarse set would have 0 points! ' + 
                        'Try a larger problem or fewer gridlevels.')
    R = sparse.lil_matrix((n, N))
    r = 0  # row index in the resulting restriction matrix
    NX = shape[0]
    if alpha >= 2:
        NY = shape[1]
    #NZ = shape[3] # don't need this
    each = 1.0 / (2 ** alpha)
    
    # columns in the restriction matrix:
    if alpha == 1:
        coarse_columns = np.array(range(N)).reshape(shape)[::2].ravel()
    elif alpha == 2:
        coarse_columns = np.array(range(N)).reshape(shape)[::2, ::2].ravel()
    elif alpha == 3:
        coarse_columns = np.array(range(N)).reshape(shape)[::2, ::2, ::2].ravel()
    else:
        raise ValueError("restriction(): Greater than 3 dimensions is not implemented.")
    rowsandcols = zip(range(n), coarse_columns)  # in cases where the geometry
    # isn't nicely divisible by 2 along one axis, this will work, but it will do
    # some very stupid reaching around to the next row 2 ** alpha
    for (r, c) in rowsandcols:
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
    #return scipy.sparse.csc_matrix(R)
    return R.tocsr()


def restrictions(problemshape, coarsest_level, dense=False, verbose=False):
    '''
    Returns a list of restriction matrices (non-square) for each level.
    It should, therefore, be a list with as many elements
    as the output of coarsen_A().
    
    Parameters
    ----------
    problemshape : tuple of ints
        The problem geometry. Only rectangular geometries of less than 3
        spatial dimensions are dealt with here. The product of the values in
        this tuple should be the total number of unknowns, N.
    coarsest_level : int
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
    '''
    if verbose: print "Generating restriction matrices; dense=%s" % dense
    levels = coarsest_level + 1
    R = list(range(levels - 1))  # We don't need R at the coarsest level.
    for level in range(levels - 1):
        if dense:
            R[level] = dense_restriction(tuple(np.array(problemshape) / (2 ** level)))
        else:
            R[level] = restriction(tuple(np.array(problemshape) / (2 ** level)))
    return R


defaults = {
    'problemshape': (200,),
    'gridlevels': 2,
    'verbose': False,         
    'threshold': 0.1,
    'cycles': 0,
    'saveProgress': False,
    'pre_iterations': 1,
    'post_iterations': 0,
    'dense': False,
    'give_info': False,
}
def mg_solve(A_in, b, parameters):
    '''
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
                problemshape : tuple of ints
                    The shape of the spatial domain.
                gridlevels : int
                    How many levels to generate in the restriction hierarchy.
                
            Optional:
                coarsest_level=gridlevels - 1 : int
                    An int larger than 0 denoting the level at which
                    to use coarse_solve().
                verbose=False : bool
                    Whether to print progress information.
                threshold=.1 : float_like
                    Absolute tolerance for doing v-cycles, if an explicit number
                     of cycles to perform is not given (i.e., for cycles=0).
                cycles=0 : int
                    How many v-cycles to perform. 0 is interpreted as meaning
                    that v-cycles should be continued until the
                    residual norm falls below threshold.
                pre_iterations=1 : int
                    How many iterations to use in the pre-smoother.
                post_iterations=0 : int
                    How many iterations to use in the post-smoother.
                give_info=False : bool
                    Whether to return the info_dict
              
            A sample parameters dictionary is available at openmg.defaults .
    
    Returns
    -------
        result : ndarray
            The solution vector; same size as b.
        info_dict : dictionary
            Only returned if give_info=True was supplied in the parameters dictionary.
            A dictionary of interesting information about how the solution was calculated.
    '''
    problemshape = parameters['problemshape']
    gridlevels = parameters['gridlevels']
    defaults['coarsest_level'] = gridlevels - 1
    tools.dictUpdateNoClobber(defaults, parameters)
    
    verbose = parameters['verbose']
    dense = parameters['dense']
    
    # Generate a list of restriction matrices; one for each level-transition.
    # Their transposes are prolongation matrices.
    R = restrictions(problemshape, parameters['coarsest_level'], dense=dense, verbose=verbose)

    # Using this list, generate a list of coefficient matrices; one for each level:
    A = coarsen_A(A_in, R, dense=dense, verbose=verbose)
    
    # do at least one cycle
    result, info_dict = amg_cycle(A, b, 0, R, parameters)
    norm = info_dict['norm']
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
        (result, info_dict) = amg_cycle(A, b, 0, R, parameters, initial=result)
        norm = info_dict['norm']
        if verbose: print "Residual norm from cycle %d is %f." % (cycle, norm)
        stopping = stop(cycle, norm)
            

    info_dict['cycle'] = cycle
    info_dict['norm'] = norm
    if verbose: print 'Returning mg_solve after %i cycle(s) with norm %f' % (cycle, norm)
    if parameters["give_info"]:
        return result, info_dict
    else:
        return result


def amg_cycle(A, b, level, R, parameters, initial=None):
    '''
    Internally used function that shows the actual multi-level solution method,
    through a recursive call within the "level < coarsest_level" block, below.
    
    Parameters
    ----------
    A : list of ndarrays
        A list of square arrays; one for each level of resolution. As returned
        by coarsen_A().
    b : ndarray
        top-level RHS vector
    level : int
        The current multigrid level. This value is 0 at the entry point for standard,
        recursive multigrid.
    R : list of ndarrays
        A list of (generally nonsquare) arrays; one for each transition between
        levels of resolution. As returned by restrictions().
    parameters : dict 
        A dictionary of parameters. See documentation for mg_solve() for details.
        
    Optional Parameters
    -------------------
    initial=np.zeros((b.size, )) : ndarray
        Initial iterate. Defaults to the zero vector, but the last supplied
        solution should be used for chained v-cycles.
    
    Returns
    -------
    
    u_out : ndarray
        solution
    
    info_dict
        Dictionary of information about the solution process. Fields:
        norm
            The final norm of the residual
    
    '''
    verbose = parameters['verbose']
    if initial is None:
        initial = np.zeros((b.size, ))
    N = b.size
    
    # The general case is a recursive call. It comprises (1) pre-smoothing,
    # (2) finding the coarse residual, (3) solving for the coarse correction
    # to account for that residual via recursive call, (4) adding that
    # correction to the pre-smoothed solution, (5) then possibly post-smoothing. 
    if level < parameters['coarsest_level']:
        # (1) pre-smoothing
        u_apx = smooth(A[level], b, initial,
                                parameters['pre_iterations'], verbose=verbose)
        
        # (2) coarse residual
        b_coarse = flexible_mmult(R[level], b.reshape((N, 1)))
        NH = len(b_coarse)
        
        if verbose: print level * " " + "calling amg_cycle at level %i" % level
        residual = tools.getresidual(b, A[level], u_apx, N)
        coarse_residual = flexible_mmult(R[level], residual.reshape((N, 1))).reshape((NH,))
        
        # (3) correction for residual via recursive call
        coarse_correction = amg_cycle(A, coarse_residual, level+1, R, parameters)[0]
        correction = (flexible_mmult(R[level].transpose(), coarse_correction.reshape((NH, 1)))).reshape((N, ))
        
        if parameters['post_iterations'] > 0:
            # (5) post-smoothing
            u_out = smooth(A[level],
                                    b,
                                    u_apx + correction, # (4)
                                    parameters['post_iterations'],
                                    verbose=verbose)
        else:
            u_out = u_apx + correction # (4)
            
        # Save norm, to monitor for convergence at topmost level.
        norm = sparse.base.np.linalg.norm(tools.getresidual(b, A[level], u_out, N))
    
    # The recursive calls only end when we're at the coarsest level, where
    # we simply apply the chosen coarse solver.
    else:
        norm = 0
        if verbose: print level * " " + "direct solving at level %i" % level
        u_out = coarse_solve(A[level], b.reshape((N, 1)))
        
    return u_out, {'norm': norm}


def coarsen_A(A_in, R, dense=False, verbose=False):
    '''Returns a list of coeffecient matrices.
    
    Parameters
    ----------
    A_in : ndarray
        The (square) coeffecient matrix for the original problem.
    R : list of ndarrays    
        A list of (generally nonsquare) restriction matrices, as produced by
        restrictions(N, problemshape, coarsest_level)
        
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
    '''
    if verbose: print "Generating coefficient matrices; dense=%s ..." % dense,
    coarsest_level = len(R)
    levels = coarsest_level + 1
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
        A[level] = flexible_mmult(
                                  flexible_mmult(R[level-1], A[level-1]),
                                                             R[level-1].T)
    if verbose: print 'made %i A matrices' % len(A)
    return A
