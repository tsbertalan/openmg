import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

from solvers import iterative_solve, iterative_solve_to_threshold, coarse_solve
import tools
import geometry
flexible_mmult = tools.flexible_mmult


def dense_restriction(N, shape):
    '''
    It would be nice to make this rely only on N, not shape.
    '''
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
        print "restriction(): Greater than 3 dimensions is not implemented. (shape was", shape, ".)"
        exit()
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


def restriction(N, shape, verbose=False):
    '''
    shape is a tuple in the shape of the problem domain.
    Requiring the shape of the domain is bad for the black-boxibility.
    Implementing a separate Ruge Steuben restriction method would avoid this.
    '''
    # Alpha is the dimensionality of the problem.
    alpha = len(shape)
    n = N / (2 ** alpha)
    if n == 0:
        print 'New restriction matrix would have shape', (n,N), '.'
        print 'coarse set would have 0 points! ' + \
              'Try a larger problem or fewer gridlevels.'
        exit()
#    print 'New restriction matrix will have shape', (n,N), '.'
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
        print "restriction(): Greater than 3 dimensions is not implemented."
        exit()
    rowsandcols = zip(range(n), coarse_columns)  # in cases where the geometry
    # isn't nicely divisible by 2 along one axis,this will work, but it will do
    # some very stupid reaching around to the next row 2 ** alpha
#    if verbose: print ''
#    if verbose: print 'problemshape is',shape
#    if verbose: print 'N is',N
#    if verbose: print 'iterable is',rowsandcols
#    if verbose: print 'Rshape is ',R.shape
    for (r, c) in rowsandcols:
#        print 'r =',r
#        print 'c =',c
#        print 'R.shape =',R.shape
#        print 'coarse_columns =',coarse_columns
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


def restrictions(N, problemshape, coarsest_level, dense=False, verbose=False):
    '''
    Returns a list of restriction matrices (non-square) for each level.
    It should, therefore, be a list with as many elements
    as the output of coarsen_A().
    '''
    alpha = np.array(problemshape).size
    levels = coarsest_level + 1
    R = list(range(levels - 1))  # We don't need R at the coarsest level.
#    print 'R initialization is ',R
#    if verbose: print 'range(levels) is',range(levels)
    for level in range(levels - 1):
#        if verbose: print ''
#        if verbose: print 'level is',level
        newsize = N / (2 ** (alpha * level))
#        if verbose: print 'newsize is ',newsize
        if dense:
            R[level] = dense_restriction(newsize,
                                 tuple(np.array(problemshape) / (2 ** level)))
#            if verbose: print 'R[%i].shape ='%level, R[level].shape
        else:
#            if verbose: print 'R[%i] (initialization) ='%level, R[level]
            R[level] = restriction(newsize,
                                   tuple(np.array(problemshape) / (2 ** level)),
                                   verbose=verbose
                                  )
#            if verbose: print 'R[%i].shape ='%level, R[level].shape
    #make_sparsity_graph('A[%i].png'%level, 'A[%i]'%level, A[level])
    return R


def interpolate(u_coarse):
    '''No longer used. Only works for 1D problems,  anyway.
    Instead,  premultiply u_coarse by the transpose of R[level].
    '''
    N = u_coarse.size * 2
    u_fine = empty(N)
    u_fine[0::2] = u_coarse
    u_fine[1:N - 1:2] = (u_fine[0:N - 2:2] + u_fine[2::2]) / 2
    u_fine[-1] = 2 * u_fine[-2] - u_fine[-3]
    return u_fine


def coarsen(u_fine):
    '''No longer used. Only works for 1D problems,  anyway.
    Instead,  premultiply u_coarse by R[level].
    '''
    u_coarse = np.empty(u_fine.size * 0.5)
    u_coarse = u_fine[::2]
    return u_coarse


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
    Main externally usable function.
    Arguments:
        A_in        the coefficient matrix
        b           the righthand side
        parameters  a dictionary,  including these fields:
          required:
            problemshape      a tuple describing the shape of the domain
            gridlevels        how many levels to generate in the hierarchy.

          optional:
            coarsest_level=gridlevels - 1
                              an int larger than 0 denoting the level at which
                              to use coarse_solve().
            verbose=False     Whether to print progres information.
            threshold=.1      Absolute tolerance for doing v-cycles
                              without an explicit number of cycles to perform
                              (i.e., for cycles=0).
            cycles=0          0 is interpreted as meaning
                              that v-cycles should be continued until the
                              residual norm falls below threshold.
            pre_iterations=1  How many iterations to use in the pre-smoother.
            post_iterations=0 How many iterations to use in the post-smoother.
            give_info=False   Whether to return the info_dict
            
        A sample parameters dictionary is available at openmg.defaults .
    
    
    if give_info, returns a tuple containing:
        result        the solution vector
        info_dict     a dictionary of interesting information about
                      how the solution was calculated
    else, just returns result.
    '''
    problemshape = parameters['problemshape']
    gridlevels = parameters['gridlevels']
    defaults['coarsest_level'] = gridlevels - 1
    tools.dictUpdateNoClobber(defaults, parameters)
    
    verbose = parameters['verbose']
    dense = parameters['dense']
    
    # A list of restriction matrices for each level
    if verbose: print "Generating restriction matrices; dense=%s" % dense
    # their transposes are prolongation matrices:
    R = restrictions(b.size,
                     problemshape,
                     parameters['coarsest_level'],
                     dense=dense,
                     verbose=verbose)

    # a list of coefficient matrices; defined using R:
    if verbose: print "Generating coefficient matrices; dense=%s" % dense,
    A = coarsen_A(A_in,
                  parameters['coarsest_level'],
                  R,
                  dense=dense)
    cycle = 1
    if verbose: print '... made %i A matrices' % len(A)
    
    if verbose: print 'calling amg_cycle No.%i' % cycle
    result, info_dict = amg_cycle(A, b, 0, R, parameters)
    
    norm = info_dict['norm']
    if verbose: print "Residual norm from cycle %d is %f." % (cycle, norm)
    
    if parameters['cycles'] > 0:  # Do v-cycles until we're past the assigned number.
        # Set cycless = [0, ] in the test definition in time_test_grid.py
        # to allow cycles to continue until convergence,
        # as defined with the threshold parameter.
        while cycle < parameters['cycles']:
            if verbose: print 'cycle %i < cycles %i' % (cycle, parameters['cycles'])
            cycle += 1
            (result, info_dict) = amg_cycle(A,
                                            b,
                                            0,
                                            R,
                                            parameters,
                                            initial=result)
            
            norm = info_dict['norm']
            if verbose: print "Residual norm from cycle %d is %f." % (cycle, norm)
            
    else:  # Do v-cycles until the solution has converged
        while norm > parameters['threshold']:
            cycle += 1
            if verbose: print 'calling amg_cycle No.%i' % cycle
            (result, info_dict) = amg_cycle(A,
                                            b,
                                            0,
                                            R,
                                            parameters,
                                            initial=result)
            
            norm = info_dict['norm']
            if verbose: print "Residual norm from cycle %d is %f." % (cycle, norm)

    info_dict['cycle'] = cycle
    info_dict['norm'] = norm
    if verbose: print 'Returning mg_solve after %i cycle(s) with norm %f' % (cycle, norm)
    if parameters["give_info"]:
        return result, info_dict
    else:
        return result


def amg_cycle(A, b, level, R, parameters, initial='None'):
    '''
    Internally used function that shows the actual multi-level solution method,
    through a recursive call within the "level < coarsest_level" block, below.
    Returns a tuple:
        solution
        dictionary of interesting things
    '''
    verbose = parameters['verbose']
    if initial == 'None':
        initial = np.zeros((b.size, ))
    coarsest_level = parameters['gridlevels'] - 1
    N = b.size
    if level < coarsest_level:
        u_apx = iterative_solve(A[level], b, initial,
                                parameters['pre_iterations'], verbose=verbose)
        b_coarse = flexible_mmult(R[level], b.reshape((N, 1)))
        NH = len(b_coarse)
        b_coarse.reshape((NH, ))
        
        if verbose: print "calling amg_cycle recursively at level %i" % level
        residual = tools.getresidual(b, A[level], u_apx, N)
        coarse_residual = flexible_mmult(R[level], residual.reshape((N, 1))).reshape((NH,))
        coarse_correction = amg_cycle(
                                      A,
                                      coarse_residual,
                                      level + 1,
                                      R,
                                      parameters,
                                     )[0]
        correction = (flexible_mmult(R[level].transpose(), coarse_correction.reshape((NH, 1)))).reshape((N, ))
        if parameters['post_iterations'] > 0:
            u_out = iterative_solve(A[level],
                                    b,
                                    u_apx + correction,
                                    parameters['post_iterations'],
                                    verbose=verbose)
        else:
            u_out = u_apx + correction
        norm = sparse.base.np.linalg.norm(tools.getresidual(b, A[level], u_out, N))
    else:
        norm = 0
        if verbose: print "direct solving at level %i" % level
        u_out = coarse_solve(A[level], b.reshape((N, 1)))
    return (u_out, {'norm': norm})

#def coarsen_vector(): # it might be nice to define something like this,
#but it would need to be a new class whose constuctor included R,  and perhaps,  somehow,  level


def coarsen_A(A_in, coarsest_level, R, dense=False):
    '''Returns a list.'''
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
    return A
