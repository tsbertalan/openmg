#from numpy import *
#from tom_viz import *
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from sys import exit

from time import strftime, localtime
isolocaltime = localtime()
#import scipy.sparse.linalg as sparselinalg # there are some alternative
#    iterative solvers in here we could use
sparsenorm = sparse.base.np.linalg.norm


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
    u_coarse = empty(u_fine.size * 0.5)
    u_coarse = u_fine[::2]
    return u_coarse


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
#    print x.shape
#    help(x)
#    x = x.reshape((N, ))
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
                x[i] = x[i] + (b[i] - np.dot(A[i, :], x.reshape((N, 1)))                                      ) / A[i, i]
    return x


def flexible_mmult(x, y):
    '''Dot two 2D arrays
    '''
    if (not sparse.issparse(x)) and (not sparse.issparse(y)):
        return np.dot(x, y)
    else:
        # we can do overloaded-asterisk multiplication
        # as long as at least one matrix is sparse
        return x * y


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


def mg_solve(A_in, b, parameters):
    '''
    Main externally usable function.
    Arguments:
        A_in        the coefficient matrix
        b           the righthand side
        a parameters dictionary,  including these fields:
            problemshape    a tuple describing the shape of the domain
            gridlevels      how many levels to generate in the hierarchy.

        Some optional extra parameters to include in the dictionary include:
            iterations      how many iterations to use in the pre-smoother.
                            Usually only 1 is fine.
            coarsest_level  Defaults to gridlevels - 1
                            an int larger than 0 denoting the level at which
                            to use coarse_solve().
            verbose         Defaults to False. Print a bunch of stuff.
            threshold       Defaults to .1 . Tolerance for doing v-cycles
                            without an explicit number of cycles to perform.
            cycles          Defaults to 0,  which is interpreted as meaning
                            that v-cycles should be continued until the
                            residual norm falls below threshold.
    Returns a tuple containing:
        the solution vector
        a dictionary of interesting information about
        how the solution was calculated
    '''
    # unpack the parameters into the local namespace:
    exec ', '.join(parameters) + ',  = parameters.values()'
    if 'coarsest_level' not in parameters: coarsest_level = gridlevels - 1
    if 'verbose'        not in parameters: verbose = False
    if 'threshold'      not in parameters: threshold = .1
    if 'cycles'         not in parameters: cycles = 0
    if 'dense'          not in parameters:
        dense = False
        parameters['dense'] = False
    if 'iterations'     not in parameters:
        if 'pre_iterations' not in parameters:
            pre_iterations = 1
            parameters['pre_iterations'] = 1
            iterations = 1
            parameters['iterations'] = 1
        else:
            iterations = pre_iterations
            parameters['iterations'] = pre_iterations
    else:
        if 'pre_iterations' not in parameters:
            pre_iterations = iterations
            parameters['pre_iterations'] = iterations
        else:
            pass
    if 'post_iterations' not in parameters:
        parameters['post_iterations'] = 0

    if verbose: print "Doing a(n) %i-grid scheme." % gridlevels

    if verbose:
        if dense:
            sparsity = 'dense'
        else:
            sparsity = 'sparse'
        print "Generating restriction matrices; using %s mechanism." % sparsity

    # A list of restriction matrices for each level
    # their transposes are prolongation matrices:
    R = restrictions(b.size,
                     problemshape,
                     coarsest_level,
                     dense=dense,
                     verbose=verbose)
    if verbose: print "Generating coefficient matrices using Galerkin expression, using %s mechanism..." % sparsity
    # a list of coefficient matrices; defined using R:
    A = coarsen_A(A_in,
                  coarsest_level,
                  R,
                  dense=dense)
    if verbose: print '... got A with length %i.' % len(A)
    cycle = 1
    if verbose: print 'calling amg_cycle No.%i' % cycle
    (result, info_dict) = amg_cycle(A, b, 0, R, parameters)
    if verbose: print 'info_dict from amg_cycle has these fields: ' + (',  '.join(info_dict))
    exec ', '.join(info_dict) + ',  = info_dict.values()'
    if verbose: print "Residual norm from cycle %d is %f." % (cycle, norm)
    isodate = strftime('%Y%m%d', isolocaltime)
    isotime = strftime('T%H%M%S', isolocaltime)
    cycles_progress_file_name = 'output/cyclesvsresid-%s%s.csv' % (isodate,
                                                                   isotime)
    cycles_progress_file = open(cycles_progress_file_name, 'a')
#    cycles_progress_file.write(', '.join(parameters))
#    cycles_progress_file.write(', '.join(parameters.values()))
    cycles_progress_file.write('cycle, residual_norm\n')
    cycles_progress_file.write('%d, %.012f\n' % (cycle, norm))
    cycles_progress_file.flush()
    if cycles > 0:  # Do v-cycles until we're past the assigned number.
        # Set cycless = [0, ] in the test definition in time_test_grid.py
        # to allow cycles to continue until convergence,
        # as defined with the threshold parameter.
        while cycle < cycles:
            if verbose: print 'cycle %i < cycles %i' % (cycle, cycles)
            cycle += 1
            (result, info_dict) = amg_cycle(A,
                                            b,
                                            0,
                                            R,
                                            parameters,
                                            initial=result)
            exec ', '.join(info_dict) + ',  = info_dict.values()'
            cycles_progress_file.write('%d, %.012f\n' % (cycle, norm))
            cycles_progress_file.flush()
            if verbose: print "Residual norm from cycle %d is %f." % (cycle, norm)
    else:  # Do v-cycles until the solution has converged
        while norm > threshold:
            cycle += 1
            if verbose: print 'calling amg_cycle No.%i' % cycle
            (result, info_dict) = amg_cycle(A,
                                            b,
                                            0,
                                            R,
                                            parameters,
                                            initial=result)
            if verbose: print 'result shape is', result.shape
            exec ', '.join(info_dict) + ',  = info_dict.values()'
            cycles_progress_file.write('%d, %.012f\n' % (cycle, norm))
            cycles_progress_file.flush()
            if verbose: print "Residual norm from cycle %d is %f." % (cycle, norm)
    info_dict['cycle'] = cycle
    info_dict['norm'] = norm
    if verbose: print 'Returning mg_solve after %i cycle(s) with norm, ' % cycle, norm
    cycles_progress_file.close()
    return (result, info_dict)


def getresidual(b, A, x, N):
    '''b and x are dense ndarrays; N is an int,  and A is a sparse matrix
    '''
    return b.reshape((N, 1)) - flexible_mmult(A, x.reshape((N, 1)))


def amg_cycle(A, b, level, R, parameters, initial='None'):
    '''
    Internally used function that shows the actual multi-level solution method,
    through a recursive call within the "level < coarsest_level" block, below.
    Returns a tuple:
        solution
        dictionary of interesting things
    '''
    if parameters['verbose']: print "unpacking %i parameters" % len(parameters)
    # unpack the parameters into the local namespace:
    exec ', '.join(parameters) + ',  = parameters.values()'
    if initial == 'None':
        initial = np.zeros((b.size, ))
    coarsest_level = gridlevels - 1
    N = b.size
    if level < coarsest_level:
        if verbose: print "level,  %i,  is less than coarsest_level,  %i." % (level, coarsest_level)
        u_apx = iterative_solve(A[level], b, initial, pre_iterations, verbose=verbose)
        if verbose: print "coarsening RHS"
        b_coarse = flexible_mmult(R[level], b.reshape((N, 1)))
        NH = len(b_coarse)
        b_coarse.reshape((NH, ))
        if verbose: print "finding residual"
        residual = getresidual(b, A[level], u_apx, N)
        if verbose: print "coarsening residual"
        coarse_residual = flexible_mmult(R[level], residual.reshape((N, 1))).reshape((NH,))
        if verbose: print "calling amg_cycle recursively at level %i" % level
        coarse_correction = amg_cycle(
                                      A,
                                      coarse_residual,
                                      level + 1,
                                      R,
                                      parameters,
                                     )[0]
        correction = (flexible_mmult(R[level].transpose(), coarse_correction.reshape((NH, 1)))).reshape((N, ))
        if post_iterations > 0:
            u_out = iterative_solve(A[level],
                                    b,
                                    u_apx + correction,
                                    post_iterations,
                                    verbose=verbose)
            if verbose: print 'shape of u_out after post-smoothing is ', u_out.shape
        else:
            if verbose: print "adding u_apx with shape", u_apx.shape, "to correction with shape", correction.shape
            u_out = u_apx + correction
            if verbose: print 'shape of u_out without post-smoothing is ', u_out.shape
        norm = sparsenorm(getresidual(b, A[level], u_out, N))
    else:
        norm = 0
        if verbose: print "direct solving at level %i" % level
        u_out = coarse_solve(A[level], b.reshape((N, 1)))
        if verbose: print 'shape of u_out after coarse_solve is ', u_out.shape
    if verbose: print "returning u_out with shape ", u_out.shape, " from mg_cycle at level ", level
    return (u_out, {'norm': norm})

#def coarsen_vector(): # it might be nice to define something like this,
#but it would need to be a new class whose constuctor included R,  and perhaps,  somehow,  level


def coarsen_A(A_in, coarsest_level, R, dense=False):
    '''Returns a list.'''
    levels = coarsest_level + 1
    A = list(range(levels))
#    print "creating A[0]"
    if dense:
        if sparse.issparse(A_in):
            A[0] = A_in.todense()
        else:
            A[0] = A_in
    else:
        A[0] = sparse.csr_matrix(A_in)
    for level in range(1, levels):
#        print "creating A[%i]" % level
        # This is the Galerkin "RAP" coarse-grid operator
        # A_H = R * A_h * R^T
        # or
        # A_H = R * A_h * P :
        A[level] = flexible_mmult(
                                  flexible_mmult(R[level-1], A[level-1]),
                                                             R[level-1].T)
    return A


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
    x = -1
    y =  2
    z = -1
    x = x * np.ones(n - 1)
    z = z * np.ones(n - 1)
    y = y * np.ones(n)
    return np.diag(x, -1) + np.diag(y) + np.diag(z, 1)
    

def poisson2d((NX, NY)):
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


def poisson3d((NX, NY, NZ)):
    '''Returns a dense square coefficient matrix, for the 3D Poisson equation.
    '''
    N = NX * NY
    main = np.eye(N) * -6
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
    threeup = np.hstack((
                np.zeros((NX * NY, 1 + NX)),
                np.vstack((
                    np.eye(NX * NY - 1 - NX),
                    np.zeros((1 + NX, NX * NY - 1 - NX))
                ))
            ))
    return main + oneup + twoup + oneup.T + twoup.T + threeup + threeup.T


def setup_variables(size):
    '''For some testing purpose. Probably unneeded now.
    '''
    u_zeros = np.zeros((size, 1))
    domain = np.ones((size))
    for i in range(domain.size):  # [0 1 2 ... domain.size-1]
        domain[i] = i * periods * 2 * 3.1415 / domain.size
    u_actual = (np.cos(domain))
    b = flexible_mmult(A_function(0), u_actual)
    return u_zeros, A_function, u_actual, b
