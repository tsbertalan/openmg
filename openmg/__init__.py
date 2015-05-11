"""
A demonstration implementation of standard multigrid.

github.com/tsbertalan/openmg

@author: bertalan@princeton.edu
"""
import numpy as np
import scipy.sparse
import scipy.sparse.linalg as splinalg

from solvers import smooth, smoothToThreshold, coarseSolve
import tools
import operators

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
    'minSize': 8,
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
                minSize=8 : int
                    The smallest allowed vector size after restriction.
                    
                    When the hierarchy of operators is generated, these two
                    parameters, coarsetLevel and minSize, are both used to
                    determine how deep the hierarchy should go. Whichever would
                    result in fewer levels prevails.
                    
                verbose=False : bool
                    Whether to print progress information.
                threshold=.1 : float_like
                    Absolute tolerance for doing v-cycles. Values less than or
                    equal to zero are interpreted as meaning that v-cycling will
                    only stop when parameters['cycles'] v-cycles have been done.
                cycles=0 : int
                    How many v-cycles to perform. Values less than or equal to
                    zero are interpreted as meaning that v-cycles should be
                    continued until the residual norm falls below
                    parameters['threshold'].
                    If both threshold and cycles are > 0, the first
                    condition to be met will halt the v-cycling. If both are <=0,
                    ValueError is raised.
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
            
            Includes the number of V-cycles performed, the final residual norm,
            and the operator hierarchies A and R for future examination, or deletion.
    """
    problemShape = parameters['problemShape']
    gridLevels = parameters['gridLevels']
    defaults['coarsestLevel'] = gridLevels - 1
    tools.dictUpdateNoClobber(defaults, parameters)
    
    verbose = parameters['verbose']
    dense = parameters['dense']
    
    # Generate a list of restriction matrices; one for each level-transition.
    # Their transposes are prolongation matrices.
    R = operators.restrictionList(problemShape, parameters['coarsestLevel'],
                                  parameters['minSize'], dense=dense, verbose=verbose)

    parameters['coarsestLevel'] = len(R)

    # Using this list, generate a list of coefficient matrices; one for each level:
    A = operators.coeffecientList(A_in, R, dense=dense, verbose=verbose)
    
    # do at least one cycle
    result, infoDict = mgCycle(A, b, 0, R, parameters)
    norm = infoDict['norm']
    cycle = 1
    if verbose: print "Residual norm from cycle %d is %f." % (cycle, norm)

    # set up stopping conditions for v-cycling    
    if parameters['threshold'] <= 0 and parameters['cycles'] <= 0:
        raise ValueError("Either parameters['threshold'] or parameters['cycles'] must be > 0.")
    def stop(cycle, norm):
        """Returns True if either stopping condition is met."""
        cycleStop = thresholdStop = False
        if 'cycles' in parameters and parameters['cycles'] > 0:
            if cycle >= parameters['cycles']:
                cycleStop = True
        if 'threshold' in parameters:
            if norm < parameters['threshold'] and parameters['threshold'] > 0:
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
    infoDict['R'] = R
    infoDict['A'] = A
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
        by operators.coeffecientList().
    b : ndarray
        top-level RHS vector
    level : int
        The current multigrid level. This value is 0 at the entry point for standard,
        recursive multigrid.
    R : list of ndarrays
        A list of (generally nonsquare) arrays; one for each transition between
        levels of resolution. As returned by operators.restrictionList().
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
        bCoarse = tools.flexibleMmult(R[level], b.reshape((N, 1)))
        NH = len(bCoarse)
        
        if verbose: print level * " " + "calling mgCycle at level %i" % level
        residual = tools.getresidual(b, A[level], uApx, N)
        coarseResidual = tools.flexibleMmult(R[level], residual.reshape((N, 1))).reshape((NH,))
        
        # (3) correction for residual via recursive call
        coarseCorrection = mgCycle(A, coarseResidual, level+1, R, parameters)[0]
        correction = (tools.flexibleMmult(R[level].transpose(), coarseCorrection.reshape((NH, 1)))).reshape((N, ))
        
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
        norm = scipy.sparse.base.np.linalg.norm(tools.getresidual(b, A[level], uOut, N))
    
    # The recursive calls only end when we're at the coarsest level, where
    # we simply apply the chosen coarse solver.
    else:
        norm = 0
        if verbose: print level * " " + "direct solving at level %i" % level
        uOut = coarseSolve(A[level], b.reshape((N, 1)))
        
    return uOut, {'norm': norm}
