"""
Sample code for using OpenMG. For a minimal example, see
simpleDemo(). For more explanation of what each step of the explainedDemo does, see
explainedDemo().

@author: bertalan@princeton.edu
"""

import numpy as np
import openmg
from scipy import sparse
from time import time
from sys import argv


def main():
    args = [int(i) for i in argv[1:]]
    if len(args) == 0:
        args = [200]
    print "N    method     norm            seconds cycles"
    for N in args:
        explainedDemo(N)
        print
    explainedDemo(64, verbose=False, dense=True)


def simpleDemo(verbose=False):
    """Solve A * u = b, where A is the 1D Poisson matrix,
    u=sin(x), and x is in [0, 20)"""
    N = 100
    u_true = np.array([np.sin(x / 10.0) for x in np.linspace(0, 20, N)])
    A = openmg.operators.poisson(N, sparse=True)
    b = openmg.tools.flexibleMmult(A, u_true)
    params = {'problemShape': (N,), 'gridLevels': 3, 'cycles': 10,
              'iterations': 2,       'verbose': verbose, 'dense': True,
              'threshold': 1e-2, 'giveInfo': True}
    u_mg, infoDict = openmg.mgSolve(A, b, params)
    if verbose:
        print "info:"
        print infoDict
    
    ## if verbose==True, output will look something like this:
    # Generating restriction matrices; dense=True
    # Generating coefficient matrices; dense=True ... made 3 A matrices
    # calling amg_cycle at level 0
    #  calling amg_cycle at level 1
    #   direct solving at level 2
    # Residual norm from cycle 1 is 0.805398.
    # cycle 1 < cycles 10
    # calling amg_cycle at level 0
    #  calling amg_cycle at level 1
    #   direct solving at level 2
    # Residual norm from cycle 2 is 0.107866.
    # cycle 2 < cycles 10
    # calling amg_cycle at level 0
    #  calling amg_cycle at level 1
    #   direct solving at level 2
    # Residual norm from cycle 3 is 0.018650.
    # cycle 3 < cycles 10
    # calling amg_cycle at level 0
    #  calling amg_cycle at level 1
    #   direct solving at level 2
    # Residual norm from cycle 4 is 0.003405.
    # Returning mgSolve after 4 cycle(s) with norm 0.003405
    # info:
    # {'norm': 0.0034051536498270769, 'cycle': 4}    
    return u_mg
    

def explainedDemo(N, verbose=True, dense=False):
    ## view the docstring
    #help(openmg.mgSolve)
    
    ## set up the problem
    threshold = 1e-14
    u_true = np.array([np.sin(x / 10.0) for x in range(N)])
    A = openmg.operators.poisson(N, sparse=True) 
    b = openmg.tools.flexibleMmult(A, u_true)


    ## Use only the coarse solver.
    params = {'verbose': False, 'threshold': threshold}
    start = time()
    soln  = openmg.solvers.coarseSolve(A, b)
    elapsed = time() - start
    if verbose: print N, "direct", np.linalg.norm(openmg.tools.getresidual(b, A, soln, N)), elapsed


    ## Use an iterative solver.
    ##   This Gauss-Seidel solver will be painfully slow for N larger than, say 200.
    ##   It's probably the main bottleneck in this process.
    if N <= 200:
        start = time()
        soln = openmg.smoothToThreshold(A, b, np.zeros((N, 1)),
                                                params['threshold'],
                                                verbose=params['verbose'])
        elapsed = time() - start
        if verbose: print N, "Gauss-Seidel", np.linalg.norm(openmg.tools.getresidual(b, A, soln, N)), elapsed


    ## Use a 3-grid pattern.
    params = {'problemShape': (N,), 'gridLevels': 3, 'cycles': 0,
              'iterations': 2,       'verbose': False, 'dense': dense,
              'threshold': threshold, 'giveInfo': True, 'minSize': 30}
    
    # gridLevels is set to 3, but minSize is set to 30.
    # the resulting restriction matrices would have shapes:
    #     (100, 200), (50, 100), and (25, 50)
    # since 25 < 30, the minSize parameter prevails, and only the first two
    # operators are generated.
    
    start = time()
    soln, info_dict = openmg.mgSolve(A, b, params)
    cycles = info_dict['cycle']
    elapsed = time() - start
    #print params
    if verbose: print N, "%i-grid"%params['gridLevels'],\
          np.linalg.norm(openmg.tools.getresidual(b, A, soln, N)), elapsed, cycles



if __name__=="__main__":
    simpleDemo()
    print
    main()
