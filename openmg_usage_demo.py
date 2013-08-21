import numpy as np
import openmg as omg
from scipy import sparse
from time import time
from sys import argv

def main(N):
    ## view the docstring
    #help(omg.mg_solve)

    
    ## Set up some parameters that will be true for each example.    
    dense = False  ## Should we use dense matrices for multigrid?
                   ## This is not useful for small problems like N=100
    threshold = 1e-10


    ## Set up the problem.
    #A = np.random.rand(N, N)  ## this will make the multigrid algorithm diverge!
                               ## (although the pure-iterative params will still work)
    u_true = np.array([np.sin(x / 10.0) for x in range(N)])
    if dense:
        A = omg.poisson1D(N)  ## a dense array for the 1D Poisson equation
        b = np.dot(A, u_true)
    else:
        A = omg.poisson(N)    ## sparse 
        b = A * u_true



    ## Use only the coarse solver.
    params = {'verbose': False, 'threshold': threshold}
    start = time()
    soln  = omg.coarse_solve(A, b)
    elapsed = time() - start
    print N, "direct", np.linalg.norm(omg.getresidual(b, A, soln, N)), elapsed


    ## Use an iterative solver.
    ##   This Gauss-Seidel solver will be painfully slow for N larger than, say 200.
    ##   It's probably the main bottleneck in this process, and really should be written in C.
    if N <= 200:
        start = time()
        soln = omg.iterative_solve_to_threshold(A, b, np.zeros((N, 1)),
                                                params['threshold'],
                                                verbose=params['verbose'])[0]
        elapsed = time() - start
        print N, "Gauss-Seidel", np.linalg.norm(omg.getresidual(b, A, soln, N)), elapsed


    ## Use a 2-grid pattern.
    params = {'problemshape': (N,), 'gridlevels': 3, 'cycles': 10,
              'iterations': 2,       'verbose': False, 'dense': dense, 'threshold': threshold}
    start = time()
    mg_output  = omg.mg_solve(A, b, params)
    soln = mg_output[0]
    cycles = mg_output[1]['cycle']
    elapsed = time() - start
    #print params
    print N, "%i-grid"%params['gridlevels'],\
          np.linalg.norm(omg.getresidual(b, A, soln, N)), elapsed, cycles



if __name__=="__main__":
    args = [int(i) for i in argv[1:]]
    if len(args) == 0:
        args = [200]
    print "N    method     norm            seconds cycles"
    for N in args:
        main(N)
        print
