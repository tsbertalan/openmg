import numpy as np
import openmg as omg
from scipy import sparse
from time import time
from sys import argv, stderr

def main(N):
    NX = NY = int(N ** 0.5)
    if float(N) / NX / NY != 1:
        print >> stderr, "N must be a perfect square. (N was given as %i)" % N
        return 1
    dense = False  # Should we use dense matrices for multigrid?
                # This is not useful for small problems like N=100

    #A = np.random.rand(N, N)  # this will make the multigrid algorithm diverge!
                            # (although the pure-iterative params will still work)
    u_true = np.random.rand(N).reshape((N, 1))
    if dense:
        A = omg.poisson2D(N)  # a dense array for the 2D Poisson equation. 5-point Laplacian.
        b = np.dot(A, u_true)
    else:
        A = omg.poisson(N)    # sparse 
        b = A * u_true

    threshold = 1e-10

    # view the docstring
    #help(omg.mg_solve)

    # use only the coarse solver:
    params = {'problemshape': (NX, NY), 'gridlevels': 1, 'verbose': False, 'threshold': threshold}
    a = time()
    soln  = omg.coarse_solve(A, b)
    elapsed = time() - a
    # scipy's sparse solver is a very fast direct solver for this problem:
    print N, "direct", np.linalg.norm(omg.getresidual(b, A, soln, N)), elapsed
    # error norm (direct method): 3.84255102999e-15 in 0.00324702262878 seconds.

    # This Gauss-Seidel solver will be painfully slow for N larger than, say 200.
    # It's probably the main bottleneck in this process, and really should be written in C.
    if N <= 256:
        a = time()
        soln  = omg.iterative_solve_to_threshold(A, b, np.zeros((N, 1)), threshold, verbose=False)[0]
        elapsed = time() - a
        print N, "Gauss-Seidel", np.linalg.norm(omg.getresidual(b, A, soln, N)), elapsed
        # error norm (Gauss-Seidel): 22.4064222452 in 6.83140087128 seconds.

    # use a 2-grid pattern
    g = 3  # number of gridlevels
    params = {'problemshape': (NX, NY), 'gridlevels': g, 'cycles': 10,
            'iterations': 2,       'verbose': False, 'dense': dense, 'threshold': threshold}
    a = time()
    mg_output  = omg.mg_solve(A, b, params)
    soln = mg_output[0]
    cycles = mg_output[1]['cycle']
    elapsed = time() - a
    #print params
    print N, "%i-grid"%g, np.linalg.norm(omg.getresidual(b, A, soln, N)), elapsed, cycles
    # error norm (2-grid): 1.63537088016e-15 in 9.87337112427 seconds.

if __name__=="__main__":
    args = [int(i) for i in argv[1:]]
    if len(args) == 0:
        args = [256]
    print "N    method     norm            seconds cycles"
    for N in args:
        main(N)
        print
