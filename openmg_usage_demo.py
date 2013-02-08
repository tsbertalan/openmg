import numpy as np
import openmg as omg
from scipy import sparse
from time import time

N = 200
dense = False  # Should we use dense matrices for multigrid?
               # This is not useful for small problems like N=100

#A = np.random.rand(N, N)  # this will make the multigrid algorithm diverge!
                          # (although the pure-iterative params will still work)
if dense:
    A = omg.poisson1D(N)  # a dense array for the 1D Poisson equation
else:
    A = omg.poisson(N)    # sparse 

u_true = np.array([np.sin(x / 10.0) for x in range(N)])
if dense:
    b = np.dot(A, u_true)
else:
    b = A * u_true


# view the docstring
#help(omg.mg_solve)

# use only the coarse solver:
params = {'problemshape': (N,), 'gridlevels': 1, 'verbose': False}
a = time()
soln  = omg.mg_solve(A, b, params)[0]
elapsed = time() - a
# scipy's sparse solver is a very fast direct solver for this problem:
print "error norm (direct method):", np.linalg.norm(soln - u_true), "in", elapsed, "seconds."
# error norm (direct method): 3.84255102999e-15 in 0.00324702262878 seconds.

# This Gauss-Seidel solver will be painfully slow for N larger than, say 200.
# It's probably the main bottleneck in this process, and really should be written in C.
a = time()
soln  = omg.iterative_solve(A, b, np.zeros((N, 1)), 200, verbose=False)[0]
elapsed = time() - a
print "error norm (Gauss-Seidel):", np.linalg.norm(soln - u_true), "in", elapsed, "seconds."
# error norm (Gauss-Seidel): 22.4064222452 in 6.83140087128 seconds.

# use a 2-grid pattern
g=3
params = {'problemshape': (N,), 'gridlevels': g, 'cycles': 10,
          'iterations': 2,       'verbose': False, 'dense': dense}
a = time()
soln  = omg.mg_solve(A, b, params)[0]
elapsed = time() - a
#print params
print "error norm (%i-grid):"%g, np.linalg.norm(soln - u_true), "in", elapsed, "seconds."
# error norm (2-grid): 1.63537088016e-15 in 9.87337112427 seconds.

