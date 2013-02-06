import numpy as np
import openmg as omg

N = 100

A = np.random.rand(N, N)  # this will make the multigrid algorithm diverge!
                          # (although the pure-iterative params will still work)

A = omg.poisson1D(N)  # a dense array for the 1D Poisson equation
u_true = np.random.rand(N)
b = np.dot(A, u_true)

# view the docstring
#help(omg.mg_solve)

# use only the iterative solver:
params = {'problemshape': (N,), 'gridlevels': 1,
          'iterations': 2000,    'verbose': False}
soln  = omg.mg_solve(A, b, params)[0]
#print params
print "error norm (Gauss-Seidel):", np.linalg.norm(soln - u_true)

# use a 2-grid pattern
params = {'problemshape': (N,), 'gridlevels': 2, 'cycles': 25,
          'iterations': 1,       'verbose': False}
soln  = omg.mg_solve(A, b, params)[0]
#print params
print "error norm (2-grid):", np.linalg.norm(soln - u_true)
