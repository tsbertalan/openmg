# -*- coding: utf-8 -*-
import pyamg
from numpy import *
from scipy import *
import scipy.sparse as sparse
import numpy as np
from openmg import *

def whoami():
    import sys
    return sys._getframe(1).f_code.co_name

#functions for checking memory usage. From http://stackoverflow.com/questions/938733/python-total-memory-used

def testgrid(parameters):
    #print whoami(),": parent  PID is ",os.getppid()
    #print whoami(),": |-> process PID is ",os.getpid()
    exec ','.join(parameters) + ', = parameters.values()' # unpack the parameters into the local namespace
    #print ','.join(parameters) + ', = parameters.values()' # do this to see what you're unpacking
    #if verbose:
        #print "gridlevels is %i." % gridlevels
    #solver='pyamg' # pyamg, mmg, gs, or direct
    description = '-' + description

    dt = 24.0*7 #hours per time step
    Nsteps = 1 #number of steps
    Nx = problemscale #number of cells, x-dimension
    Ny = problemscale
    Nz = problemscale
    ndims = 3
    N = Nx*Ny*Nz #number of cells, total
    size=N

    if verbose:
        print "N is %i." % N
    #h in feet
    hx=16.0 #length, x-dimension
    hy=16.0
    hz=10.0
    #perm in md, conversion factor for hours
    Perm =ones([3,Nz,Ny,Nx]).astype(np.float64)*100.0*.0002637 #mD is millidarcy=m^2/1000 #permeability
    Kinv = 1.0/Perm
    phi = .20 #unitless #porosity
    ct = 10.0E-06 #1/psi #conductivity?
    #B=1.0
    alpha = hx*hy*hz*phi*ct/dt #ft^3/hours/psi #???
    gamma = .433 #psi/ft #specific weight
    Phi =ones([Nz,Ny,Nx]).astype(np.float64)*2500.0 #Flow Potential
    Z=zeros([Nz,Ny,Nx]).astype(np.float64)
    for k in range(Nz):
        Z[k,:,:]+=k*hz # 3D field of all 0's in the bottom slice (z=0), adding 10 for each slice up (all 10s in slice where z=1, all 20s in slice where z=2, etc.)
    p = Phi-gamma*Z #potential
    Phi = Phi.flatten() #Phi is now just a row vector of 2500.0's, either Nx or Ny long. Really? Not Nx*Ny*Nz?
                        # We're switching from using a 3-tuple as an index to each cell to having a scalar between 0 and (N-1) as an index to each cell.

    TX = zeros([Nz,Ny,Nx+1]).astype(np.float64)#zeros([Nx+1,Ny,Nz])  #a 3D scalar field: x-components of transmissibility vectors
    TY = zeros([Nz,Ny+1,Nx]).astype(np.float64)#zeros([Nx,Ny+1,Nz])
    TZ = zeros([Nz+1,Ny,Nx]).astype(np.float64)#zeros([Nx,Ny,Nz+1])
                                                                    # (for Nx=3,Ny=4,Nz=5):
    TX[:,:,1:-1] = (2*hy*hz/hx)/(Kinv[0,:,:,0:-1]+Kinv[0,:,:,1: ])  # 2x4x5
    TY[:,1:-1,:] = (2*hx*hz/hy)/(Kinv [1,:,0:-1,:]+Kinv[1,:,1:,:])  # 3x3x5 Why are these not all 3x4x5?
    TZ[1:-1,:,:] = (2*hx*hy/hz)/(Kinv[2,0:-1,:,:]+Kinv[2,1:,:,:])   # 3x4x4

    x1 = TX[:,:,0:Nx].flatten()
    x2 = TX[:,:,1:].flatten()
    y1 = TY[:,0:Ny,:].flatten()
    y2 = TY[:,1:,:].flatten()
    z1 = TZ[0:Nz,:,:].flatten()
    z2 = TZ[1:,:,:].flatten()

    q = zeros([Nz,Ny,Nx]).astype(np.float64)
    q[:,4,4]=10.0*5.61/24.0
    Iw = (Nx*Ny-1)/2
    q = q.flatten()

    main_diag=x1+x2+y1+y2+z1+z2#+q2 #(when q2==1, sum is zero and vice versa)
    #DiagVecs = vstack((-z2,-y2,-x2,main_diag,-x1,-y1,-z1))
    DiagVecs  = vstack((-z2,-y2,-x2,          -x1,-y1,-z1))
    #DiagIndx = array([-Nx*Ny,-Nx,-1,0,1,Nx,Nx*Ny])
    DiagIndx  = array([-Nx*Ny,-Nx,-1,1,  Nx,Nx*Ny])
    Phi_w = zeros(Nsteps+1).astype(np.float64)
    Phi_w[0] = Phi[Iw]
    Phi_c = zeros(Nsteps+1).astype(np.float64)
    Phi_c[0] = Phi[-1]

    Phi_cl = zeros(Nsteps+1).astype(np.float64)
    Phi_cl[0] = Phi[0]

    level=0
    #whoami()
    #global N
    #global ndims
    A = sparse.spdiags(DiagVecs,DiagIndx,N/(2**ndims)**level,N/(2**ndims)**level,format='csr')

    DM =  sparse.spdiags(main_diag,array([0]),N/(2**ndims)**level,N/(2**ndims)**level,format='csr')

    D0 =  sparse.spdiags(ones(N/(2**ndims)**level)*alpha,array([0]),N/(2**ndims)**level,N/(2**ndims)**level,format='csr')

    A = sparse.spdiags(DiagVecs,DiagIndx,N/(2**ndims)**level,N/(2**ndims)**level,format='csr')
    DM =  sparse.spdiags(main_diag,array([0]),N/(2**ndims)**level,N/(2**ndims)**level,format='csr')
    D0 =  sparse.spdiags(ones(N/(2**ndims)**level)*alpha,array([0]),N/(2**ndims)**level,N/(2**ndims)**level,format='csr')
    LHS = (A+D0+DM).toarray().astype(np.float64)
    if verbose:
        print 'LHS shape is:'
        print LHS.shape


    # Main Timestep Loop
    info_dict = {'test_grid':True}
    for k in range(Nsteps):
        #print "test_USS_etc.: timestep %i of %i." % (k+1,Nsteps)
        Qa = (-A.astype(np.float64)*Phi.astype(np.float64)).astype(np.float64)
        Qa-= DM*Phi

        if verbose: print 'solving with %s...' % solver
        ##CHOOSE THY SOLVER:
        if solver == 'pyamg':                                           ##PyAMG: Ruge-Steuben
            #make_sparsity_graph('graphs/pyamg_LHS.png','pyamg LHS',A+D0+DM)
            ml = pyamg.ruge_stuben_solver(A+D0+DM,max_levels=gridlevels,presmoother=('gauss_seidel',{'iterations':iterations}))
            u = ml.solve(q+Qa,cycle='F',tol=1e-06, accel='cg')
            info_dict['solverstring']='PyAMG-%icells' % N
            info_dict['cycle']=0
            info_dict['norm']=np.linalg.norm(q+Qa-np.dot(LHS,u))
        if solver == 'pyamg-linagg':                                           ##PyAMG; linear aggregation:
            #make_sparsity_graph('graphs/pyamg_LHS.png','pyamg LHS',A+D0+DM)
            ml=pyamg.smoothed_aggregation_solver(A+D0+DM)#,mat_flag= 'symmetric',strength= 'symmetric')
            u = ml.solve(q+Qa,cycle='F',tol=1e-06, accel='cg')
            info_dict['solverstring']='PyAMG-%icells' % N
            info_dict['cycle']=0
            info_dict['norm']=np.linalg.norm(q+Qa-np.dot(LHS,u))
        elif solver == 'mmg':                                           ##Our Multi-Multi-Grid
            if verbose: print 'calling mg_solve()'
            #parameters['verbose']=False #uncomment to disable verbosity within the OpenMG code.
            if graph_pressure: parameters['cycles']=1
            (u,info_dict) = mg_solve(LHS,q+Qa,parameters) # info_dict here contains both cycle and norm

            if graph_pressure: parameters['cycles']=2
            if graph_pressure: (utwo,info_dict) = mg_solve(LHS,q+Qa,parameters) # info_dict here contains both cycle and norm
            if graph_pressure: parameters['cycles']=3
            if graph_pressure: (uthree,info_dict) = mg_solve(LHS,q+Qa,parameters) # info_dict here contains both cycle and norm

            if verbose: print "got ",",".join(info_dict)," from mg_solve."
            info_dict['solverstring'] = '%igrid-%iiter-%icells%sthreshold-%icycles' % (gridlevels,iterations,N,threshold,cycles)
            if verbose: print 'mg_solve() returned this solverstring: %s' % info_dict['solverstring']
        elif solver == 'gs':                                            ##Our Gauss-Seidel iterative solver
            (u,gs_iterations) =   iterative_solve_to_threshold(LHS, q+Qa, np.zeros((q.size,)), threshold,verbose=verbose)
            info_dict['solverstring']='GaussSeidel-%iiter-%icells' % (gs_iterations,N)
            info_dict['cycle']=0
            info_dict['norm']=np.linalg.norm(q+Qa-np.dot(LHS,u))
        elif solver == 'gs_xiter':                                            ##Our Gauss-Seidel iterative solver, with a specified number of iterations
            #def iterative_solve(A,b,x,iterations,verbose=False):
            u =   iterative_solve(LHS, q+Qa, np.zeros((q.size,)), iterations,verbose)
            info_dict={}
            gs_iterations = iterations
            info_dict['solverstring']='GaussSeidel-%iiter-%icells' % (gs_iterations,N)
            info_dict['cycle']=0
            info_dict['norm']=np.linalg.norm(q+Qa-np.dot(LHS,u))
        else:                                                           ##Numpy's (pretty good) direct solver.
            solver = 'direct'
            u = np.linalg.solve(LHS,q+Qa)
            info_dict['solverstring']= 'direct-%icells' % N
            info_dict['cycle']=0
            info_dict['norm']=np.linalg.norm(q+Qa-np.dot(LHS,u))
        if graph_pressure: oldPhi = Phi.copy()
        Phi += u
        Phi_w[k+1]=Phi[Iw]
        Phi_cl[k+1]=Phi[0]
        Phi_c[k+1]=Phi[-1]
        if verbose: print "adding description", description, " to solverstring"
        if verbose: print 'info_dict now contains', ','.join(info_dict)
        info_dict['solverstring'] += description
        if graph_pressure:
            u_correct = np.linalg.solve(LHS,q+Qa)
            Phi_correct = oldPhi + u_correct
            Phi_graph_correct = reshape(Phi_correct,[Nz,Ny,Nx])

            Phi_two = oldPhi + utwo
            Phi_two_graph=reshape(Phi_two,[Nz,Ny,Nx])
            Phi_three = oldPhi + uthree
            Phi_three_graph=reshape(Phi_three,[Nz,Ny,Nx])

            Phi_graph=reshape(Phi,[Nz,Ny,Nx])
            filename='graphs/USS_output-%s.png' % (info_dict['solverstring'],)
            import visualization as viz
            viz.make_multiple_3d_graphs([Phi_graph_correct.T,Phi_graph.T,Phi_two_graph.T,Phi_three_graph.T],filename)
            del [Phi,Phi_c,Phi_cl,Phi_graph,Phi_w,u,Qa,DM,D0,LHS,A]
        else:
            del [Phi,Phi_c,Phi_cl,Phi_w,u,Qa,DM,D0,LHS,A]
        if verbose: print "completed solverstring is", info_dict['solverstring']
    return info_dict
