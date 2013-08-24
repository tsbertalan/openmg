from datetime import datetime
from time import strftime, localtime, sleep
import sys
import os
from multiprocessing import Pool
from random import shuffle
import logging
#import resource # unix-only linux-only?

# from scipy import *
import scipy.sparse as sparse
import numpy as np
from openmg import mgSolve, smoothToThreshold

def whoami():
    '''
    >> print whoami(),": parent  PID is ",os.getppid()
    >> print whoami(),": |-> process PID is ",os.getpid()
    '''
    import sys
    return sys._getframe(1).f_code.co_name

def testgrid(parameters):
    exec ','.join(parameters) + ', = parameters.values()' # unpack the parameters into the local namespace
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
    Perm = np.ones([3,Nz,Ny,Nx]).astype(np.float64)*100.0*.0002637 #mD is millidarcy=m^2/1000 #permeability
    Kinv = 1.0/Perm
    phi = .20 #unitless #porosity
    ct = 10.0E-06 #1/psi #conductivity?
    #B=1.0
    alpha = hx*hy*hz*phi*ct/dt #ft^3/hours/psi #???
    gamma = .433 #psi/ft #specific weight
    Phi = np.ones([Nz,Ny,Nx]).astype(np.float64)*2500.0 #Flow Potential
    Z=np.zeros([Nz,Ny,Nx]).astype(np.float64)
    for k in range(Nz):
        Z[k,:,:]+=k*hz # 3D field of all 0's in the bottom slice (z=0), adding 10 for each slice up (all 10s in slice where z=1, all 20s in slice where z=2, etc.)
    p = Phi-gamma*Z #potential
    Phi = Phi.flatten() #Phi is now just a row vector of 2500.0's, either Nx or Ny long. Really? Not Nx*Ny*Nz?
                        # We're switching from using a 3-tuple as an index to each cell to having a scalar between 0 and (N-1) as an index to each cell.

    TX = np.zeros([Nz,Ny,Nx+1]).astype(np.float64)#np.zeros([Nx+1,Ny,Nz])  #a 3D scalar field: x-components of transmissibility vectors
    TY = np.zeros([Nz,Ny+1,Nx]).astype(np.float64)#np.zeros([Nx,Ny+1,Nz])
    TZ = np.zeros([Nz+1,Ny,Nx]).astype(np.float64)#np.zeros([Nx,Ny,Nz+1])
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

    q = np.zeros([Nz,Ny,Nx]).astype(np.float64)
    q[:,4,4]=10.0*5.61/24.0
    Iw = (Nx*Ny-1)/2
    q = q.flatten()

    main_diag=x1+x2+y1+y2+z1+z2#+q2 #(when q2==1, sum is zero and vice versa)
    #DiagVecs = np.vstack((-z2,-y2,-x2,main_diag,-x1,-y1,-z1))
    DiagVecs  = np.vstack((-z2,-y2,-x2,          -x1,-y1,-z1))
    #DiagIndx = np.array([-Nx*Ny,-Nx,-1,0,1,Nx,Nx*Ny])
    DiagIndx  = np.array([-Nx*Ny,-Nx,-1,1,  Nx,Nx*Ny])
    Phi_w = np.zeros(Nsteps+1).astype(np.float64)
    Phi_w[0] = Phi[Iw]
    Phi_c = np.zeros(Nsteps+1).astype(np.float64)
    Phi_c[0] = Phi[-1]

    Phi_cl = np.zeros(Nsteps+1).astype(np.float64)
    Phi_cl[0] = Phi[0]

    level=0
    #whoami()
    #global N
    #global ndims
    A = sparse.spdiags(DiagVecs,DiagIndx,N/(2**ndims)**level,N/(2**ndims)**level,format='csr')

    DM =  sparse.spdiags(main_diag,np.array([0]),N/(2**ndims)**level,N/(2**ndims)**level,format='csr')

    D0 =  sparse.spdiags(np.ones(N/(2**ndims)**level)*alpha,np.array([0]),N/(2**ndims)**level,N/(2**ndims)**level,format='csr')

    A = sparse.spdiags(DiagVecs,DiagIndx,N/(2**ndims)**level,N/(2**ndims)**level,format='csr')
    DM =  sparse.spdiags(main_diag,np.array([0]),N/(2**ndims)**level,N/(2**ndims)**level,format='csr')
    D0 =  sparse.spdiags(np.ones(N/(2**ndims)**level)*alpha,np.array([0]),N/(2**ndims)**level,N/(2**ndims)**level,format='csr')
    LHS = (A+D0+DM).toarray().astype(np.float64)
    if verbose:
        print 'LHS shape is:'
        print LHS.shape


    # Main Timestep Loop
    info_dict = {'test_grid':True}
    for k in range(Nsteps):
        Qa = (-A.astype(np.float64)*Phi.astype(np.float64)).astype(np.float64)
        Qa-= DM*Phi

        if verbose: print 'solving with %s...' % solver
        ##CHOOSE THY SOLVER:
        if solver == 'pyamg':                                           ##PyAMG: Ruge-Steuben
            import pyamg
            #make_sparsity_graph('graphs/pyamg_LHS.png','pyamg LHS',A+D0+DM)
            ml = pyamg.ruge_stuben_solver(A+D0+DM,max_levels=gridlevels,presmoother=('gauss_seidel',{'iterations':iterations}))
            u = ml.solve(q+Qa,cycle='F',tol=1e-06, accel='cg')
            info_dict['solverstring']='PyAMG-%icells' % N
            info_dict['cycle']=0
            info_dict['norm']=np.linalg.norm(q+Qa-np.dot(LHS,u))
        if solver == 'pyamg-linagg':                                           ##PyAMG; linear aggregation:
            #make_sparsity_graph('graphs/pyamg_LHS.png','pyamg LHS',A+D0+DM)
            ml=pyamg.smoothed_aggregation_solver(A+D0+DM)#,mat_flag= 'symmetric',strength= 'symmetric')
            u = ml.solve(q+Qa,cycle='F',tol=threshold, accel='cg')
            info_dict['solverstring']='PyAMG-%icells' % N
            info_dict['cycle']=0
            info_dict['norm']=np.linalg.norm(q+Qa-np.dot(LHS,u))
        elif solver == 'mmg':                                           ##Our Multi-Grid
            #parameters['verbose']=False #uncomment to disable verbosity within the OpenMG code.
            if graph_pressure: parameters['cycles']=1
            (u,info_dict) = mgSolve(LHS,q+Qa,parameters) # info_dict here contains both cycle and norm

            if graph_pressure: parameters['cycles']=2
            if graph_pressure: (utwo,info_dict) = mgSolve(LHS,q+Qa,parameters) # info_dict here contains both cycle and norm
            if graph_pressure: parameters['cycles']=3
            if graph_pressure: (uthree,info_dict) = mgSolve(LHS,q+Qa,parameters) # info_dict here contains both cycle and norm

            info_dict['solverstring'] = '%igrid-%iiter-%icells%sthreshold-%icycles' % (gridlevels,iterations,N,threshold,cycles)
        elif solver == 'gs':                                            ##Our Gauss-Seidel iterative solver
            (u,gs_iterations) =   smoothToThreshold(LHS, q+Qa, np.zeros((q.size,)), threshold,verbose=verbose)
            info_dict['solverstring']='GaussSeidel-%iiter-%icells' % (gs_iterations,N)
            info_dict['cycle']=0
            info_dict['norm']=np.linalg.norm(q+Qa-np.dot(LHS,u))
        elif solver == 'gs_xiter':                                            ##Our Gauss-Seidel iterative solver, with a specified number of iterations
            #def smooth(A,b,x,iterations,verbose=False):
            u =   smooth(LHS, q+Qa, np.zeros((q.size,)), iterations,verbose)
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




###############################################################################
# Choose a Test
###############################################################################
long_description = ''
doTests = {   'v_cycle_convergence':  False,\
            'domain_size':          False,\
            'sparsity':             False,\
            'presmoother':          False,\
            'gs_thresh':            False,\
            'little_gs':            False,\
            'big_gs':               False,\
            'postsmoother':         False,\
            'ngrid':                False,\
            'graph_pressure':       False,\
            'compare_solvers':      False,\
            'full':                 False,\
            'selftest':             True,\
            'little_mmg':           False,\
            'something_else':       False,\
}


test_descriptions = ''
tests_count=0
for test_description in doTests:
    if doTests[test_description]:
        tests_count +=1
        test_descriptions = test_descriptions + test_description + '-'
if tests_count > 1:
    raise ValueError(
          "Setting more than one test to True can result in confusing "+\
          "combination-doTests. Either set only one test to True, "+\
          "or write a new combination test, with the desired parameters."
          "\nNote that such combination doTests can produce interesting results, "+\
          "although they require multiple linear regression to analyze."
          )
if tests_count == 0:
    print "No explicit test defined. Proceeding with default values."


##multiprocessing parameters
processes = 18
chunksize = 1

##general parameters
start_parameters = datetime.now()
isolocaltime = localtime()

##Defaults
verbose = False
threshold = .00021 # When should the v-cycling stop?
repeats = 4
gridlevelss = [2,] # there are two s's on this and some other vectors just below this because it had to be a plural of a plural.
problemscales = [6,]
iterationss = [1,]
pre_iterationss = [1,]
post_iterationss = [0,]
solvers = ['mmg',]
denses = [False,]
cycless =[0,] # if cycles=0 (i.e., cycless = [0,] ), OpenMG will do v-cycles until converged, as defined by threshold
graph_pressures = [False,]
give_infos = [True,]

###############################################################################
# Definitions of Tests
###############################################################################
##Overwrite default values with lists of parameters to test.
##Create a new 'elif' block here for each new test created in the 'doTests' dict above.
# TODO We need another test here about sigma factors in front of the prolongation (or was it restriction?) matrices.
if doTests['selftest']:
    repeats = 1
elif doTests['v_cycle_convergence']:
    denses = [True,]
    cycless = range(1,200)
elif doTests['domain_size']:
    problemscales = [24,32,36,40,44,48,52,56,60,64]
    repeats = 1
elif doTests['sparsity']:
    verbose = True
    denses = [True,False,]
    problemscales = [24]
    repeats = 4
    threshold = 7
elif doTests['presmoother']:
    pre_iterationss = range(1,200)
    repeats = 16
    denses = [True,]
elif doTests['postsmoother']:
    #post_iterationss = [0,]
    #post_iterationss.extend(range(20,501)[::20])
    post_iterationss = range(0,200)
elif doTests['compare_solvers']:
#    verbose = True
    problemscales = [20,]
    threshold = .73 # .695 for ... 12 unknowns?
    denses = [True,]
    #solvers = ['pyamg', 'pyamg-linagg', 'mmg', 'gs', 'direct']
    solvers = [ 'pyamg-linagg', 'mmg', 'gs', 'direct']
#    solvers = ['mmg',]
    repeats = 5
elif doTests['big_gs']:
    solvers = ['gs_xiter',]
    problemscales = [12,]
    denses = [True,]
    repeats = 2
    iterationss = range(1,100)[::1]
    iterationss.extend(range(100,1000)[::10])
    iterationss.extend(range(1000,200000)[::100])
elif doTests['little_gs']:
    solvers = ['gs_xiter',]
    problemscales = [12,]
    denses = [True,]
    repeats = 2
    iterationss = range(1,100)[::10]
elif doTests['gs_thresh']:
    solvers = ['gs',]
    verbose = True
    problemscales = [12,]
#    denses = [True,]
    repeats = 1
elif doTests['full']: # probably not a good idea
    gridlevelss = range(2,5)
    problemscales = [8,16] #These must not only be multiples of 4 but also powers of 2. And also >= 4*gridlevels, to provide enough coarse points.
    iterationss = range(1,36)[::4]
    solvers = ['pyamg', 'mmg', 'gs', 'direct']
    cycless = range(1,42)[::2]
elif doTests['ngrid']:
    #verbose = True
    threshold = 2
    denses = [True,]
    problemscales = [24,]
    processes = 8
    gridlevelss = [3,4,5]
elif doTests['little_mmg']:
    verbose = True
    problemscales = [12,]
    gridlevelss = [2,]
elif doTests['graph_pressure']:
    problemscales = [12,]
    repeats = 1
    cycless = [2,]
    graph_pressures = [True,]
    verbose = False
else:
    raise ValueError('No doTests defined.')

if False:
    print 'Test Definition:'
    print '    cycles in',cycless
    print '    gridlevels in',gridlevelss
    print '    iterations in',iterationss
    print '    dense in',denses
    print '    pre_iterations in',pre_iterationss
    print '    post_iterations in',post_iterationss
    print '    solver in',solvers
    print '    problemscale in',problemscales
    print '    graph_pressure in',graph_pressures
#Generate a list of all parameter combinations using vectors (lists) from above.
parameterslist = []
for cycles in cycless:
    for gridlevels in gridlevelss:
        coarsest_level = gridlevels-1
        for iterations in iterationss:
            for dense in denses:
                for pre_iterations in pre_iterationss:
                    for post_iterations in post_iterationss:
                        for solver in solvers:
                            for problemscale in problemscales:
                                for graph_pressure in graph_pressures:
                                    for give_info in give_infos:
                                        #if solver == 'direct' and problemscale >= 20: #might not alwasy be necessary. But our cores are only 2.40 GHz.
                                            #pass
                                        #elif solver == 'gs' and problemscale >= 20:
                                            #pass
                                        #else:
                                        parameterslist.append({'graph_pressure':graph_pressure,
                                                               'iterations':iterations,'dense':dense,
                                                               'solver':solver,'problemscale':problemscale,
                                                               'coarsest_level':coarsest_level,'gridlevels':gridlevels,
                                                               'problemshape':(problemscale,problemscale,problemscale),
                                                               'threshold':threshold,'verbose':verbose,
                                                               'description':test_descriptions,'cycles':cycles,
                                                               'pre_iterations':pre_iterations,
                                                               'post_iterations':post_iterations,
                                                               'give_info':give_info
                                                               })
parameterslist_once = list(parameterslist) # see this lovely article: http://henry.precheur.org/python/copy_list
repeats = repeats.__abs__() # to overcome excessive cleverness
for repeat in range(1,repeats):
    parameterslist.extend(parameterslist_once) # this is similar to numpy's tile operation.
shuffle(parameterslist) # makes the output of percentdone() (below) seem a little more reasonable to the impatient user (me).

#generate filename
isodate         = strftime('%Y%m%d',isolocaltime)
isotime         = strftime('T%H%M%S',isolocaltime)
seconds_prefix  = strftime('%S',isolocaltime)
filelabel = seconds_prefix + long_description + '-' + test_descriptions + "chunksize%i_multip%i-"%(chunksize,processes) + isodate + isotime # use this to make unique log and csv output files for a given test

jobcount = parameterslist.__len__()
print 'jobcount is %i.'%jobcount
filelabel = filelabel + '-%i_jobs'%jobcount
print 'Test description and filename seed is:',filelabel

#start logging
f = open('output/%s.log'%filelabel, 'a')
csv = open('output/%s.csv'%filelabel, 'a')
csv.write('%s\n' % start_parameters)
csv.write('%s\n' % filelabel)
columnheadings = ','.join([\
                        'solver',\
                        'solvernumber',\
                        'problemscale',\
                        'coarsest_level',\
                        'gridlevels',\
                        'threshold',\
                        'time [sec]',\
                        'memory [bytes]',\
                        'resident [bytes]',\
                        'stacksize [bytes]',\
                        'final_cycle',\
                        'final_residual_norm',\
                        'iterations',\
                        'pre_iterations',\
                        'post_iterations',\
                        'sparsity',\
                       ])
csv.write(columnheadings)
f.write('%s\n' % start_parameters)
f.write('%s\n' % filelabel)
f.flush()
csv.flush()


def _VmB(VmKey,pid):
    '''Private.
    '''
#    _proc_status = '/proc/%d/status' % pid
#    _scale = {'kB': 1024.0, 'mB': 1024.0*1024.0, 'KB': 1024.0, 'MB': 1024.0*1024.0}
#     # get pseudo file  /proc/<pid>/status
#    try:
#        t = open(_proc_status)
#        v = t.read()
#        t.close()
#    except:
#        return 0.0  # non-Linux?
#     # get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
#    i = v.index(VmKey)
#    v = v[i:].split(None, 3)  # whitespace
#    if len(v) < 3:
#        return 0.0  # invalid format?
#     # convert Vm value to bytes
#    return float(v[1]) * _scale[v[2]]
    return 0

def memory(pid,since=0.0,):
    '''Return memory usage in bytes.
    '''
#    return _VmB('VmSize:',pid) - since
    return 0

def resident(pid,since=0.0):
    '''Return resident memory usage in bytes.
    '''
#    return _VmB('VmRSS:',pid) - since
    return 0

def stacksize(pid,since=0.0):
    '''Return stack size in bytes.
    '''
#    return _VmB('VmStk:',pid) - since
    return 0


def runparameters(parameters):
    '''
    Function for running a dictionary of parameters. Required (in this functional form)
    so that we can use a map operation with multiprocessing, below.
    That is, you map this function across the list of all the parameter dictionaries:
        map( f(dict), list_of_dicts)
    '''
    # Log everything, and send it to stderr.
    # http://blog.tplus1.com/index.php/2007/09/28/
    logging.basicConfig(level=logging.DEBUG)
    #f.write("solver %s:\n" % solver)
    #sectimes = range(repeats)
    #f.write("Timed trial #%i:" % i)
    #print "runparameters: parent  PID is ",os.getppid()
    #print "runparameters: |-> process PID is ",os.getpid()
    startTime = datetime.now()
    if parameters['solver']=='gs': parameters['verbose'] = True
    try:
        #print 'from the resource module (runparameters, before testgrid):',resource.getrusage(resource.RUSAGE_SELF).ru_isrss,'problemscale',parameters['problemscale']
        returned = testgrid(parameters) #return a dictionary
        #print 'from the resource module (runparameters, after  testgrid):',resource.getrusage(resource.RUSAGE_SELF).ru_isrss,'problemscale',parameters['problemscale']
        toexec = ','.join(returned) + ', = returned.values()' # unpack the returned stuff into the local namespace. print join(returned) to see what those stuffs are.
        exec toexec
        delta=(datetime.now()-startTime)
        trial_time=delta.microseconds/1000000.+delta.seconds*1.
        if verbose: print 'Trial time was %.4f seconds.'%trial_time
        csv.write("\n")
        solvernumbers = {'pyamg':1, 'pyamg-linagg':5, 'mmg':2, 'gs':3, 'direct':4, 'gs_xiter':3}
        data_list = []
        data_list.append('%s'%parameters['solver'])
        data_list.append('%i'%solvernumbers[parameters['solver']]) # this should make it possible to group data together in openoffice, for quick analysis.
        data_list.append('%i'%parameters['problemscale'])
        data_list.append('%i'%parameters['coarsest_level'])
        data_list.append('%i'%parameters['gridlevels'])
        data_list.append('%.6f'%parameters['threshold'])
        data_list.append('%.3f'%trial_time)
        data_list.append('%i'%memory(os.getpid()))
        data_list.append('%i'%resident(os.getpid()))
        data_list.append('%i'%stacksize(os.getpid()))
        data_list.append('%i'%cycle)
        data_list.append('%.20f'%norm)
        data_list.append('%i'%parameters['iterations'])
        data_list.append('%i'%parameters['pre_iterations'])
        data_list.append('%i'%parameters['post_iterations'])
        if parameters['dense']: sparsity = '0'
        else: sparsity = '1'
        data_list.append(sparsity)
        data = ','.join(data_list)
        csv.write(data) # data is a string
        csv.flush()
    except:
        #solverstring = sys.exc_info()[0]
        solverstring = 'ERROR: ' + str(sys.exc_info()[0]) + ' with parameters ' + str(parameters)
        print ""
        logging.exception('error in runparameters():')
#        logging.debug("debug: Something awful happened!")
        print solverstring
        print ""
    f.write( "%s\n" %solverstring)
    f.write("\n")
    f.flush()
    #print 'from the resource module (runparameters, before return  ):',resource.getrusage(resource.RUSAGE_SELF).ru_isrss,'problemscale',parameters['problemscale']
    if verbose: print ''
    if verbose: print "finished %s solver in %.3f seconds."%(parameters['solver'],trial_time)
    return solverstring

if verbose: print 'Spawning a pool of',processes,'processes.'
pool = Pool(processes) # see http://docs.python.org/library/multiprocessing.html
poolresult = pool.map_async(runparameters,parameterslist,chunksize)
remaining = poolresult._number_left
#jobcount = jobcount/repeats
def percentdone(jobcount, remaining):
    return '%.2f%%'%(    (jobcount.__float__() - remaining.__float__()) / jobcount.__float__() * 100.    )
#print "Waiting for", remaining, "tasks to complete..."
oldremaining = remaining
while True:
    if (poolresult.ready()): break
    remaining = poolresult._number_left
    if remaining < oldremaining:
        print '%s' % percentdone(jobcount,remaining)
        #print "Waiting for", remaining, "tasks to complete..."
    oldremaining = remaining
    sleep(0.5)
parameters_delta=(datetime.now()-start_parameters)
f.write('%s\n' % datetime.now())
f.write('parameters_delta of %s\n' % parameters_delta)
f.write("\n")

csv.write("\n")
csv.write('%s\n' % datetime.now())
csv.write('parameters_delta of %s\n' % parameters_delta)
print 'Test description and filename seed is:',filelabel
print 'This script took %s to run.' % parameters_delta
csv.write("\n")
csv.close()
f.close()

if doTests['compare_solvers']:
    import make_scipy_graphs as msg # I should probably try to reduce this for health reasons.
    msg.make_scipy_n_up('%s.csv'%filelabel) # generates output/n-up.png from this and other data.
