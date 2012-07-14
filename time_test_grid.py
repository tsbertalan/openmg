from datetime import datetime
from time import strftime, localtime, sleep
import sys
import os
from multiprocessing import Pool
from test_grid_AMG_USS_Pressure import testgrid
from random import shuffle
import logging
#import resource # unix-only

def whoami():
    import sys
    return sys._getframe(1).f_code.co_name

#print whoami(),": parent  PID is ",os.getppid()
#print whoami(),": |-> process PID is ",os.getpid()

###############################################################################
# Choose a Test
###############################################################################
long_description = ''
tests = {   'v_cycle_convergence':  False,\
            'domain_size':          False,\
            'sparsity':             False,\
            'presmoother':          False,\
            'gs_thresh':            False,\
            'little_gs':            False,\
            'big_gs':               False,\
            'postsmoother':         False,\
            'ngrid':                False,\
            'graph_pressure':       True,\
            'compare_solvers':      False,\
            'full':                 False,\
            'selftest':             False,\
            'little_mmg':           False,\
            'something_else':       False,\
}


test_descriptions = ''
tests_count=0
for test_description in tests:
    if tests[test_description]:
        tests_count +=1
        test_descriptions = test_descriptions + test_description + '-'
if tests_count > 1:
    print "Setting more than one test to True can result in confusing "+\
          "combination-tests. Either set only one test to True, "+\
          "or write a new combination test, with the desired parameters. "
    print "Note that such combination tests can produce interesting results,"+\
          "although they require multiple linear regression to analyze."
    sys.exit()
if tests_count == 0:
    print "No explicit test defined. Proceeding with default values."


##multiprocessing parameters
processes = 2
chunksize = 1

##general parameters
start_parameters = datetime.now()
isolocaltime = localtime()

##Defaults
verbose = False
threshold = .00021 # When should the v-cycling stop?
repeats = 4
gridlevelss = [2,] # there are two s's on this and some other vectors just below this because it had to be a plural of a plural.
problemscales = [8,]
iterationss = [1,]
pre_iterationss = [1,]
post_iterationss = [0,]
solvers = ['mmg',]
denses = [False,]
cycless =[0,] # if cycles=0 (i.e., cycless = [0,] ), OpenMG will do v-cycles until converged, as defined by threshold
graph_pressures = [False,]

###############################################################################
# Definitions of Tests
###############################################################################
##Overwrite default values with lists of parameters to test.
##Create a new 'elif' block here for each new test created in the 'tests' dict above.
# TODO We need another test here about sigma factors in front of the prolongation (or was it restriction?) matrices.
if tests['selftest']:
    repeats = 1
elif tests['v_cycle_convergence']:
    denses = [True,]
    cycless = range(1,200)
elif tests['domain_size']:
    problemscales = [24,32,36,40,44,48,52,56,60,64]
    repeats = 1
elif tests['sparsity']:
    verbose = True
    denses = [True,False,]
    problemscales = [24]
    repeats = 4
    threshold = 7
elif tests['presmoother']:
    pre_iterationss = range(1,200)
    repeats = 16
    denses = [True,]
elif tests['postsmoother']:
    #post_iterationss = [0,]
    #post_iterationss.extend(range(20,501)[::20])
    post_iterationss = range(0,200)
elif tests['compare_solvers']:
#    verbose = True
    problemscales = [20,]
    threshold = .73 # .695 for ... 12 unknowns?
    denses = [True,]
    #solvers = ['pyamg', 'pyamg-linagg', 'mmg', 'gs', 'direct']
    solvers = [ 'pyamg-linagg', 'mmg', 'gs', 'direct']
#    solvers = ['mmg',]
    repeats = 5
elif tests['big_gs']:
    solvers = ['gs_xiter',]
    problemscales = [12,]
    denses = [True,]
    repeats = 2
    iterationss = range(1,100)[::1]
    iterationss.extend(range(100,1000)[::10])
    iterationss.extend(range(1000,200000)[::100])
elif tests['little_gs']:
    solvers = ['gs_xiter',]
    problemscales = [12,]
    denses = [True,]
    repeats = 2
    iterationss = range(1,100)[::10]
elif tests['gs_thresh']:
    solvers = ['gs',]
    verbose = True
    problemscales = [12,]
#    denses = [True,]
    repeats = 1
elif tests['full']: # probably not a good idea
    gridlevelss = range(2,5)
    problemscales = [8,16] #These must not only be multiples of 4 but also powers of 2. And also >= 4*gridlevels, to provide enough coarse points.
    iterationss = range(1,36)[::4]
    solvers = ['pyamg', 'mmg', 'gs', 'direct']
    cycless = range(1,42)[::2]
elif tests['ngrid']:
    problemscales = [20,]
    gridlevelss = [2,3,4]
elif tests['little_mmg']:
    verbose = True
    problemscales = [12,]
    gridlevelss = [2,]
elif tests['graph_pressure']:
    problemscales = [12,]
    repeats = 1
    cycless = [2,]
    graph_pressures = [True,]
    verbose = False
else:
    print 'No tests defined.'
    sys.exit()

if True:
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
parameterslist = range(0)
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
                                    #if solver == 'direct' and problemscale >= 20: #might not alwasy be necessary. But our cores are only 2.40 GHz.
                                        #pass
                                    #elif solver == 'gs' and problemscale >= 20:
                                        #pass
                                    #else:
                                    parameterslist.append({'graph_pressure':graph_pressure,'iterations':iterations,'dense':dense,'solver':solver,'problemscale':problemscale,'coarsest_level':coarsest_level,'gridlevels':gridlevels,'problemshape':(problemscale,problemscale,problemscale),'threshold':threshold,'verbose':verbose,'description':test_descriptions,'cycles':cycles,'pre_iterations':pre_iterations,'post_iterations':post_iterations})
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
        if verbose: print "unpacking testgrid results"
        toexec = ','.join(returned) + ', = returned.values()' # unpack the returned stuff into the local namespace. print join(returned) to see what those stuffs are.
        if verbose: print toexec
        exec toexec
        if verbose:            print 'unpacked info_dict from test_grid()'
        delta=(datetime.now()-startTime)
        if verbose:            print 'calculated delta'
        trial_time=delta.microseconds/1000000.+delta.seconds*1.
        if verbose: print 'Trial time was %.4f seconds.'%trial_time
        #print "%s:    trial %i of %i: %f sec" % (parameters['solver'],trialnumber,repeats,trial_time)
        csv.write("\n")
        solvernumbers = {'pyamg':1, 'pyamg-linagg':5, 'mmg':2, 'gs':3, 'direct':4, 'gs_xiter':3}
        data_list = []
        data_list.append('%s'%parameters['solver'])
        data_list.append('%i'%solvernumbers[parameters['solver']]) # this should make it possible to group data together in openoffice, for quick analysis.
        data_list.append('%i'%parameters['problemscale'])
        data_list.append('%i'%parameters['coarsest_level'])
        data_list.append('%i'%parameters['gridlevels'])
        data_list.append('%.6f'%parameters['threshold'])
#        if verbose: print 'trial_time'
        data_list.append('%.3f'%trial_time)
#        if verbose: print 'memory'
        data_list.append('%i'%memory(os.getpid()))
#        if verbose: print 'resident'
        data_list.append('%i'%resident(os.getpid()))
#        if verbose: print 'stacksize'
        data_list.append('%i'%stacksize(os.getpid()))
#        if verbose: print 'cycle'
        data_list.append('%i'%cycle)
#        if verbose: print 'norm'
        data_list.append('%.20f'%norm)
        data_list.append('%i'%parameters['iterations'])
        data_list.append('%i'%parameters['pre_iterations'])
        data_list.append('%i'%parameters['post_iterations'])
#        if verbose: print 'joining up data to one string'
        if parameters['dense']: sparsity = '0'
        else: sparsity = '1'
        data_list.append(sparsity)
        data = ','.join(data_list)
        if verbose: print 'data string is %s'%data
        csv.write(data) # data is a string
#        if verbose: print 'Wrote data to csv file.'
        csv.flush()
    except:
        solverstring = sys.exc_info()[0]
        print ""
        logging.exception('error in runparameters():')
#        logging.debug("debug: Something awful happened!")
        print "ERROR:", solverstring, "with parameters:"
        print parameters
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
print '%s' % percentdone(jobcount,remaining)
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
print 'parameters_delta of %s' % parameters_delta
csv.write("\n")
csv.close()
f.close()

if tests['compare_solvers']:
    import make_scipy_graphs as msg # I should probably try to reduce this for health reasons.
    msg.make_scipy_n_up('%s.csv'%filelabel) # generates output/n-up.png from this and other data.
