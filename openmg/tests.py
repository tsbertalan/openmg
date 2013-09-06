"""
Tests for OpenMG. Mostly, they just check that interfaces are all still
functional. Coverage appears to be close to 100%, but there is certainly still
potential for logical testing holes.

@author: bertalan@princeton.edu
"""


import unittest
import numpy as np
from __init__ import mgSolve
from solvers import smooth, smoothToThreshold, gaussSeidel
import operators
import tools
from sys import _getframe, exc_info


def whoami(level=1):
    '''Return the name of the calling function. Specifying a level greater than
    1 will return the name of the calling function's caller.'''
    return _getframe(level).f_code.co_name


def mpl_test_data(delta=0.25):
    '''
    Return a tuple X, Y, Z with a test data set.
    Modified from matplotlib's mpl_toolkits.mplot3d.axes3d.get_test_data
    '''
    from matplotlib.mlab import  bivariate_normal
    x = y = np.arange(-4.0, 4.0, delta)
    X, Y = np.meshgrid(x, y)

    Z1 = bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
    Z2 = bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
    Z = Z2 - Z1

    X = X * 16
    Y = Y * 16
    Z = Z * 512
    return X, Y, Z


class TestOpenMG(unittest.TestCase):

    def setUp(self):
        self.verbose = False
        self.saveFig = False
        self.parameters = {
                           'problemShape': (512,),
                           'gridLevels': 3,
                           'iterations': 1,
                           'verbose': False,
                           'threshold': 4,
                           'giveInfo': True,
                          }
    
    def test_a(self):
        '''
        Poisson equation on a sine curve. In 3D?
        It wasn't originally, but the restriciton algorithm was having
        trouble with 1D. This might bear investigation.
        '''
        problemscale = 12
        size = problemscale ** 3
        gridLevels = 4
        u_zeros = np.zeros((size,))
        u_actual = np.sin(np.array(range(int(size))) * 3.0 / size).T
        A = operators.poisson((size,))
        b = tools.flexibleMmult(A, u_actual)
        uSmoothed = smooth(A, b, u_zeros, iterations=1)
        parameters = {'coarsestLevel': gridLevels - 1,
                      'problemShape': (problemscale, problemscale, problemscale),
                      'gridLevels': gridLevels,
                      'threshold': 8e-3,
                      }
        u_mmg = mgSolve(A, b, parameters)
        if self.verbose:
            print 'norm is', (np.linalg.norm(tools.getresidual(b, A, uSmoothed.reshape((size, 1)), size)))
        residual_norm = np.linalg.norm(tools.flexibleMmult(A, u_mmg) - b)
        assert parameters['threshold'] > residual_norm
    
    def test_b(self):
        '''Demonstrate the restriction matrix generator.
        '''
        N = 4096
        shape = (16, 16, 16,)
        if self.verbose:
            print 'for N = %i' % N, 'restriction matrix has shape', \
                                     operators.restriction(N, shape).shape
    
    def test_c(self):
        '''Run the multifreq_1d test with four different N arguments'''
        # thousands = [10, 30, ]
        thousands = [1, 3, ]
        frames = 5  # output gets compiled into a movie!
        multifreq_Ns = map(lambda x: int(x * 10 ** 3), thousands)
        for N in multifreq_Ns:
            self.multifreq_1d(N, frames)
    
    def test_d(self):
        '''Run the multifreq_2d test with four different N arguments'''
        # thousands = [10,30,]
        # multifreq_problemscales = map(lambda x: int(x * 10 ** 3), thousands)
        tens = [1, 2]
        multifreq_problemscales = map(lambda x: int(x * 10 ** 1), tens)
        frames = 5  # output gets compiled into a movie!
        for NX in multifreq_problemscales:
            self.multifreq_2d(NX, frames)
    
    def multifreq_1d(self, N, finaliterations, spectralxscale='linear',
                     solutionname='whitenoise', save=False):
        '''Generates 1-dimensional uniform noise with N unknowns, then applies
        Gauss-Siedel smoothing iterations to an initial zero-vector guess until
        finaliterations is reached, graphing the Fourier transform of the error
        each time. Does so pretty inefficiently, by doing 1 iteration, 2 iterations,
        3, iterations, etc. until finaliterations, instead of just doing
        finaliterations iterations, and graphing after each sweep.
    
        Uses ffmpeg and mplayer.'''
        if solutionname == 'summedsine':  # Summed sine waves of several frequencies.
            data = np.zeros((N,))  # initialize
            domainwidthsexponents = range(10)
            domainwidths = range(10)  # initialize
            for i in range(10):
                domainwidths[i] = float(N) / (2.0 ** domainwidthsexponents[i])
            for domainwidth in domainwidths:
                sininput = np.arange(0.0, domainwidth, domainwidth / N)
                xdata = (np.sin(sininput) / domainwidth * 10.0)[0:N:]
                data = data + xdata
        else:
            solutionname = 'whitenoise'
            data = np.random.random((N,))  # 1D white noise
        gif_output_name = 'spectral_solution_rate-%i_unknowns-%s_xscale-%s_solution.gif' % (N, spectralxscale, solutionname)

        A = operators.poisson((N,))
        b = tools.flexibleMmult(A, data)

        if self.saveFig:
            from scipy import fftpack
            import matplotlib.pyplot as plt
        #    Prepare for Graphing
            fig = plt.figure(figsize=(7.5, 7))
            fig.suptitle(gif_output_name)
            fig.subplots_adjust(wspace=.2)
            fig.subplots_adjust(hspace=.2)
            solnax = fig.add_subplot(211)  # subplot to hold the solution
            solnaxylim = [0, 1]
            specax = fig.add_subplot(212)  # subplot to hold the spectrum
            specaxylim = [10.0 ** (-18), 10.0 ** 3]

        solutions = []
        spectra = []
        filenames = []
        loop = 1  # for monitoring progress

        iterationslist = range(finaliterations)
    #    from random import shuffle
    #    shuffle(iterationslist)
        if save:
            iterations_to_save = range(1, finaliterations + 1)
            str_tosave = []
            for x in iterations_to_save:
                str_tosave.append(str(x))
            from os import system; system("mkdir -p output")
            csvfilelabel = 'iterative_spectra-%i_N' % N
            csvfile = open('output/%s.csv' % csvfilelabel, 'a')
            csvfile.write('frequency,' + ','.join(str_tosave) + '\n')
            csvfile.flush()
        for iterations in iterationslist:
            solutions.append(smooth(A,
                                             b,
                                             np.zeros((N,)),
                                             iterations)
                                            )
            if self.saveFig:
                spectra.append(fftpack.fft(b - tools.flexibleMmult(A,
                                                                   solutions[iterations]
                                                                   )
                                          )
                              )
            if self.saveFig:
                filebasename = '%i_unknowns-%i_iterations' % (N, iterations)
                filename = filebasename + '-solution.png'
                solnax.set_ylim(solnaxylim)
                solnax.set_autoscaley_on(False)
                solnax.set_title(filebasename + ': Solution')
                solnax.plot(solutions[iterations])
    
                specax.set_yscale('log')
                specax.set_xscale(spectralxscale)
                specax.set_ylim(specaxylim)
                specax.set_autoscaley_on(False)
                specax.set_title(filebasename + ': Solution Error (frequency domain)')
                specax.plot(spectra[iterations])

                filename = 'temp/' + filebasename + '-combined.png'
                filenames.append(filename)
                if self.saveFig: 
                    print "%i of %i: " % (loop, finaliterations) + "saving " + filename
                    fig.savefig(filename, dpi=80)
                solnax.cla()
                specax.cla()
            loop += 1
        frequencies = range(len(solutions[0]))
        for i in frequencies:
            towrite = '%i,' % frequencies[i]
            for spectrum in spectra:
                towrite += '%.8f,' % abs(spectrum[i])
            if save:
                csvfile.write(towrite + '\n')
                csvfile.flush()
        if save:
            csvfile.close()

        # make .gif and .mp4 files:
        if self.saveFig:
            # TODO (Tom) Might use subprocess to suppress super verbose ffmpeg
        #    import subprocess
        #    s = subprocess.Popen(['cowsay', 'hello']
        #    stderr=subprocess.STDOUT
        #    stdout=subprocess.PIPE).communicate()[0]
        #    print s
            torun = []
            torun.extend([
                'convert ' + ' '.join(filenames) + ' output/%s' % gif_output_name,
                'rm ' + ' '.join(filenames),
                'cp %s simple.gif' % ('output/' + gif_output_name),
                'mplayer -vo jpeg simple.gif > /dev/null 2> /dev/null',
                'ffmpeg -r 12 -i %%08d.jpg -y -an output/%s.avi > /dev/null' % gif_output_name,
                'rm simple.gif 0*.jpg',
                'ffmpeg -y -i output/%s.avi output/%s.mp4 > /dev/null && rm output/%s.avi' % (gif_output_name, gif_output_name, gif_output_name),
                ])
            for command in torun:
                system(command)
    
#     def test_multifreq_1d(self):
#         """Uses ffmpeg and mplayer, with dumb text."""
#         self.multifreq_1d(18000, 4)
     
    def multifreq_2d(self, problemscale, finaliterations):
        '''Attempts to generate a pretty 2D solution, by summing several 2D
        sine waves of different frequencies. Then, proceeds to apply a GS smoother
        to the generated problem, saving pretty pictures after each iteration.
        This might work OK if problemscale is big enough.
        '''
        if self.saveFig:
            from scipy import fftpack
            import matplotlib.pyplot as plt
            from matplotlib import cm
            from mpl_toolkits.mplot3d import Axes3D

        # Generate test problem. Summed sine waves of several resolutions:
        NX = problemscale
        floatNX = float(NX)
        data = np.zeros((NX, NX))
        domainwidths = [floatNX, floatNX / 2.0, 10., 3.1415]
        for domainwidth in domainwidths:
            sininput = np.arange(0.0, domainwidth, domainwidth / NX)
            xdata = (np.sin(sininput) / domainwidth * 10.0)[0:NX:]
            data = data + xdata[:, np.newaxis] + xdata[:, np.newaxis].T
        X = np.tile(np.array(range(NX)), (NX, 1))
        Y = X.T

        if self.saveFig:
            # Prepare for Graphing:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.set_zlim3d([-1, 6])
            specfig = plt.figure()
            specax = specfig.add_subplot(111)
            specax.set_ylim([.000001, 10000])
            specax.set_autoscaley_on(False)
            specax.set_xscale('log')

        # Problem Setup:
        A = operators.poisson((NX, NX))
        solnforb = data.ravel().reshape((NX ** 2, 1))
        b = tools.flexibleMmult(A, solnforb)

        # Initial "Solution" placeholder:
        solutions = range(finaliterations)
        spectra = range(finaliterations)
        iterations = 0
        solutions[iterations] = np.zeros((NX, NX)).ravel()
        spectra[iterations] = b - tools.flexibleMmult(A, solutions[iterations])

        if self.saveFig:
            # Initial graphs:
            filebasename = 'output/N_%i-iterations_%i' % (NX ** 2, iterations)
            filename = filebasename + '.png'
            ax.set_title(filebasename)
            # surf = ax.plot_wireframe(X, Y, solutions[iterations].reshape(NX,NX),rstride=stridelength,cstride=stridelength)
            surf = ax.plot_surface(X, Y, data, cmap=cm.jet)
    #        print "saving", filename
            if self.saveFig: fig.savefig(filename)
            surf.remove()
            ax.cla()

            # Error Spectrum
            specax.plot(spectra[iterations])
            specax.set_title(filebasename + ': Error Spectrum')
            filename = filebasename + '-error.png'
            if self.saveFig: specfig.savefig(filename)
            del specax.lines[0]
            specax.cla()

        verbose = False
        for iterations in range(1, finaliterations):
            solutions[iterations] = smooth(A,
                                                    b,
                                                    np.zeros(NX ** 2),
                                                    iterations,
                                                    verbose=verbose
                                                   ).reshape(NX, NX)
            if self.saveFig:
                spectra[iterations] = fftpack.fft2(solutions[iterations])
            if self.saveFig:
                filebasename = 'output/N_%i-iterations_%i' % (NX ** 2, iterations)
                filename = filebasename + '-solution.png'
                ax.set_zlim3d([-1, 6])
                surf = ax.plot_surface(X, Y, solutions[iterations], cmap=cm.jet)
                ax.set_title(filebasename)
                surf = ax.plot_wireframe(X, Y, solutions[iterations])
    #            print "saving", filename
                if self.saveFig: fig.savefig(filename)
                surf.remove()
                ax.cla()
    
                specax.set_yscale('log')
                specax.set_ylim([.000001, 10000])
                specax.set_autoscaley_on(False)
                specax.set_xscale('log')
                specax.plot(spectra[iterations])
                specax.set_title(filebasename + ': Error Spectrum')
                filename = filebasename + '-error.png'
    #            print "saving", filename
                if self.saveFig: specfig.savefig(filename)
                del specax.lines[0]
                specax.cla()
    
    def test_gs_thresh(self):
        '''Test the Gauss-Siedel to-threshold solver.'''
        NX = 12
        NY = NX
        N = NX * NY
        A = operators.poisson((NX, NY,))
        b = np.random.random((N,))
        x_init = np.zeros((N,))
        threshold = 0.0001
        verbose = False
        x = smoothToThreshold(A, b, x_init,
                                                    threshold,
                                                    verbose=verbose)
        if self.verbose:
            print 'norm is', (np.linalg.norm(tools.getresidual(b, A, x, N)))
    
    def test_gs_nothresh(self):
        '''Test the Gauss-Siedel with no threshold or iterations specified.'''
        NX = N = 12
        A = operators.poisson((NX,))
        b = np.random.random((N,))
        x_init = np.zeros((N,))
        x = gaussSeidel(A, b, x_init,)
    
    def test_1d_noise_mg(self, parameters=None):
        '''Solve a 2D Poisson equation, with the solution being 2D white noise,
        i.e., np.random.random((NX,NY)). Useful as a sanity check.
        '''
        if parameters is None:
            parameters = self.parameters
        N = NX = parameters['problemShape'][0]
        u_actual = np.random.random((NX,)).reshape((N, 1))
        A_in = operators.poisson((NX,))
        b = tools.flexibleMmult(A_in, u_actual)
        u_mg, info_dict = mgSolve(A_in, b, parameters)
        if self.verbose:
            print 'test_1d_noise_mg: final norm is ', info_dict['norm']
    
    def test_3d_noise_mg(self):
        '''Solve a 3D Poisson equation, with the solution being 2D white noise,
        i.e., np.random.random((NX, NY, NZ)). Useful as a sanity check.
        '''
        NX = NY = NZ = 8
        N = NX * NY * NZ
        shape = NX, NY, NZ
        u_actual = np.random.random(shape).reshape((N, 1))
        A_in = operators.poisson(shape)
        b = tools.flexibleMmult(A_in, u_actual)
        u_mg, info_dict = mgSolve(A_in, b, {
                                               'problemShape': shape,
                                               'gridLevels': 3,
                                               'iterations': 1,
                                               'verbose': False,
                                               'threshold': 4,
                                               'giveInfo': True,
                                              }
                                    )
        if self.verbose:
            print 'test_1d_noise_mg: final norm is ', info_dict['norm']
    
    def test_2d_noise_mg(self):
        '''
        Solve a 2D Poisson equation, with the solution being 2D white noise, i.e.,
        np.random.random((NX,NY)). Useful as a sanity check.
        '''
        NX = 32
        NY = NX
        N = NX * NY
        u_actual = np.random.random((NX, NY)).reshape((N, 1))
        A_in = operators.poisson((NX, NY))
        b = tools.flexibleMmult(A_in, u_actual)
        u_mg, info_dict = mgSolve(A_in, b, {
                                               'problemShape': (NX, NY),
                                               'gridLevels': 3,
                                               'iterations': 1,
                                               'verbose': False,
                                               'threshold': 4,
                                               'giveInfo': True,
                                              }
                                    )
        if self.verbose:
            print 'test_2d_noise_mg: final norm is ', info_dict['norm']
    
#     def test_3d_noise_mg_multi_N(self):
#         '''
#         Solve a 3D Poisson equation, with the solution being 3D white noise, i.e.,
#         np.random.random((NX,NY,NZ)). Useful as a sanity check, and as a check
#         on what is the allowable maximum problemsize with your memory.
#         '''
#         saytest()
#         NXes = range(1, 16)
#     #    NXes = range(8, 16)[::4]
#     #    NXes = range(8, 22)[::2]
#     #    NXes = range(8, 22)[::4]
#         print ''
#         for NX in NXes:
#             NY = NZ = NX
#             N = NX * NY * NZ
#             print 'NX is', NX, 'and N is ', N, ':'
#             u_actual = np.random.random((NX, NY, NZ)).reshape((N, 1))
#             A_in = poisson((NX, NY, NZ))
#             b = tools.flexibleMmult(A_in, u_actual)
#             (u_mg, info_dict) = mgSolve(A_in, b,
#                                 {'problemShape': (NX, NY, NZ),
#                                  'gridLevels': 3,
#                                  'iterations': 1,
#                                  'verbose': False,
#                                  'threshold': .0001
#                                 })
#             print '    test_3d_noise_mg:', info_dict
#             print ''
    
    def test_2d_mpl_demo(self):
        '''
        Plot a pretty 2D surface, and the result of solving a 2D Poisson equation
        with that surface as the solution.
        '''
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        sizemultiplier = 2
        NX = 8 * sizemultiplier
        cycles = 3
        actualtraces = 8
        mgtraces = 16
        NY = NX
        N = NX * NY
        (X, Y, u_actual) = mpl_test_data(delta=1 / float(sizemultiplier))
        A_in = operators.poisson((NX, NY))
        b = tools.flexibleMmult(A_in, u_actual.ravel())
        u_mg = mgSolve(A_in, b, {
                                  'problemShape': (NX, NY),
                                  'gridLevels': 2,
                                  'iterations': 1,
                                  'verbose': False,
                                  'cycles': 300,
                                  'dense': True
                                 }
                       )

        fig = plt.figure()
        ax = Axes3D(fig)
        fig.suptitle('blue is actual; red is %i v-cycles; %i unknowns' % (cycles, N))
        # ax = fig.add_subplot(111,projection='3d') #not in matplotlib version 0.99, only version 1.x
        ax.plot_wireframe(X, Y, u_actual,
                          rstride=NX / actualtraces,
                          cstride=NY / actualtraces,
                          color=((0, 0, 1.0),)
                         )
        # ax.scatter(X,Y,u_mg[0],c=((1.,0,0),),s=40)
        ax.plot_wireframe(X, Y, u_mg.reshape((NX, NY)),
                          rstride=NX / mgtraces,
                          cstride=NY / mgtraces,
                          color=((1.0, 0, 0),)
                         )
        filename = 'output/test_2d_mpl_demo-%i_vycles-%i_N.png' % (cycles, N)
        if self.saveFig: fig.savefig(filename)
        # plt.show()
        if self.verbose:
            print 'norm is', (np.linalg.norm(openmg.tools.getresidual(b, A_in, u_mg[0], N)))

    def test_threshStop(self):
        size = 36
        u_actual = np.sin(np.array(range(int(size))) * 3.0 / size).T
        A = operators.poisson((size,))
        b = tools.flexibleMmult(A, u_actual)
        parameters = {
                      'problemShape': (size,),
                      'gridLevels': 2,
                      'threshold': 8e-3,
                      'giveInfo': True, 
                      }
        u_mmg, info_dict = mgSolve(A, b, parameters)
        residual_norm = np.linalg.norm(tools.flexibleMmult(A, u_mmg) - b)
        assert parameters['threshold'] > residual_norm

    def test_cycleStop(self):
        size = 36
        u_actual = np.sin(np.array(range(int(size))) * 3.0 / size).T
        A = operators.poisson((size,))
        b = tools.flexibleMmult(A, u_actual)
        parameters = {
                      'problemShape': (size,),
                      'gridLevels': 2,
                      'cycles': 3,
                      'threshold': 1e-10,
                      'giveInfo': True, 
                      }
        u_mmg, info_dict = mgSolve(A, b, parameters)
        residual_norm = np.linalg.norm(tools.flexibleMmult(A, u_mmg) - b)
        assert info_dict['cycle'] == parameters['cycles']

    def test_poisson4d(self):
        def testor():
            operators.poisson((1, 2, 3, 4))
        self.assertRaises(ValueError, testor)
        
    def test_123DRestriction(self):
        for dense in True, False:
            R = [operators.restriction(tuple(
                   [4 for i in range(alpha)]
                                   ), dense=dense) for alpha in range(1, 4)]
    
    def test_4DRestriction(self):
        for dense in True, False:
            def testor():
                R = operators.restriction((4, 4, 4, 4), dense=dense)
            self.assertRaises(ValueError, testor)
            
    def test_neitherStopValueError(self):
        parameters = self.parameters.copy()
        parameters["cycles"] = 0
        parameters["threshold"] = 0
        def testor():
            self.test_1d_noise_mg(parameters=parameters)
        self.assertRaises(ValueError, testor)

    def test_minSize(self):
        parameters = self.parameters.copy()
        shape = parameters["problemShape"] = (1024,)
        parameters["gridLevels"] = 24
        parameters["minSize"] = 23
        u_actual = np.random.random(shape).ravel()
        A_in = operators.poisson(shape)
        b = tools.flexibleMmult(A_in, u_actual)
        soln, info_dict = mgSolve(A_in, b, parameters)
        if self.verbose:
            print "R shapes:", [r.shape for r in info_dict['R']]
            print "c.v. minSize = ", parameters['minSize']
        assert min(info_dict['R'][-1].shape) > parameters["minSize"]


def doTests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOpenMG)
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    doTests()
