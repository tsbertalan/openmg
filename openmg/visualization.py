"""
Some plotting code used in the early development of OpenMG.

It is pretty rough, but contains a some methods I used to generate plots for
papers and presentations. When I clean it up later, I'll remove this paragraph.
Until then, sorry.

@author: bertalan@princeton.edu
""" 

import matplotlib
from matplotlib import *
from matplotlib.pyplot import figure, title, savefig
#from enthought.mayavi.api import OffScreenEngine
#from enthought.mayavi.modules.api import Outline, IsoSurface

## Create the MayaVi offscreen engine and start it.
#e = OffScreenEngine()


def make_3d_graph(scalarfield, filename):
    import enthought.mayavi.mlab as mlab
    mlab.figure(bgcolor=(1,1,1),fgcolor=(0,0,0))
    #mlab.options.offscreen = True
    #win = e.new_scene()
    #src = mlab.pipeline.scalar_scatter(Phi_graph)
    #e.add_source(src)
    #e.add_module(IsoSurface())
    #e.add_module(
    #win.scene.isometric_view()
    #win.scene.save(filename,size=(800,600))
    mlab.clf()
    mlab.contour3d(scalarfield,opacity=.5,transparent=True)#,contours=[1.0])
#    mlab.outline()
    mlab.zlabel('Z')
    mlab.xlabel('X')
    mlab.ylabel('Y')
    print 'saving %s' % filename
    mlab.savefig(filename)
    #del mlab
    #mayavi.save_visualization(filename)
    #wait = raw_input('Press ENTER to quit.')

def make_multiple_3d_graphs(data, filename):
    '''data is a list of 3d scalar fields. filename is a string, probably ending in .png'''
    import enthought.mayavi.mlab as mlab
    mlab.figure(bgcolor=(1,1,1),fgcolor=(0,0,0),size=(1000,600))

#    f = mlab.gcf()
#    camera = f.scene.camera
#    cam.parallel_scale = 9

#    view = mlab.view()
#    roll = mlab.roll()
#    print 'camera view is:',view
#    print 'camera roll is:',roll

    #mlab.options.offscreen = True
    #win = e.new_scene()
    #src = mlab.pipeline.scalar_scatter(Phi_graph)
    #e.add_source(src)
    #e.add_module(IsoSurface())
    #e.add_module(
    #win.scene.isometric_view()
    #win.scene.save(filename,size=(800,600))
    i=0
#    print 'mean is: ',data[i].mean()
#    print 'min is: ',data[i].min()
    for contourpercent in [.232,]:  # [.16, .20,.22,.232,.25,.28,.32,.36,.40,.44,.48,.52,.56,.60,.64,.68,.72]:
        mlab.clf()
#        contourpercent = .232 # .28, .24 is good
        value = (data[i].max()-data[i].min())*contourpercent+data[i].min()
        print 'graphing contours at',contourpercent,'of min-max distance, where value is ',value,'(versus a max of',data[i].max(),'and a min of',data[i].min(),', and a range of ',data[i].max()-data[i].min(),').'
    #    print 'graphing data[%i].shape='%i,data[i].shape
        mlab.contour3d(data[i],contours=[value],opacity=.36,transparent=True,colormap='bone')
        for i in range(1,len(data)):
    #        print data[i]
    #        print 'graphing data[%i].shape='%i,data[i].shape
            mlab.contour3d(data[i],contours=[(data[i].max()-data[i].min())*contourpercent+data[i].min()],opacity=.56,transparent=True)
    #    mlab.view(distance=36.0)
    #    mlab.outline()
    #    mlab.zlabel('Z')
    #    mlab.xlabel('X')
    #    mlab.ylabel('Y')
        tosave = filename + '%.03f'%contourpercent + '.png'
        print 'saving %s' % tosave
        mlab.savefig(tosave)
        #wait = raw_input('Press ENTER to quit.')

def make_sparsity_graph(filename,graph_title,x):
    resolution_factor=2
    fig = figure(figsize=(8*2**resolution_factor,6*2**resolution_factor), dpi=300)
    ax3 = fig.add_subplot(111)
    #fontsize=min(max(x.shape[0]*25/4000,10),30)
    fontsize=25
    #markersize=min(max(x.shape[0]*.05/4000,.08),1)
    markersize=.005
    #print "b length=%i, so fontsize=%i and markersize=%s." % (x.shape[0],fontsize,markersize)
    ax3.spy(x,markersize=markersize)
    for tick in ax3.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label2.set_fontsize(fontsize)
    for tick in ax3.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label2.set_fontsize(fontsize)

    title(graph_title,fontsize=fontsize)
    print 'saving %s' % filename
    savefig(filename)

def make_graph(filename,title,y):
    import matplotlib
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    x = range(y.size)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.plot(x,y_actual,color='red')
    ax.scatter(x,y,color='blue')
    #ax.set_ylim(-1,1)
    ax.set_title(title,fontsize=11)
    print "Saving %s" % filename
    fig.savefig(filename)
