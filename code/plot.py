#-*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pdb import set_trace as debug


def plot(x, y, title='', xlabel='', ylabel=''):
    plt.clf()
    figure = plt
    figure.scatter(x, y)
    figure.title(title)
    figure.xlabel(xlabel)
    figure.ylabel(ylabel)
    figure.autoscale(tight=True)
    figure.grid()
    figure.savefig(title + '.png', format='png')
    plt.close()


def multiPlot(plotArray, title='', xlabel='', ylabel='', xscale='linear', yscale='linear'):
    """
        plotArray is of the form: [[X1, Y1], [X2, Y2]]
        Where Xi, Yi are arrays of of the values to plot.
    """
    plt.clf()
    figure = plt
    figure.title(title)
    figure.xlabel(xlabel)
    figure.ylabel(ylabel)
    markers = '+,.1234'
    colors = 'rbgyo'
    if len(plotArray) > 5:
        print 'More than 5 plots given as parameters (plotArray)'
        return False
    for i, (Xi, Yi) in enumerate(plotArray):
        figure.scatter(
            Xi,
            Yi,
            marker=markers[i],
            c=colors[i]
        )
    figure.xscale(xscale)
    figure.yscale(yscale)
    figure.autoscale(tight=True)
    figure.grid()
    figure.savefig(title + '.png', format='png')
    plt.close()


def plotLines(plotArray, title='', xlabel='', ylabel='', xscale='linear', yscale='linear'):
    """
        plotArray is of the form: [[X1, Y1], [X2, Y2]]
        Where Xi, Yi are arrays of of the values to plot.
    """
    plt.clf()
    figure = plt
    figure.title(title)
    figure.xlabel(xlabel)
    figure.ylabel(ylabel)
    markers = '+,.1234'
    colors = 'rbgyo'
    if len(plotArray) > 5:
        print 'More than 5 plots given as parameters (plotArray)'
        return False
    for i, (Xi, Yi) in enumerate(plotArray):
        figure.plot(
            Xi,
            Yi,
            marker=markers[i],
            c=colors[i]
        )
    figure.xscale(xscale)
    figure.yscale(yscale)
    figure.autoscale(tight=True)
    figure.grid()
    figure.savefig(title + '.png', format='png')
    plt.close()

def plot3D(plotArray, title='', xlabel='', ylabel='', zlabel=''):
    """
        Given a plotArray = [X, Y, Z] this will save a 3D
        figure of the scattered data.
    """
    X, Y, Z = plotArray
    figure = plt.figure()
    ax = Axes3D(figure)
    ax.scatter(X, Y, Z)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    figure.savefig(title + '.png', format='png')
    plt.close(figure)


def plotLines3D(plotArray, title='', xlabel='', ylabel='', zlabel=''):
    """
        Given a plotArray = [X, Y, Z] this will save a 3D
        figure of the plotted data.
    """
    X, Y, Z = plotArray
    figure = plt.figure()
    ax = Axes3D(figure)
    ax.plot_wireframe(X, Y, Z)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    figure.savefig(title + '.png', format='png')
    plt.close(figure)

def plotSurface3D(plotArray, title='', xlabel='', ylabel='', zlabel=''):
    """
        Given a plotArray = [X, Y, Z] this will save a 3D surface
        figure of the plotted data.
    """
    X, Y, Z = plotArray
    figure = plt.figure()
    ax = Axes3D(figure)
    ax.plot_surface(X, Y, Z)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    figure.savefig(title + '.png', format='png')
    plt.close(figure)


if __name__ == '__main__':
    x = [i for i in xrange(10)]
    y = [i**2 for i in xrange(10)]
    x1 = [i for i in xrange(10)]
    y2 = [i**3 for i in xrange(10)]
    x3 = [i for i in xrange(10)]
    y3 = [i*3 for i in xrange(10)]
    plotLines3D([x, y, y3], title="test")
