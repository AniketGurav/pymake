import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import os
from multiprocessing import Process
from itertools import cycle

from local_utils import *

def display(block=False):
    #p = Process(target=_display)
    #p.start()
    plt.show(block=block)

def _display():
    os.setsid()
    plt.show()

def tag_from_csv(c):
    ## loglikelihood_Y, loglikelihood_Z, alpha, sigma, _K, Z_sum, ratio_MH_F, ratio_MH_W
    if c == 0:
        ylabel = 'log(P(Y))'
        label = 'log likelihood'
    elif c == 1:
        ylabel = 'log(P(F))'
        label = 'posterior F'
    elif c == 2:
        ylabel = 'alpha'
        label = 'alpha'
    elif c == 3:
        ylabel = 'sigma_w'
        label = 'sigma_w'
    elif c == 4:
        ylabel = 'K'
        label = 'K'
    elif c == 5:
        ylabel = 'Z sum'
        label = 'Z sum'
    elif c == 6:
        ylabel = 'MH ratio new features and weights'
        label = 'MH IBP'
    elif c == 7:
        ylabel = 'MH ratio weights'
        label = 'MH weights'

    return ylabel, label

def csv_row(s):
    if s == 'll':
        row = 0
    elif s == 'lp':
        row = 1
    elif s == 'K':
        row = 4
    elif s == 'alpha':
        row = 2
    elif s == 'sigma':
        row = 3
    return row

def plot_csv(target_dir='', columns=0, sep=' ', separate=False):
    if type(columns) is not list:
        columns = [columns]

    title = "MCMC ILFRM"
    xlabel = 'Iterations'
    markers = cycle([ '+', '*', ',', 'o', '.', '1', 'p', ])
    if not separate:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel(xlabel)
        ax1.set_title(title)
    for column in columns:

        if separate:
            fig = plt.figure()
            plt.title(title)
            plt.xlabel('xlabel')
            ax1 = plt.gca()

        target = target_dir + '/mcmc'
        filen = os.path.dirname(__file__) + "/../../output/" +target
        with open(filen) as f:
            data = f.read()

        data = filter(None, data.split('\n'))
        data = [x.strip() for x in data if not x.startswith(('#', '%'))]

        ll_y = [row.split(sep)[column] for row in data]

        ylabel, label = tag_from_csv(column)
        ax1.set_ylabel(ylabel)

        #ax1.plot(ll_y, c='r',marker='x', label='log likelihood')
        ax1.plot(ll_y, marker=next(markers), label=label)
        leg = ax1.legend()
    plt.draw()

class ColorMap:
    def __init__(self, mat, cmap=None, pixelspervalue=20, minvalue=None, maxvalue=None, title='', ax=None):
        """ Make a colormap image of a matrix
        :key mat: the matrix to be used for the colormap.
        """
        if minvalue == None:
            minvalue = np.amin(mat)
            if maxvalue == None:
                maxvalue = np.amax(mat)
            if not cmap:
                cmap = plt.cm.hot
                if not ax:
                    #figsize = (np.array(mat.shape) / 100. * pixelspervalue)[::-1]
                    #self.fig = plt.figure(figsize=figsize)
                    #self.fig.set_size_inches(figsize)
                    #plt.axes([0, 0, 1, 1]) # Make the plot occupy the whole canvas
                    self.fig = plt.figure()
                    plt.axis('off')
                    plt.title(title)
                    implot = plt.imshow(mat, cmap=cmap, clim=(minvalue, maxvalue), interpolation='nearest')
                else:
                    ax.axis('off')
                    implot = ax.imshow(mat, cmap=cmap, clim=(minvalue, maxvalue), interpolation='nearest')
        def show(self):
            """ have the image popup """
            plt.show()
        def save(self, filename):
            """ save colormap to file"""
            plt.savefig(filename, fig=self.fig, facecolor='white', edgecolor='black')

def draw_adjmat(Y, title=''):
    plt.figure()
    plt.axis('off')
    plt.title('Adjacency matrix')
    plt.imshow(Y, cmap="Greys", interpolation="none", origin='upper')
    title = 'Adjacency matrix, N = %d\n%s' % (Y.shape[0], title)
    plt.title(title)

# Assume one mcmc file by directory
def plot_K_fix(sep=' ', columns=[0,'K'], target_dir='K_test'):
    bdir = os.path.join(os.path.dirname(__file__), "../../output", target_dir)

    # figure
    markers = cycle([ '+', '*', ',', 'o', '.', '1', 'p', ])
    fig = plt.figure()
    fig.canvas.set_window_title(target_dir)

    # for compared curves
    extra_c = []
    for i, column in enumerate(columns):

        # subplot
        ylabel, label = tag_from_csv(i)
        xlabel = 'iterations' if column == 0 else 'K'
        stitle = 'Likelihood convergence' if column == 0 else 'Likelihood comparaison'
        ax1 = fig.add_subplot(1, 2, i+1)
        plt.title(stitle)
        #ax1.set_title(stitle)
        ax1.set_xlabel(xlabel)
        if  column is 'K':
            support = np.arange(min(k_order),max(k_order)+1) # min max of K curve.
            k_order = sorted(range(len(k_order)), key=lambda k: k_order[k])
            extra_c = np.array(extra_c)[k_order]
            ax1.plot(support, extra_c, marker=next(markers))
            continue
        ax1.set_ylabel(ylabel)

        k_order = []
        # Assume one mcmc file by directory
        for dirname, dirnames, filenames in os.walk(bdir):
            if not 'mcmc' in filenames:
                continue

            _k = dirname.split('_')[-1]
            k_order.append(int(_k))
            filen = os.path.join(dirname, 'mcmc')
            with open(filen) as f:
                data = f.read()

            data = filter(None, data.split('\n'))
            data = [x.strip() for x in data if not x.startswith(('#', '%'))]
            curve = [row.split(sep)[column] for row in data]
            curve = np.ma.masked_invalid(np.array(curve, dtype='float'))
            extra_c.append(curve.max())
            ax1.plot(curve, marker=next(markers), label=_k)
            #leg = ax1.legend()

    plt.draw()

# several plot to see random consistancy
def plot_K_dyn(sep=' ', columns=[0,'K', 'K_hist'], target_dir='K_dyn'):
    bdir = os.path.join(os.path.dirname(__file__), "../../output", target_dir)

    # figure
    markers = cycle([ '+', '*', ',', 'o', '.', '1', 'p', ])
    fig = plt.figure()
    fig.canvas.set_window_title(target_dir)

    # for compared curves
    extra_c = []
    for i, column in enumerate(columns):

        # subplot
        ax1 = fig.add_subplot(2, 2, i+1)
        if column is 'K':
            plt.title('LL end point')
            ax1.set_xlabel('run')
            ax1.plot(extra_c, marker=next(markers))
            continue
        elif column is 'K_hist':
            plt.title('K distribution')
            ax1.set_xlabel('K')
            ax1.set_ylabel('P(K)')
            bins = int( len(set(k_order)) * 1)
            #k_order, _ = np.histogram(k_order, bins=bins, density=True)
            ax1.hist(k_order, bins, normed=True, range=(min(k_order), max(k_order)))
            continue
        else:
            ylabel, label = tag_from_csv(i)
            plt.title('Likelihood consistency')
            ax1.set_xlabel('iterations')
            ax1.set_ylabel(ylabel)

        k_order = []
        # Assume one mcmc file by directory
        for dirname, dirnames, filenames in os.walk(bdir):
            if not 'mcmc' in filenames:
                continue

            filen = os.path.join(dirname, 'mcmc')
            with open(filen) as f:
                data = f.read()

            data = filter(None, data.split('\n'))
            data = [x.strip() for x in data if not x.startswith(('#', '%'))]
            _k = data[csv_row('K')][-1]
            k_order.append(int(_k))
            curve = [row.split(sep)[column] for row in data]
            curve = np.ma.masked_invalid(np.array(curve, dtype='float'))
            extra_c.append(curve.max())
            ax1.plot(curve, marker=next(markers), label=_k)
            #leg = ax1.legend()

    plt.draw()

def plot_ibp(model, target_dir=None, block=False, columns=[0], separate=False, K=4):

    G = nx.from_numpy_matrix(model.Y(), nx.DiGraph())
    F = model.leftordered()
    W = model._W

    # Plot Adjacency Matrix
    draw_adjmat(model._Y)
    # Plot Log likelihood
    plot_csv(target_dir=target_dir, columns=columns, separate=separate)
    #W[np.where(np.logical_and(W>-1.6, W<1.6))] = 0
    #W[W <= -1.6]= -1
    #W[W >= 1.6] = 1

    # KMeans test
    clusters = kmeans(F, K=K)
    nodelist_kmeans = [k[0] for k in sorted(zip(range(len(clusters)), clusters), key=lambda k: k[1])]
    adj_mat_kmeans = nx.adjacency_matrix(G, nodelist=nodelist_kmeans).A
    draw_adjmat(adj_mat_kmeans, title='KMeans on feature matrix')
    # Adjacency matrix generation
    draw_adjmat(model.generate(nodelist_kmeans), title='Generated Y from ILFRM')

    # training Rescal
    R = rescal(model._Y, K)
    R = R[nodelist_kmeans, :][:, nodelist_kmeans]
    draw_adjmat(R, 'Rescal generated')

    # Networks Plots
    f = plt.figure()

    ax = f.add_subplot(121)
    title = 'Features matrix, K = %d' % model._K
    ax.set_title(title)
    ColorMap(F, pixelspervalue=5, title=title, ax=ax)

    ax = f.add_subplot(122)
    ax.set_title('W')
    img = ax.imshow(W, interpolation='None')
    plt.colorbar(img)

    f = plt.figure()
    ax = f.add_subplot(221)
    ax.set_title('Spectral')
    nx.draw_spectral(G, axes=ax)
    ax = f.add_subplot(222)
    ax.set_title('Spring')
    nx.draw(G, axes=ax)
    ax = f.add_subplot(223)
    ax.set_title('Random')
    nx.draw_random(G, axes=ax)
    ax = f.add_subplot(224)
    ax.set_title('graphviz')
    try:
        nx.draw_graphviz(G, axes=ax)
    except:
        pass


    display(block=block)

    def plot_ibp():
        from ibp import IBP
        ibp = IBP()
        ibp.plot_matrices()

