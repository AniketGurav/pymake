#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
from scipy.stats import kstest, ks_2samp
from scipy.special import zeta
import matplotlib.pyplot as plt
import networkx as nx
from utils.utils import *
from utils.math import *
from frontend import frontend

import re
import os
from multiprocessing import Process
from itertools import cycle

_markers = cycle([ '+', '*', '|','x', 'o', '.', '1', 'p', '<', '>', 's' ])
_colors = cycle(['r', 'g','b','y','c','m','k'])

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
        ylabel = 'Iteration'
        label = 'Iteration'
    elif c in (1,2, 3):
        ylabel = 'loglikelihood'
        label = 'loglikelihood'

    return ylabel, label

def csv_row(s):
    #csv_typo = '# mean_eta, var_eta, mean_alpha, var_alpha, log_perplexity'
    if s == 'Iteration':
        row = 0
    if s == 'Timeit':
        row = 1
    elif s in ('loglikelihood', 'likelihood', 'perplexity'):
        row = 2
    elif s in ('loglikelihood_t', 'likelihood_t', 'perplexity_t'):
        row = 3
    elif s == 'K':
        row = 4
    elif s == 'alpha':
        row = 5
    elif s == 'gamma':
        row = 6
    else:
        row = s
    return row

def plot_degree(y, title=None, noplot=False):
    if len(y) > 6000:
        return
    G = nxG(y)
    degree = sorted(nx.degree(G).values(), reverse=True)
    if noplot:
        return degree
    #plt.plot(degree)
    x = np.arange(1, y.shape[0] + 1)
    fig = plt.figure()
    plt.loglog(x, degree)
    if title:
        plt.title(title)
    plt.draw()

def plot_degree_(y, title=None):
    if len(y) > 6000:
        return
    G = nxG(y)
    degree = sorted(nx.degree(G).values(), reverse=True)
    x = np.arange(1, y.shape[0] + 1)
    plt.loglog(x, degree)
    if title:
        plt.title(title)

def log_binning(counter_dict,bin_count=35):
    max_x = np.log10(max(counter_dict.keys()))
    max_y = np.log10(max(counter_dict.values()))
    max_base = max([max_x,max_y])

    min_x = np.log10(min(drop_zeros(counter_dict.keys())))

    bins = np.logspace(min_x,max_base,num=bin_count)

    # Based off of: http://stackoverflow.com/questions/6163334/binning-data-in-python-with-scipy-numpy
    #bin_means_y = (np.histogram(counter_dict.keys(),bins,weights=counter_dict.values())[0] / np.histogram(counter_dict.keys(),bins)[0])
    #bin_means_x = (np.histogram(counter_dict.keys(),bins,weights=counter_dict.keys())[0] / np.histogram(counter_dict.keys(),bins)[0])
    bin_means_y = np.histogram(counter_dict.keys(),bins,weights=counter_dict.values())[0]
    bin_means_x = np.histogram(counter_dict.keys(),bins,weights=counter_dict.keys())[0]
    return bin_means_x,bin_means_y

from collections import Counter

def degree_hist_to_list(d, dc):
    degree = np.repeat(np.round(d).astype(int), np.round(dc).astype(int))
    return degree


# Ref: Clauset, Aaron, Cosma Rohilla Shalizi, and Mark EJ Newman. "Power-law distributions in empirical data."
# @todo:cut-off
def gofit(x, y, model='powerlaw'):
    """ (x,y): the empirical distribution with x the values and y **THE COUNTS** """

    # Reconstruct the data samples
    d = degree_hist_to_list(x, y)

    y = y.astype(float)
    #### Power law Goodness of fit
    # Estimate x_min
    x_min = x[y.argmax()]

    ### cutoff ?
    x_max = x.max()

    # Estimate \alpha
    N = len(d)
    n_tail = float((d>x_min).sum())
    alpha = 1 + n_tail * (np.log(d[d>x_min] / (x_min -0.5)).sum())**-1
    print 'x_min', x_min

    ### Build The hypothesis
    if model == 'powerlaw':
        ### Discrete CDF (gives worse p-value)
        cdf = lambda x: 1 - zeta(alpha, x) / zeta(alpha, x_min)
        ### Continious CDF
        #cdf = lambda x:1-(x/x_min)**(-alpha+1)
    else:
        lgg.error('Godfit: Hypothese Unknow %s' % model)
        return

    # Number of synthetic datasets to generate
    precision = 0.03
    S = int(0.25 * (precision)**-2)
    pvalue = []

    # Ignore head data
    if False:
        N = n_tail # bad effect
        ks_d = kstest(d[d>x_min], cdf)
    else:
        ks_d = kstest(d, cdf)

    for s in range(S):
        ### p-value with Kolmogorov-Smirnov, for each synthetic dataset
        # Each synthetic dataset has following size:
        powerlow_samples_size = np.random.binomial(N, n_tail/N)
        # plus random sample from before the cut
        out_empirical_samples_size = N - powerlow_samples_size

        # Generate synthetic dataset
        out_samples = np.random.choice((d[d<=x_min]), size=out_empirical_samples_size)
        powerlaw_samples = random_powerlaw(alpha, x_min, powerlow_samples_size*2) # *2 because the cut-off will remove potentienlly number of sample

        ### Cutoff ?!
        powerlaw_samples = powerlaw_samples[powerlaw_samples <= x_max]

        sync_samples = np.hstack((out_samples, powerlaw_samples))

        #ks_2 = ks_2samp(sync_samples, d)
        ks_s = kstest(sync_samples, cdf)
        pvalue.append(ks_s.statistic > ks_d.statistic)

    #frontend.DataBase.save(d, 'd.pk')
    #frontend.DataBase.save(sync_samples, 'sc.pk')

    pvalue = float(sum(pvalue)) / len(pvalue)
    estim = {'alpha': alpha, 'x_min':x_min, 'n_tail': n_tail,'n_head': N-n_tail,  'pvalue':pvalue}
    print 'KS data: ', ks_d
    print 'KS synthetic: ', ks_s
    print estim
    estim.update({'sync':sync_samples.astype(int)})
    return estim

def adj_to_degree(y):
    # To convert normalized degrees to raw degrees
    #ba_c = {k:int(v*(len(ba_g)-1)) for k,v in ba_c.iteritems()}
    G = nxG(y)
    #degree = sorted(nx.degree(G).values(), reverse=True)

    #ba_c = nx.degree_centrality(G)
    return nx.degree(G)

def degree_hist(_degree):
    degree = _degree.values() if type(_degree) is dict else _degree
    bac = dict(Counter(degree))

    #ba_x,ba_y = log_binning(bac,50)
    d = np.array(bac.keys())  # Degrees
    dc = np.array(bac.values()) # Degree counts

    if d[0] == 0:
        print '%d unconnected vertex' % dc[0]
        d = d[1:]
        dc = dc[1:]

    if len(d) != 0:
        d, dc = zip(*sorted(zip(d, dc)))
    return np.round(d), np.round(dc)

def random_degree(Y, params=None):
    _X = []
    _Y = []
    N = Y[0].shape[0]
    size = []
    for y in Y:
        ba_c = adj_to_degree(y)
        d, dc = degree_hist(ba_c)

        _X.append(d)
        _Y.append(dc)
        size.append(len(_Y[-1]))

    min_d = min(size)
    for i, v in enumerate(_Y):
        if len(v) > min_d:
            _X[i] = _X[i][:min_d]
            _Y[i] = _Y[i][:min_d]

    X = np.array(_X)
    Y = np.array(_Y)
    x = X.mean(0)
    y = Y.mean(0)
    yerr = Y.std(0)
    return np.round(x), np.round(y), yerr

def plot_degree_poly(y, scatter=True):
    """ Degree plot along with a linear regression of the distribution.
    if scatter is false, draw only the linear approx"""
    # To convert normalized degrees to raw degrees
    #ba_c = {k:int(v*(len(ba_g)-1)) for k,v in ba_c.iteritems()}
    ba_c = adj_to_degree(y)
    d, dc = degree_hist(ba_c)

    plt.xscale('log'); plt.yscale('log')

    fit = np.polyfit(np.log(d), np.log(dc), deg=1)
    plt.plot(d,np.exp(fit[0] *np.log(d) + fit[1]), 'g--', label='power %.2f' % fit[1])
    leg = plt.legend(loc=1,prop={'size':10})

    if scatter:
        plt.scatter(d,dc,c='b',marker='o')
        #plt.scatter(ba_x,ba_y,c='r',marker='s',s=50)

    plt.xlim(left=1)
    plt.ylim((.9,1e3))
    plt.xlabel('Degree')
    plt.ylabel('Counts')

def plot_degree_poly_l(Y):
    """ Same than plot_degree_poly, but for a list of random graph ie plot errorbar."""
    x, y, yerr = random_degree(Y)

    plt.xscale('log'); plt.yscale('log')

    fit = np.polyfit(np.log(x), np.log(y), deg=1)
    plt.plot(x,np.exp(fit[0] *np.log(x) + fit[1]), 'm:', label='model power %.2f' % fit[1])
    leg = plt.legend(loc=1,prop={'size':10})

    plt.errorbar(x, y, yerr=yerr, fmt='o')

    plt.xlim(left=1)
    plt.ylim((.9,1e3))
    plt.xlabel('Degree'); plt.ylabel('Counts')

def plot_degree_2(P, logscale=False, colors=False, line=False):
    """ Plot degree distribution for different configuration"""
    x, y, yerr = P

    c = next(_colors) if colors else 'b'
    m = next(_markers) if colors else 'o'
    l = '--' if line else None

    if yerr is None:
        plt.scatter(x, y, c=c, marker=m)
        if line:
            plt.plot(x, y, c=c, marker=m, ls=l)
    else:
        plt.errorbar(x, y, yerr, c=c, fmt=m, ls=l)

    min_d, max_d = min(x), max(x)

    if logscale:
        plt.xscale('log'); plt.yscale('log')
        # Ensure that the ticks will be visbile (ie larger than in los step)
        #logspace = 10**np.arange(6)
        #lim =  np.searchsorted(logspace,min_d )
        #if lim == np.searchsorted(logspace,max_d ):
        #    min_d = logspace[lim-1]
        #    max_d = logspace[lim]

    plt.xlim((min_d, max_d+10))
    #plt.ylim((.9,1e3))
    plt.xlabel('Degree'); plt.ylabel('Counts')


def draw_graph(y, clusters='blue', ns=30):
    G = nxG(y)

    plt.figure()
    nx.draw_spring(G, cmap = plt.get_cmap('jet'), node_color = clusters, node_size=30, with_labels=False)

def draw_blocks(y, clusters='blue', ns=30):
    G = nxG(y)

     nx.draw(H,G.position,
             node_size=[G.population[v] for v in H],
             node_color=node_color,
             with_labels=False)

try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
except ImportError:
    try:
        import pydotplus
        from networkx.drawing.nx_pydot import graphviz_layout
    except ImportError:
        lgg.error("This example needs Graphviz and either "
                  "PyGraphviz or PyDotPlus")
def draw_graph_circular(y, clusters='blue', ns=30):
    G = nxG(y)
    pos = graphviz_layout(G, prog='twopi', args='')
    plt.figure()
    nx.draw(G, pos, node_size=ns, alpha=0.8, node_color=clusters, with_labels=False)
    plt.axis('equal')

def draw_graph_spectral(y, clusters='blue', ns=30):
    G = nxG(y)
    pos = graphviz_layout(G, prog='twopi', args='')
    plt.figure()
    nx.draw_spectral(G, cmap = plt.get_cmap('jet'), node_color = clusters, node_size=30, with_labels=False)

def adjmat(Y, title=''):
    plt.figure()
    plt.axis('off')
    plt.title('Adjacency matrix')
    plt.imshow(Y, cmap="Greys", interpolation="none", origin='upper')
    title = 'Adjacency matrix, N = %d\n%s' % (Y.shape[0], title)
    plt.title(title)

def adjshow(Y, cmap=None, pixelspervalue=20, minvalue=None, maxvalue=None, title='', ax=None):
        """ Make a colormap image of a matrix
        :key Y: the matrix to be used for the colormap.
        """
        # Artefact
        np.fill_diagonal(Y, 0)

        if minvalue == None:
            minvalue = np.amin(Y)
        if maxvalue == None:
            maxvalue = np.amax(Y)
        if not cmap:
            cmap = plt.cm.hot
            if not ax:
                #figsize = (np.array(Y.shape) / 100. * pixelspervalue)[::-1]
                #fig = plt.figure(figsize=figsize)
                #fig.set_size_inches(figsize)
                #plt.axes([0, 0, 1, 1]) # Make the plot occupy the whole canvas
                plt.axis('off')
                implot = plt.imshow(Y, cmap=cmap, clim=(minvalue, maxvalue), interpolation='nearest')
                plt.title(title)
            else:
                ax.axis('off')
                implot = ax.imshow(Y, cmap=cmap, clim=(minvalue, maxvalue), interpolation='nearest')
                plt.title(title)
        #plt.savefig(filename, fig=fig, facecolor='white', edgecolor='black')

def adjshow_l(Y,title=[], pixelspervalue=20):
        minvalue = np.amin(Y)
        maxvalue = np.amax(Y)
        cmap = plt.cm.hot

        fig = plt.figure()
        plt.subplot(1,2,1)
        plt.axis('off')
        implot = plt.imshow(Y[0], cmap=cmap, clim=(minvalue, maxvalue), interpolation='nearest')
        plt.title(title[0])

        plt.subplot(1,2,2)
        plt.axis('off')
        implot = plt.imshow(Y[1], cmap=cmap, clim=(minvalue, maxvalue), interpolation='nearest')
        plt.title(title[1])

        plt.draw()

def adjshow_ll(Y,title=[], pixelspervalue=20):
        minvalue = np.amin(Y)
        maxvalue = np.amax(Y)
        cmap = plt.cm.hot

        fig = plt.figure()
        plt.subplot(2,2,1)
        plt.axis('off')
        implot = plt.imshow(Y[0], cmap=cmap, clim=(minvalue, maxvalue), interpolation='nearest')
        plt.title(title[0])

        plt.subplot(2,2,2)
        plt.axis('off')
        implot = plt.imshow(Y[1], cmap=cmap, clim=(minvalue, maxvalue), interpolation='nearest')
        plt.title(title[1])

        plt.subplot(2,2,3)
        plt.axis('off')
        implot = plt.imshow(Y[2], cmap=cmap, clim=(minvalue, maxvalue), interpolation='nearest')
        plt.title(title[2])

        plt.subplot(2,2,4)
        plt.axis('off')
        implot = plt.imshow(Y[3], cmap=cmap, clim=(minvalue, maxvalue), interpolation='nearest')
        plt.title(title[3])

        plt.draw()

def plot_csv(target_dirs='', columns=0, sep=' ', separate=False, title=None, twin=False, iter_max=None):
    if type(columns) is not list:
        columns = [columns]

    if type(target_dirs) is not list:
        target_dirs = [target_dirs]

    title = title or 'Inference'
    xlabel = 'Iterations'

    prop_set = get_expe_file_set_prop(target_dirs)
    id_plot = 0
    old_prop = 0
    fig = None
    _Ks = None
    is_nonparam = False
    #if separate is False:
    #    fig = plt.figure()
    #    ax1 = fig.add_subplot(111)
    #    ax1.set_xlabel(xlabel)
    #    ax1.set_title(title)

    for i, target_dir in enumerate(target_dirs):

        filen = os.path.join(os.path.dirname(__file__), "../data/", target_dir)
        print 'plotting in %s ' % (filen, )
        with open(filen) as f:
            data = f.read()

        data = filter(None, data.split('\n'))
        if iter_max:
            data = data[:iter_max]
        data = [re.sub("\s\s+" , " ", x.strip()) for l,x in enumerate(data) if not x.startswith(('#', '%'))]
        #data = [x.strip() for x in data if not x.startswith(('#', '%'))]

        prop = get_expe_file_prop(target_dir)
        for column in columns:
            if type(column) is str:
                column = csv_row(column)

			### Optionnal row...?%
            if separate is False:
                if fig is None:
                    fig = plt.figure()
                plt.subplot(1,1,1)
                plt.title(title)
                plt.xlabel(xlabel)
                ax1 = plt.gca()
            elif separate is True:
                fig = plt.figure()
                plt.title(title)
                plt.xlabel(xlabel)
                ax1 = plt.gca()
            elif type(separate) is int:
                if i % (separate*2) == 0:
                    fig = plt.figure()
                    plt.subplot(1,2,1)
                if i % (separate*2) == 2:
                    plt.subplot(1,2,2)
                plt.title(title)
                plt.xlabel(xlabel)
                ax1 = plt.gca()
            elif type(separate) is str:
                if fig is None:
                    fig = plt.figure()
                if prop['N'] != old_prop:
                    id_plot += 1
                old_prop = prop['N']
                plt.subplot(1,prop_set['N'],id_plot)
                plt.title(title)
                plt.xlabel(xlabel)
                ax1 = plt.gca()

            ll_y = [row.split(sep)[column] for row in data]
            ll_y = np.ma.masked_invalid(np.array(ll_y, dtype='float'))

            Ks = [int(float(row.split(sep)[csv_row('K')])) for row in data]
            Ks = np.ma.masked_invalid(np.array(Ks, dtype='int'))

            ylabel, label = tag_from_csv(column)
            ax1.set_ylabel(ylabel)

            model_label = target_dir.split('/')[-1][len('inference-'):]
            #label = target_dir + ' ' + label
            label = target_dir.split('/')[-3] +' '+ model_label
            if model_label.startswith(('ilda', 'immsb')):
                is_nonparam = True
                label += ' K -> %d' % (float(Ks[-1]))

            ax1.plot(ll_y, marker=next(_markers), label=label)
            leg = ax1.legend(loc=1,prop={'size':10})

            if not twin:
                continue
            if is_nonparam or _Ks is None:
                _Ks = Ks
            Ks = _Ks
            ax2 = ax1.twinx()
            ax2.plot(Ks, marker='*')

        #plt.savefig('../results/debug1/%s.pdf' % (prop['corpus']))
    plt.draw()

if __name__ ==  '__main__':
    block = True

    display(block)

