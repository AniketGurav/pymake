#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import re
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
    if (y == y.T).all():
        # Undirected Graph
        typeG = nx.Graph()
    else:
        # Directed Graph
        typeG = nx.DiGraph()
    G = nx.from_numpy_matrix(y, typeG)
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
    if (y == y.T).all():
        # Undirected Graph
        typeG = nx.Graph()
    else:
        # Directed Graph
        typeG = nx.DiGraph()
    G = nx.from_numpy_matrix(y, typeG)
    degree = sorted(nx.degree(G).values(), reverse=True)
    x = np.arange(1, y.shape[0] + 1)
    plt.loglog(x, degree)
    if title:
        plt.title(title)

def plot_degree_3(y):
    if len(y) > 6000:
        return
    if (y == y.T).all():
        # Undirected Graph
        typeG = nx.Graph()
    else:
        # Directed Graph
        typeG = nx.DiGraph()
    G = nx.from_numpy_matrix(y, typeG)
    degree = sorted(nx.degree(G).values(), reverse=True)
    degree, _ = np.histogram(degree ,bins=len(degree), density=True)
    x = np.arange(1, y.shape[0] + 1)
    plt.loglog(x, degree)
    if title:
        plt.title(title)

def drop_zeros(a_list):
    return [i for i in a_list if i>0]

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
def plot_degree_2(y, ax=None, scatter=True):
    # To convert normalized degrees to raw degrees
    #ba_c = {k:int(v*(len(ba_g)-1)) for k,v in ba_c.iteritems()}
    if (y == y.T).all():
        # Undirected Graph
        typeG = nx.Graph()
    else:
        # Directed Graph
        typeG = nx.DiGraph()
    G = nx.from_numpy_matrix(y, typeG)
    #degree = sorted(nx.degree(G).values(), reverse=True)

    #ba_c = nx.degree_centrality(G)
    ba_c = nx.degree(G)
    ba_c2 = dict(Counter(ba_c.values()))

    #ba_x,ba_y = log_binning(ba_c2,50)
    x = np.array(ba_c2.keys())
    y = np.array(ba_c2.values())

    if x[0] == 0:
        print '%d unconnected vertex' % y[0]
        x = x[1:]
        y = y[1:]

    #plt.scatter(ba_x,ba_y,c='r',marker='s',s=50)
    plt.xscale('log')
    plt.yscale('log')

    fit = np.polyfit(np.log(x), np.log(y), deg=1)
    plt.plot(x,np.exp(fit[0] *np.log(x) + fit[1]), 'g--', label='power %.2f' % fit[1])
    leg = plt.legend(loc=1,prop={'size':10})

    if scatter:
        plt.scatter(x,y,c='b',marker='o')

    #plt.xlim((1,1e4))
    plt.ylim((.9,1e3))
    plt.xlabel('Degree')
    #plt.ylabel('Counts of degree')
    #plt.show()

def plot_degree_2_l(Y, ax=None):
    _X = []
    _Y = []
    size = []
    for y in Y:
        # To convert normalized degrees to raw degrees
        #ba_c = {k:int(v*(len(ba_g)-1)) for k,v in ba_c.iteritems()}
        if (y == y.T).all():
            # Undirected Graph
            typeG = nx.Graph()
        else:
            # Directed Graph
            typeG = nx.DiGraph()
        G = nx.from_numpy_matrix(y, typeG)
        #degree = sorted(nx.degree(G).values(), reverse=True)

        #ba_c = nx.degree_centrality(G)
        ba_c = nx.degree(G)
        ba_c2 = dict(Counter(ba_c.values()))

        #ba_x,ba_y = log_binning(ba_c2,50)
        _X.append(ba_c2.keys())
        _Y.append(ba_c2.values())
        # Remove degree 0
        if _X[-1][0] == 0:
            _X[-1] = _X[-1][1:]
            _Y[-1] = _Y[-1][1:]
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

    #plt.scatter(ba_x,ba_y,c='r',marker='s',s=50)
    plt.xscale('log')
    plt.yscale('log')

    fit = np.polyfit(np.log(x), np.log(y), deg=1)
    plt.plot(x,np.exp(fit[0] *np.log(x) + fit[1]), 'm:', label='model power %.2f' % fit[1])
    leg = plt.legend(loc=1,prop={'size':10})

    plt.errorbar(x, y, yerr=yerr, fmt='o')
    #plt.scatter(_X, _Y, marker='o')

    #plt.xlim((1,1e4))
    plt.ylim((.9,1e3))
    plt.xlabel('Degree')
    #plt.ylabel('Counts of degree')
    #plt.show()

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
    markers = cycle([ '+', '*', '|','x', 'o', '.', '1', 'p', '<', '>', 's' ])
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

            ax1.plot(ll_y, marker=next(markers), label=label)
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

def basic_plot():

    columns = 1
    targets = ['text/nips12/debug/inference-ilda_10_auto_100',
               'text/nips12/debug/inference-lda_cgs_1_auto_100',
               'text/nips12/debug/inference-lda_cgs_2_auto_100',
               'text/nips12/debug/inference-lda_cgs_5_auto_100',
               'text/nips12/debug/inference-lda_cgs_10_auto_100000000', ]
    plot_csv(targets, columns, separate=False)

def complex_plot(spec):
	sep = 'corpus'
	separate = 'N'
	targets = make_path(spec, sep=sep)
	json_extract(targets)
	if sep:
		for t in targets:
			plot_csv(t, spec['columns'], separate=separate, twin=False, iter_max=spec['iter_max'])
	else:
		plot_csv(targets, spec['columns'], separate=True, iter_max=spec['iter_max'])

def make_path(spec, sep=None, ):
    targets = []
    if sep:
        tt = []
    for base in spec['base']:
        for hook in spec['hook_dir']:
            for c in spec['corpus']:
                p = os.path.join(base, c, hook)
                for n in spec['Ns']:
                    for m in spec['models']:
                        for k in spec['Ks']:
                            for h in spec['hyper']:
                                for hm in spec['homo']:
                                    t = 'inference-%s_%s_%s_%s_%s' % (m, k, h, hm,  n)
                                    t = os.path.join(p, t)
                                    filen = os.path.join(os.path.dirname(__file__), "../data/", t)
                                    if not os.path.isfile(filen) or os.stat(filen).st_size == 0:
                                        continue
                                    if sum(1 for line in open(filen)) <= 1:
                                        # empy file
                                        continue
                                    targets.append(t)

                if sep == 'corpus' and targets:
                    tt.append(targets)
                    targets = []

    if sep:
        return tt
    else:
        return targets

# Return dictionary of property for an expe file. (format inference-model_K_hyper_N)
def get_expe_file_prop(target):
    _id = target.split('_')
    model = ''
    st = 0
    for s in _id:
        try:
            int(s)
            break
        except:
            st += 1
            model += s

    _id = _id[st:]
    prop = dict(
        corpus = target.split('/')[-3],
        model = model.split('-')[-1],
        K     = _id[0],
        hyper = _id[1],
        homo = _id[2],
        N     = _id[3],)
    return prop

# Return size of proportie in a list if expe files
def get_expe_file_set_prop(targets):
    c = []
    for t in targets:
        c.append(get_expe_file_prop(t))

    sets = {}
    for p in ('N', 'K'):
        sets[p] = len(set([ _p[p] for _p in c ]))

    return sets

def json_extract(targets):
    l = []
    for t in targets:
        for _f in t:
            f = os.path.join(os.path.dirname(__file__), "../data/", _f) + '.json'
            d = os.path.dirname(f)
            corpus_type = ('/').join(d.split('/')[-2:])
            f = os.path.basename(f)[len('inference-'):]
            fn = os.path.join(d, f)
            try:
                d = json.load(open(fn,'r'))
                l.append(d)
                density = d['density'] # excepte try density_all
                mask_density = d['mask_density']
                #print density
                #print mask_density
                precision = d['Precision']
                rappel = d['Rappel']
                K = len(d['Local_Attachment'])
                h_s = d.get('homo_ind1_source', np.inf)
                h_l = d.get('homo_ind1_learn', np.inf)
                nmi = d.get('NMI', np.inf)
                print '%s %s; \t K=%s,  global precision: %.3f, local precision: %.3f, rappel: %.3f, homsim s/l: %.3f / %.3f, NMI: %.3f' % (corpus_type, f, K, d.get('g_precision'), precision, rappel, h_s, h_l, nmi )
            except Exception, e:
                print e
                pass

    print
    if len(l) == 1:
        return l[0]
    else:
        return l


if __name__ ==  '__main__':
    block = True
    conf = argParse()

    spec = dict(
        base = ['networks'],
        hook_dir = ['debug5/'],
        #corpus   = ['kos', 'nips12', 'nips', 'reuter50', '20ngroups'],
        #corpus   = ['generator/Graph1', 'generator/Graph2', 'clique3'],
        #corpus   = ['generator/Graph3', 'generator/Graph4'],
        corpus   = ['generator/Graph4', 'generator/Graph10', 'generator/Graph12', 'generator/Graph13'],
        columns  = ['perplexity'],
        #models   = ['ibp', 'ibp_cgs'],
        #models   = ['ibp_cgs', 'immsb'],
        ##models   = ['immsb', 'mmsb_cgs'],
        models   = [ 'ibp', 'immsb'],
        #Ns       = [250, 1000, 'all'],
        Ns       = ['all',],
        #Ks       = [5, 10, 15, 20, 25, 30],
        Ks       = [5, 10],
        #Ks       = [5, 10, 30],
        #Ks       = [10],
        #homo     = [0,1,2],
        homo     = [0],
        hyper    = ['fix', 'auto'],
        #hyper    = ['auto'],
        iter_max = 500 ,
    )

    sep = 'corpus'
    separate = 'N'
    #separate = 'N' and False
    targets = make_path(spec, sep=sep)
    json_extract(targets)
    exit()
    if sep:
        for t in targets:
            plot_csv(t, spec['columns'], separate=separate, twin=False, iter_max=spec['iter_max'])
    else:
        plot_csv(targets, spec['columns'], separate=True, iter_max=spec['iter_max'])


    ### Basic Plots
    #basic_plot()

    display(block)

