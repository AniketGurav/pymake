#!/usr/bin/python -u
# -*- coding: utf-8 -*-

import logging
import os.path

from random import choice
import itertools

import utils.algo as A
from utils.algo import gofit
from utils.utils import *
from utils.math import *
from expe.spec import _spec_; _spec = _spec_()

from plot import *
from plot import _markers, _colors

from tabulate import tabulate
import numpy as np
import scipy as sp
#np.set_printoptions(threshold='nan')
from sklearn.metrics import roc_curve, auc, precision_recall_curve



""" **kwargs is passed to the format function.
    The attributes curently in used in the globals dict are:
    * model_name (str)
    * corpus_name (str)
    * model (the model [ModelBase]
    * y (the data [Frontend])
    etc..
"""


def savefig_debug(**kwargs):
    # @Debug: does not make the variable accessible
    #in the current scope.
    globals().update(kwargs)
    path = '../results/networks/generate/'

    #################################################
    ### Plot Degree
    figsize=(3.8, 4.3)
    plt.figure(figsize=figsize)
    plot_degree_2_l(Y)
    plot_degree_poly(data, scatter=False)
    plt.title(title)

    #fn = path+fn+'_d_'+ globals()['K'] +'.pdf'
    fn = os.path.join(path, '%s_d_%s.pdf' % (fn, globals()['K']))
    print('saving %s' % fn)
    plt.savefig(fn, facecolor='white', edgecolor='black')

    return

def zipf(**kwargs):
    """ Local/Global Preferential attachment effect analysis """
    globals().update(kwargs)
    y = kwargs['y']
    N = y.shape[0]
    if Model['model'] == 'ibp':
        title = 'N=%s, K=%s alpha=%s, lambda:%s'% ( N, K, alpha, delta)
    elif Model['model'] == 'immsb':
        title = 'N=%s, K=%s alpha=%s, gamma=%s, lambda:%s'% (N, K, alpha, gmma, delta)
    elif Model['model'] == 'mmsb_cgs':
        title = 'N=%s, K=%s alpha=%s, lambda:%s'% ( N, K, alpha, delta)
    else:
        raise NotImplementedError

    ##############################
    ### Global degree
    ##############################
    #plot_degree_poly_l(Y)
    #plot_degree_poly(data, scatter=False)
    d, dc, yerr = random_degree(Y)
    god = gofit(d, dc)
    #plt.figure()
    #plot_degree_2((d,dc,yerr))
    #plt.title(title)
    plt.figure()
    plot_degree_2((d,dc,yerr), logscale=True)
    #plot_degree_2((d,dc,None), logscale=True)
    plt.title(title)

    if False:
        ### Just the gloabl degree.
        return

    print 'Computing Local Preferential attachment'
    ##############################
    ### Z assignement method
    ##############################
    now = Now()
    if model_name == 'immsb':
        ZZ = []
        #for _ in [Y[0]]:
        for _ in Y[:5]: # Do not reflect real local degree !
            Z = np.empty((2,N,N))
            order = np.arange(N**2).reshape((N,N))
            if frontend.is_symmetric():
                triu = np.triu_indices(N)
                order = order[triu]
            else:
                order = order.flatten()
            order = zip(*np.unravel_index(order, (N,N)))

            for i,j in order:
                Z[0, i,j] = categorical(theta[i])
                Z[1, i,j] = categorical(theta[j])
            Z[0] = np.triu(Z[0]) + np.triu(Z[0], 1).T
            Z[1] = np.triu(Z[1]) + np.triu(Z[1], 1).T
            ZZ.append( Z )
        ellapsed_time('Z formation', now)

    ##############################
    ### Plot all local degree
    ##############################
    plt.figure()
    # **max_assignement** evalutaion gives the degree concentration
    # for all clusters, when counting for
    # all the interaction for all other classess.
    # **modularity** counts only degree for interaction between two classes.
    # It appears that the modularity case concentration, correspond
    # to the interactions of concentration
    # of the maxèassignement case.
    #Clustering = ['modularity', 'max_assignement']
    clustering = 'modularity'
    comm = model.communities_analysis(data=Y[0], clustering=clustering)
    print 'clustering method: %s, active clusters ratio: %f' % (clustering, len(comm['block_hist']>0)/float(theta.shape[1]))

    local_degree_c = {}
    ### Iterate over all classes couple
    if frontend.is_symmetric():
        #k_perm = np.unique( map(list, map(set, itertools.product(np.unique(clusters) , repeat=2))))
        k_perm =  np.unique(map(list, map(list, map(set, itertools.product(range(theta.shape[1]) , repeat=2)))))
    else:
        #k_perm = itertools.product(np.unique(clusters) , repeat=2)
        k_perm = itertools.product(range(theta.shape[1]) , repeat=2)
    for i, c in enumerate(k_perm):
        if i > 10:
            break
        if len(c) == 2:
            # Stochastic Equivalence (extra class bind
            k, l = c
            #continue
        else:
            # Comunnities (intra class bind)
            k = l = c.pop()

        degree_c = []
        YY = []
        if model_name == 'immsb':
            for y, z in zip(Y, ZZ): # take the len of ZZ if < Y
                y_c = y.copy()
                phi_c = np.zeros(y.shape)
                # UNDIRECTED !
                phi_c[(z[0] == k) & (z[1] == l)] = 1 #; phi_c[(z[0] == l) & (z[1] == k)] = 1
                y_c[phi_c != 1] = 0
                #degree_c += adj_to_degree(y_c).values()
                #yerr= None
                YY.append(y_c)
        elif model_name == 'ibp':
            for y in Y:
                YY.append((y == np.outer(theta[:,k], theta[:,l])).astype(int))

        ## remove ,old issue
        #if len(degree_c) == 0: continue
        #d, dc = degree_hist(degree_c)

        d, dc, yerr = random_degree(YY)
        if len(dc) == 0: continue
        #local_degree_c[str(k)+str(l)] = filter(lambda x: x != 0, degree_c)
        god =  gofit(d, dc)
        plot_degree_2((d,dc,yerr), logscale=True, colors=True, line=True)
    plt.title('Local Preferential attachment (Stochastic Block)')


    ##############################
    ### Blockmodel Analysis
    ##############################
    # Class Ties

    #plt.figure()
    ##local_degree = comm['local_degree']
    ##local_degree = local_degree_c # strong concentration on degree 1 !
    #label, hist = zip(*model.blockmodel_ties(Y[0]))
    #bins = len(hist)
    #plt.bar(range(bins), hist)
    #label_tick = lambda t : '-'.join(t)
    #plt.xticks(np.arange(bins)+0.5, map(label_tick, label))
    #plt.tick_params(labelsize=5)
    #plt.xlabel('Class Interactions')
    #plt.title('Weighted Harmonic mean of class interactions ties')


    if model_name == "immsb":

        # Class burstiness
        plt.figure()
        hist, label = clusters_hist(comm['clusters'])
        bins = len(hist)
        plt.bar(range(bins), hist)
        plt.xticks(np.arange(bins)+0.5, label)
        plt.xlabel('Class labels')
        plt.title('Blocks Size (max assignement)')
    elif model_name == "ibp":
        # Class burstiness
        plt.figure()
        hist, label = sorted_perm(comm['block_hist'], reverse=True)
        bins = len(hist)
        plt.bar(range(bins), hist)
        plt.xticks(np.arange(bins)+0.5, label)
        plt.xlabel('Class labels')
        plt.title('Blocks Size (max assignement)')


    #draw_graph_spring(y, clusters)
    #draw_graph_spectral(y, clusters)
    #draw_graph_circular(y, clusters)
    #adjshow(y, title='Adjacency Matrix')

    #adjblocks(y, clusters=comm['clusters'], title='Blockmodels of Adjacency matrix')
    #draw_blocks(comm)

    print 'density: %s' % (float(y.sum()) / (N**2))


def pvalue(**kwargs):
    """ similar to zipf but compute pvalue and print table
        Parameters
        ==========
        type: pvalue type in (global, local, feature)

    """
    globals().update(kwargs)
    _type = kwargs.get('_type', 'global')
    y = kwargs['y']
    N = y.shape[0]

    if _type == 'global':
        try:
            Table
            print '11111111111'
        except NameError:
            Meas = [ 'pvalue', 'alpha', 'x_min', 'n_tail']; headers = Meas
            Table = np.empty((len(Corpuses), len(Meas), len(Y)))
            print '22222222222'

        ### Global degree
        d, dc, yerr = random_degree(Y)
        for it_dat, data in enumerate(Y):
            d, dc = degree_hist(adj_to_degree(data))
            gof = gofit(d, dc)


            for i, v in enumerate(Meas):
                Table[corpus_pos, i, it_dat] = gof[v]

        print Table

    elif _type == 'local':
        ### Z assignement method
        now = Now()
        if model_name == 'immsb':
            ZZ = []
            #for _ in [Y[0]]:
            for _ in Y[:5]: # Do not reflect real local degree !
                Z = np.empty((2,N,N))
                order = np.arange(N**2).reshape((N,N))
                if frontend.is_symmetric():
                    triu = np.triu_indices(N)
                    order = order[triu]
                else:
                    order = order.flatten()
                order = zip(*np.unravel_index(order, (N,N)))

                for i,j in order:
                    Z[0, i,j] = categorical(theta[i])
                    Z[1, i,j] = categorical(theta[j])
                Z[0] = np.triu(Z[0]) + np.triu(Z[0], 1).T
                Z[1] = np.triu(Z[1]) + np.triu(Z[1], 1).T
                ZZ.append( Z )
            ellapsed_time('Z formation', now)

        clustering = 'modularity'
        comm = model.communities_analysis(data=Y[0], clustering=clustering)
        print 'clustering method: %s, active clusters ratio: %f' % (clustering, len(comm['block_hist']>0)/float(theta.shape[1]))

        local_degree_c = {}
        ### Iterate over all classes couple
        if frontend.is_symmetric():
            #k_perm = np.unique( map(list, map(set, itertools.product(np.unique(clusters) , repeat=2))))
            k_perm =  np.unique(map(list, map(list, map(set, itertools.product(range(theta.shape[1]) , repeat=2)))))
        else:
            #k_perm = itertools.product(np.unique(clusters) , repeat=2)
            k_perm = itertools.product(range(theta.shape[1]) , repeat=2)
        for i, c in enumerate(k_perm):
            if i > 10:
                break
            if len(c) == 2:
                # Stochastic Equivalence (extra class bind
                k, l = c
                #continue
            else:
                # Comunnities (intra class bind)
                k = l = c.pop()

            degree_c = []
            YY = []
            if model_name == 'immsb':
                for y, z in zip(Y, ZZ): # take the len of ZZ if < Y
                    y_c = y.copy()
                    phi_c = np.zeros(y.shape)
                    # UNDIRECTED !
                    phi_c[(z[0] == k) & (z[1] == l)] = 1 #; phi_c[(z[0] == l) & (z[1] == k)] = 1
                    y_c[phi_c != 1] = 0
                    #degree_c += adj_to_degree(y_c).values()
                    #yerr= None
                    YY.append(y_c)
            elif model_name == 'ibp':
                for y in Y:
                    YY.append((y == np.outer(theta[:,k], theta[:,l])).astype(int))

            ## remove ,old issue
            #if len(degree_c) == 0: continue
            #d, dc = degree_hist(degree_c)

            d, dc, yerr = random_degree(YY)
            if len(dc) == 0: continue
            #local_degree_c[str(k)+str(l)] = filter(lambda x: x != 0, degree_c)
            god =  gofit(d, dc)
            plot_degree_2((d,dc,yerr), logscale=True, colors=True, line=True)

    elif _type == "feature":
        raise NotImplementedError
        ### Blockmodel Analysis
        if model_name == "immsb":
            # Class burstiness
            hist, label = clusters_hist(comm['clusters'])
            bins = len(hist)
        elif model_name == "ibp":
            # Class burstiness
            hist, label = sorted_perm(comm['block_hist'], reverse=True)
            bins = len(hist)
    else:
        raise NotImplementedError


    ### Table Format Printing
    if _end is True:

        # Function in (utils. ?)
        table_mean = np.char.array(np.around(Table.mean(2), decimals=3)).astype("|S20")
        table_std = np.char.array(np.around(Table.std(2), decimals=3)).astype("|S20")
        Table = table_mean + ' p2m ' + table_std

        Table = np.column_stack((_spec.name(Corpuses), Table))
        tablefmt = 'latex' # 'latex'
        print
        print tabulate(Table, headers=headers, tablefmt=tablefmt, floatfmt='.3f')


def debug(**kwargs):
    y = kwargs['y']
    model = kwargs['model']

    clustering = 'modularity'
    comm = model.communities_analysis(data=y, clustering=clustering)

    clusters = comm['clusters']

    #y, l = reorder_mat(y, clusters, labels=True)
    #clusters = clusters[l]

    #adjblocks(y, clusters=clusters, title='Blockmodels of Adjacency matrix')
    #adjshow(reorder_mat(y, comm['clusters']), 'test reordering')

    draw_graph_circular(y, clusters)


def roc_test(**kwargs):
    globals().update(kwargs)
    y_true, probas = model.mask_probas(data)
    fpr, tpr, thresholds = roc_curve(y_true, probas)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC %s (area = %0.2f)' % (_spec.name(model_name), roc_auc))

    #precision, recall, thresholds = precision_recall_curve( y_true, probas)
    #plt.plot(precision, recall, label='PR curve; %s' % (model_name ))

def perplexity(**kwargs):
    globals().update(kwargs)

    data = model.load_some()
    burnin = 5
    sep = ' '
    # Test perplexity not done for masked data. Usefull ?!
    #column = csv_row('likelihood_t')
    column = csv_row('likelihood')
    ll_y = [row.split(sep)[column] for row in data][5:]
    ll_y = np.ma.masked_invalid(np.array(ll_y, dtype='float'))
    plt.plot(ll_y, label=_spec.name(model_name))

_algo = 'Louvain'
_algo = 'Annealing'
def clustering(algo=_algo, **kwargs):
    globals().update(kwargs)

    mat = data
    #mat = phi

    alg = getattr(A, algo)(mat)
    clusters = alg.search()

    mat = draw_boundary(alg.hi_phi(), alg.B)
    #mat = draw_boundary(mat, clusters)

    adjshow(mat, algo)
    plt.colorbar()
