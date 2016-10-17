#!/usr/bin/python -u
# -*- coding: utf-8 -*-

from random import choice
from utils.utils import *
from utils.math import *
from plot import *

import numpy as np
import scipy as sp
#np.set_printoptions(threshold='nan')

import logging
import os
import os.path

#path = '../../../papers/personal/relational_models/git/img/'
path = '../results/networks/generate/'

### Expe Spec
corpus_ = dict((
    ('manufacturing'  , ('Manufacturing', 'manufacturing')),
    ('fb_uc'          , ('UC Irvine', 'irvine')),
    ('generator4'     , ('Network 4', 'g4')),
    ('generator10'    , ('Network 3', 'g3')),
    ('generator12'    , ('Network 2', 'g2')),
    ('generator7'     , ('Network 1', 'g1')),
))

model_ = dict(ibp = 'ilfm',
              immsb = 'immsb')

def generate_icdm(**kwargs):
    # @Debug: does not make the variable accessible
    #in the current scope.
    globals().update(kwargs)
    model_name = kwargs['model_name']
    corpus_name = kwargs['corpus_name']
    Y = kwargs['Y']

    #### Expe ID
    model_name = model_[model_name]
    title = model_name +' | '+ corpus_[corpus_name][0]
    fn = model_name +'_'+ corpus_[corpus_name][1]

    #################################################
    ### Plot Degree
    figsize=(3.8, 4.3)
    plt.figure(figsize=figsize)
    plot_degree_2_l(Y)
    plot_degree_poly(data, scatter=False)
    plt.title(title)

    fn = path+fn+'_d'+'.pdf'
    print('saving %s' % fn)
    plt.savefig(fn, facecolor='white', edgecolor='black')

    return


### Expe
corpus_ = dict ((
    ('generator4'     , ('Network 4 -b/h'  , 'g4'))            ,
    ('generator10'    , ('Network 3 -b/-h' , 'g3'))            ,
    ('generator12'    , ('Network 2 b/-h'  , 'g2'))            ,
    ('generator7'     , ('Network 1  b/h ' , 'g1'))            ,
    ('manufacturing'  , ('Manufacturing'   , 'manufacturing')) ,
    ('fb_uc'          , ('UC Irvine'       , 'irvine' ))       ,
))

def corpus_icdm(**kwargs):
    globals().update(kwargs)
    corpus_name = kwargs['corpus_name']
    data = kwargs['data']

    #### Expe ID
    title = corpus_[corpus_name][0]
    fn = corpus_[corpus_name][1]

    #################################################
    ### Plotting

    ### Filtering
    if fn != 'manufacturing':
        ddata = dilate(data)
    else:
        ddata = data

    ### Plot Adjacency matrix
    figsize=(4.7, 4.7)
    plt.figure(figsize=figsize)
    adjshow(ddata, title=title)
    #plt.figtext(.15, .1, homo_text, fontsize=12)
    plt.savefig(path+fn+'.pdf', facecolor='white', edgecolor='black')

    ### Plot Degree
    figsize=(3.8, 4.3)
    plt.figure(figsize=figsize)
    plot_degree_poly(data)

    fn = path+fn+'_d'+'.pdf'
    plt.savefig(fn, facecolor='white', edgecolor='black')


def generate_icdm_debug(**kwargs):
    # @Debug: does not make the variable accessible
    #in the current scope.
    globals().update(kwargs)
    model_name = kwargs['model_name']
    corpus_name = kwargs['corpus_name']
    Y = kwargs['Y']

    #### Expe ID
    model_name = model_[model_name]
    title = model_name +' | '+ corpus_[corpus_name][0]
    fn = model_name +'_'+ corpus_[corpus_name][1]

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

def preferential_attachment(**kwargs):
    globals().update(kwargs)
    ##############################
    ### Global degree
    ##############################
    d, dc, yerr = random_degree(Y)
    god = gofit(d, dc)
    plt.figure()
    plot_degree_2((d,dc,yerr))
    plt.title(title)
    plt.figure()
    plot_degree_2((d,dc,None), logscale=True)
    plt.title(title)

    if False:
        ### Just the gloabl degree.
        return

    print 'Computing Local Preferential attachment'
    ##############################
    ### Z assignement method
    ##############################
    now = Now()
    ZZ = []
    for _ in [Y[0]]:
    #for _ in Y: # Do not reflect real loacal degree !
        Z = np.empty((2,N,N))
        order = np.arange(N**2).reshape((N,N))
        triu = np.triu_indices(N)
        order = order[triu]
        order = zip(*np.unravel_index(order, (N,N)))

        for i,j in order:
            Z[0, i,j] = categorical(theta[i])
            Z[1, i,j] = categorical(theta[j])
        Z[0] = np.triu(Z[0]) + np.triu(Z[0], 1).T
        Z[1] = np.triu(Z[1]) + np.triu(Z[1], 1).T
        ZZ.append( Z )
    ellapsed_time('Z formation', now)

    comm_distrib, local_degree, clusters = model.communities_analysis(theta, data=Y[0])

    ##############################
    ### Plot all local degree
    ##############################
    plt.figure()
    bursty_class = []
    ### Iterate over all classes couple
    for c in np.unique(map(set, itertools.product(range(len(comm_distrib)) , repeat=2))):
        if len(c) == 2:
            # Stochastic Equivalence (extra class bind
            k, l = c
            continue
        else:
            # Comunnities (intra class bind)
            l = c.pop()
            k = l

        degree_c = []
        for y, z in zip(Y, ZZ): # take the len of ZZ
            y_c = y.copy()
            z_c = z.copy()
            z_c[0][z_c[0] != k] = -1; z_c[1][z_c[1] != l] = -1
            y_c[z_c[0] == -1] = 0; y_c[z_c[1] == -1] = 0
            degree_c += adj_to_degree(y_c).values()

        d, dc = degree_hist(degree_c)
        if  len(dc) == 0: continue
        god =  gofit(d, dc)
        if god['pvalue'] > 0.1:
            bursty_class.append((d,dc, god))
        plot_degree_2((d,dc,None), logscale=True, colors=True, line=True)
    plt.title('Local Prefrential attachment (Stochastic Block)')
    return clusters

    ##############################
    #### Plot Bursty Class
    ##############################
    #for d,dc,god in bursty_class:
    #    plt.figure()
    #    plt.xscale('log'); plt.yscale('log')
    #    plt.scatter(d, dc, c=next(_colors), marker=next(_markers))
    #    d, dc = degree_hist(god['sync'])
    #    #d, dc = zip(*sorted(zip(d, dc)))
    #    #plt.scatter(d, dc, c=next(_colors), marker=next(_markers))

    ##############################
    ### Max cluster assignemet
    ##############################
    #deg_l = defaultdict(list)
    #for y in Y:
    #    comm_distrib, local_degree, clusters = model.communities_analysis(theta, data=y)
    #    deg_l = {key: value + deg_l[key] for key, value in local_degree.iteritems()}
    print 'active cluster (max assignement): %d' % len(comm_distrib)
    plt.figure()
    #plt.loglog( sorted(comm_distrib, reverse=True))
    for c in local_degree.values():
        d, dc = degree_hist(c)
        if  len(dc) == 0: continue
        plot_degree_2((d,dc,None), logscale=True, colors=True, line=True)
        plt.title('Local Prefrential attachment (Max Assignement)')
