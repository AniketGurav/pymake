#!/usr/bin/python -u
# -*- coding: utf-8 -*-

from random import choice
from utils.utils import *
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
    plot_degree_2(data, scatter=False)
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
    plot_degree_2(data)

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
    plot_degree_2(data, scatter=False)
    plt.title(title)

    #fn = path+fn+'_d_'+ globals()['K'] +'.pdf'
    fn = os.path.join(path, '%s_d_%s.pdf' % (fn, globals()['K']))
    print('saving %s' % fn)
    plt.savefig(fn, facecolor='white', edgecolor='black')

    return
