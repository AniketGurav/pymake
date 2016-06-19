#!/usr/bin/python -u
# -*- coding: utf-8 -*-

from random import choice
from local_utils import *
from vocabulary import Vocabulary, parse_corpus
from util.frontend import ModelManager, FrontendManager
from util.frontendnetwork import frontendNetwork
from plot import *
from util.frontend_io import *

import numpy as np
import scipy as sp
#np.set_printoptions(threshold='nan')

import logging
import os
import os.path

_USAGE = '''?'''

####################################################
### Config
config = defaultdict(lambda: False, dict(
    ##### Global settings
    verbose     = 0,
    ##### Input Features / Corpus
    limit_train = None,
    ###### I/O settings
    bdir = '../data',
    generative = 'predictive',
    K = 10,
    gen_size = 10000,
    epoch = 20
))
config.update(argParse(_USAGE))

### Bursty
#corpuses = ( 'generator3', 'generator11', 'generator12', 'generator7', 'generator14',)

### Non Bursty
#corpuses = ( 'generator4', 'generator5', 'generator6', 'generator9', 'generator10',)

### Expe Spec

Corpuses = (('manufacturing', '| Manufacturing', 'manufacturing'),
            ('fb_uc', '| UC Irvine', 'irvine' )
           )

Corpuses = (('generator4', ' | Network 4', 'g4'),
            ('generator10',' | Network 3', 'g3'),
            ('generator12',' | Network 2', 'g2'),
            ('generator7' ,' | Network 1 ', 'g1'),
            )

### Models
Models = [
        dict ((
            ('debug'        , 'debug11') ,
            ('model'        , 'ibp')   ,
            ('K'            , 10)        ,
            ('N'            , 'all')     ,
            ('hyper'        , 'fix')     ,
            ('homo'         , 0)         ,
            #('repeat'      , '*')       ,
            )),
        [ dict ((
            ('debug'        , 'debug11') ,
            ('model'        , 'immsb')   ,
            ('K'            , 10)        ,
            ('N'            , 'all')     ,
            ('hyper'        , 'auto')     ,
            ('homo'         , 0)         ,
            #('repeat'      , '*')       ,
            ))
            ]

if config.get('arg'):
    try:
        Models =  [get_conf_from_file(config['arg'])]
    except:
        Models = [None]
        pass

############################################################
##### Simulation Output
if config.get('simul'):
    print '''--- Simulation settings ---
    Model : %s
    Corpus : %s
    K : %s
    N : %s
    hyper : %s
    Output : %s''' % (config['model'], config['corpus_name'],
                     config['K'], config['N'], config['hyper'],
                     config['output_path'])
    exit()

for corpus_ in Corpuses:
    for Model in Models:
        #### Expe ID
        path = '../../../papers/personal/relational_models/git/img/'
        model_name = Model['model']
        corpus_name = corpus_[0]
        title = model_name + corpus_[1]
        fn = model_name +'_'+corpus_[2]

        ### Initializa Model
        frontend = frontendNetwork(config)
        data = frontend.load_data(corpus_name)
        data = frontend.sample()
        model = ModelManager(config=config)

        Y = []
        N = config['gen_size']
        ### Setup Models
        if config['generative'] == 'predictive':
            ### Generate data from a fitted model
            Model.update(corpus=corpus_name)
            model = model.load(Model)
        elif config['generative'] == 'evidence':
            ### Generate data from a un-fitted model
            model = model.model
            alpha = .5
            gmma = 1.
            delta = .1
            if Model['model'] == 'ibp':
                hyper = (alpha, delta)
            elif Model['model'] == 'immsb':
                hyper = (alpha, gmma, delta)
            else:
                raise NotImplementedError
            model.update_hyper(hyper)
        else:
            raise NotImplementedError

        for i in range(config.get('epoch',1)):
            y, theta, phi = model.generate(N, config['K'], _type=config['generative'])
            Y += [y]

        ### Baselines
        #R = rescal(data, config['K'])
        R = None

        K = theta.shape[1]
        if frontend.is_symmetric():
            frontend.symmetrize(y)
            frontend.symmetrize(R)
        ###############################################################
        ### Expe Wrap up debug
        print 'corpus: %s, model: %s, K = %s, N =  %s'.replace(',','\n') % (frontend.corpus_name, model.model_name, model.K, frontend.N)

        #################################################
        ### Plotting
        ### Plot Degree
        figsize=(3.8, 4.3)
        #figsize=(3.3, 4.3)
        plt.figure(figsize=figsize)
        plot_degree_2_l(y)
        plt.title(title)

        plt.savefig(path+fn+'_d'+'.pdf', facecolor='white', edgecolor='black')
#    display(False)
#display(True)


