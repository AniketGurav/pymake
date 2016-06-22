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

####################################################
### Config
config = defaultdict(lambda: False, dict(
    ##### Global settings
    ###### I/O settings
    generative = 'predictive',
    gen_size = 1000,
    epoch = 100
))
config.update(argParse())

### Bursty
#corpuses = ( 'generator3', 'generator11', 'generator12', 'generator7', 'generator14',)

### Non Bursty
#corpuses = ( 'generator4', 'generator5', 'generator6', 'generator9', 'generator10',)

### Expe Spec

Corpuses = ( 'manufacturing', 'fb_uc',)
Corpuses = ( 'generator4', 'generator10', 'generator12', 'generator7',)
Corpuses = ( 'generator10', )

### Models
Models = [ dict ((
('data_type'    , 'networks'),
('debug'        , 'debug10') ,
('model'        , 'ibp')   ,
('K'            , 10)        ,
('N'            , 'all')     ,
('hyper'        , 'auto')     ,
('homo'         , 0)         ,
#('repeat'      , '*')       ,
))]

if config.get('arg'):
    try:
        Models =  [get_conf_from_file(config['arg'])]
    except:
        Models = [None]
        pass

alpha = .01
gmma = 0.5
delta = 10
keys_hyper = ('alpha','gmma','delta')
hyper = (alpha, gmma, delta)
for corpus_name in Corpuses:
    for Model in Models:
        #### Expe ID
        path = '../../../papers/personal/relational_models/git/img/'
        model_name = Model['model']
        corpus_name = corpus_[0]
        title = model_name + corpus_[1]
        fn = model_name +'_'+corpus_[2]

        # Initializa Model
        frontend = frontendNetwork(config)
        data = frontend.load_data(corpus_name)
        data = frontend.sample()

        Y = []
        N = config['gen_size']
        ### Setup Models
        if config['generative'] == 'predictive':
            ### Generate data from a fitted model
            Model.update(corpus=corpus_name)
            model = ModelManager(config=config).load(Model)
            #model = model.load(Model)
        elif config['generative'] == 'evidence':
            ### Generate data from a un-fitted model
            if Model['model'] == 'ibp':
                keys_hyper = ('alpha','gmma','delta')
                hyper = (alpha, delta)
            Model['hyperparams'] = dict(zip(keys_hyper, hyper))
            Model['hyper'] = 'fix'
            #model.data = np.zeros((1,1))
            model = ModelManager(config=config).load(Model, init=True)
            #model.update_hyper(hyper)
        else:
            raise NotImplementedError

        for i in range(config.get('epoch',1)):
            y, theta, phi = model.generate(N, Model['K'], _type=config['generative'])
            Y += [y]

        ### Baselines
        #R = rescal(data, config['K'])
        R = None

        N = theta.shape[0]
        K = theta.shape[1]
        if frontend.is_symmetric():
            for y in Y:
                frontend.symmetrize(y)
                frontend.symmetrize(R)

        #################################################
        ### Plot Degree
        figsize=(3.8, 4.3)
        plt.figure(figsize=figsize)
        plot_degree_2_l(Y)
        plot_degree_2(data, scatter=False)

        fn = corpus_name+'.pdf'
        plt.savefig(path+fn+'_d'+'.pdf', facecolor='white', edgecolor='black')


