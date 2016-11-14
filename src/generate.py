#!/usr/bin/python -u
# -*- coding: utf-8 -*-

from frontend.frontend import ModelManager, FrontendManager
from frontend.frontendnetwork import frontendNetwork
from utils.utils import *
from utils.math import *
from utils.algo import Annealing
from plot import *
from frontend.frontend_io import *
from expe.spec import _spec_
from expe import format
from utils.argparser import argparser

from collections import Counter, defaultdict

USAGE = '''\
# Usage:
    generate [-w] [-k K] [-n N] [--[hypername]] [-g|-p]] [analysis]

-g: generative model (evidence)
-p: predicted data (model fitted)

analysis in [clustering, zipf, (to complete)]

# Examples
    parallel ./generate.py -w -k {}  ::: $(echo 5 10 15 20)
    ./generate.py --alpha 1 --gmma 1 -n 1000 --seed
'''

####################################################
### Config
config = defaultdict(lambda: False, dict(
    write_to_file = False,
    do            = 'zipf',
    generative    = 'evidence',
    #generative    = 'predictive',
    gen_size      = 1000,
    epoch         = 10 , #20
))
config.update(argparser.generate(USAGE))

alpha = 1
gmma = 1
delta = (0.8, 0.1)

# Corpuses
Corpuses = _spec_.CORPUS_REAL_ICDM_1
Corpuses = _spec_.CORPUS_SYN_ICDM_1
Corpuses = ('generator10',)

### Models
#Models = _spec_.MODELS_GENERATE
Models = [dict ((
    ('data_type'    , 'networks'),
    ('debug'        , 'debug11') , # ign in gen
    #('model'        , 'mmsb_cgs')   ,
    ('model'        , 'immsb')   ,
    ('K'            , 10)        ,
    ('N'            , 'all')     , # ign in gen
    ('hyper'        , 'auto')    , # ign in ge
    ('homo'         , 0)         , # ign in ge
    #('repeat'      , '*')       ,
))]

for m in Models:
    m['debug'] = 'debug11'

if config.get('K'):
    for m in Models:
        m['K'] = config['K']

for opt in ('alpha','gmma', 'delta'):
    if config.get(opt):
        globals()[opt] = config[opt]

for corpus_name in Corpuses:
    frontend = frontendNetwork(config)
    data = frontend.load_data(corpus_name)
    data = frontend.sample()
    for Model in Models:

        ###################################
        ### Setup Models
        ###################################
        if config['generative'] == 'predictive':
            ### Generate data from a fitted model
            Model.update(corpus=corpus_name)
            model = ModelManager(config=config).load(Model)
            #model = model.load(Model)
            Model['hyperparams'] = model.get_hyper()
            N = data.shape[0]
        elif config['generative'] == 'evidence':
            N = config['gen_size']
            ### Generate data from a un-fitted model
            if Model['model'] == 'ibp':
                keys_hyper = ('alpha','delta')
                hyper = (alpha, delta)
            else:
                keys_hyper = ('alpha','gmma','delta')
                hyper = (alpha, gmma, delta)
            Model['hyperparams'] = dict(zip(keys_hyper, hyper))
            Model['hyper'] = 'fix' # dummy
            model = ModelManager(config=config).load(Model, init=True)
            #model.update_hyper(hyper)
        else:
            raise NotImplementedError('What generation context ? evidence/generative..')

        if model is None:
            continue

        ###################################
        ### Generate data
        ###################################
        ### Defaut random graph (Evidence), is directed
        y, theta, phi = model.generate(N, Model['K'], _type=config['generative'])
        Y = [y]
        for i in range(config.get('epoch',1)-1):
            ### Mean and var on the networks generated
            pij = model.link_expectation(theta, phi)
            pij = np.clip(model.link_expectation(theta, phi), 0, 1)
            Y += [sp.stats.bernoulli.rvs(pij)]
            ### Mean and variance  on the model generated
            #y, theta, phi = model.generate(N, Model['K'], _type=config['generative'])
            #Y += [y]
        #y = data
        #Y = [y]

        ### @TODO: Baselines / put in args input.
        #R = rescal(data, config['K'])
        R = None

        N = theta.shape[0]
        K = theta.shape[1]
        if frontend.is_symmetric():
            for y in Y:
                frontend.symmetrize(y)
                frontend.symmetrize(R)

        ###################################
        ### Expe Show Setup
        ###################################
        model_name = Model['model']
        model_hyper = Model['hyperparams']
        print 'Mode: %s' % config['generative']
        print('corpus: %s, model: %s, K = %s, N =  %s, hyper: %s'.replace(',','\n') % (corpus_name, model_name, K, N, str(model_hyper)) )

        ###################################
        ### Visualize
        ###################################
        if config.get('write_to_file'):
            #format.generate_icdm(data=data, Y=Y, corpus_name=corpus_name, model_name=Model['model'])
            format.generate_icdm_debug(data=data, Y=Y, corpus_name=corpus_name, model_name=model_name, K=K)
            continue

        g = None
        if config['generative'] == 'predictive':
            y = data
        else:
            y = Y[0]

        analysis = getattr(format, config['do'])
        analysis(**globals())

        display(False)

if not config.get('write_to_file'):
    display(True)

