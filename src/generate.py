#!/usr/bin/python -u
# -*- coding: utf-8 -*-

from frontend.frontend import ModelManager, FrontendManager
from frontend.frontendnetwork import frontendNetwork
from utils.utils import *
from utils.math import *
from plot import *
from frontend.frontend_io import *
from expe.spec import _spec_
from expe.format import generate_icdm, generate_icdm_debug
from utils.argparser import argparser

from collections import Counter, defaultdict
import itertools

USAGE = '''\
# Usage:
    generate [-w] [-k K]

# Examples
    parallel ./generate.py -w -k {}  ::: $(echo 5 10 15 20)
'''

####################################################
### Config
config = defaultdict(lambda: False, dict(
    write_to_file = False,
    generative    = 'evidence',
    #generative    = 'predictive',
    gen_size      = 10,
    epoch         = 10 , #200
))
config.update(argparser.generate(USAGE))

alpha = 0.5
gmma = 0.5
delta = 0.1
delta = (1, 1)

# Corpuses
Corpuses = _spec_.CORPUS_REAL_ICDM_1
Corpuses = _spec_.CORPUS_SYN_ICDM_1
### Models
Models = _spec_.MODELS_GENERATE

Corpuses = ('generator7',)
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
            raise NotImplementedError

        if model is None:
            continue

        ### Defaut random graph (Evidence), is directed
        y, theta, phi = model.generate(N, Model['K'], _type=config['generative'])
        params = (theta, phi)
        Y = [y]
        for i in range(config.get('epoch',1)-1):
            ### Mean and var on the networks generated
            pij = model.link_expectation(theta, phi)
            pij = sp.stats.threshold(model.link_expectation(theta, phi), threshmax=1, newval=1)
            Y += [sp.stats.bernoulli.rvs(pij)]
            ### Mean and variance  on the model generated
            #y, theta, phi = model.generate(N, Model['K'], _type=config['generative'])
            #Y += [y]

        ### Baselines
        #R = rescal(data, config['K'])
        R = None

        N = theta.shape[0]
        K = theta.shape[1]
        if frontend.is_symmetric():
            for y in Y:
                frontend.symmetrize(y)
                frontend.symmetrize(R)

        ###############################################################
        ### Expe Wrap up debug
        model_name = Model['model']
        model_hyper = Model['hyperparams']
        if Model['model'] == 'ibp':
            title = '%s, N=%s, K=%s alpha=%s, lambda:%s'% (model_name, N, K, alpha, delta)
        elif Model['model'] == 'immsb':
            title = '%s, N=%s, K=%s alpha=%s, gamma=%s, lambda:%s'% (model_name, N, K, alpha, gmma, delta)
        elif Model['model'] == 'mmsb_cgs':
            title = '%s, N=%s, K=%s alpha=%s, lambda:%s'% (model_name, N, K, alpha, delta)
        else:
            raise NotImplementedError
        print('corpus: %s, model: %s, K = %s, N =  %s, hyper: %s'.replace(',','\n') % (corpus_name, model_name, K, N, str(model_hyper)) )

        #################################################
        ### Plot Degree
        from plot import _markers, _colors
        if config.get('write_to_file'):
            #generate_icdm(data=data, Y=Y, corpus_name=corpus_name, model_name=Model['model'])
            generate_icdm_debug(data=data, Y=Y, corpus_name=corpus_name, model_name=model_name, K=K)
            continue

        #plot_degree_(y, title='Overall Degree')
        if config['generative'] == 'predictive':
            plt.figure()
            plot_degree_2_l(Y)
            plot_degree_2(data, scatter=False)
            plt.title('%s on %s'% (Model['model'], corpus_name))
        elif config['generative'] == 'evidence':
            d, dc, yerr = random_degree(Y)
            print gofit(degree_hist_to_list(d, dc), d, dc)
            plt.figure()
            plot_degree_2_l_e((d,dc,yerr))
            plt.title(title)
            plt.figure()
            plot_degree_2_l_e((d,dc,yerr), logscale=True)
            plt.title(title)

            print 'Computing Local Preferential attachment'
            ### Z assignement method
            Z = np.empty((2,N,N))
            order = np.arange(N**2).reshape((N,N))
            triu = np.triu_indices(N, +1)
            order = order[triu]
            order = zip(*np.unravel_index(order, (N,N)))

            now = now()
            for i,j in order:
                Z[0, i,j] = categorical(theta[i])
                Z[1, i,j] = categorical(theta[j])
            ellapsed_time('Z formation', now)
            Z[0] = np.triu(Z[0]) + np.triu(Z[0], 1).T
            Z[1] = np.triu(Z[1]) + np.triu(Z[1], 1).T
            comm_distrib, local_degree, clusters = model.communities_analysis(theta, data=y)

            plt.figure()
            lgg.info('computing local class degree, can take a while...')
            for c in np.unique(map(set, itertools.product(range(len(comm_distrib)) , repeat=2))):
                if len(c) == 2:
                    # Stochastic Equivalence (extra class bind
                    k, l = c
                    continue
                else:
                    # Comunnities (intra class bind)
                    l = c.pop()
                    k = l

                y_c = y.copy()
                z_c = Z.copy()
                z_c[0][z_c[0] != k] = 0
                z_c[1][z_c[1] != l] = 0
                y_c[z_c[0] != z_c[1]] = 0
                degree_c = adj_to_degree(y_c)
                d, dc = degree_hist(degree_c)
                print  gofit(degree_c, d, dc)
                plt.xscale('log')
                plt.yscale('log')
                plt.scatter(d, dc, c=next(_colors), marker=next(_markers))
                #plot_degree_2_l_e((d,dc,yerr), logscale=True, colors=True)
            plt.title('Local Prefrential attachment')

            ### Max cluster assignemet
            #model/max_clusters_assignement(y)

        display(False)

if not config.get('write_to_file'):
    display(True)

