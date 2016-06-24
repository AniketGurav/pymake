#!/usr/bin/python -u
# -*- coding: utf-8 -*-

from util.frontend import ModelManager, FrontendManager
from util.frontendnetwork import frontendNetwork
from local_utils import *
from plot import *
from util.frontend_io import *
from expe.spec import *


####################################################
### Config
config = defaultdict(lambda: False, dict(
    generative = 'predictive',
    gen_size = 1000,
    epoch = 10
))
config.update(argParse())


Corpuses = NETWORKS_DD
Models = MODELS_DD

### Models

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
    frontend = frontendNetwork(config)
    data = frontend.load_data(corpus_name)
    data = frontend.sample()
    for Model in Models:

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

        ###############################################################
        ### Expe Wrap up debug
        print 'corpus: %s, model: %s, K = %s, N =  %s'.replace(',','\n') % (frontend.corpus_name, Model['model'], K, N)

        #################################################
        ### Plot Degree
        plt.figure()
        #plot_degree_(y, title='Overall Degree')
        plot_degree_2_l(Y)
        plot_degree_2(data, scatter=False)

        fn = corpus_name+'.pdf'
        #plt.savefig(fn, facecolor='white', edgecolor='black')
        display(False)

display(True)

