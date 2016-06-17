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

_USAGE = '''zipf [-vhswp] [-k [rvalue] [-n N] [-d basedir] [-lall] [-l type] [-m model] [-c corpus] [-i iterations]

Default load corpus and run a model !!

##### Argument Options
--hyper|alpha  : hyperparameter optimization ( asymmetric | symmetric | auto)
-lall          : load all; Corpus and LDA model
-l type        : load type ( corpus | lda)
-i iterations  : Iterations number
-c corpus      : Pickup a Corpus (20ngroups | nips12 | wiki | lucene)
-m model       : Pickup a Model (ldamodel | ldafullbaye)
-n | --limit N : Limit size of corpus
-d basedir     : base directory to save results.
-k K           : Number of topics.
##### Single argument
-p           : Do prediction on test data
-s           : Simulate output
-w|-nw           : Write/noWrite convergence measures (Perplexity etc)
-h | --help  : Command Help
-v           : Verbosity level

Examples:
./zipf.py -n 1000 -k 30 --alpha fix -m immsb -c generator3 -l model --refdir debug5
./zipf.py -n 1000 -k 10 --alpha auto --homo 0 -m ibp_cgs -c generator3 -l model --refdir debug5 -nld


'''


if __name__ == '__main__':

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
        gen_size = 2000,
    ))
    config.update(argParse(_USAGE))

    ### Bursty
    #corpuses = ( 'generator3', 'generator11', 'generator12', 'generator7', 'generator14',)

    ### Non Bursty
    #corpuses = ( 'generator4', 'generator5', 'generator6', 'generator9', 'generator10',)

    ### Expe
    Corpuses = ( 'generator4', )
    Corpuses = ( 'manufacturing', 'fb_uc',)
    Corpuses = ( 'generator4', 'generator10', 'generator12', 'generator7', 'generator13')
    Corpuses = ( 'generator4', 'generator10', 'generator12', 'generator7',)
    Corpuses = ( 'generator10', )

    ### Models
    Model = dict ((
    ('debug'        , 'debug11') ,
    ('model'        , 'ibp')   ,
    ('K'            , 10)        ,
    ('N'            , 'all')     ,
    ('hyper'        , 'fix')     ,
    ('homo'         , 0)         ,
    #('repeat'      , '*')       ,
    ))

    if config.get('arg'):
        Model =  get_conf_from_file(config['arg'])
        try:
            Model =  get_conf_from_file(config['arg'])
        except:
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

    for corpus_name in Corpuses:

        # Initializa Model
        frontend = frontendNetwork(config)
        data = frontend.load_data(corpus_name)
        data = frontend.sample()
        model = ModelManager(config=config)

        N = config['gen_size']
        if config['generative'] == 'predictive':
            ### Generate data from a fitted model
            Model.update(corpus=corpus_name)
            model.load(Model)
            y, theta, phi = model.model.generate(N)
            if frontend.is_symmetric():
                y = np.triu(y) + np.triu(y, 1).T
                phi = np.triu(phi) + np.triu(phi, 1).T
        elif config['generative'] == 'evidence':
            ### Generate data from a un-fitted model
            alpha = .5
            gmma = 1.
            delta = .1
            if Model['model'] == 'ibp':
                hyper = (alpha, delta)
            elif Model['model'] == 'immsb':
                hyper = (alpha, gmma, delta)
            else:
                raise NotImplementedError
            model = model.model
            model.update_hyper(hyper)
            y, theta, phi = model.generate(N, config['K'])
        else:
            raise NotImplementedError

        ### Baselines
        #R = rescal(data, config['K'])
        R = None

        K = theta.shape[1]
        ###############################################################
        ### Expe Wrap up debug
        print 'corpus: %s, model: %s, K = %s, N =  %s'.replace(',','\n') % (frontend.corpus_name, model.model_name, model.K, frontend.N)

        #################################################
        ### Zipf Analisis

        ### Get the Class/Cluster and local degree information
        try:
            'Getting Cluster from Dataset...'
            community_distribution, local_attach, clusters = frontend.communities_analysis()
        except TypeError:
            'Getting Latent Classes from Latent Models...'
            #d = frontend.get_json()
            #local_attach_source = d['Local_Attachment']
            #community_distribution_source = d['Community_Distribution']
            ### In the future
            #cluster_source = d['clusters']
            model = ModelManager(config=config).load()
            clusters_source = model.get_clusters()
        except Exception, e:
            print 'Skypping reordering adjacency matrix: %s' % e
            data_r = data

        ### Reordering Adjacency Matrix based on Clusters/Class/Communities
        if globals().get('clusters'):
            nodelist = [k[0] for k in sorted(zip(range(len(clusters)), clusters), key=lambda k: k[1])]
        y_r = y[nodelist, :][:, nodelist]

        #################################################
        ### Plotting

        ### Plot Adjacency matrix
        #figsize=(30, 40)
        plt.figure(figsize=None)
        plt.suptitle(corpus_name)
        plt.subplot(1,2,1)
        adjshow(y_r, title='Adjacency Matrix')
        #plt.figtext(.15, .1, homo_text, fontsize=12)

        ### Plot Degree
        plt.subplot(1,2,2)
        #plot_degree_(y, title='Overall Degree')
        plot_degree_2(y)

        fn = corpus_name+'.pdf'
        #plt.savefig(fn, facecolor='white', edgecolor='black')
        display(False)
    display(True)


