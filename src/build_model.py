#!/usr/bin/python -u
# -*- coding: utf-8 -*-

from local_utils import *
from vocabulary import Vocabulary, parse_corpus
from util.frontendtext import frontEndText
from settings import *

import numpy as np
import scipy as sp
#np.set_printoptions(threshold='nan')

import os
import logging
import os.path

_USAGE = '''build_model [-vhswp] [-k [rvalue] [-n N] [-d basedir] [-lall] [-l type] [-m model] [-c corpus] [-i iterations]

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
# Load corpus and infer modef (eg LDA)
./lda_run.py -k 6 -m ldafullbaye -p
# Load corpus and model
./lda_run.py -k 6 -m ldafullbaye -lall -p

'''

from hdp.hdp2 import *

class modelManager(object):
    def __init__(self, corpus, config):
        self.corpus = corpus

        self.model_name = config.get('model')
        #models = {'ilda' : HDP_LDA_CGS,
        #          'lda_cgs' : LDA_CGS, }
        self.hyperparams = config.get('hyperparams')
        self.output_path = config.get('output_path')
        self.K = config.get('K')
        self.inference_method = '?'

        self.write = config.get('write', False)
        self.config = config

        if self.model_name:
            self.model = self.loadgibbs(self.model_name)

    # Base class for Gibbs, VB ... ?
    def loadgibbs(self, target, likelihood=None):
        delta = self.hyperparams['delta']
        alpha = self.hyperparams['alpha']
        gmma = self.hyperparams['gmma']
        K = self.K

        if likelihood is None:
            now = datetime.now()
            likelihood = DirMultLikelihood(delta, self.corpus)
            last_d = ellapsed_time('Corpus Preprocessing Time', now)

        if target == 'ilda':
            zsampler = ZSampler(alpha, likelihood, K_init=K)
            msampler = MSampler(zsampler)
            betasampler = BetaSampler(gmma, msampler)
            jointsampler = HDP_LDA_CGS(zsampler, msampler, betasampler, hyper=config['hyper'],)
        elif target == 'lda_cgs':
            jointsampler = LDA_CGS(ZSamplerParametric(alpha, likelihood, K))
        else:
            raise NotImplementedError()

        return GibbsRun(jointsampler, iterations=config['iterations'],
                        burnin=5, thinning_interval=1,
                        output_path=self.output_path, write=self.write)

    def run(self):
        now = datetime.now()
        self.model.run()
        last_d = ellapsed_time('Inference Time: %s'%(self.output_path), now)

    # Measure perplexity on different initialization
    def init_loop_test(self):
        niter = 2
        pp = []
        likelihood = self.model.s.zsampler.likelihood
        for i in xrange(niter):
            self.model.s.zsampler.estimate_latent_variables()
            pp.append( self.model.s.zsampler.perplexity() )
            self.model = self.loadgibbs(self.model_name, likelihood)

        print self.output_path
        np.savetxt('t.out', np.log(pp))


##################
###### MAIN ######
##################
if __name__ == '__main__':
    config = defaultdict(lambda: False, dict(
        ##### Global settings
        verbose                       = 0,
        host                          = 'localhost',
        index                         = 'search',
        ##### Input Features / Corpus
        corpus_name                   = 'kos',
        vsm                           = 'tf',
        limit_train                   = 10000,
        limit_predict                 = None,
        extra_feat                    = False,
        ##### Models Hyperparameters
        #model                         = 'lda_cgs',
        model                         = 'ilda',
        hyper                         = 'auto',
        K                             = 10,
        N                             = 3,
        chunk                         = 10000,
        iterations                    = 2,
        ###### I/O settings
        refdir                        = 'debug',
        bdir                          = '../data',
        load_corpus                   = True,
        save_corpus                   = True,
        load_model                    = False,
        save_model                    = True,
        write                         = False, # -w/-nw
        #####
        predict                       = False,
    ))
    config.update(argParse(_USAGE))

    # Silly ! think different
    if config.get('lall'):
        # Load All
        config.update(load_corpus=True, load_model=True)
    if config.get('load') == 'corpus':
        config['load_corpus'] = True
    elif config.get('load') == 'model':
        config['load_model'] = True

    frontend = frontEndText(config)

    ############################################################
    ##### Simulation Output
    if config.get('simul'):
        print '''--- Simulation settings ---
        Model : %s
        Corpus : %s
        K: %s
        N: %s
        hyper: %s
        Output: %s''' % (config['model'], config['corpus_name'],
                         config['K'], config['N'], config['hyper'],
                         config['output_path'])
        exit()

    ############################################################
    ##### Load Corpus
    corpus = frontend.load_corpus()
    if config.get('N'):
        corpus = corpus[:config['N']]
        # Here we come to streaming problem !
        # To manage one day :s // plus mega non optimal avec data.A pour le caclul de la perplexité bref a revoir
        # @DEBUG manage id2word
        corpus = corpus.A
        empty_words =  np.where(corpus.sum(0) == 0)[0]
        corpus = np.delete(corpus, empty_words, axis=1)
        corpus = sp.sparse.csr_matrix(corpus)


    ############################################################
    ##### Load Model
    #models = ('ilda_cgs', 'lda_cgs', 'immsb', 'mmsb', 'ilfm_gs', 'lda_vb', 'ldafull_vb')
    # Hyperparameter
    delta = .1
    alpha = 0.2
    gmma = 0.5
    hyperparams = {'alpha': alpha, 'delta': delta, 'gmma': gmma}
    config['hyperparams'] = hyperparams

    #### Debug
    #config['write'] = False
    #model = modelManager(corpus, config)
    #model.init_loop_test()
    #exit()

    # Initializa Model
    model = modelManager(corpus, config)

    ##### Run Inference
    model.run()

