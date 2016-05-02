#!/usr/bin/env python

import random, os
from datetime import datetime
import numpy as np
import scipy as sp
import networkx as nx

from ibp.ilfm_gs import IBPGibbsSampling
from util.plot import *
from local_utils import *

_USAGE = '''netwks [-vhs] [-r [rvalue] [-n N] [-d basedir]
basedir: base directory to save results.
N: Number of nodes.
rvalue: 0 - random
        1 - clique
        2 - barabasi-albert
'''

_rvalue = {0: 'uniform', 1: 'clique', 2: 'barabasi-albert'}

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.interactive(True)
    #import numpy
    #numpy.set_printoptions(threshold=numpy.nan)

    seed = 44
    random.seed(seed)
    np.random.seed(seed)

    conf = {
        'niterations': 32,
        'N' : 80,
        'K': 4,
        'random' : 1,
        'lookup_path' : ['Graph3/t0.graph', ],
        #'lookup_path' : ['Graph1/t0.graph', 'Graph3/t0.graph'],
    }; conf.update(argParse(_USAGE))

    block = False
    if conf.get('random'):
        lookup = [ 'RAND', ]
    else: # Load
        lookup = conf.get('lookup_path', [''])

    if conf.get('simul'):
        print '--- Global settings ---'
        print '\trandom: %s, N: %d, K: %d' % (_rvalue.get(conf.get('random')), conf['N'], conf['K'])
        print '\tLookup: %s' % (lookup)
        if 'load' in  conf:
            print '\tState: loading lookup...'
        else:
            print '\tState: MCMC on %s iter' % (conf['niterations'])

    KKs = [None]
    #KKs = [3]
    #KKs = range(2,30)
    for KK in KKs:

        for i, target in enumerate(lookup):
            # Initialize the Model

            # Set up the hyper-parameter for sampling
            metropolis_hastings_k_new = True
            alpha_hyper_parameter = (1., 1.)
            sigma_w_hyper_parameter = None #(1., 1.)
            symmetric_relation = True
            W_is_diag = False

            # Hyper parameter init
            alpha = 1.
            sigma_w = 1.
            niterations = conf['niterations']
            N = conf['N']
            K = conf['K']

            target_dir = target.replace('/','-') + '_' + str(alpha) + '_' + str(sigma_w) + '_' + str(niterations)
            target_dir = target_dir + '_' + str(KK) if len(KKs) > 1 else target_dir
            if 'bdir' in conf:
                target_dir =  conf['bdir'] + target_dir
            if conf.get('random'):
                target_dir += '_' + str(N)

            if conf.get('simul'):
                print '--- Simulation settings ---\n\tTarget dir: %s\n\tMH K new: %s\n\tW is diag: %s' % (target_dir, metropolis_hastings_k_new, W_is_diag)
                exit()

            if 'load' in conf:
                #### Load it with a regex ;)
                ibp = IBPGibbsSampling.load(target_dir=target_dir)
            else:
                # Generate Data
                if type(conf.get('random')) is int:
                    rvalue = _rvalue.get(conf['random'])
                    if rvalue == 'uniform':
                        data = np.random.randint(0, 2, (N, N))
                        np.fill_diagonal(data, 1)
                    elif rvalue == 'clique':
                        data = getClique(N, K=K)
                        G = nx.from_numpy_matrix(data, nx.DiGraph())
                        data = nx.adjacency_matrix(G, np.random.permutation(range(N))).A
                    elif rvalue == 'barabasi-albert':
                        data = nx.adjacency_matrix(nx.barabasi_albert_graph(N, m=13)).A
                else:
                    data = getGraph(target=target, **conf)
                    np.fill_diagonal(data, 1)

                print data

                # Training IBP
                ibp = IBPGibbsSampling(symmetric_relation, W_is_diag, alpha_hyper_parameter, sigma_w_hyper_parameter, metropolis_hastings_k_new)
                ibp._initialize(data, alpha, sigma_w, KK=KK)
                try:
                    ibp.sample(niterations, target_dir=target_dir)
                except KeyboardInterrupt:
                    pass
                ibp.save(target_dir=target_dir)
                print 'Sampling duration: %s' % ibp.time_sampling
                print 'Target dir: %s' % target_dir

            # Plot Things
            if KKs and len(KKs) < 2 and conf.get('noprint') is not True:
                if i == len(lookup) -1:
                    block = True
                plot_ibp(ibp, target_dir=target_dir, block=block, columns=[0,1], K=K)
                #plot_ibp(ibp, target_dir=target_dir, block=block, columns=range(len(ibp.csv_typo.split(','))), separate=True)


