#!/usr/bin/env python
# -*- coding: utf-8 -*-

from local_utils import *
from vocabulary import Vocabulary, parse_corpus
from util.frontend import ModelManager, FrontendManager
from util.frontendnetwork import frontendNetwork

import numpy as np
import scipy as sp
#np.set_printoptions(threshold='nan')

_USAGE = '-s'

##################
###### MAIN ######
##################
if __name__ == '__main__':
    config = defaultdict(lambda: False, dict(
        ##### Global settings
        verbose     = 0,
        ##### Input Features / Corpus
        limit_train = None,
        ###### I/O settings
        bdir = '../data',
        load_data = False, # Fasle because if not cluster and feature are not loaded
        save_data = False,
    ))
    config.update(argParse(_USAGE))

    # Bursty
    #corpuses = ( 'generator3', 'generator11', 'generator12', 'generator7', 'generator14',)

    # Non Bursty
    #corpuses = ( 'generator4', 'generator5', 'generator6', 'generator9', 'generator10',)

    # Expe
    corpuses = ( 'generator4', )
    corpuses = ( 'manufacturing', 'fb_uc',)
    corpuses = ( 'generator4', 'generator10', 'generator12', 'generator7',)
    corpuses = ( 'generator4', 'generator10', 'generator12', 'generator7', 'generator13')

    ############################################################
    ##### Simulation Output
    if config.get('simul'):
        print '''--- Simulation settings ---
        Build Corpuses %s''' % (str(corpuses))
        exit()

    from plot import *
    for corpus_name in corpuses:
        frontend = frontendNetwork(config)
        data = frontend.load_data(corpus_name)
        prop = frontend.get_data_prop()
        msg = frontend.template(prop)
        #print msg

        print corpus_name
        homo_euclide_o_old, homo_euclide_e_old = frontend.homophily(sim='euclide_old')
        diff2 = homo_euclide_o_old - homo_euclide_e_old
        homo_euclide_o_abs, homo_euclide_e_abs = frontend.homophily(sim='euclide_abs')
        diff3 = homo_euclide_o_abs - homo_euclide_e_abs
        homo_euclide_o_dist, homo_euclide_e_dist = frontend.homophily(sim='euclide_dist')
        diff4 = homo_euclide_o_dist - homo_euclide_e_dist
        homo_comm_o, homo_comm_e = frontend.homophily(sim='comm')
        diff1 = homo_comm_o - homo_comm_e
        homo_text =  '''Similarity | Hobs | Hexp | diff\
                       \ncommunity   %s  %s %s\
                       \neuclide_old     %s  %s %s\
                       \neuclide_abs     %s  %s %s\
                       \neuclide_dist     %s  %s %s\
        ''' % ( homo_comm_o, homo_comm_e ,diff1,
               homo_euclide_o_old, homo_euclide_e_old, diff2,
               homo_euclide_o_abs, homo_euclide_e_abs,diff3,
               homo_euclide_o_dist, homo_euclide_e_dist, diff4)
        print homo_text

        prop = frontend.get_data_prop()
        print frontend.template(prop)

        ### Zipf Analisis

        # Degree
        #degree_source = sorted(frontend.degree().values(), reverse=True)
        try:
            community_distribution_source, local_attach_source, clusters_source = frontend.communities_analysis()

            # Re-order adjacency matrix
            nodelist = [k[0] for k in sorted(zip(range(len(clusters_source)), clusters_source), key=lambda k: k[1])]
            data_r = data[nodelist, :][:, nodelist]
        except:
            data_r = data

        #figsize=(30, 40)
        plt.figure(figsize=None)
        plt.suptitle(corpus_name)
        plt.subplot(1,2,1)
        adjshow_(data_r, title='Adjacency Matrix')
        #plt.figtext(.15, .1, homo_text, fontsize=12)

        plt.subplot(1,2,2)
        #plot_degree_(data, title='Overall Degree')
        plot_degree_2(data)

        fn = corpus_name+'.pdf'
        #plt.savefig(fn, facecolor='white', edgecolor='black')
        display(False)
    display(True)
