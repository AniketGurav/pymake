#!/usr/bin/env python
# -*- coding: utf-8 -*-

from frontend.frontend import ModelManager
from frontend.frontendnetwork import frontendNetwork
from utils.utils import *
from plot import *
from expe.spec import _spec_
from expe.format import corpus_icdm
from utils.argparser import argparser
from utils.algo import reorder_mat

""" Inspect data on disk, for checking
    or updating results

    params
    ------
    zipf : degree based analysis
    homo : homophily based analysis
"""

### Config
config = defaultdict(lambda: False, dict(
    write_to_file = False,
    do           = 'homo', # homo/zipf
    clusters      = 'source' # source/model
))
config.update(argparser.generate(''))

### Specification
Corpuses = _spec_.CORPUS_SYN_ICDM_1
Corpuses += _spec_.CORPUS_REAL_ICDM_1

#Model = _spec_.MODEL_FOR_CLUSTER_IBP
Model = _spec_.MODEL_FOR_CLUSTER_IMMSB

### Simulation Output
if config.get('simul'):
    print '''--- Simulation settings ---
    Build Corpuses %s''' % (str(Corpuses))
    exit()

for corpus_name in Corpuses:
    frontend = frontendNetwork(config)
    data = frontend.load_data(corpus_name)
    data = frontend.sample()
    prop = frontend.get_data_prop()
    msg = frontend.template(prop)
    #print msg

    if config['do'] == 'homo':
        ###################################
        ### Homophily Analysis
        ###################################

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
                       \ncommunity   %.3f  %.3f %.3f\
                       \neuclide_old   %.3f  %.3f %.3f\
                       \neuclide_abs   %.3f  %.3f %.3f\
                       \neuclide_dist  %.3f  %.3f %.3f\
        ''' % ( homo_comm_o, homo_comm_e ,diff1,
               homo_euclide_o_old, homo_euclide_e_old, diff2,
               homo_euclide_o_abs, homo_euclide_e_abs,diff3,
               homo_euclide_o_dist, homo_euclide_e_dist, diff4)
        print homo_text

        #prop = frontend.get_data_prop()
        #print frontend.template(prop)

    else:
        ###################################
        ### Zipf Analisis
        ###################################
        degree = adj_to_degree(data)
        gofit(*degree_hist(adj_to_degree(data)))

        ### Get the Class/Cluster and local degree information
        data_r = data
        clusters = None
        K = None
        try:
            msg =  'Getting Cluster from Dataset.'
            clusters = frontend.get_clusters()
            if config.get('clusters') == 'model':
                if clusters is not None:
                    class_hist = np.bincount(clusters)
                    K = (class_hist != 0).sum()
                raise TypeError
        except TypeError:
            msg =  'Getting Latent Classes from Latent Models %s' % Model['model']
            Model.update(corpus=corpus_name)
            model = ModelManager(config=config).load(Model)
            #clusters = model.get_clusters(K, skip=1)
            clusters = model.get_communities(K)
        except Exception, e:
            msg = 'Skypping reordering adjacency matrix: %s' % e

        ##############################
        ### Reordering Adjacency Mmatrix based on Clusters/Class/Communities
        ##############################
        if clusters is not None:
            print 'Reordering Adj matrix:'
            print 'corpus: %s, %s, Clusters size: %s' % (corpus_name, msg, K)
            data_r = reorder_mat(data, clusters)
        else:
            print 'No Reordering !'

        ###################################
        ### Plotting
        ###################################
        if config.get('write_to_file'):
            corpus_icdm(data=data_r, corpus_name=corpus_name)
            continue

        ###################################
        ### Plot Adjacency matrix
        ###################################
        plt.figure()
        plt.suptitle(corpus_name)
        plt.subplot(1,2,1)
        adjshow(data_r, title='Adjacency Matrix', fig=False)
        #plt.figtext(.15, .1, homo_text, fontsize=12)

        ###################################
        ### Plot Degree
        ###################################
        plt.subplot(1,2,2)
        #plot_degree_(data, title='Overall Degree')
        plot_degree_poly(data)

        display(False)

if not config.get('write_to_file'):
    display(True)
