#!/usr/bin/env python
# -*- coding: utf-8 -*-

from local_utils import *
from vocabulary import Vocabulary, parse_corpus
from util.frontend import ModelManager, FrontendManager
from util.frontendnetwork import frontendNetwork
from plot import *

import numpy as np
import scipy as sp
#np.set_printoptions(threshold='nan')

####################################################
### Config
config = defaultdict(lambda: False, dict(
    ##### Global settings
    verbose     = 0,
    ##### Input Features / Corpus
    limit_train = None,
    ###### I/O settings
    bdir = '../data',
    load_data = False, # Fasle because if not cluster and feature are not loaded
))
config.update(argParse())

### Bursty
#corpuses = ( 'generator3', 'generator11', 'generator12', 'generator7', 'generator14',)

### Non Bursty
#corpuses = ( 'generator4', 'generator5', 'generator6', 'generator9', 'generator10',)

### Expe
Corpuses = (('generator4', 'Network 4 -b/h', 'g4'),
            ('generator10','Network 3 -b/-h', 'g3'),
            ('generator12','Network 2 b/-h', 'g2'),
            ('generator7' ,'Network 1  b/h ', 'g1'),
            )

Corpuses = (('manufacturing', 'Manufacturing', 'manufacturing'),
            ('fb_uc', 'UC Irvine', 'irvine' )
           )

### Models
Model = dict ((
('data_type'    , 'networks'),
('debug'        , 'debug11') ,
('model'        , 'ibp')   ,
('K'            , 20)        ,
('N'            , 'all')     ,
('hyper'        , 'fix')     ,
('homo'         , 0)         ,
#('repeat'      , '*')       ,
))

############################################################
##### Simulation Output
if config.get('simul'):
    print '''--- Simulation settings ---
    Build Corpuses %s''' % (str(Corpuses))
    exit()

for corpus_ in Corpuses:
    path = '../../../../papers/personal/relational_models/git/img/'
    corpus_name = corpus_[0]
    title = corpus_[1]
    fn = corpus_[2]

    frontend = frontendNetwork(config)
    data = frontend.load_data(corpus_name)
    data = frontend.sample()
    prop = frontend.get_data_prop()
    msg = frontend.template(prop)
    #print msg

    #################################################
    ### Homophily Analysis

    #print corpus_name
    #homo_euclide_o_old, homo_euclide_e_old = frontend.homophily(sim='euclide_old')
    #diff2 = homo_euclide_o_old - homo_euclide_e_old
    #homo_euclide_o_abs, homo_euclide_e_abs = frontend.homophily(sim='euclide_abs')
    #diff3 = homo_euclide_o_abs - homo_euclide_e_abs
    #homo_euclide_o_dist, homo_euclide_e_dist = frontend.homophily(sim='euclide_dist')
    #diff4 = homo_euclide_o_dist - homo_euclide_e_dist
    #homo_comm_o, homo_comm_e = frontend.homophily(sim='comm')
    #diff1 = homo_comm_o - homo_comm_e
    #homo_text =  '''Similarity | Hobs | Hexp | diff\
    #               \ncommunity   %s  %s %s\
    #               \neuclide_old     %s  %s %s\
    #               \neuclide_abs     %s  %s %s\
    #               \neuclide_dist     %s  %s %s\
    #''' % ( homo_comm_o, homo_comm_e ,diff1,
    #       homo_euclide_o_old, homo_euclide_e_old, diff2,
    #       homo_euclide_o_abs, homo_euclide_e_abs,diff3,
    #       homo_euclide_o_dist, homo_euclide_e_dist, diff4)
    #print homo_text

    #prop = frontend.get_data_prop()
    #print frontend.template(prop)

    #################################################
    ### Zipf Analisis

    ### Get the Class/Cluster and local degree information
    try:
        'Getting Cluster from Dataset...'
        community_distribution_source, local_attach_source, clusters_source = frontend.communities_analysis()
    except TypeError:
        'Getting Latent Classes from Latent Models...'
        #d = frontend.get_json()
        #local_attach_source = d['Local_Attachment']
        #community_distribution_source = d['Community_Distribution']
        ### In the future
        #cluster_source = d['clusters']
        Model.update(corpus=corpus_name)
        model = ModelManager(config=config).load(Model)
        clusters_source = model.get_clusters()
    except Exception, e:
        print 'Skypping reordering adjacency matrix: %s' % e
        data_r = data

    ### Reordering Adjacency Matrix based on Clusters/Class/Communities
    if globals().get('clusters_source') is not None:
        nodelist = [k[0] for k in sorted(zip(range(len(clusters_source)), clusters_source), key=lambda k: k[1])]
        data_r = data[nodelist, :][:, nodelist]

    ### Filtering
    if fn != 'manufacturing':
        data_r = dilate(data_r)

    #################################################
    ### Plotting

    ### Plot Adjacency matrix
    figsize=(4.7, 4.7)
    plt.figure(figsize=figsize)
    #plt.subplot(1,2,1)
    np.fill_diagonal(data_r, 0)
    adjshow(data_r, title=title)
    #plt.figtext(.15, .1, homo_text, fontsize=12)
    plt.savefig(path+fn+'.pdf', facecolor='white', edgecolor='black')

    ### Plot Degree
    figsize=(3.8, 4.3)
    #figsize=(3.3, 4.3)
    plt.figure(figsize=figsize)
    plot_degree_2(data)

    plt.savefig(path+fn+'_d'+'.pdf', facecolor='white', edgecolor='black')
#     display(False)
# display(True)
