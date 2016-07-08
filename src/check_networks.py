#!/usr/bin/env python
# -*- coding: utf-8 -*-

from util.frontend import ModelManager, FrontendManager
from util.frontendnetwork import frontendNetwork
from local_utils import *
from plot import *
from expe.spec import _spec_
from expe.format import corpus_icdm
from util.argparser import argparser


### Config
config = defaultdict(lambda: False, dict(
    write_to_file = False,
))
config.update(argparser.generate())

### Specification
Corpuses = _spec_.CORPUS_SYN_ICDM_1
Corpuses += _spec_.CORPUS_REAL_ICDM_1
Model = _spec_.MODEL_FOR_CLUSTER_IBP

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
        msg =  'Getting Cluster from Dataset...'
        community_distribution_source, local_attach_source, clusters_source = frontend.communities_analysis()
    except TypeError:
        msg =  'Getting Latent Classes from Latent Models...'
        #d = frontend.get_json()
        #local_attach_source = d['Local_Attachment']
        #community_distribution_source = d['Community_Distribution']
        ### In the future
        #cluster_source = d['clusters']
        Model.update(corpus=corpus_name)
        model = ModelManager(config=config).load(Model)
        clusters_source = model.get_clusters()
    except Exception, e:
        msg = 'Skypping reordering adjacency matrix: %s' % e
        data_r = data
    print msg

    ### Reordering Adjacency Mmatrix based on Clusters/Class/Communities
    if globals().get('clusters_source') is not None:
        nodelist = [k[0] for k in sorted(zip(range(len(clusters_source)), clusters_source), key=lambda k: k[1])]
        data_r = data[nodelist, :][:, nodelist]


    #################################################
    ### Plotting
    if config.get('write_to_file'):
        corpus_icdm(data=data_r, corpus_name=corpus_name)
        continue

    ### Plot Adjacency matrix
    plt.figure()
    plt.suptitle(corpus_name)
    plt.subplot(1,2,1)
    adjshow(data_r, title='Adjacency Matrix')
    #plt.figtext(.15, .1, homo_text, fontsize=12)

    ### Plot Degree
    plt.subplot(1,2,2)
    #plot_degree_(data, title='Overall Degree')
    plot_degree_2(data)

    display(False)

if not config.get('write_to_file'):
    display(True)
