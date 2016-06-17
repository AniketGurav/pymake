#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import re, os, sys, json
from multiprocessing import Process
from itertools import cycle
from collections import OrderedDict

from util.frontend_io import *
from local_utils import *

def display(block=False):
    #p = Process(target=_display)
    #p.start()
    plt.show(block=block)

def _display():
    os.setsid()
    plt.show()

def tag_from_csv(c):
    ## loglikelihood_Y, loglikelihood_Z, alpha, sigma, _K, Z_sum, ratio_MH_F, ratio_MH_W
    if c == 0:
        ylabel = 'Iteration'
        label = 'Iteration'
    elif c in (1,2, 3):
        ylabel = 'loglikelihood'
        label = 'loglikelihood'

    return ylabel, label

def csv_row(s):
    #csv_typo = '# mean_eta, var_eta, mean_alpha, var_alpha, log_perplexity'
    if s == 'Iteration':
        row = 0
    if s == 'Timeit':
        row = 1
    elif s in ('loglikelihood', 'likelihood', 'perplexity'):
        row = 2
    elif s in ('loglikelihood_t', 'likelihood_t', 'perplexity_t'):
        row = 3
    elif s == 'K':
        row = 4
    elif s == 'alpha':
        row = 5
    elif s == 'gamma':
        row = 6
    else:
        row = s
    return row


def get_expe_json(fn):
    try:
        d = json.load(open(fn,'r'))
        return d
    except Exception, e:
        return None

def results_tensor(target_files, map_parameters, verbose=True):

    # Expe analyser / Tabulyze It

    # res shape ([expe], [model], [measure]
    # =================================================================================
    # Expe: [debug, corpus] -- from the dirname
    # Model: [name, K, hyper, homo] -- from the expe filename
    # measure:
    #   * 0: global precision,
    #   * 1: local precision,
    #   * 2: recall

    ### Output: rez.shape rez_map_l rez_map
    # Those valuea are get from file expe see get_expe_file()
    dim = get_expe_file_set_prop(target_files)
    map_parameters = map_parameters

    rez_map = map_parameters.keys() # order !
    # Expert knowledge value
    new_dims = [{'measure':4}]
    # Update Mapping
    [dim.update(d) for d in new_dims]
    [rez_map.append(n.keys()[0]) for n in new_dims]

    # Create the shape of the Ananisys/Resulst Tensor
    #rez_map = dict(zip(rez_map_l, range(len(rez_map_l))))
    shape = []
    for n in rez_map:
        shape.append(dim[n])

    # Create the numpy array to store all experience values, whith various setings
    rez = np.zeros(shape) * np.nan

    not_finished = []
    info_file = []
    for _f in target_files:
        prop = get_expe_file_prop(_f)
        pt = np.empty(rez.ndim)

        assert(len(pt) - len(new_dims) == len(prop))
        for k, v in prop.items():
            try:
                v = int(v)
            except:
                pass
            try:
                idx = map_parameters[k].index(v)
            except Exception, e:
                print e
                print k, v
                raise ValueError
            pt[rez_map.index(k)] = idx

        #--
        f = os.path.join(os.path.dirname(__file__), "../data/", _f) + '.json'
        d = os.path.dirname(f)

        corpus_type = ('/').join(d.split('/')[-2:])

        f = os.path.basename(f)[len('inference-'):]
        fn = os.path.join(d, f)
        d = get_expe_json(fn)
        #--
        if not d:
            not_finished.append( '%s not finish...\n' % fn)
            continue

        g_precision = d.get('g_precision')
        precision = d.get('Precision')
        recall = d.get('Recall')
        K = len(d['Local_Attachment'])
        #density = d['density_all']
        #mask_density = d['mask_density']
        #h_s = d.get('homo_ind1_source', np.inf)
        #h_l = d.get('homo_ind1_learn', np.inf)
        #nmi = d.get('NMI', np.inf)

        pt = list(pt.astype(int))
        pt[-1] = 0
        rez[zip(pt)] = g_precision
        pt[-1] = 1
        rez[zip(pt)] = precision
        pt[-1] = 2
        rez[zip(pt)] = recall
        pt[-1] = 3
        rez[zip(pt)] = K

        info_file.append( '%s %s; \t K=%s\n' % (corpus_type, f, K) )

    if verbose:
        [ sys.stdout.write(m) for m in not_finished ]
        print
        #[ sys.stdout.write(m) for m in info_file]
    return rez

def save_files(files):
    f =  open( "ffiles.out", "wb" )
    f.write('\n'.join(target_files))
    print 'ffiles.out writen.'
    exit()

if __name__ ==  '__main__':
    from tabulate import tabulate
    #import argparse
    largs = sys.argv
    model = None
    K = None
    if len(largs) > 1:
        for arg in largs[1:]:
            try:
                K = int(arg)
            except:
                model = arg
    ###################################################################
    # Data Forest config
    #

    ### Expe Forest
    map_parameters = OrderedDict((
        ('debug'  , ('debug10', 'debug11')),
        #('corpus' , ('fb_uc', 'manufacturing')),
        ('corpus' , ('Graph7', 'Graph12', 'Graph10', 'Graph4')),
        ('model'  , ('immsb', 'ibp')),
        ('K'      , (5, 10, 15, 20)),
        ('N'      , ('all',)),
        ('hyper'  , ('fix', 'auto')),
        ('homo'   , (0, 1)),
        #('repeat'   , (0, 1, 2, 4, 5)),
    ))

    ### Seek experiments results
    target_files = make_forest_path(map_parameters, 'json', sep=None)
    ### Make Tensor Forest of results
    rez = results_tensor(target_files, map_parameters, verbose=False)

    ###################################################################
    # Experimentation
    #

    ### Expe 1 settings
    # debug10, immsb
    expe_1 = OrderedDict((
        ('debug' , 'debug10') ,
        ('corpus', '*'),
        ('model' , 'immsb')   ,
        ('K'     , 5)         ,
        ('N'     , 'all')     ,
        ('hyper' , 'auto')     ,
        ('homo'  , 0) ,
        #('repeat', '*'),
        ('measure', '*'),
        ))
    if model:
        expe_1.update(model=model)
        if model == 'ibp':
            expe_1.update(hyper='fix')
    if K:
        expe_1.update(K=K)
    assert(expe_1.keys()[:len(map_parameters)] == map_parameters.keys())

    ###################################
    ### Extract Resulst *** in: setting - out: table

    ### Make the ptx index
    ptx = []
    for i, o in enumerate(expe_1.items()):
        k, v = o[0], o[1]
        if v in ( '*', ':'): #wildcar / indexing ...
            ptx.append(slice(None))
        else:
            ptx.append(map_parameters[k].index(v))
    ptx = tuple(ptx)

    ### Output
    ## Forest setting
    #print 'Forest:'
    #print tabulate(map_parameters, headers="keys")
    #finished =  1.0* rez.size - np.isnan(rez).sum()
    #print '%.3f%% results over forest experimentations' % (finished / rez.size)

    ## Expe setting
    #ptx = np.index_exp[0, :, 0, 0, 0, 1, 0, :]
    print 'Expe 1:'
    print tabulate([expe_1.keys(), expe_1.values()])
    # Headers
    headers = [ 'global', 'precision', 'recall', 'K->']
    h_mask = 'mask all' if '11' in expe_1['debug'] else 'mask sub1'
    h = expe_1['model'].upper() + ' / ' + h_mask
    headers.insert(0, h)
    # Row
    keys = map_parameters['corpus']
    keys = [''.join(k) for k in zip(keys, [' b/h', ' b/-h', ' -b/h', ' -b/-h'])]
    ## Results
    table = rez[ptx]
    table = np.column_stack((keys, table))
    print
    #print tabulate(table, headers=headers)
    print tabulate(table, headers=headers, tablefmt='latex', floatfmt='.4f')


