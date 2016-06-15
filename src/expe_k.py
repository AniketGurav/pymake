#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import re, os, sys, json
from multiprocessing import Process
from itertools import cycle
from collections import OrderedDict

from local_utils import *
from expe_icdm import *


if __name__ ==  '__main__':
    from tabulate import tabulate
    #import argparse
    largs = sys.argv
    model = None
    if len(largs) > 1:
        model = largs[-1]
    ###################################################################
    # Data Forest config
    #

    ### Expe Forest
    map_parameters = OrderedDict((
        ('debug'  , ('debug10', 'debug11')),
        ('corpus' , ('Graph7', 'Graph12', 'Graph4', 'Graph10')),
        ('model'  , ('immsb', 'ibp')),
        ('K'      , (5, 10, 15, 20)),
        ('N'      , ('all',)),
        ('hyper'  , ('fix', 'auto')),
        ('homo'   , (0, 1, 2)),
    ))

    ### Tensor Forest
    target_files = make_path_v2(map_parameters, sep=None)
    rez = results_tensor(target_files, map_parameters, verbose=False)


    ###################################################################
    # Experimentation
    #

    ### Expe 1 settings
    # debug10, immsb
    expe_1 = OrderedDict((
        ('debug' , 'debug11') ,
        ('corpus', '*'),
        ('model' , 'immsb')   ,
        ('K'     , '*')         ,
        ('N'     , 'all')     ,
        ('hyper' , 'auto')     ,
        ('homo'  , 0) ,
        ('measure', 0),
        ))
    if model:
        expe_1.update(model=model)
        if model == 'ibp':
            expe_1.update(hyper='fix')
    assert(expe_1.keys()[:len(map_parameters)] == map_parameters.keys())

    ###################################
    ### Extract Resulst *** in: setting - out: table
    headers = [ 'global', 'precision', 'rappel', 'K->']

    ### Make the ptx index
    ptx = []
    for i, o in enumerate(expe_1.items()):
        k, v = o[0], o[1]
        if v in ( '*', ':'): #wildcar / indexing ...
            ptx.append(slice(None))
        else:
            if k in map_parameters:
                ptx.append(map_parameters[k].index(v))
            elif type(v) is int:
                ptx.append(v)
            else:
                raise ValueError('Unknow data type for tensor forest')

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
    headers = list(map_parameters['K'])
    h_mask = 'mask all' if '11' in expe_1['debug'] else 'mask sub1'
    h = expe_1['model'] + ' / ' + h_mask
    headers.insert(0, h)
    # Row
    keys = map_parameters['corpus']
    keys = [''.join(k) for k in zip(keys, [' b/h', ' b/-h', ' -b/h', ' -b/-h'])]
    ## Results
    table = rez[ptx]
    table = np.column_stack((keys, table))
    print
    #print tabulate(table, headers=headers)
    print tabulate(table, headers=headers, tablefmt='latex')
    print '\t\t--> precision'



