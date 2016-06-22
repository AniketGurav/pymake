#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from collections import OrderedDict
import numpy as np

from util.frontend_io import *
from local_utils import *


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
        ('data_type', ('networks',)),
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
    target_files = make_forest_path(map_parameters, 'json')
    ### Make Tensor Forest of results
    rez = forest_tensor(target_files, map_parameters)

    ###################################################################
    # Experimentation
    #

    ### Expe 1 settings
    # debug10, immsb
    expe_1 = OrderedDict((
        ('data_type', 'networks'),
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

