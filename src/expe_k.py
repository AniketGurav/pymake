#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from tabulate import tabulate
from collections import OrderedDict
from util.frontend_io import *
from local_utils import *
from util.argparser import argparser

USAGE = '''\
# Usage:
    expe_k [model]
'''

zyvar = argparser.expe_tabulate(USAGE)
model = zyvar.get('model')

###################################################################
# Data Forest config
#

### Expe Forest
map_parameters = OrderedDict((
    ('data_type', ('networks',)),
    ('corpus' , ('fb_uc', 'manufacturing')),
    #('corpus' , ('Graph7', 'Graph12', 'Graph10', 'Graph4')),
    ('debug'  , ('debug10', 'debug11')),
    ('model'  , ('immsb', 'ibp')),
    ('K'      , (5, 10, 15, 20)),
    ('hyper'  , ('fix', 'auto')),
    ('homo'   , (0, 1, 2)),
    ('N'      , ('all',)),
    #('repeat'   , (0, 1, 2, 4, 5)),
))

### Seek experiments results
target_files = make_forest_path(map_parameters, 'json',  sep=None)
### Make Tensor Forest of results
rez = forest_tensor(target_files, map_parameters)

###################################################################
# Experimentation
#

### Expe 1 settings
# debug10, immsb
expe_1 = OrderedDict((
    ('data_type', 'networks'),
    ('corpus', '*'),
    ('debug' , 'debug10') ,
    ('model' , 'immsb')   ,
    ('K'     , '*')         ,
    ('hyper' , 'auto')     ,
    ('homo'  , 0) ,
    ('N'     , 'all')     ,
    #('repeat', '*'),
    ('measure', 0),
    ))
if model:
    expe_1.update(model=model)
    if model == 'ibp':
        expe_1.update(hyper='fix')
assert(expe_1.keys()[:len(map_parameters)] == map_parameters.keys())

###################################
### Extract Resulst *** in: setting - out: table
headers = [ 'global', 'precision', 'recall', 'K->']

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
print tabulate(table, headers=headers, tablefmt='latex', floatfmt='.4f')
print '\t\t--> precision'



