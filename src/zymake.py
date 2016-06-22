#!/usr/bin/python -u
# -*- coding: utf-8 -*-

from local_utils import *
from vocabulary import Vocabulary, parse_corpus
from util.frontend import ModelManager, FrontendManager
from util.frontendnetwork import frontendNetwork
from plot import *
from util.frontend_io import *

import numpy as np
import scipy as sp
#np.set_printoptions(threshold='nan')

from joblib import Parallel, delayed
import multiprocessing

import logging
import os
import os.path
from collections import OrderedDict
from random import choice

_USAGE = '''?'''

####################################################
### Config
config = defaultdict(lambda: False, dict(
    ##### Global settings
    #generative = 'predictive',
    generative = 'predictive',
    gen_size = 1000,
    epoch = 100
))
config.update(argParse(_USAGE))

### Bursty
#corpuses = ( 'generator3', 'generator11', 'generator12', 'generator7', 'generator14',)

### Non Bursty
#corpuses = ( 'generator4', 'generator5', 'generator6', 'generator9', 'generator10',)

### Expe Spec

### Expe Forest
map_parameters = OrderedDict((
    ('data_type', ('networks',)),
    ('debug'  , ('debug10',)),
    #('corpus' , ('fb_uc', 'manufacturing')),
    ('corpus' , ('Graph7', 'Graph12', 'Graph10', 'Graph4')),
    ('model'  , ('immsb', 'ibp')),
    ('K'      , (10,)),
    ('N'      , ('all',)),
    ('hyper'  , ('fix', 'auto')),
    ('homo'   , (0, 1)),
    #('repeat'   , (0, 1, 2, 4, 5)),
))

alpha = .01
gmma = 0.5
delta = 10
keys_hyper = ('alpha','gmma','delta')
hyper = (alpha, gmma, delta)

### Makes taget files
source_files = make_forest_path(map_parameters, 'pk', status=None)

### Task to build figure
#from expe import *
#expe_figures = generete_icdm
#
#### Makes figures on remote / parallelize
#num_cores = int(multiprocessing.cpu_count() / 4)
#results_files = Parallel(n_jobs=num_cores)(delayed(expe_figures)(i) for i in source_files)

### Retrieve the figure
#rsync the results_files
print '\n'.join(source_files)


