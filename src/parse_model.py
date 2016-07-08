#!/usr/bin/env python
# -*- coding: utf-8 -*-

from util.frontend import ModelManager, FrontendManager
from util.frontendnetwork import frontendNetwork
from util.frontend_io import *
from expe.spec import _spec_
from util.argparser import argparser

USAGE = '''\
# Usage:
    generate [-w] [-k K]

# Examples
    parallel ./generate.py -w -k {}  ::: $(echo 5 10 15 20)
'''

config = defaultdict(lambda: False, dict(
    write_to_file = False,
))
config.update(argparser.generate(USAGE))

####################################################
### Config
#spec = _spec_.SPEC_TO_PARSE
spec = _spec_.EXPE_ICDM

#spec['model'] = ['ibp_cgs']
#spec['homo'] = ['2']
#spec['hyper'] = ['auto']
#spec['K'] = ['10']
#spec['N'] = ['1000']

def exception_config(config):
    if config['model'] in ('mmsb_cgs', 'immsb'):
        if config['hyper'] == 'fix':
            return False
    if config['model'] in ('ibp_cgs', 'ibp'):
        if config['hyper'] == 'auto':
            return False
    return True

def_conf = {'load_data':False, # Need to compute feature and communities ground truth (no stored in pickle)
            'load_model': True, # Load model vs Gnerate random data
            #'save_data': True,
           }
configs = make_forest_conf(spec)

for config in configs:
    config.update(def_conf)

    test = exception_config(config)
    if not test:
        continue

    # Generate Data source
    #frontend = FrontendManager.get(config)
    frontend = frontendNetwork(config)
    data = frontend.load_data(randomize=False)
    data = frontend.sample()
    model = ModelManager(config=config)

    # Generate Data model
    if config.get('load_model'):
        ### Generate data from a fitted model
        model = model.load()
        if model is None:
            continue
    else:
        ### Generate data from a un-fitted model
        model = model.model

    ### Homophily measures
    #d = frontend.assort(model)
    #d = frontend.homophily(model=model, sim=sim)
    #print(d)
    #frontend.update_json(d)

    print('Model: %s , K: %s' % (config['model'], config['K']))
    homo_dot_o, homo_dot_e = frontend.homophily(model=model, sim='dot')
    diff2 = homo_dot_o - homo_dot_e
    homo_comm_o, homo_comm_e = frontend.homophily(model=model, sim='comm')
    diff1 = homo_comm_o - homo_comm_e
    homo_text =  '''Similarity | Hobs | Hexp | diff\
                   \ncommunity   %s  %s %s\
                   \ndot         %s  %s %s\
    ''' % ( homo_comm_o, homo_comm_e ,diff1,
           homo_dot_o, homo_dot_e, diff2)
    print homo_text

