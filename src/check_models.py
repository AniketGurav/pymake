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

def_conf = defaultdict(lambda: False, dict(
    write_to_file = False,
    do           = 'homo',
))
def_conf.update(argparser.generate(USAGE))

####################################################
### Config
#spec = _spec_.SPEC_TO_PARSE
spec = _spec_.EXPE_ICDM
#spec = _spec_.EXPE_ALL_3_IBP

#spec = dict((
#    ('data_type', ('networks',)),
#    ('debug'  , ('debug10')),
#    #('corpus' , ('fb_uc', 'manufacturing')),
#    ('corpus' , ('generator7', 'generator12', 'generator10', 'generator4')),
#    #('model'  , ('immsb', 'ibp')),
#    ('model'  , ('ibp', )),
#    ('K'      , (5,10,15, 20)),
#    ('N'      , ('all',)),
#    ('hyper'  , ('fix', 'auto')),
#    ('homo'   , (0,)),
#    #('repeat'   , (0, 1, 2, 4, 5)),
#))

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

def_conf.update( {'load_data':False, # Need to compute feature and communities ground truth (no stored in pickle)
            'load_model': True, # Load model vs Gnerate random data
            #'save_data': True,
           } )
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

    ###############################################################
    ### Expe Wrap up debug
    print('corpus: %s, model: %s, K = %s, N =  %s' % (frontend.corpus_name, config['model'], config['K'], config['N']) )

    if config['do'] == 'homo':
        d = {}
        d['homo_dot_o'], d['homo_dot_e'] = frontend.homophily(model=model, sim='dot')
        diff2 = d['homo_dot_o'] - d['homo_dot_e']
        d['homo_model_o'], d['homo_model_e'] = frontend.homophily(model=model, sim='model')
        diff1 = d['homo_model_o'] - d['homo_model_e']
        homo_text =  '''Similarity | Hobs | Hexp | diff\
                       \nmodel   %.4f  %.4f %.4f\
                       \ndot     %.4f  %.4f %.4f\
        ''' % ( d['homo_model_o'], d['homo_model_e'] ,diff1,
               d['homo_dot_o'], d['homo_dot_e'], diff2)
        print homo_text
        if config.get('write_to_file'):
            try:
                frontend.update_json(d)
            except Exception as e:
                print e
                pass

