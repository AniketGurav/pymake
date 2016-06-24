#!/usr/bin/env python
# -*- coding: utf-8 -*-

from util.frontend import ModelManager, FrontendManager
from util.frontendnetwork import frontendNetwork
from util.frontend_io import *
from expe.spec import *


if __name__ == '__main__':

    ####################################################
    ### Config
    spec = SPEC_TO_PARSE

    #spec['model'] = ['ibp_cgs']
    #spec['homo'] = ['2']
    #spec['hyper'] = ['auto']
    #spec['K'] = ['10']
    #spec['N'] = ['1000']

    def exception_config(config):
        if config['model'] is 'mmsb_cgs':
            if config['hyper'] is 'fix':
                return False
        if config['model'] is 'ibp_cgs':
            if config['hyper'] is 'fix':
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
        frontend = FrontendManager.get(config)
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
            y, theta, phi = model.generate(config['N'], config['K'])

        ### Homophily measures
        d = frontend.assort(model)

        print d
        #frontend.update_json(d)

