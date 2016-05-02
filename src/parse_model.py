#!/usr/bin/env python
# -*- coding: utf-8 -*-

from util.frontend import ModelManager, FrontendManager
from util.frontendnetwork import frontendNetwork

import numpy as np
import scipy as sp
import os

if __name__ == '__main__':

    spec = dict(
        base = ['networks'],
        hook_dir = ['debug5/'],
        #corpus   = ['kos', 'nips12', 'nips', 'reuter50', '20ngroups'],
        #corpus   = ['generator3', 'generator4'],
        #corpus   = ['generator5', 'generator6'],
        corpus   = [ 'generator6'],
        #models   = ['ibp', 'ibp_cgs', 'immsb', 'mmsb_cgs' ],
        #models   = ['ibp', 'ibp_cgs', 'mmsb_cgs', 'immsb'],
        models   = ['ibp_cgs', 'mmsb_cgs'],
        #Ns       = [250, 1000, 'all'],
        Ns       = [1000,],
        #Ks       = [2, 3, 6, 10, 20, 30],
        Ks       = [5, 10, 30],
        homo     = [0,1,2],
        hyper    = ['fix', 'auto'],
    )

    #spec['models'] = ['ibp_cgs']
    #spec['homo'] = ['2']
    #spec['hyper'] = ['auto']
    #spec['Ks'] = ['10']
    #spec['Ns'] = ['1000']

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
                'bdir': '../data',
                #'save_data': True,
               }
    configs = frontendNetwork.make_conf(spec)

    for config in configs:
        config.update(def_conf)

        test = exception_config(config)
        if not test:
            continue

        # Generate Data source
        frontend = FrontendManager.get(config)
        data = frontend.load_data(randomize=False)
        data = frontend.sample()
        model = ModelManager(None, config)

        ouf = config['output_path'][:-len('.out')] + '.pk'
        if not os.path.isfile(ouf) or os.stat(ouf).st_size == 0:
            continue

        #if frontend.get_json():
        #    continue

        # Generate Data model
        if config.get('load_model'):
            ### Generate data from a fitted model
            model = model.load()

            ## For zipf: do the script
            #y, theta, phi = model.generate(config['N'])
            #y = np.triu(y) + np.triu(y, 1).T
        else:
            ### Generate data from a un-fitted model
            model = model.model
            y, theta, phi = model.generate(config['N'], config['K'])

        d = frontend.assort(model)
        frontend.update_json(d)

