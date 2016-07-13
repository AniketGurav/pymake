#!/usr/bin/env python
# -*- coding: utf-8 -*-

from local_utils import *
from vocabulary import Vocabulary, parse_corpus
from util.frontend import ModelManager, FrontendManager
from util.frontendnetwork import frontendNetwork

import numpy as np
import scipy as sp
np.set_printoptions(threshold='nan')

_USAGE = '-s'

if __name__ == '__main__':
    config = defaultdict(lambda: False, dict(
        ##### Global settings
        ###### I/O settings
        load_data = False,
        save_data = True,
    ))
    config.update(argParse(_USAGE))

    # Expe
    corpuses = ( 'fb_uc', 'manufacturing', )
    corpuses = ('generator7',)
    corpuses = ( 'generator4', 'generator10', 'generator12', 'generator7',)

    ############################################################
    ##### Simulation Output
    if config.get('simul'):
        print '''--- Simulation settings ---
        Build Corpuses %s''' % (str(corpuses))
        exit()

    ask_sure_exit('Sure to overwrite corpus / networks ?')

    fn_corpus_build = os.path.join(config['bdir'], 'networks', 'Corpuses.txt')
    _f = open(fn_corpus_build, 'a')
    _f.write('/**** %s ****/\n\n' % (datetime.now()))

    for corpus_name in corpuses:
        startt = datetime.now()
        frontend = frontendNetwork(config)
        frontend.load_data(corpus_name)
        building_corpus_time = (ellapsed_time('Prepropressing %s'%corpus_name, startt) - startt)
        prop = frontend.get_data_prop()
        prop.update(time='%0.3f' % (building_corpus_time.total_seconds()/60) )
        msg = frontend.template(prop)
        print msg
        _f.write(msg)
        _f.flush()

