#!/usr/bin/env python
# -*- coding: utf-8 -*-

from local_utils import *
from vocabulary import Vocabulary, parse_corpus
from util.frontend import ModelManager, FrontendManager
from util.frontendtext import frontendText

import numpy as np
import scipy as sp
np.set_printoptions(threshold='nan')

_USAGE = '-s'

##################
###### MAIN ######
##################
if __name__ == '__main__':
    config = defaultdict(lambda: False, dict(
        ##### Global settings
        verbose     = 0,
        ##### Input Features / Corpus
        limit_train = None,
        ###### I/O settings
        bdir = '../data',
        load_corpus = False,
        save_corpus = True,
    ))
    config.update(argParse(_USAGE))

    corpuses = ('nips12',)
    corpuses = ('nips12', 'kos','reuter50', 'nips', 'enron', 'nytimes', 'pubmed', '20ngroups')
    print 'Sure to overwrite corpus ?'; exit()

    ############################################################
    ##### Simulation Output
    if confif.get('simul'):
        print '''--- Simulation settings ---
        Build Corpuses %s''' % (str(corpuses))
        exit()

    fn_corpus_build = os.path.join(config['bdir'], 'text','Corpuses.txt')
    _f = open(fn_corpus_build, 'a')
    _f.write('/**** %s ****/\n\n' % (datetime.now()))

    frontend = frontendText(config)

    for corpus_name in corpuses:
        startt = datetime.now()
        frontend.load_data(corpus_name)
        building_corpus_time = (ellapsed_time('Prepropressing %s'%corpus_name, startt) - startt)
        prop = frontend.get_data_prop()
        prop.update(time='%0.3f' % (building_corpus_time.total_seconds()/60) )
        msg = frontend.template(prop)
        _f.write(msg)
        _f.flush()


