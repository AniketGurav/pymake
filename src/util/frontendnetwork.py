import sys, os
from itertools import chain
from string import Template

from local_utils import *
from vocabulary import Vocabulary, parse_corpus
from util.frontend import frontEndBase

sys.path.insert(1, '../../gensim')
import gensim
from gensim.models import ldamodel, ldafullbaye
Models = { 'ldamodel': ldamodel, 'ldafullbaye': ldafullbaye, 'hdp': 1}

############################################################
############################################################
#### Aim at be a frontend for corpus data manipulation.
####   * First, purpose is to be the frontend for model/algorithm input,
####   * Second, frontend for data observation. States of corpus or results analysis,
####   * Third, operate on corpus various operation as filtering, merging etc.

class frontEndNetwork(frontEndBase):

    def __init__(self, **conf):
        conf['bdir'] = os.path.join(conf['bdir'], 'networks')
        super(frontEndNetwork, self).__init__(**conf)

    def load_corpus(self, corpus_name=None):
        return True

    def get_netw_prop(self):
        raise NotImplementedError()

    def template(self, dct):
        netw_templ = '''###### $corpus_name
        Building: $time minutes
        Nodes: $instances
        Links: $nnz
        Degree mean: $nnz_mean
        Degree var: $nnz_var
        Diameter: $diameter
        Communities: $communities
        Relations: $features
        train: $train_size
        test: $test_size
        \n'''
        return super(frontEndNetwork, self).template(dct, netw_templ)

