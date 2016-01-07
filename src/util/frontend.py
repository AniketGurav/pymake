import sys, os
from itertools import chain
from string import Template
from collections import defaultdict

from local_utils import *
from vocabulary import Vocabulary, parse_corpus

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

### @Debug : update config path !

class frontEndBase(object):

    def __init__(self, config):
        self.seed = np.random.get_state()
        self.corpus_name = config.get('corpus_name')
        self.model_name = config.get('model')

        self.K = config.get('K', 10)
        self.C = config.get('C', 10)
        self.N = config.get('N')
        self.hyper_optimiztn = config.get('hyper')
        self.true_classes = None
        self.corpus = None
        self.corpus_t = None

        self._load_corpus = config.get('load_corpus')
        self._save_corpus = config.get('save_corpus')

        # Read Directory
        self.basedir = os.path.join(config.get('bdir', 'tmp-data'), config.get('corpus_name', ''))
        # Write Path (for models results)
        fname_out = '%s_%s_%s_%s.out' % ( self.model_name, self.K, self.hyper_optimiztn, self.N)
        config['output_path'] = os.path.join(self.basedir, config.get('refdir', ''),  fname_out)

        self.config = config

    def loade_corpus(self):
        raise NotImplementedError()

    def get_corpus_prop(self):
        prop = defaultdict()
        prop.update( {'corpus_name': self.corpus_name,
                'instances' : self.corpus.shape[1] })
        return prop

    # Template for corpus information: Instance, Nnz, features etx
    def template(self, dct, templ):
        return Template(templ).substitute(dct)


    def shuffle_instances(self):
        np.random.shuffle(self.corpus)

    def shuffle_features(self):
        raise NotImplemented

    # Return a vector with document generated from a count matrix.
    # Assume sparse matrix
    @staticmethod
    def sparse2stream(corpus):
        #new_corpus = []
        #for d in corpus:
        #    new_corpus.append(d[d.nonzero()].A1)
        bow = []
        for doc in corpus:
            # Also, see collections.Counter.elements() ...
            bow.append( np.array(list(chain(* [ doc[0,i]*[i] for i in doc.nonzero()[1] ]))))
        bow = np.array(bow)
        map(np.random.shuffle, bow)
        return bow
