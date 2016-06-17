import sys, os
import cPickle, json
from itertools import chain
from string import Template
from collections import defaultdict
import logging
lgg = logging.getLogger('root')

from frontend_io import *
from local_utils import *
from vocabulary import Vocabulary, parse_corpus

### @Debug :
#   * update config path !
#   * Separate attribute of the frontend: dataset / Learning / IHM ...

# review that:
#    * separate better load / save and preprocessing (input can be file or array...)
#    * view of file confif.... and path creation....

class DataBase(object):
    """
###################################################################
### Root Class for Frontend Manipulation over Corpuses and Models.

    Given Data Y, and Model M = {\theta, \Phi}
    E[Y] = \theta \phi^T

    Fonctionality are of the frontend decline as:
    1. Frontend for model/algorithm I/O,
    2. Frontend for Corpus Information, and Result Gathering for
        Machine Learning Models.
    3. Data Analisis and Prediction..

    load_corpus -> load_text_corpus -> text_loader
    (frontent)  ->   (choice)       -> (adapt preprocessing)

"""

    def __init__(self, config):
        if config.get('seed'):
            np.random.seed(config.get('seed'))
        self.seed = np.random.get_state()
        self.config = config
        self.corpus_name = config.get('corpus_name')
        self.model_name = config.get('model_name')

        self.K = config.get('K', 10)
        self.C = config.get('C', 10)
        self.N = config.get('N')
        self.homo = int(config.get('homo', 0))
        self.hyper_optimiztn = config.get('hyper')
        self.true_classes = None
        self.data = None
        self.data_t = None

        self._load_data = config.get('load_data')
        self._save_data = config.get('save_data')

        # Read Directory
        #self.make_output_path()


    def make_output_path(self, corpus_name=None):
        # Write Path (for models results)
        config = self.config
        if not hasattr(self, 'basedir') or corpus_name:
            self.basedir = os.path.join(self.bdir, corpus_name)
        corpus_name = corpus_name or self.corpus_name
        fname_out = '%s_%s_%s_%s_%s.out' % (self.model_name,
                                            self.K,
                                            self.hyper_optimiztn,
                                            self.homo,
                                            self.N)

        config['output_path'] = os.path.join(self.basedir,
                                             config.get('refdir', ''),
                                             config.get('repeat', ''),
                                             fname_out)

    @staticmethod
    def corpus_walker(bdir):
        raise NotImplementedError()

    def load_data(self):
        raise NotImplementedError()

    def get_data_prop(self):
        prop = defaultdict()
        prop.update( {'corpus_name': self.corpus_name,
                'instances' : self.data.shape[1] })
        return prop

    # Template for corpus information: Instance, Nnz, features etx
    def template(self, dct, templ):
        return Template(templ).substitute(dct)

    def shuffle_instances(self):
        index = np.arange(np.shape(self.data)[0])
        np.random.shuffle(index)
        self.data =  self.data[index, :]
        #if hasattr(self.data, 'A'):
        #    data = self.data.A
        #    np.random.shuffle(data)
        #    self.data = sp.sparse.csr_matrix(data)
        #else:
        #    np.random.shuffle(self.data)

    def shuffle_features(self):
        raise NotImplemented

    # Return a vector with document generated from a count matrix.
    # Assume sparse matrix
    @staticmethod
    def sparse2stream(data):
        #new_data = []
        #for d in data:
        #    new_data.append(d[d.nonzero()].A1)
        bow = []
        for doc in data:
            # Also, see collections.Counter.elements() ...
            bow.append( np.array(list(chain(* [ doc[0,i]*[i] for i in doc.nonzero()[1] ]))))
        bow = np.array(bow)
        #map(np.random.shuffle, bow)
        return bow

    # Pickle class
    def save(self, data, f):
        with open(f, 'w') as _f:
            return cPickle.dump(data, _f)

    def load(self, f):
        with open(f, 'r') as _f:
            return cPickle.load(_f)


from util.frontendtext import frontendText
from util.frontendnetwork import frontendNetwork
class FrontendManager(object):
    """ Utility Class who aims at mananing the frontend at the higher level.
    """
    @staticmethod
    def get(config):
        """ Return: The frontend suited for the given configuration"""
        corpus = config.get('corpus_name')
        corpus_typo = {'network': ['facebook','generator', 'bench', 'clique', 'fb_uc', 'manufacturing'],
                       'text': ['reuter50', 'nips12', 'nips', 'enron', 'kos', 'nytimes', 'pubmed', '20ngroups', 'odp', 'wikipedia', 'lucene']}

        frontend = None
        for key, cps in corpus_typo.items():
            if corpus.startswith(tuple(cps)):
                if key == 'text':
                    frontend = frontendText(config)
                elif key == 'network':
                    frontend = frontendNetwork(config)

        if frontend is None:
            raise ValueError('Unknown Corpus `%s\'!' % corpus)
        return frontend

# @debug, have to be called before the ModelManager import
class ModelBase(object):
    """"  Root Class for all the Models.

    * Suited for unserpervised model
    * Virtual methods for the desired propertie of models
    """
    def __init__(self):
        self._samples = []
        super(ModelBase, self).__init__()

    # Write data with buffer manager
    def write_some(self, samples, buff=20):
        f = self._f
        fmt = self.fmt

        if samples is None:
            buff=1
        else:
            self._samples.append(samples)

        if len(self._samples) >= buff:
            samples = np.array(self._samples)
            np.savetxt(f, samples, fmt=fmt)
            f.flush()
            self._samples = []

    def close(self):
        if not hasattr(self, '_f'):
            return
        # Write remaining data
        if self._samples:
            self.write_some(None)
        self._f.close()


    def similarity_matrix(self, theta=None, phi=None, sim='cos'):
        if theta is None:
            theta = self.theta
        if phi is None:
            phi = self.phi

        features = theta
        if sim == 'dot':
            sim = np.dot(features, features.T)
        elif sim == 'cos':
            norm = np.linalg.norm(features, axis=1)
            sim = np.dot(features, features.T)/norm/norm.T
        return sim

    def get_params(self):
        return self.reduce_latent()
        if hasattr(self, 'theta') and hasattr(self, 'phi'):
            return self.theta, self.phi
        else:
            return self.reduce_latent()

    def get_clusters(self):
        theta, phi = self.get_params()
        return np.argmax(theta, axis=1)

    # Remove variable that are non serializable.
    def purge(self):
        return

    def update_hyper(self):
        raise NotImplementedError
    def get_hyper(self):
        raise NotImplementedError
    # Just for MCMC ?():
    def reduce_latent(self):
        raise NotImplementedError
    def communities_analysis():
        raise NotImplementedError
    def generate(self):
        raise NotImplementedError
    def predict(self):
        raise NotImplementedError
    def run(self):
        raise NotImplementedError


# Model Import
from hdp import mmsb, lda
from ibp.ilfm_gs import IBPGibbsSampling

sys.path.insert(1, '../../gensim')
import gensim as gsm
from gensim.models import ldamodel, ldafullbaye
Models = {'ldamodel': ldamodel, 'ldafullbaye': ldafullbaye, 'hdp': 1}

class ModelManager(object):
    """ Utility Class for Managing I/O and debugging Models
    """
    def __init__(self, data=None, config=None, data_t=None):
        self.data = data
        self.data_t = data_t

        self.model_name = config.get('model_name')
        #models = {'ilda' : HDP_LDA_CGS,
        #          'lda_cgs' : LDA_CGS, }
        self.hyperparams = config.get('hyperparams')
        self.output_path = config.get('output_path')
        self.K = config.get('K')
        self.inference_method = '?'

        self.write = config.get('write', False)
        self.config = config

        if self.config.get('load_model'):
            return
        if not self.model_name:
            return

        if self.model_name in ('ilda', 'lda_cgs', 'immsb', 'mmsb_cgs'):
            self.model = self.loadgibbs(self.model_name)
        elif self.model_name in ('lda_vb'):
            self.model = self.lda_gensim(model='ldafullbaye')
        elif self.model_name in ('ilfrm', 'ibp', 'ibp_cgs'):
            if '_cgs' in self.model_name:
                metropolis_hastings_k_new = False
            else:
                metropolis_hastings_k_new = True
                if self.config['homo'] == 2:
                    raise NotImplementedError('Warning !: Metropolis Hasting not implemented with matrix normal. Exiting....')
            alpha_hyper_parameter = self.config['hyper']
            sigma_w_hyper_parameter = None #(1., 1.)
            symmetric = self.config['symmetric']
            assortativity = self.config['homo']
            # Hyper parameter init
            alpha = self.hyperparams['alpha']
            sigma_w = 1.
            self.model = IBPGibbsSampling(symmetric, assortativity, alpha_hyper_parameter, sigma_w_hyper_parameter, metropolis_hastings_k_new,
                                         iterations=self.config['iterations'], output_path=self.output_path, write=self.write)
            self.model._initialize(data, alpha, sigma_w, KK=self.K)
            print 'Warning: K is IBP initialized...'
            #self.model._initialize(data, alpha, sigma_w, KK=None)
        else:
            raise NotImplementedError()

    # Base class for Gibbs, VB ... ?
    def loadgibbs(self, target, likelihood=None):
        delta = self.hyperparams['delta']
        alpha = self.hyperparams['alpha']
        gmma = self.hyperparams['gmma']
        mask = self.config['mask']
        symmetric = self.config.get('symmetric',False)
        assortativity = self.config['homo']
        K = self.K

        if 'mmsb' in target:
            kernel = mmsb
        elif 'lda' in target:
            kernel = lda

        if likelihood is None:
            likelihood = kernel.DirMultLikelihood(delta, self.data, symmetric=symmetric, assortativity=assortativity)

        if target.split('_')[-1] == 'cgs':
            # Parametric case
            jointsampler = kernel.CGS(kernel.ZSamplerParametric(alpha, likelihood, K, data_t=self.data_t))
        else:
            # Nonparametric case
            zsampler = kernel.ZSampler(alpha, likelihood, K_init=K, data_t=self.data_t)
            msampler = kernel.MSampler(zsampler)
            betasampler = kernel.BetaSampler(gmma, msampler)
            jointsampler = kernel.NP_CGS(zsampler, msampler, betasampler, hyper=self.config['hyper'],)

        return kernel.GibbsRun(jointsampler, iterations=self.config['iterations'],
                        output_path=self.output_path, write=self.write, data_t=self.data_t)

    def lda_gensim(self, id2word=None, save=False, model='ldamodel', load=False, updatetype='batch'):
        #lgg.setLevel(logging.DEBUG)
        #logging.getLogger().setLevel(logging.DEBUG)
        #logging.basicConfig(format='Gensim : %(message)s', level=logging.DEBUG)
        fname = self.output_path if self.write else None
        iter = self.config['iterations']
        data = self.data
        heldout_data = self.data_t
        delta = self.hyperparams['delta']
        #alpha = self.hyperparams['alpha']
        alpha = 'asymmetric'
        K = self.K
        if load:
            return Models[model].LdaModel.load(fname)

        if hasattr(data, 'tocsc'):
            # is csr sparse matrix
            data = data.tocsc()
            data = gsm.matutils.Sparse2Corpus(data, documents_columns=False)
            if heldout_data is not None:
                heldout_data = heldout_data.tocsc()
                heldout_data = gsm.matutils.Sparse2Corpus(heldout_data, documents_columns=False)
        elif isanparray:
            # up tocsc ??!!! no !
            dense2corpus
        # Passes is the iterations for batch onlines and iteration the max it in the gamma treshold test loop
        # Batch setting !
        if updatetype == 'batch':
            lda = Models[model].LdaModel(data, id2word=id2word, num_topics=K, alpha=alpha, eta=delta,
                                         iterations=100, eval_every=None, update_every=None, passes=iter, chunksize=200000,
                                         fname=fname, heldout_corpus=heldout_data)
        elif updatetype == 'online':
            lda = Models[model].LdaModel(data, id2word=id2word, num_topics=K, alpha=alpha, eta=delta,
                                         iterations=100, eval_every=None, update_every=1, passes=1, chunksize=2000,
                                         fname=fname, heldout_corpus=heldout_data)

        if save:
            lda.expElogbeta = None
            lda.sstats = None
            lda.save(fname)
        return lda

    def run(self):
        if hasattr(self.model, 'run'):
            self.model.run()

    def predict(self, frontend):
        if not hasattr(self.model, 'predict'):
            print 'No predict method for self._name_ ?'
            return

        res = self.model.predict()

        ### Global Measure
        # Global Preferential Aattachment
        # Overall Links Density
        res['density_all'] = frontend.density()
        res['degree_all'] = jsondict(frontend.degree())
        res['degree_hist'] = frontend.degree_histogram()
        res['clustering_coefficient'] = frontend.clustering_coefficient()

        if self.write:
            # Write ...
            fn = self.output_path[:-len('.out')] + '.json'
            json.dump(res, open(fn,'w'))
            self.save()
        else:
            print res

    # Measure perplexity on different initialization
    def init_loop_test(self):
        niter = 2
        pp = []
        likelihood = self.model.s.zsampler.likelihood
        for i in xrange(niter):
            self.model.s.zsampler.estimate_latent_variables()
            pp.append( self.model.s.zsampler.perplexity() )
            self.model = self.loadgibbs(self.model_name, likelihood)

        print self.output_path
        np.savetxt('t.out', np.log(pp))

    # Pickle class
    def save(self):
        fn = self.output_path[:-len('.out')] + '.pk'
        ### Debug for non serializable variables
        #for u, v in vars(self.model).items():
        #    with open(f, 'w') as _f:
        #        try:
        #            cPickle.dump(v, _f)
        #        except:
        #            print 'not serializable here: %s, %s' % (u, v)
        self.model._f = None
        self.model.purge()

        with open(fn, 'w') as _f:
            return cPickle.dump(self.model, _f)

    # Debug classmethod and ecrasement d'object en jeux.
    #@classmethod
    def load(self, spec=None):
        if spec == None:
            fn = self.output_path[:-len('.out')] + '.pk'
        else:
            fn = make_output_path(spec, 'pk')
        if not os.path.isfile(fn) or os.stat(fn).st_size == 0:
            print 'No file for this model: %s' %fn
            for f in model_walker(os.path.dirname(fn), fmt='list'):
                print f
            print 'Pick the right models, maestro !'
            raise IOError
        with open(fn, 'r') as _f:
            return cPickle.load(_f)



