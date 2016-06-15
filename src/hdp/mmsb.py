# -*- coding: utf-8 -*-
import collections
import warnings
from datetime import datetime
import logging
lgg = logging.getLogger('root')

import numpy as np
from numpy import ma
import scipy as sp
import sympy as sym
#import sppy

from scipy.special import gammaln
from numpy.random import dirichlet, multinomial, gamma, poisson, binomial, beta
from sympy.functions.combinatorial.numbers import stirling

from util.frontend import DataBase, ModelBase

from util.compute_stirling import load_stirling
_stirling_mat = load_stirling()

#import sys
#sys.setrecursionlimit(10000)

# Implementation of Teh et al. (2005) Gibbs sampler for the Hierarchical Dirichlet Process: Direct assignement.

#warnings.simplefilter('error', RuntimeWarning)

""" @Todo
* sparse masked array ?!@$*

* Add constant to count matrix by default will win some precious second

"""

def lognormalize(x):
    a = np.logaddexp.reduce(x)
    return np.exp(x - a)

def categorical(params):
    return np.where(multinomial(1, params) == 1)[0]


# Broadcast class based on numpy matrix
# Assume delta a scalar.
class DirMultLikelihood(object):

    def __init__(self, delta, data, nodes_list=None, symmetric=False, assortativity=False):

        if nodes_list is None:
            self.nodes_list = [np.arange(data.shape[0]), np.arange(data.shape[1])]
        else:
            self.nodes_list = nodes_list
            raise ValueError('re order the networks ! to avoid using _nmap')

        if type(data) is not np.ma.masked_array:
            data = np.ma.array(data)
        self.data_ma = data
        self.symmetric = symmetric
        self.data_dims = self.get_data_dims()
        self.nnz = self.get_nnz()
        # Vocabulary size
        self.nfeat = self.get_nfeat()

        # assert for coocurence matric
        #assert(self.data_mat.shape[1] == self.nfeat)
        #self.data_mat = sppy.csarray(self.data_mat)

        # Cst for CGS of DM and scala delta as prior.
        self.assortativity = assortativity
        self.delta = delta if isinstance(delta, np.ndarray) else np.asarray([delta] * self.nfeat)
        self.w_delta = self.delta.sum()
        if assortativity == 1:
            raise NotImplementedError('assort 2')
        elif assortativity == 2:
            self.epsilon = 0.01
        else:
            pass

        # for loglikelihood bernoulli computation
        self.data_A = self.data_ma.copy()
        self.data_A.data[self.data_A.data == 0] = -1
        self.data_B = np.ones(self.data_ma.shape) - self.data_ma


    def __call__(self, j, i, k_ji, k_ij):
        return self.loglikelihood(j, i, k_ji, k_ij)

    def get_nfeat(self):
        nfeat = self.data_ma.max() + 1
        if nfeat == 1:
            print 'Warning, only zeros in adjacency matrix...'
            nfeat = 2
        return nfeat

    # Contains the index of nodes with who it interact.
    # @debug no more true for bipartite networks
    def get_data_dims(self):
        #data_dims = np.vectorize(len)(self.data)
        #data_dims = [r.count() for r in self.data_ma]
        data_dims = []
        for i in xrange(self.data_ma.shape[0]):
            data_dims.append(self.data_ma[i,:].count() + self.data_ma[:,i].count())
        return data_dims

    def get_nnz(self):
        return len(self.data_ma.compressed())

    # Need it in the case of sampling sysmetric networks. (only case where ineed to map ?)
    # return the true node index corresponding to arbitrary index i of matrix count/data position
    # @pos: 0 indicate line picking, 1 indicate rows picking
    def _nmap(self, i, pos):
        return self.nodes_list[pos][i]

    # @debug: symmetric matrix ?
    def make_word_topic_counts(self, z, K):
        word_topic_counts = np.zeros((self.nfeat, K, K))

        for j, i in self.data_iter():
            z_ji = z[j,i,0]
            z_ij = z[j,i,1]
            word_topic_counts[self.data_ma[j, i], z_ji, z_ij] += 1
            if self.symmetric:
                word_topic_counts[self.data_ma[j, i], z_ij, z_ji] += 1

        self.word_topic_counts = word_topic_counts.astype(int)

    # Interface to properly iterate over data
    def data_iter(self, randomize=True):
        if not hasattr(self, '_order'):
            order = np.arange(len(self.data_dims)**2).reshape(self.data_ma.shape)
            masked = order[self.data_ma.mask]

            if self.symmetric:
                tril = np.tril_indices_from(self.data_ma, -1)
                tril = order[tril]
                masked =  np.append(masked, tril)

            # Remove masked value to the iteration list
            order = np.delete(order, masked)
            # Get the indexes of nodes (i,j) for each observed interactions
            order = zip(*np.unravel_index(order, self.data_ma.shape))
            self._order = order
        else:
            order = self._order

        if randomize is True:
            np.random.shuffle(order)
        return order

    # @debug: symmetric matrix ?
    def loglikelihood(self, j, i, k_ji, k_ij):
        w_ji = self.data_ma[j, i]
        self.word_topic_counts[w_ji, k_ji, k_ij] -= 1
        self.total_w_k[k_ji, k_ij] -= 1
        if self.symmetric:
            self.word_topic_counts[w_ji, k_ij, k_ji] -= 1
            self.total_w_k[k_ij, k_ji] -= 1

        if self.assortativity == 2:
            if k_ji == k_ij:
                log_smooth_count_ji = np.log(self.word_topic_counts[w_ji] + self.delta[w_ji])
                ll = log_smooth_count_ji - np.log(self.total_w_k + self.w_delta)
            else:
                ll = self.epsilon

        else:
            log_smooth_count_ji = np.log(self.word_topic_counts[w_ji] + self.delta[w_ji])
            ll = log_smooth_count_ji - np.log(self.total_w_k + self.w_delta)

        return ll

class GibbsSampler(object):

    def __iter__(self):
        return self

    def sample(self):
        return self()

class ZSampler(GibbsSampler):
    # Alternative is to keep the two count matrix and
    # The docment-word topic array to trace get x(-ij) topic assignment:
        # C(word-topic)[w, k] === n_dotkw
        # C(document-topic[j, k] === n_jdotk
        # z DxN_k topic assignment matrix
        # --- From this reconstruct theta and phi

    def __init__(self, alpha_0, likelihood, K_init=0, data_t=None):
        self.K_init = K_init
        self.alpha_0 = alpha_0
        self.likelihood = likelihood
        self.symmetric_pt = (self.likelihood.symmetric&1) +1 # the increment for Gibbs iteration
        self._nmap = likelihood._nmap
        self.nodes_list = likelihood.nodes_list
        self.data_dims = self.likelihood.data_dims
        self.J = len(self.data_dims)
        self.z = self._init_topics_assignement()
        self.doc_topic_counts = self.make_doc_topic_counts().astype(int)
        if not hasattr(self, 'K'):
            # Nonparametric Case
            self.purge_empty_topics()
        self.likelihood.make_word_topic_counts(self.z, self.get_K())
        self.likelihood.total_w_k = self.likelihood.word_topic_counts.sum(0)

        # if a tracking of topics indexis pursuit,
        # pay attention to the topic added among those purged...(A topic cannot be added and purged in the same sample iteration !)
        self.last_purged_topics = []

    # @debug: symmetric matrix ?
    def _init_topics_assignement(self):
        dim = (self.J, self.J, 2)
        alpha_0 = self.alpha_0

        # Poisson way
        #z = np.array( [poisson(alpha_0, size=dim) for dim in data_dims] )

        # Random way
        K = self.K_init
        z = np.random.randint(0, K, (dim))

        if self.likelihood.symmetric:
            z[:, :, 0] = np.triu(z[:, :, 0]) + np.triu(z[:, :, 0], 1).T
            z[:, :, 1] = np.triu(z[:, :, 1]) + np.triu(z[:, :, 1], 1).T

        # LDA way
        # improve local optima ?
        #theta_j = dirichlet([1, gmma])
        #todo ?

        return z

    def __call__(self):
        # Add pnew container
        self._update_log_alpha_beta()
        self.update_matrix_shape()

        lgg.info('Sample z...')
        lgg.debug('#J \t #I \t  #topic')
        doc_order = np.random.permutation(self.J)
        # @debug: symmetric matrix !
        for j, i in self.likelihood.data_iter(randomize=True):
            lgg.debug( '%d \t %d \t %d' % ( j , i, self.doc_topic_counts.shape[1]-1))
            params = self.prob_zji(j, i, self._K + 1)
            sample_topic_raveled = categorical(params)
            k_j, k_i = np.unravel_index(sample_topic_raveled, (self._K+1, self._K+1))
            k_j, k_i = k_j[0], k_i[0] # beurk :(
            self.z[j, i, 0] = k_j
            self.z[j, i, 1] = k_i

            # Regularize matrices for new topic sampled
            if k_j == self.doc_topic_counts.shape[1] - 1 or k_i == self.doc_topic_counts.shape[1] - 1:
                self._K += 1
                #print 'Simplex probabilities: %s' % (params)
                self.update_matrix_shape(new_topic=True)

            self.update_matrix_count(j,i,k_j, k_i)

        # Remove pnew container
        self.doc_topic_counts = self.doc_topic_counts[:, :-1]
        self.likelihood.word_topic_counts = self.likelihood.word_topic_counts[:, :-1, :-1]
        self.purge_empty_topics()

        return self.z

    def update_matrix_shape(self, new_topic=False):
        if new_topic is True:
            # Updata alpha
            self.log_alpha_beta = np.append(self.log_alpha_beta, self.log_alpha_beta[-1])
            self.alpha = np.append(self.alpha, np.exp(self.log_alpha_beta[-1]))

        # Update Doc-topic count
        new_inst = np.zeros((self.J, 1), dtype=int)
        self.doc_topic_counts = np.hstack((self.doc_topic_counts, new_inst))

        # Update word-topic count
        new_feat_1 = np.zeros((self.likelihood.nfeat, self._K), dtype=int)
        new_feat_2 = np.zeros((self.likelihood.nfeat, self._K+1), dtype=int)
        self.likelihood.word_topic_counts = np.dstack((self.likelihood.word_topic_counts, new_feat_1))
        self.likelihood.word_topic_counts = np.hstack((self.likelihood.word_topic_counts, new_feat_2[:, None]))
        # sum all to update to fit the shape (a bit nasty if operation (new topic) occur a lot)
        self.likelihood.total_w_k = self.likelihood.word_topic_counts.sum(0)

    def update_matrix_count(self, j, i, k_j, k_i):
        self.doc_topic_counts[j, k_j] += self.symmetric_pt
        self.doc_topic_counts[i, k_i] += self.symmetric_pt
        self.likelihood.word_topic_counts[self.likelihood.data_ma[j,i], k_j, k_i] += 1
        self.likelihood.total_w_k[k_j, k_i] += 1
        if self.likelihood.symmetric:
            self.likelihood.word_topic_counts[self.likelihood.data_ma[j,i], k_i, k_j] += 1
            self.likelihood.total_w_k[k_i, k_j] += 1

    # @debug: symmetric matrix ?
    def make_doc_topic_counts(self):
        K = self.get_K()
        counts = np.zeros((self.J, K))

        for j, i in self.likelihood.data_iter(randomize=False):
            k_j = self.z[j, i, 0]
            k_i = self.z[j, i, 1]
            counts[j, k_j] += self.symmetric_pt
            counts[i, k_i] += self.symmetric_pt
        return counts

    def _update_log_alpha_beta(self):
        self.log_alpha_beta = np.log(self.alpha_0) + np.log(self.betasampler.beta)
        self.alpha = np.exp(self.log_alpha_beta)

    # Remove empty topic in nonparametric case
    # @debug: symmetric matrix ?
    def purge_empty_topics(self):
        counts = self.doc_topic_counts

        dummy_topics = []
        # Find empty topics
        for k, c in enumerate(counts.T):
            if c.sum() == 0:
                dummy_topics.append(k)
        for k in sorted(dummy_topics, reverse=True):
            counts = np.delete(counts, k, axis=1)
            self._K -= 1
            if hasattr(self.likelihood, 'word_topic_counts'):
                self.likelihood.word_topic_counts = np.delete(self.likelihood.word_topic_counts, k, axis=1)
                self.likelihood.word_topic_counts = np.delete(self.likelihood.word_topic_counts, k, axis=2)
            # Regularize Z
            for d in self.z:
                d[d > k] -= 1
            # Regularize alpha_beta, minus one the pnew topic
            if hasattr(self, 'log_alpha_beta') and k < len(self.log_alpha_beta)-1:
                self.log_alpha_beta = np.delete(self.log_alpha_beta, k)
                self.betasampler.beta = np.delete(self.betasampler.beta, k)

        self.last_purged_topics = dummy_topics
        if len(dummy_topics) > 0:
            lgg.info( 'zsampler: %d topics purged' % (len(dummy_topics)))
        self.doc_topic_counts =  counts

    def add_beta_sampler(self, betasampler):
        self.betasampler = betasampler
        self._update_log_alpha_beta()

    def get_K(self):
        if not hasattr(self, '_K'):
            self._K =  np.max(self.z) + 1
        return self._K

    # Compute probabilityy to sample z_ij = k for each [K].
    # K is would be fix or +1 for nonparametric case
    def prob_zji(self, j, i, K):
        k_jji = self.z[j, i, 0]
        k_jij = self.z[j, i, 1]
        self.doc_topic_counts[j, k_jji] -= self.symmetric_pt
        self.doc_topic_counts[i, k_jij] -= self.symmetric_pt

        # Keep the outer product in memory
        p_jk = self.doc_topic_counts[j] + self.alpha
        p_ik = self.doc_topic_counts[i] + self.alpha
        outer_kk = np.outer(p_jk, p_ik)

        params = np.log(outer_kk) + self.likelihood(j, i, k_jji, k_jij)
        params = params[:K, :K].ravel()
        return lognormalize(params)

    def get_log_alpha_beta(self, k):
        old_max = self.log_alpha_beta.shape[0]

        if k > (old_max - 1):
            return self.log_alpha_beta[old_max - 1]
        else:
            return self.log_alpha_beta[k]

    def clean(self):
        self.K = self.doc_topic_counts.shape[1]

    def predictive_topics(self, data):
        pass

    def estimate_latent_variables(self):
        # check if perplexity is equal if removing dummy empty topics...
        if not hasattr(self, 'logalpha'):
            log_alpha_beta = self.log_alpha_beta
            new_k = self.get_K()+1 - len(log_alpha_beta)
            if new_k > 0:
                gmma = log_alpha_beta[-1]
                log_alpha_beta = np.hstack((log_alpha_beta, np.ones((new_k,))*gmma))
            # Remove empty possibly new topic
            alpha = np.exp(log_alpha_beta[:-1])
        else:
            alpha = np.exp(self.logalpha)
        delta = self.likelihood.delta
        K = len(alpha)

        # Recontruct Documents-Topic matrix
        _theta = self.doc_topic_counts + np.tile(alpha, (self.J, 1))
        self._theta = (_theta.T / _theta.sum(axis=1)).T

        # Recontruct Words-Topic matrix
        _phi = self.likelihood.word_topic_counts + np.tile(delta, (K, K, 1)).T
        self._phi = (_phi / np.linalg.norm(_phi, axis=0))[1]

        return self._theta, self._phi

    # Mean can be arithmetic or geometric
    def perplexity(self, data=None, mean='arithmetic'):
        phi = self._phi
        if data is None:
            data = self.likelihood.data_ma
            nnz = self.likelihood.nnz
            theta = self._theta
        else:
            nnz = data.sum()
            theta = self.predictive_topics(data)

        ### based on aritmetic mean

        ### Loop Approach
        #entropy = 0.0
        #_indices = lambda x: x.nonzero()[1]
        #for j in xrange(self.J):
        #    data_j = [ (i, data[j, i]) for i in _indices(data[j]) ]
        #    entropy += np.sum( cnt_wi * np.log(theta[j] * phi[w_ji]).sum() for w_ji, cnt_wi in data_j )

        ### Vectorized approach
        # < 2s for kos and nips k=50, quite fast
        p_ji =  theta.dot(phi).dot(theta.T)
        # p_ji if a links, (1 - p_ji) if not a links
        p_ji = self.likelihood.data_A * p_ji + self.likelihood.data_B

        # Commented because count matrices are kept symmetric. So normalized over all to keep coherence.
        #if self.likelihood.symmetric:
        #    entropy =  np.log(p_ji[self.likelihood.triu]).sum()
        #else:
        #    entropy =  np.log(p_ji).sum()
        entropy =  np.log(p_ji).sum()

        #perplexity = np.exp(-entropy / nnz)
        entropy = - entropy / nnz
        return entropy

class ZSamplerParametric(ZSampler):
    # Parametric Version of HDP sampler. Number of topics fixed.

    def __init__(self, alpha_0, likelihood, K, alpha='asymmetric', data_t=None):
        self.K = self.K_init = self._K =  K
        if 'alpha' in ('symmetric', 'fix'):
            alpha = np.ones(K)*1/K
        elif 'alpha' in ('asymmetric', 'auto'):
            alpha = np.asarray([1.0 / (i + np.sqrt(K)) for i in xrange(K)])
            alpha /= alpha.sum()
        else:
            alpha = np.ones(K)*alpha_0
        self.logalpha = np.log(alpha)
        self.alpha = alpha
        super(ZSamplerParametric, self).__init__(alpha_0, likelihood, self.K, data_t=data_t)

    def __call__(self):
        print('Sample z...')
        lgg.debug('#J \t #I \t #topic')
        doc_order = np.random.permutation(self.J)
        # @debug: symmetric matrix !
        for j, i in self.likelihood.data_iter(randomize=True):
            lgg.debug( '%d \t %d \t %d' % (j , i, self.doc_topic_counts.shape[1]-1))
            params = self.prob_zji(j, i, self.K)
            sample_topic_raveled = categorical(params)
            k_j, k_i = np.unravel_index(sample_topic_raveled, (self._K, self._K))
            k_j, k_i = k_j[0], k_i[0] # beurk :(
            self.z[j, i, 0] = k_j
            self.z[j, i, 1] = k_i
            nodes_classes_ass = [(j, k_j), (i, k_i)]

            self.update_matrix_count(j, i, k_j, k_i)
        return self.z

    def get_K(self):
        return self.K

    def get_log_alpha_beta(self, k):
        return self.logalpha[k]

    def clean(self):
        pass


class MSampler(GibbsSampler):

    def __init__(self, zsampler):
        self.stirling_mat = _stirling_mat
        self.zsampler = zsampler
        self.get_log_alpha_beta = zsampler.get_log_alpha_beta
        self.count_k_by_j = zsampler.doc_topic_counts

        # We don't know the preconfiguration of tables !
        self.m = np.ones(self.count_k_by_j.shape, dtype=int)
        self.m_dotk = self.m.sum(axis=0)

    def __call__(self):
        self._update_m()

        indices = np.ndenumerate(self.count_k_by_j)

        lgg.info( 'Sample m...')
        for ind in indices:
            j, k = ind[0]
            count = ind[1]

            if count > 0:
                # Sample number of tables in j serving dishe k
                params = self.prob_jk(j, k)
                sample = categorical(params) + 1
            else:
                sample = 0

            self.m[j, k] = sample

        self.m_dotk = self.m.sum(0)
        self.purge_empty_tables()

        return self.m

    def _update_m(self):
        # Remove tables associated with purged topics
        for k in sorted(self.zsampler.last_purged_topics, reverse=True):
            self.m = np.delete(self.m, k, axis=1)

        # Passed by reference, but why not...
        self.count_k_by_j = self.zsampler.doc_topic_counts
        K = self.count_k_by_j.shape[1]
        # Add empty table for new fancy topics
        new_k = K - self.m.shape[1]
        if new_k > 0:
            lgg.info( 'msampler: %d new topics' % (new_k))
            J = self.m.shape[0]
            self.m = np.hstack((self.m, np.zeros((J, new_k), dtype=int)))

    # Removes empty table.
    def purge_empty_tables(self):
        # cant be.
        pass

    def prob_jk(self, j, k):
        # -1 because table of current sample topic jk, is not conditioned on
        njdotk = self.count_k_by_j[j, k]
        if njdotk == 1:
            return np.ones(1)

        possible_ms = np.arange(1, njdotk) # +1-1
        log_alpha_beta_k = self.get_log_alpha_beta(k)
        alpha_beta_k = np.exp(log_alpha_beta_k)

        normalizer = gammaln(alpha_beta_k) - gammaln(alpha_beta_k + njdotk)
        log_stir = self.stirling_mat[njdotk, possible_ms]
        #log_stir = sym.log(stirling(njdotk, m, kind=1)).evalf() # so long.

        params = normalizer + log_stir + possible_ms*log_alpha_beta_k

        return lognormalize(params)

class BetaSampler(GibbsSampler):

    def __init__(self, gmma, msampler):
        self.gmma = gmma
        self.msampler = msampler

        # Initialize restaurant with just one table.
        self.beta = dirichlet([1, gmma])

    def __call__(self):
        lgg.info( 'Sample Beta...')
        self._update_dirichlet_params()
        self.beta = dirichlet(self.dirichlet_params)

        return self.beta

    def _update_dirichlet_params(self):
        m_dotk_augmented = np.append(self.msampler.m_dotk, self.gmma)
        lgg.info( 'Beta Dirichlet Prior: %s' % (m_dotk_augmented))
        self.dirichlet_params = m_dotk_augmented

class NP_CGS(GibbsSampler):

    # Joint Sampler of topic Assignement, table configuration, and beta proportion.
    # ref to direct assignement Sampling in HDP (Teh 2006)
    def __init__(self, zsampler, msampler, betasampler, hyper='auto'):
        zsampler.add_beta_sampler(betasampler)

        self.zsampler = zsampler
        self.msampler = msampler
        self.betasampler = betasampler

        msampler.sample()
        betasampler.sample()

        if hyper.startswith( 'auto' ):
            self.hyper = hyper
            self.a_alpha = 1
            self.b_alpha = 1
            self.a_gmma = 1
            self.b_gmma = 1
            self.optimize_hyper_hdp()
        elif hyper.startswith( 'fix' ):
            self.hyper = hyper
        else:
            raise NotImplementedError('Hyperparameters optimization ?')

    def optimize_hyper_hdp(self):
        # Optimize \alpha_0
        m_dot = self.msampler.m_dotk.sum()
        alpha_0 = self.zsampler.alpha_0
        n_jdot = np.array(self.zsampler.data_dims) # @debug add row count + line count for masked !
        #p = np.power(n_jdot / alpha_0, np.arange(n_jdot.shape[0]))
        #norm = np.linalg.norm(p)
        #u_j = binomial(1, p/norm)
        u_j = binomial(1, alpha_0/(n_jdot + alpha_0))
        #u_j = binomial(1, n_jdot/(n_jdot + alpha_0))
        v_j = beta(alpha_0 + 1, n_jdot)
        new_alpha0 = gamma(self.a_alpha + m_dot - u_j.sum(), 1/( self.b_alpha - np.log(v_j).sum()), size=3).mean()
        self.zsampler.alpha_0 = new_alpha0

        # Optimize \gamma
        K = self.zsampler._K
        #m_dot = self.msampler.m_dotk
        gmma = self.betasampler.gmma
        #p = np.power(m_dot / gmma, np.arange(m_dot.shape[0]))
        #norm = np.linalg.norm(p)
        #u = binomial(1, p/norm)
        u = binomial(1, gmma / (m_dot + gmma))
        #u = binomial(1, m_dot / (m_dot + gmma))
        v = beta(gmma + 1, m_dot)
        new_gmma = gamma(self.a_gmma + K -1 + u, 1/(self.b_gmma - np.log(v)), size=3).mean()
        self.betasampler.gmma = new_gmma

        #print 'm_dot %d, alpha a, b: %s, %s ' % (m_dot, self.a_alpha + m_dot - u_j.sum(), 1/( self.b_alpha - np.log(v_j).sum()))
        #print 'gamma a, b: %s, %s ' % (self.a_gmma + K -1 + u, 1/(self.b_gmma - np.log(v)))
        lgg.info( 'hyper sample: alpha_0: %s gamma: %s' % (new_alpha0, new_gmma))
        return

    def __call__(self):
            z = self.zsampler.sample()
            m = self.msampler.sample()
            beta = self.betasampler.sample()

            if self.hyper.startswith('auto'):
                self.optimize_hyper_hdp()

            return z, m, beta

class CGS(GibbsSampler):

    def __init__(self, zsampler):
        self.zsampler = zsampler

    def __call__(self):
        return self.zsampler.sample()

class GibbsRun(ModelBase):

    def __init__(self, sampler, iterations=100, burnin=0.05, thinning_interval=1, output_path=None, write=False, data_t=None):
        super(GibbsRun, self).__init__()
        self.s = sampler
        self.iterations = iterations
        self.burnin = iterations - int(burnin*iterations)
        self.thinning = thinning_interval
        self.samples = []

        self.data_t = data_t
        self.write = write
        if output_path and self.write:
            import os
            bdir = os.path.dirname(output_path) or 'tmp-output'
            fn = os.path.basename(output_path)
            try: os.makedirs(bdir)
            except: pass
            self.fname_i = bdir + '/inference-' + fn.split('.')[0]
            csv_typo = '# it it_time entropy_train entropy_test K alpha gamma alpha_mean delta_mean alpha_var delta_var'
            self.fmt = '%d %.4f %.8f %.8f %d %.8f %.8f %.4f %.4f %.4f %.4f'
            self._f = open(self.fname_i, 'w')
            self._f.write(csv_typo + '\n')

    def measures(self):
        pp = self.evaluate_perplexity()
        if self.data_t is not None:
            pp_t = self.predictive_likelihood()
        else:
            pp_t = np.nan
        k = self.s.zsampler._K
        alpha_0 = self.s.zsampler.alpha_0
        try:
            gmma = self.s.betasampler.gmma
            alpha = np.exp(self.s.zsampler.log_alpha_beta)
        except:
            gmma = np.nan
            alpha = np.exp(self.s.zsampler.logalpha)

        alpha_mean = alpha.mean()
        alpha_var = alpha.var()
        delta_mean = self.s.zsampler.likelihood.delta.mean()
        delta_var = self.s.zsampler.likelihood.delta.var()

        measures = [pp, pp_t, k, alpha_0, gmma, alpha_mean, delta_mean, alpha_var, delta_var]
        return measures

    def run(self):
        time_it = 0
        self.evaluate_perplexity()
        for i in xrange(self.iterations):
            ### Output / Measures
            measures = self.measures()
            sample = [i, time_it] + measures
            k = self.s.zsampler._K
            lgg.info('Iteration %d, K=%d Entropy: %s ' % (i, k, measures[0]))
            if self.write:
                self.write_some(sample)

            begin = datetime.now()
            ### Sampling
            self.s.sample()
            time_it = (datetime.now() - begin).total_seconds() / 60

            if i >= self.iterations:
                s.clean()
                break
            if i >= self.burnin:
                if i % self.thinning == 0:
                    self.samples.append([self.theta, self.phi])

        ### Clean Things
        if not self.samples:
            self.samples.append([self.theta, self.phi])
        self.close()
        return

    def generate(self, N, K=None, alpha=0.01, delta=[0.4, 0.2]):
        #alpha = self.s.zsampler.alpha
        N = int(N)
        if K is not None:
            K = int(K)
            #alpha = np.ones(K) * alpha
            alpha = np.ones(K) * 1/(K*10)
            ##alpha = np.asarray([1.0 / (i + np.sqrt(K)) for i in xrange(K)])
            #alpha /= alpha.sum()
            #delta = self.s.zsampler.likelihood.delta
            delta = delta
            theta = dirichlet(alpha, size=N)
            phi = beta(delta[0], delta[1], size=(K,K))
            phi = np.triu(phi) + np.triu(phi, 1).T
        else:
            theta, phi = self.reduce_latent()
            K = theta.shape[1]

        Y = np.empty((N,N))
        pij = theta.dot(phi).dot(theta.T)
        #pij[pij >= 0.5 ] = 1
        #pij[pij < 0.5 ] = 0
        #Y = pij
        Y = sp.stats.bernoulli.rvs(pij)

        #for j in xrange(N):
        #    print 'j %d' % j
        #    for i in xrange(N):
        #        zj = categorical(theta[j])
        #        zi = categorical(theta[i])
        #        Y[j, i] = sp.stats.bernoulli.rvs(B[zj, zi])
        self.theta = theta
        self.phi = phi
        return Y, theta, phi

    # Nasty hack to make serialisation possible
    def purge(self):
        try:
            self.s.mask = self.s.zsampler.likelihood.data_ma.mask
        except:
            pass

        self.s.zsampler.betasampler = None
        self.s.zsampler._nmap = None
        self.s.msampler = None
        self.s.betasampler = None
        self.s.zsampler.likelihood = None

    def evaluate_perplexity(self, data=None):
        self.theta, self.phi = self.s.zsampler.estimate_latent_variables()
        return self.s.zsampler.perplexity(data)

    # keep only the most representative dimension (number of topics) in the samples
    def reduce_latent(self):
        theta, phi = map(list, zip(*self.samples))
        ks = [ mat.shape[1] for mat in theta]
        bn = np.bincount(ks)
        k_win = np.argmax(bn)
        lgg.info('K selected: %d' % k_win)

        ind_rm = []
        [ind_rm.append(i) for i, v in enumerate(theta) if v.shape[1] != k_win]
        for i in sorted(ind_rm, reverse=True):
            theta.pop(i)
            phi.pop(i)

        lgg.info('Samples Selected: %d over %s' % (len(theta), len(theta)+len(ind_rm) ))

        theta = np.mean(theta, 0)
        phi = np.mean(phi, 0)
        return theta, phi

    # * Precision on masked data
    # -- On Gen Y
    # * Local preferential attachement
    # * Global preferential attachement
    def predict(self):
        lgg.info('Reducing latent variables...')
        theta, phi = self.reduce_latent()

        ### Computing Precision
        likelihood = self.s.zsampler.likelihood
        masked = likelihood.data_ma.mask
        ### @Debug Ignore the Diagonnal
        np.fill_diagonal(masked, False)
        ground_truth = likelihood.data_ma.data[masked]
        data = likelihood.data_ma.data
        test_size = float(ground_truth.size)

        p_ji = theta.dot(phi).dot(theta.T)
        prediction = p_ji[masked]
        #prediction[prediction >= 0.5 ] = 1
        #prediction[prediction < 0.5 ] = 0
        prediction = sp.stats.bernoulli.rvs( prediction )

        good_1 = ((prediction + ground_truth) == 2).sum()
        precision = good_1 / float(prediction.sum())
        rappel = good_1 / float(ground_truth.sum())
        g_precision = (prediction == ground_truth).sum() / test_size
        mask_density = ground_truth.sum() / test_size

        ### Finding Communities
        lgg.info('Finding Communities...')
        community_distribution, local_attach, c = self.communities_analysis(theta)

        res = {'Precision': precision,
               'Rappel': rappel,
               'g_precision': g_precision,
               'mask_density': mask_density,
               'Community_Distribution': community_distribution,
               'Local_Attachment': local_attach
              }

        return res

    def communities_analysis(self, theta, data=None):
        if data is None:
            likelihood = self.s.zsampler.likelihood
            data = likelihood.data_ma.data
            symmetric = likelihood.symmetric
        else:
            symmetric = True

        clusters = np.argmax(theta, axis=1)
        community_distribution = list(np.bincount(clusters))

        local_attach = {}
        for n, _comm in enumerate(clusters):
            comm = str(_comm)
            local = local_attach.get(comm, [])
            degree_n = data[n, :].sum()
            if symmetric:
                degree_n += data[:, n].sum()
            local.append(degree_n)
            local_attach[comm] = local

        return community_distribution, local_attach, clusters



