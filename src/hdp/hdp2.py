# -*- coding: utf-8 -*-
import collections
import warnings

import numpy as np
import scipy as sp
import sympy as sym
#import sppy

from scipy.special import gammaln
from numpy.random import dirichlet, multinomial, gamma, poisson, binomial, beta
from sympy.functions.combinatorial.numbers import stirling

from util.frontend import frontEndBase
sparse2stream = frontEndBase.sparse2stream

from util.compute_stirling import load_stirling
_stirling_mat = load_stirling()

import sys
sys.setrecursionlimit(10000)

# Implementation of Teh et al. (2005) Gibbs sampler for the Hierarchical Dirichlet Process: Direct assignement.

#warnings.simplefilter('error', RuntimeWarning)

""" @Todo
* Controm the calcul of perplexity at each iterations. Costly !

* HDP direct assignement:
Network implementation (2*N restaurant, symmetric, p(k,l|.) = p(k)p(l) etc)

"""

def lognormalize(x):
    a = np.logaddexp.reduce(x)
    return np.exp(x - a)

def categorical(params):
    return np.where(multinomial(1, params) == 1)[0]

# Broadcast class based on numpy matrix
class DirMultLikelihood(object):

    def __init__(self, delta, data):
        self.delta = delta
        if type(data) is np.ndarray:
            # Error when streaming ? sppy ?
            #self.data_mat = sp.sparse.csr_matrix(data)
            self.data_mat = data
        elif data.format == 'csr':
            self.data_mat = data
        else:
            raise NotImplementedError('type %s unknow as corpus' % type(data))
        self.data = sparse2stream(self.data_mat)
        assert(len(self.data) > 1)
        for i, j in enumerate(self.data):
            if len(j) == 0:
                print i, j
                print data[i]

        self.data_dims = self.get_data_dims()
        self.nnz = self.get_nnz()
        # Vocabulary size
        self.nfeat = self.get_nfeat()

        #print self.data_mat.shape[1],  self.nfeat
        assert(self.data_mat.shape[1] == self.nfeat)
        #self.data_mat = sppy.csarray(self.data_mat)

    def __call__(self, j, i, k, k_ji):
        return self.loglikelihood(j, i, k, k_ji)

    def get_nfeat(self):
        return np.vectorize(max)(self.data).max() + 1

    def get_data_dims(self):
        data_dims = np.vectorize(len)(self.data)
        return list(data_dims)

    def get_nnz(self):
        return sum(self.data_dims)

    def words_k(self, where_k):
        for group in xrange(len(self.data)):
            word_gen = self.data[group][where_k[group]]
            for word in word_gen:
                yield word

    def make_word_topic_counts(self, z, K):
        word_topic_counts = np.zeros((self.nfeat, K))

        for k in xrange(K):
            where_k =  np.array([np.where(zj == k)[0] for zj in z])
            words_k_dict = collections.Counter(self.words_k(where_k))
            word_topic_counts[words_k_dict.keys(), k] = words_k_dict.values()

        self.word_topic_counts = word_topic_counts.astype(int)

    def loglikelihood(self, j, i, k, k_ji):
        w_ji = self.data[j][i]
        if k == k_ji:
            self.word_topic_counts[w_ji, k_ji] -= 1
        count_ji = self.word_topic_counts[w_ji, k_ji]
        total_k = self.word_topic_counts[:, k_ji].sum()

        log_smooth_count_ji = np.log(count_ji + self.delta)

        #return log_smooth_count_ji - np.log(total + (total + 1)*self.delta)
        return log_smooth_count_ji - np.log(total_k + self.nfeat*self.delta)

class GibbsSampler(object):

    def __iter__(self):
        return self

    def sample(self):
        return self()

class ZSampler(GibbsSampler):
    # Alternative is to keep the two count matrix and
    # the the docment-word topic array to trace get x(-ij) topic assignment:
        # C(word-topic)[w, k] === n_dotkw
        # C(document-topic[j, k] === n_jdotk
        # z DxN_k topic assignment matrix
        # --- From this reconstruct theta and phi

    def __init__(self, alpha_0, likelihood, K_init=1):
        self.K_init = K_init
        self.alpha_0 = alpha_0
        self.likelihood = likelihood
        self.data_dims = likelihood.data_dims
        self.J = len(self.data_dims)
        self.z = self._init_topics_assignement()
        self.doc_topic_counts = self.make_doc_topic_counts().astype(int)
        if not hasattr(self, 'K'):
            # Nonparametric Case
            self.purge_empty_topics()
        self.likelihood.make_word_topic_counts(self.z, self.get_K())

        # if a tracking of topics indexis pursuit,
        # pay attention to the topic added among those purged...
        self.last_purged_topics = []

    # @data_dims: list of number of observations per document/instance
    def _init_topics_assignement(self):
        data_dims = self.data_dims
        alpha_0 = self.alpha_0

        # Poisson way
        #z = np.array( [poisson(alpha_0, size=dim) for dim in data_dims] )

        # Random way
        # Why perplexity so low ?????????????
        K = 1
        #z = np.array( [np.zeros(dim, dtype=int) for dim in data_dims] )

        K = self.K_init
        z = np.array( [np.random.randint(0, K, (dim)) for dim in data_dims] )

        # LDA way
        #todo ?

        return z

    def __call__(self):
        # Add pnew container
        self._update_log_alpha_beta()
        self.doc_topic_counts =  np.column_stack((self.doc_topic_counts, np.zeros(self.J, dtype=int)))
        self.likelihood.word_topic_counts =  np.column_stack((self.likelihood.word_topic_counts, np.zeros(self.likelihood.nfeat, dtype=int)))

        print 'Sample z...'
        print '#Doc \t nnz \t  #topic'
        for j in xrange(self.J):
            nnz =  self.data_dims[j]
            print '%d \t %d \t %d' % ( j , nnz, self.doc_topic_counts.shape[1]-1 )
            for i in xrange(nnz):
                params = self.prob_zji(j, i)

                sample = categorical(params)
                self.z[j][i] = sample

                # Regularize matrices for new topic sampled
                if sample == self.doc_topic_counts.shape[1] - 1:
                    self._K += 1
                    print 'Simplex probabilities: %s' % (params)
                    col_doc = np.zeros((self.J, 1), dtype=int)
                    col_word = np.zeros((self.likelihood.nfeat, 1), dtype=int)
                    self.doc_topic_counts = np.hstack((self.doc_topic_counts, col_doc))
                    self.likelihood.word_topic_counts = np.hstack((self.likelihood.word_topic_counts, col_word))

                # Update count matrixes
                self.doc_topic_counts[j, sample] += 1
                self.likelihood.word_topic_counts[self.likelihood.data[j][i], sample] += 1

        # Remove pnew container
        self.doc_topic_counts = self.doc_topic_counts[:, :-1]
        self.likelihood.word_topic_counts = self.likelihood.word_topic_counts[:,:-1]
        self.purge_empty_topics()

        return self.z

    def make_doc_topic_counts(self):
        K = self.get_K()
        counts = np.zeros((self.J, K))
        for j, d in enumerate(self.z):
            counts[j] = np.bincount(d, minlength=K)

        return counts

    def _update_log_alpha_beta(self):
        self.log_alpha_beta = np.log(self.alpha_0) + np.log(self.betasampler.beta)

    # Remove empty topic in nonparametric case
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
            # Regularize Z
            for d in self.z:
                d[d > k] -= 1
            # Regularize alpha_beta, minus one the pnew topic
            if hasattr(self, 'log_alpha_beta') and k < len(self.log_alpha_beta)-1:
                self.log_alpha_beta = np.delete(self.log_alpha_beta, k)
                self.betasampler.beta = np.delete(self.betasampler.beta, k)

        self.last_purge_topics = dummy_topics
        if len(dummy_topics) > 0:
            print 'zsampler: %d topics purged' % (len(dummy_topics))
        self.doc_topic_counts =  counts

    def add_beta_sampler(self, betasampler):
        self.betasampler = betasampler
        self._update_log_alpha_beta()

    def get_K(self):
        if not hasattr(self, '_K'):
            self._K =  np.max(np.vectorize(np.max)(self.z)) + 1
        return self._K


    def prob_zji(self, j, i):
        njdot_minusji = self.doc_topic_counts[j]
        k_ji = self.z[j][i]
        njdot_minusji[k_ji] -= 1

        new_t = 0 if hasattr(self, 'K') else 1
        possible_ks = range(self.get_K() + new_t)

        params = [self.prob_zji_eq_k(k, j, i, njdot_minusji[k]) for k in possible_ks]

        return lognormalize(params)

    def prob_zji_eq_k(self, k, j, i, njdotk_minusji):
        log_alpha_beta_k = self.get_log_alpha_beta(k)

        if njdotk_minusji > 0:
            prior = np.logaddexp(np.log(njdotk_minusji), log_alpha_beta_k)
        else:
            # new topic
            prior = log_alpha_beta_k

        k_ji = self.z[j][i]
        ll = self.likelihood(j, i, k, k_ji)

        return prior + ll

    def get_log_alpha_beta(self, k):
        old_max = self.log_alpha_beta.shape[0]

        if k > (old_max - 1):
            return self.log_alpha_beta[old_max - 1]
        else:
            return self.log_alpha_beta[k]

    def clean(self):
        self.K = self.doc_topic_counts.shape[1]

    def predictive_likelihood(self, data=None):
        pass

    def predictive_topics(self, data):
        pass

    def estimate_latent_variables(self):
        # check if perplecxy is equal if removing dummy empty topics...
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
        delta = np.array([self.likelihood.delta]*self.likelihood.nfeat)
        K = len(alpha)

        # Recontruct Documents-Topic matrix
        _theta = self.doc_topic_counts + np.tile(alpha, (self.J, 1))
        self._theta = (_theta.T / _theta.sum(axis=1)).T

        # Recontruct Words-Topic matrix
        _phi = self.likelihood.word_topic_counts.T + np.tile(delta, (K, 1))
        self._phi = (_phi.T / _phi.sum(axis=1))

    # Mean can be arithmetic or geometric
    def perplexity(self, data=None, mean='arithmetic'):
        phi = self._phi
        if data is None:
            data = self.likelihood.data_mat
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
        entropy = (data.A * np.log( theta.dot(phi.T) )).sum()
        #entropy = (data.hadamard( sppy.csarray(np.log( theta.dot(phi.T) )) )).sum()

        entropy /= nnz
        perplexity = np.exp(-entropy)
        return perplexity

class ZSamplerParametric(ZSampler):
    # Parametric Version of HDP sampler. Number of topics fixed.

    def __init__(self, alpha_0, likelihood, K, alpha='asymmetric'):
        self.K = self.K_init = self._K =  K
        if 'alpha' is 'symmetric':
            alpha = np.ones(K)*1/K
        elif 'alpha' == 'asymmetric':
            alpha = numpy.asarray([1.0 / (i + numpy.sqrt(K)) for i in xrange(K)])
            alpha /= alpha.sum()
        else:
            alpha = np.ones(K)*alpha_0
        self.logalpha = np.log(alpha)
        super(ZSamplerParametric, self).__init__(alpha_0, likelihood)

    def __call__(self):
        print 'Sample z...'
        print '#Doc \t #nnz\t #Topic'
        for j in xrange(self.J):
            nnz =  self.data_dims[j]
            print '%d \t %d \t %d' % ( j , nnz, self.K )
            for i in xrange(nnz):
                params = self.prob_zji(j, i)
                sample = categorical(params)
                self.z[j][i] = sample

                self.doc_topic_counts[j, sample] += 1
                self.likelihood.word_topic_counts[self.likelihood.data[j][i], sample] += 1

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
        self.m = np.zeros(self.count_k_by_j.shape, dtype=int)
        self.m_dotk = self.m.sum(axis=0)

    def __call__(self):
        self._update_m()

        indices = np.ndenumerate(self.count_k_by_j)

        print 'Sample m...'
        for ind in indices:
            j, k = ind[0]
            count = ind[1]

            if count > 0:
                params = self.prob_mjk(j, k)
                sample = categorical(params)
            else:
                sample = -1

            self.m[j, k] = sample + 1

        self.purge_empty_tables()

        return self.m

    def _update_m(self):
        # Passed by reference, but why not...
        self.count_k_by_j = self.zsampler.doc_topic_counts
        K = self.count_k_by_j.shape[1]

        # Add empty table for new fancy topics
        new_k = K - self.m.shape[1]
        if new_k > 0:
            print 'msampler: %d new topics' % (new_k)
            J = self.m.shape[0]
            self.m = np.hstack((self.m, np.zeros((J, new_k), dtype=int)))

    # Removes empty table.
    def purge_empty_tables(self):
        self.m_dotk = self.m.sum(0)

        K = self.count_k_by_j.shape[1]
        dummy_k = self.m.shape[1] - K
        if dummy_k > 0:
            #self.m = np.delete(self.m, index_nullvalues, axis=1)
            self.m = self.m[:, :-dummy_k]
            self.m_dotk = self.m_dotk[:-dummy_k]

        assert(self.m.shape == self.count_k_by_j.shape)
        index_nullvalues = np.where(self.m_dotk == 0)[0]
        assert(index_nullvalues.size == 0)

    def prob_mjk(self, j, k):
        # -1 because table of current sample topic jk, is not conditioned on
        possible_ms = range(1, self.count_k_by_j[j, k]) # +1-1
        params = np.array([self.prob_mjk_eq_m(m, j, k) for m in possible_ms], dtype=np.float64)

        if sym.zoo in params or np.isnan(params).any() or np.isinf(params).any():
            assert( 'whaaaaatttss up !!!!' == 'zoo')
            all_mass = np.where(params == sym.zoo)[0][0]
            log_norm_params = np.zeros(params.shape)
            log_norm_params[all_mass] = 1
            return log_norm_params
        elif params.shape[0] == 0:
            return np.ones(1)
        else:
            return lognormalize(params)

    def prob_mjk_eq_m(self, m, j, k):
        njdotk = self.count_k_by_j[j, k]

        log_alpha_beta_k = self.get_log_alpha_beta(k)
        alpha_beta_k = np.exp(log_alpha_beta_k)

        #log_stir = sym.log(stirling(njdotk, m, kind=1)).evalf()
        log_stir = self.stirling_mat[njdotk, m]

        normalizer = gammaln(alpha_beta_k) - gammaln(alpha_beta_k + njdotk)
        log_prob = normalizer + log_stir + m*log_alpha_beta_k

        #print normalizer, log_prob

        return log_prob

class BetaSampler(GibbsSampler):

    def __init__(self, gmma, msampler):
        self.gmma = gmma
        self.msampler = msampler

        # Initialize restaurant with just one table.
        self.beta = dirichlet([1, gmma])

    def __call__(self):
        print 'Sample Beta...'
        self._update_dirichlet_params()
        self.beta = dirichlet(self.dirichlet_params)

        return self.beta

    def _update_dirichlet_params(self):
        m_dotk_augmented = np.append(self.msampler.m_dotk, self.gmma)
        print 'Beta Dirichlet Prior: %s' % (m_dotk_augmented)
        self.dirichlet_params = m_dotk_augmented


class HDP_LDA_CGS(GibbsSampler):

    # Joint Sampler of topic Assignement, table configuration, and beta proportion.
    # ref to direct assignement Sampling in HDP (Teh 2006)
    def __init__(self, zsampler, msampler, betasampler, hyper='auto'):
        zsampler.add_beta_sampler(betasampler)

        self.zsampler = zsampler
        self.msampler = msampler
        self.betasampler = betasampler

        if hyper == 'auto':
            self.hyper = hyper
            self.a_alpha = 1
            self.b_alpha = 1
            self.a_gmma = 1
            self.b_gmma = 1
        else:
            raise NotImplementedError('Hyperparameters optimization ?')


    def optimize_hyper_hdp(self):
        # Optimize \alpha_0
        alpha_0 = self.zsampler.alpha_0
        n_jdot = np.array(self.zsampler.data_dims)
        m_dot = self.msampler.m_dotk.sum()
        u_j = binomial(1, n_jdot/(n_jdot + alpha_0))
        v_m = beta(alpha_0 + 1, n_jdot)
        new_alpha0 = gamma(self.a_alpha + m_dot - u_j.sum(), self.b_alpha - np.log(v_m).sum())
        self.zsampler.alpha_0 = new_alpha0

        # Optimize \gamma
        K = self.zsampler._K
        gmma = self.betasampler.gmma
        u = binomial(1, m_dot / (m_dot + gmma))
        v = beta(gmma + 1, m_dot)
        new_gmma = gamma(self.a_gmma + K -1 - u, self.b_gmma - np.log(v))
        self.betasampler.gmma = new_gmma
        return

    def __call__(self):
            z = self.zsampler.sample()
            m = self.msampler.sample()
            beta = self.betasampler.sample()

            self.optimize_hyper_hdp()

            return z, m, beta

class LDA_CGS(GibbsSampler):

    def __init__(self, zsampler):
        self.zsampler = zsampler

    def __call__(self):
            return self.zsampler.sample()

class GibbsRun(object):

    def __init__(self, sampler, iterations=1000, burnin=100, thinning_interval=10, output_path=None, write=False):
    #def __init__(self, sampler, iterations=10000, burnin=1000, thinning_interval=100):
        self.s = sampler

        self.iterations = iterations
        self.burnin = burnin
        self.thinning = thinning_interval

        self.write = write
        if output_path and self.write:
            import os
            bdir = os.path.dirname(output_path) or 'tmp-output'
            fn = os.path.basename(output_path)
            try: os.makedirs(bdir)
            except: pass
            self.fname_i = bdir + '/inference-' + fn.split('.')[0]
            csv_typo = '# it entropy K'
            self._f = open(self.fname_i, 'w')
            self._f.write(csv_typo + '\n')

    def run(self):
        self.samples = []

        for i in xrange(self.iterations):

            ### Output
            pp =  np.log(self.evaluate_perplexity())
            k = self.s.zsampler._K
            alpha = self.s.zsampler.alpha_0
            try:
                gmma = self.s.betasampler.gmma
            except:
                gmma = self.s.zsampler.likelihood.delta
            sample = [i, pp, k, alpha, gmma]
            print 'Iteration %d, Perplexity: %s ' % (i, pp)
            if self.write:
                self.write_some(sample)

            _sample = self.s.sample()

            if i >= self.iterations:
                s.clean()
                break
            if i >= self.burnin:
                if i % self.thinning == 0:
                    self.samples.append(sample)

    def write_some(self, samples, f=None):
        if not f:
            f = self._f
        samples = np.array([samples])
        np.savetxt(f, samples, fmt="%.8f")

    def evaluate_perplexity(self, data=None):
        self.s.zsampler.estimate_latent_variables()
        return self.s.zsampler.perplexity(data)


