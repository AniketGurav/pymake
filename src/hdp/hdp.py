import collections
import functools
import warnings
import copy

import numpy as np
import scipy as sp
import sympy as sym

from scipy.special import gammaln
from numpy.random import dirichlet, multinomial, gamma, poisson
from sympy.functions.combinatorial.numbers import stirling

from util.frontend import frontEndBase
sparse2stream = frontEndBase.sparse2stream

import sys
sys.setrecursionlimit(10000)

# Implementation of Teh et al. (2005) Gibbs sampler for the Hierarchical Dirichlet Process: Direct assignement.
#https://github.com/aaronstevenwhite/HDP

#warnings.simplefilter('error', RuntimeWarning)

def lognormalize(x):
    a = np.logaddexp.reduce(x)
    return np.exp(x - a)

def categorical(params):
    return np.where(multinomial(1, params) == 1)[0]

class DirMultLikelihood(object):

    def __init__(self, delta, data):
        if data.format == 'csr':
            self.data_csr = data
        else:
            self.data_csr = sp.sparse.csr_matrix(data)
        self.data = sparse2stream(data)
        self.delta = delta
        self.voc_size = np.vectorize(max)(data).max() + 1

        self.counts = {}

    def __call__(self, j, i, k, where_k):
        return self.log_likelihood(j, i, k, where_k)

    def get_data_dims(self):
        shape_vect = np.vectorize(np.shape)

        return shape_vect(self.data)

    def words_k(self, where_k):
        for group in xrange(self.data.shape[0]):
            word_gen = self.data[group][where_k[group]]
            for word in word_gen:
                yield word

    def k_counts(self, j, i, k, where_k):

        try:
            old_where_k = self.counts[k]['where_k']

            if i-1 in where_k[j] and i-1 not in old_where_k[j]: # if z_{i-1, j} got changed to a k last sample
                self.counts[k]['counts'][self.data[j][i-1]] += 1
                self.counts[k]['total'] += 1

            counts_ji = self.counts[k]['counts'][self.data[j][i]]
            total = self.counts[k]['total']

            if i in old_where_k[j]: # if the item you are popping out was a k
                counts_ji -= 1
                total -= 1

        except KeyError:
            words_k = self.words_k(where_k)
            counts = collections.Counter(words_k)
            total = np.sum(counts.values())

            self.counts[k] = {'counts' : counts,
                              'total' : total}
            counts_ji = self.counts[k]['counts'][self.data[j][i]]

        self.counts[k]['where_k'] = where_k

        return total, counts_ji

    def log_likelihood(self, j, i, k, where_k):
        total, count_ji = self.k_counts(j, i, k, where_k)

        log_smooth_count_ji = np.log(count_ji + self.delta)

        #return log_smooth_count_ji - np.log(total + (total + 1)*self.delta)
        return log_smooth_count_ji - np.log(total + self.voc_size*self.delta)


class GibbsSampler(object):

    def __iter__(self):
        return self

    def sample(self):
        return self()

class ZSampler(GibbsSampler):
    # Alternative is to keep the two count matrix:
        # C(word-topic)[w, k] === n_dotkw
        # C(document-topic[j, k] === n_jdotk
        # From this reconstruct theta and phi

    def __init__(self, alpha_0, likelihood):
        data_dims = likelihood.get_data_dims()
        self.z = np.array([poisson(alpha_0, size=dim) for dim in data_dims])
        print 'Z shape: %s' % self.z.shape

        self.alpha_0 = alpha_0
        self.likelihood = likelihood

    def __call__(self):
        self._update_log_alpha_beta()

        print 'Sample z...'
        print '#Doc \t #topic'
        for j in xrange(self.z.shape[0]):
            print '%d \t %d' % ( j , self.get_max_k() )
            for i in xrange(self.z[j].shape[0]):
                params = self.prob_zji(j, i)

                sample = categorical(params)
                self.z[j] = np.insert(self.z[j], i, sample)

        return self.z

    def _update_log_alpha_beta(self):
        self.log_alpha_beta = np.log(self.alpha_0) + np.log(self.betasampler.beta)

    def _remove_zji(self, j, i):
        self.z[j] = np.delete(self.z[j], i)

    def _insert_zji(self, j, i, k):
        self.z[j] = np.insert(self.z[j], i, k)

    def add_beta_sampler(self, betasampler):
        self.betasampler = betasampler

    def _where_k(self, k):
        return np.array([np.where(zj == k)[0] for zj in self.z])

    def get_max_k(self):
        return np.max(np.vectorize(np.max)(self.z))

    def get_count_k_by_j(self):
        num_groups = self.z.shape[0]

        return np.array([self._count_k_in_j(j) for j in range(num_groups)])

    def _count_k_in_j(self, j):
        return np.bincount(self.z[j], minlength=self.get_max_k()+2)

    def prob_zji(self, j, i):
        self._remove_zji(j, i)

        njdot_minusji = self._count_k_in_j(j)

        possible_ks = range(self.get_max_k()+2)

        params = [self.prob_zji_eq_k(k, j, i, njdot_minusji[k]) for k in possible_ks]

        return lognormalize(params)

    def prob_zji_eq_k(self, k, j, i, njdotk_minusji):
        # Add a dummy k for shape consistence with likelihood count
        self._insert_zji(j, i, k+1)

        where_k = self._where_k(k)

        log_alpha_beta = self.get_log_alpha_beta(k)

        prior = np.logaddexp(np.log(njdotk_minusji), log_alpha_beta)

        ll = self.likelihood(j, i, k, where_k)

        # Remove the dummy k
        self._remove_zji(j, i)

        return prior + ll

    def get_log_alpha_beta(self, k):
        old_max = self.log_alpha_beta.shape[0]

        if k > (old_max - 1):
            return self.log_alpha_beta[old_max - 1]
        else:
            return self.log_alpha_beta[k]


class MSampler(GibbsSampler):

    def __init__(self, zsampler):
        self.zsampler = zsampler

        self.count_k_by_j = zsampler.get_count_k_by_j()
        self.max_k = zsampler.get_max_k()
        self.get_log_alpha_beta = zsampler.get_log_alpha_beta

        self.m = np.zeros(self.count_k_by_j.shape)

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

        return self.m

    def _update_m(self):
        self.count_k_by_j = self.zsampler.get_count_k_by_j()
        self.max_k = self.zsampler.get_max_k()

        while self.m.shape[1] < self.max_k + 2:
            self.m = np.insert(self.m, self.m.shape[1], 0, axis=1)

        self.m = self.m[:,:self.max_k+2]

    def prob_mjk(self, j, k):
        # -1 because table of current sample topic jk, is not conditioned on
        possible_ms = range(1, self.count_k_by_j[j, k]+1-1)
        params = np.array([self.prob_mjk_eq_m(m, j, k) for m in possible_ms], dtype=np.float64)

        if sym.zoo in params:
            print 'whaaaaatttss up !!!!'
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

        log_stir = sym.log(stirling(njdotk, m, kind=1)).evalf()

        normalizer = gammaln(alpha_beta_k) - gammaln(alpha_beta_k + njdotk)
        log_prob = normalizer + log_stir + m*log_alpha_beta_k

        #print normalizer, log_prob

        return log_prob


class BetaSampler(GibbsSampler):

    def __init__(self, gmma, msampler):
        self.gmma = gmma
        self.msampler = msampler

        self.beta = dirichlet([1, gmma])

    def __call__(self):
        print 'Sample Beta...'
        self._update_dirichlet_params()
        self.beta = dirichlet(self.dirichlet_params)

        return self.beta

    def _update_dirichlet_params(self):
        #mdot = np.vectorize(np.sum)(self.msampler.m)
        mdot = self.msampler.m.sum(axis=0)

        self.dirichlet_params = np.append(mdot, self.gmma)


class JointSampler(GibbsSampler):

    def __init__(self, zsampler, msampler, betasampler):
        zsampler.add_beta_sampler(betasampler)

        self.zsampler = zsampler
        self.msampler = msampler
        self.betasampler = betasampler

    def __call__(self):
        z = self.zsampler.sample()
        m = self.msampler.sample()
        beta = self.betasampler.sample()

        return z, m, beta


class GibbsRun(object):

    def __init__(self, sampler, iterations=1000, burnin=100, thinning_interval=10):
    #def __init__(self, sampler, iterations=10000, burnin=1000, thinning_interval=100):
        self.sampler = sampler

        self.iterations = iterations
        self.burnin = burnin
        self.thinning = thinning_interval

    def run(self):
        self.samples = []

        for i in xrange(self.iterations):
        #for i, sample in enumerate(self.sampler):
            print 'Iteration %d: ' % (i)
            sample = self.sampler.sample()
            if i >= self.iterations:
                break
            if i >= self.burnin:
                if i % self.thinning == 0:
                    self.samples.append(sample)


if __name__ == '__main__':
    import os

    delta = .5 # used in Teh et al. (2005) for the parameters of H

    bdir = '../data/text/nips12/'
    corpus_dir = os.listdir(bdir)

    def parse_document(doc_name):
        return np.array([word.strip('.').lower() for line in open(doc_name).readlines() for word in line.strip().split()])

    corpus = np.array([parse_document(bdir + doc_name) for doc_name in corpus_dir])

    corpus = corpus[0:50]

    #corpus = np.array([poisson(1, k) for k in poisson(50, 20)])

    dmlike = DirMultLikelihood(delta, corpus)

    # @Todo: Static or Guruu call inside Gibbs class.
    def create_new_run(likelihood=dmlike):
        alpha = gamma(1, 1)
        gmma = gamma(1, .1)

        zsampler = ZSampler(alpha, likelihood)
        msampler = MSampler(zsampler)
        betasampler = BetaSampler(gmma, msampler)

        jointsampler = JointSampler(zsampler, msampler, betasampler)

        run = GibbsRun(jointsampler, iterations=100, burnin=10, thinning_interval=10)

        return run

    gibbs_run = create_new_run(dmlike)
    gibbs_run.run()
