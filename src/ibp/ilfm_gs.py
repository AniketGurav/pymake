"""
@author: Adrien Dulac (adrien.dulac@imag.fr)

Implements MCMC inference for the Infinite Latent Feature Relationnal Model [1].
This code was modified from the code originally written by Zhai Ke (kzhai@umd.edu).

[1] Kurt Miller, Michael I Jordan, and Thomas L Griffiths. Nonparametric latent feature models for link prediction. In Advances in neural information processing systems 2009.
"""

""" @TODO
* 2 parameter ibp
* poisson-gamma ipb for real valued relation
* Oprimization:
*   - updating only the subset of relation affected by feature modified (active in both sides).
"""

import random, cPickle
from datetime import datetime
from os.path import dirname
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
sp_dot = csr_matrix.dot
from ibp import IBP
from local_utils import *

# We will be taking log(0) = -Inf, so turn off this warning
np.seterr(divide='ignore')

W_diag = -20

class IBPGibbsSampling(IBP):
    #@param snapshot_interval: the interval for exporting a snapshot of the model
    default_output = 'tmp-output/'

    def __init__(self, symmetric=False, W_is_diag=False, alpha_hyper_parameter=None, sigma_w_hyper_parameter=None, metropolis_hastings_k_new=True,
                 snapshot_interval=100):
        self._sigma_w_hyper_parameter = sigma_w_hyper_parameter
        assert(type(symmetric) is bool)
        self.bilinear_matrix = None
        self.log_likelihood = None
        self.ll_inc = 0
        self.symmetric = symmetric
        self.W_is_diag = W_is_diag
        self._snapshot_interval = snapshot_interval
        self._overflow = 1.0
        self.ratio_MH_F = 0.0
        self.ratio_MH_W = 0.0
        self.time_sampling = 0
        self.csv_typo = '# loglikelihood_Y, loglikelihood_Z, alpha, sigma, _K, Z_sum, ratio_MH_F, ratio_MH_W'
        #self.snap_attr = ['_Z', '_W', '_alpha', '_alpha_hyper_parameter', '_sigma_w' ]
        super(IBPGibbsSampling, self).__init__(alpha_hyper_parameter, metropolis_hastings_k_new)

    """
    @param data: a NxD np data matrix
    @param alpha: IBP hyper parameter
    @param sigma_w: standard derivation of the feature
    @param initialize_Z: seeded Z matrix """
    def _initialize(self, data, alpha=1.0, sigma_w=1, initial_Z=None, initial_W=None, KK=None):
        # Data matrix
        #super(IBPGibbsSampling, self)._initialize(self.center_data(data), alpha, initial_Z)
        super(IBPGibbsSampling, self)._initialize(data, alpha, initial_Z, KK=KK)

        assert(type(sigma_w) is float)
        self._sigma_w = sigma_w
        self._sigb = 1 # Carreful make overflow in exp of sigmoid !

        self._W_prior = np.zeros((1, self._K))
        if initial_W != None:
            self._W = initial_W
        else:
            self._W = np.random.normal(0, self._sigma_w, (self._K, self._K))
            if self.symmetric:
                self._W = np.tril(self._W) + np.tril(self._W, -1).T
                np.fill_diagonal(self._W, 1)
            elif self.W_is_diag:
                self._W  = (np.ones((self._K, self._K))*W_diag) * (np.ones((self._K)) + np.eye(self._K)*-2)

        #self._Z = csr_matrix(self._Z)
        #self._Z = lil_matrix(self._Z)

        assert(self._W.shape == (self._K, self._K))

    """
    sample the corpus to train the parameters """
    def sample(self, iteration=30, target_dir=None):
        self.iteration = iteration
        target_dir = target_dir or self.default_output
        bdir = dirname(__file__) + '/../../output/' + target_dir + '/'

        import os, shutil
        if os.path.exists(bdir):
            shutil.rmtree(bdir)
        os.makedirs(bdir)
        _f = open(bdir + 'mcmc', 'w')
        _f.write(self.csv_typo + '\n')

        assert(self._Z.shape == (self._N, self._K))
        assert(self._W.shape == (self._K, self._K))
        assert(self._Y.shape == (self._N, self._D))

        # Sample the total data
        begin = datetime.now()

        likelihood_Y = self.log_likelihood_Y()
        print 'Init Likelihood: %f' % likelihood_Y
        for iter in xrange(iteration):
            # Can't get why I need this !!!!!
            self.log_likelihood_Y()
            # Sample every object
            order = np.random.permutation(self._N)
            for (object_counter, object_index) in enumerate(order):
                singleton_features = self.sample_Zn(object_index)

                if self._metropolis_hastings_k_new:
                    if self.metropolis_hastings_K_new(object_index, singleton_features):
                        self.ratio_MH_F += 1

            self.ratio_MH_F /= len(order)

            # Regularize matrices
            self.regularize_matrices()

            if self.W_is_diag:
                self._W  = (np.ones((self._K, self._K))*W_diag) * (np.ones((self._K)) + np.eye(self._K)*-2)
            else:
                self.sample_W()

            if self._alpha_hyper_parameter != None:
                self._alpha = self.sample_alpha()

            if self._sigma_w_hyper_parameter != None:
                self._sigma_w = self.sample_sigma_w(self._sigma_w_hyper_parameter)

            if likelihood_Y == self.log_likelihood:
                self.ll_inc +=1
            likelihood_Y = self.log_likelihood
            likelihood_Z = self.log_likelihood_Z()
            Z_sum = (self._Z == 1).sum()
            print("iteration: %i\tK: %i\tlikelihood Y: %f\tlikelihood Z: %f" % (iter, self._K, likelihood_Y, likelihood_Z))
            print("alpha: %f\tsigma_w: %f\t Z.sum(): %i" % (self._alpha, self._sigma_w, Z_sum))

            ### Save data
            ## loglikelihood_Y, loglikelihood_Z, alpha, sigma, _K, Z_sum, ratio_MH_F, ratio_MH_W
            samples = [ likelihood_Y, likelihood_Z, self._alpha, self._sigma_w, self._K, Z_sum, self.ratio_MH_F, self.ratio_MH_W ]
            samples = np.array([samples])
            np.savetxt(_f, samples, fmt="%.4f")
            #if (iter + 1) % self._snapshot_interval == 0:
            #    self.export_snapshot(bdir, iter + 1)
        self.time_sampling = datetime.now() - begin
        return _f.close()

    """
    @param object_index: an int data type, indicates the object index (row index) of Z we want to sample """
    def sample_Zn(self, object_index):
        assert(type(object_index) == int or type(object_index) == np.int32 or type(object_index) == np.int64)

        # calculate initial feature possess counts
        m = self._Z.sum(axis=0)

        # remove this data point from m vector
        new_m = (m - self._Z[object_index, :]).astype(np.float)

        #m = np.array(m).reshape(-1)
        #new_m = np.array(new_m).reshape(-1)

        # compute the log probability of p(Znk=0 | Z_nk) and p(Znk=1 | Z_nk)
        log_prob_z0 = np.log(1.0 - new_m / self._N)
        log_prob_z1 = np.log(new_m / self._N)
        log_prob_z = {0: log_prob_z0, 1: log_prob_z1}

        # find all singleton features possessed by current object
        singleton_features = [nk for nk in range(self._K) if self._Z[object_index, nk] != 0 and new_m[nk] == 0]
        non_singleton_features = [nk for nk in range(self._K) if nk not in singleton_features]

        order = np.random.permutation(self._K)
        for (feature_counter, feature_index) in enumerate(order):
            if feature_index in non_singleton_features:
                old_znk = self._Z[object_index, feature_index]
                new_znk = 1 if old_znk == 0 else 0

                # compute the log likelihood when Znk = old
                #log_old = self.log_likelihood_Y(object_index=(object_index, feature_index))
                log_old = self.log_likelihood
                bilinear = self.bilinear_matrix.copy()
                #log_old_1 = self.log_likelihood_Y(object_index=(object_index, feature_index))
                #if (not (bilinear == self.bilinear_matrix).all()):
                #    print feature_counter, object_index, feature_index
                #    print np.where(self.bilinear_matrix != bilinear)
                self._overflow = - log_old -1
                prob_old = log_old + log_prob_z[old_znk][feature_index]

                # compute the log likelihood when Znk = new
                self._Z[object_index, feature_index] = new_znk
                log_new = self.log_likelihood_Y(object_index=(object_index, feature_index))
                if log_new > -self._overflow:
                    self._overflow = - log_new -1
                prob_new = log_new + log_prob_z[new_znk][feature_index]

                prob_new = np.exp(prob_new + self._overflow)
                prob_old = np.exp(prob_old + self._overflow)
                Znk_is_new = prob_new / (prob_new + prob_old)
                if Znk_is_new > 0 and random.random() <= Znk_is_new:
                    # gibbs_accept ++
                    pass
                else:
                    self._Z[object_index, feature_index] = old_znk
                    self.bilinear_matrix = bilinear
                    self.log_likelihood = log_old

        return singleton_features

    """
    sample K_new using metropolis hastings algorithm """
    def metropolis_hastings_K_new(self, object_index, singleton_features):
        if type(object_index) != list:
            object_index = [object_index]

        # sample K_new from the metropolis hastings proposal distribution, i.e., a poisson distribution with mean \frac{\alpha}{N}
        K_temp = sp.stats.poisson.rvs(self._alpha / self._N)

        if K_temp <= 0 and len(singleton_features) <= 0:
            return False

        # generate new weight from a normal distribution with mean 0 and variance sigma_w, a K_new-by-K matrix
        K_new = self._K + K_temp - len(singleton_features)
        non_singleton_features = [k for k in xrange(self._K) if k not in singleton_features]
        W_temp_v = np.random.normal(0, self._sigma_w, (K_temp, self._K - len(singleton_features)))
        W_temp_h = np.random.normal(0, self._sigma_w, (K_new, K_temp))
        W_new = np.delete(self._W, singleton_features,0)
        W_new = np.vstack((np.delete(W_new, singleton_features,1), W_temp_v))
        W_new = np.hstack((W_new, W_temp_h))
        # generate new features z_i row
        Z_new_r = np.hstack((self._Z[[object_index], non_singleton_features], np.ones((len(object_index), K_temp))))
        #print K_temp, object_index, non_singleton_features

        # compute the probability of generating new features / old featrues
        log_old = self.log_likelihood
        Z_new = np.hstack((self._Z[:, non_singleton_features], np.zeros((self._N, K_temp))))
        Z_new[object_index, :] = Z_new_r
        bilinear_matrix = self.bilinear_matrix.copy()
        log_new = self.log_likelihood_Y(Z=Z_new, W=W_new)
        self._overflow = - log_new -1
        prob_new = np.exp(log_new + self._overflow)
        prob_old = np.exp(log_old + self._overflow)

        assert(W_new.shape == (K_new, K_new))
        assert(Z_new.shape == (self._N, K_new))

        # compute the probability of generating new features
        r = prob_new / prob_old
        MH_accept = min(1, r)

        # if we accept the proposal, we will replace old W and Z matrices
        if random.random() <= MH_accept and not np.isnan(r):
            # construct W_new and Z_new
            #print 'MH_accept: %s, singleton feature: %s, k_new: %s' % (MH_accept, len(singleton_features), K_temp)
            self._W = W_new
            self._Z = Z_new
            self._K = K_new
            return True
        else:
            self.bilinear_matrix = bilinear_matrix
            self.log_likelihood = log_old
            return False

    """
    Sample W using metropolis hasting """
    def sample_W(self):
        # sample every weight
        sigma_rw = 1.0
        if self.symmetric:
            order = np.arange(self._K**2).reshape(self._W.shape)
            iu = np.triu_indices(self._K)
            order = np.random.permutation(order[iu])
        else:
            order = np.random.permutation(self._K**2)
        for (observation_counter, observation_index) in enumerate(order):
            w_old = self._W.flat[observation_index]
            w_new = np.random.normal(w_old, sigma_rw)
            j_new = sp.stats.norm(w_old, sigma_rw).pdf(w_new)
            j_old = sp.stats.norm(w_new, sigma_rw).pdf(w_old)
            pw_new = sp.stats.norm(0, self._sigma_w).pdf(w_new)
            pw_old = sp.stats.norm(0, self._sigma_w).pdf(w_old)

            log_old = self.log_likelihood
            bilinear_matrix = self.bilinear_matrix.copy()
            self._overflow = - log_old -1
            self._W.flat[observation_index] = w_new
            if self.symmetric:
                indm = np.unravel_index(observation_index, self._W.shape)
                self._W.T[indm] = w_new
            log_new = self.log_likelihood_Y(object_index=observation_index)
            if -log_new < self._overflow:
                self._overflow = - log_new -1
            likelihood_new = np.exp(log_new + self._overflow)
            likelihood_old = np.exp(log_old + self._overflow)
            r = likelihood_new * pw_new * j_old / ( likelihood_old * pw_old * j_new )
            MH_accept = min(1, r)

            if random.random() <= MH_accept and not np.isnan(r):
                self.ratio_MH_W += 1
            else:
                self.bilinear_matrix = bilinear_matrix
                self.log_likelihood = log_old
                self._W.flat[observation_index] = w_old
                if self.symmetric:
                    self._W.T[indm] = w_old

        try:
            self.ratio_MH_W /= len(order)
        except:
            pass
        return self.ratio_MH_W

    """
    sample feature variance, i.e., sigma_w """
    def sample_sigma_w(self, sigma_w_hyper_parameter):
        return self.sample_sigma(self._sigma_w_hyper_parameter, self._W - np.tile(self._W_prior, (self._K, 1)))

    """
    remove the empty column in matrix Z and the corresponding feature in W """
    def regularize_matrices(self):
        Z_sum = np.sum(self._Z, axis=0)
        indices = np.nonzero(Z_sum == 0)

        if 0 in Z_sum:
            print "need to regularize matrices, feature to all zeros !"

        #self._Z = self._Z[:, [k for k in range(self._K) if k not in indices]]
        #self._W = self._W[[k for k in range(self._K) if k not in indices], :]
        #self._K = self._Z.shape[1]
        #assert(self._Z.shape == (self._N, self._K))
        #assert(self._W.shape == (self._K, self._K))

    """
    compute the log-likelihood of the data Y """
    def log_likelihood_Y(self, Z=None, W=None, object_index=None):
        if W == None:
            W = self._W
        if Z == None:
            Z = self._Z

        (N, K) = Z.shape
        assert(W.shape == (K, K))

        bilinear_init = self.bilinear_matrix is not None
        #bilinear_init = False

        if type(object_index) is tuple and bilinear_init:
            # Z update
            n = object_index[0]
            k = object_index[1]
            self.bilinear_matrix[n,:] = self.logsigmoid(Z[n].dot(W).dot( Z.T ), self._Y[n,:])
            #self.bilinear_matrix[n,:] = self.logsigmoid( Z.dot(Z[n].dot(W).T).T , self._Y[n,:])
            if self.symmetric:
                self.bilinear_matrix[:,n] = self.bilinear_matrix[n,:]
            else:
                self.bilinear_matrix[:,n] = self.logsigmoid(Z.dot(W).dot(Z[n]), self._Y[:, n])
        elif np.issubdtype(object_index, np.integer) and bilinear_init:
            # W update
            indm = np.unravel_index(object_index, W.shape)
            ki, kj = indm
            sublinear = list( np.where(Z[:, ki] > 0)[0])
            sublinear = sorted(list(set(sublinear +  list(np.where(Z[:, kj] > 0)[0]) )))
            if len(sublinear) > 0:
                self.bilinear_matrix[np.ix_(sublinear, sublinear)] = self.logsigmoid( Z[sublinear].dot(W).dot(Z[sublinear].T), self._Y[np.ix_(sublinear, sublinear)] )
                #self.bilinear_matrix[np.ix_(sublinear, sublinear)] = self.logsigmoid( sp_dot(Z[sublinear].dot(W), Z[sublinear].T), Y[np.ix_(sublinear, sublinear)] )
        else:
            # Check speed with sparse matrix here !
            self.bilinear_matrix = self.logsigmoid(Z.dot(W).dot(Z.T))
            #self.bilinear_matrix = self.logsigmoid( Z.dot(Z.dot(W).T).T )

        self.log_likelihood = np.sum(self.bilinear_matrix)
        if np.abs(self.log_likelihood) == np.inf:
            if self._sigb >= 2:
                self._sigb -= 1
            else:
                self._sigb = 1
                self._W /= 2
                W = self._W
            self.bilinear_matrix = self.logsigmoid(Z.dot(W).dot(Z.T))
            self.log_likelihood = np.sum(self.bilinear_matrix)

        return self.log_likelihood

    def logsigmoid(self, X, Y=None):
        if Y is None:
            Y = self._Y
        # 1 - sigmoid(x) = sigmoid(-x) ~ Y * ...
        v = - np.log(1 + np.exp(- Y * self._sigb * X))
        return v

    """
    compute the log-likelihood of W """
    def log_likelihood_W(self):
        log_likelihood = - 0.5 * self._K * self._D * np.log(2 * np.pi * self._sigma_w * self._sigma_w)
        #for k in range(self._K):
        #    W_prior[k, :] = self._mean_a[0, :]
        W_prior = np.tile(self._W_prior, (self._K, 1))
        log_likelihood -= np.trace(np.dot((self._W - W_prior).transpose(), (self._W - W_prior))) * 0.5 / (self._sigma_w ** 2)

        return log_likelihood

    """
    compute the log-likelihood of the model """
    def log_likelihood_model(self):
        # loglikelihood_W ? Z ?
        return self.log_likelihood_Y()

    # Save numpy array of class
    #def save(self, target_dir='tmp-output/'):
    #    bdir = dirname(__file__) + '/../../output/' + target_dir
    #    for snap in self.snap_attr:
    #        value = getattr(self, snap)
    #        np.save(bdir + snap, value)
    #    return

    #def load(self, target_dir='tmp-output'):
    #    bdir = dirname(__file__) + '/../../output/' + target_dir
    #    for snap in self.snap_attr:
    #        value = np.load(bdir + snap)
    #        setattr(self, snap, value)
    #    return

    # Pickle class
    def save(self, target_dir=None):
        filen = 'ilfm_mcmc.pk'
        target_dir = target_dir or self.default_output
        bdir = dirname(__file__) + '/../../output/' + target_dir +'/'
        with open(bdir + filen, 'w') as _f:
            return cPickle.dump(self, _f)
    @classmethod
    def load(cls, target_dir=None):
        filen = 'ilfm_mcmc.pk'
        target_dir = target_dir or cls.default_output
        bdir = dirname(__file__) + '/../../output/' + target_dir +'/'
        with open(bdir + filen, 'r') as _f:
            return cPickle.load(_f)

    def generate_Y(self, nodelist=None):
        Y = self._Y
        W = self._W
        Z = self._Z

        if nodelist:
            Y = Y[nodelist, :][:, nodelist]
            Z = Z[nodelist, :]

        likelihood = np.empty_like(Y)
        bilinear_form = Z.dot(W).dot(Z.T)
        likelihood = 1 / (1 + np.exp(- self._sigb * bilinear_form))
        Yp = sp.stats.bernoulli.rvs(likelihood)
        return Yp

#if __name__ == '__main__':
#    from util.plot import plot_csv, ColorMap
#    from util.utils import argParse, getGraph
#
#    # Generate Data
#    N = 10
#    data = np.random.randint(0, 2, (N, N))
#    #data = getGraph()
#    np.fill_diagonal(data, 1)
#    print data
#
#    # Set up the hyper-parameter for sampling
#    alpha_hyper_parameter = None #(1., 1.)
#    sigma_w_hyper_parameter = None #(1., 1.)
#    symmetric_relation = True
#
#    # Hyper parameter init
#    alpha = 3
#    sigma_w = 1.0
#    niterations = 30
#
#    # Initialize the model
#    ibp = IBPGibbsSampling(symmetric_relation, alpha_hyper_parameter, sigma_w_hyper_parameter, True)
#    ibp._initialize(data, alpha, sigma_w, None)
#    ibp.sample(niterations)
#
#    print 'Sampling duration: %s' % ibp.time_sampling
#
#    # Plot Things
#    import pylab
#    plot_csv()
#    ColorMap(ibp.leftordered(), pixelspervalue=5)
#    pylab.show()
#
