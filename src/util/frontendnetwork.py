import sys, os
from itertools import chain
from operator import itemgetter
from string import Template
from numpy import ma
import networkx as nx

from local_utils import *
from util.frontend import DataBase

sys.path.insert(1, '../../gensim')
import gensim
from gensim.models import ldamodel, ldafullbaye
Models = { 'ldamodel': ldamodel, 'ldafullbaye': ldafullbaye, 'hdp': 1}

############################################################
############################################################
#### Frontend for network data.
#
#   Symmetric network support.

class frontendNetwork(DataBase):

    def __init__(self, config):
        self.bdir = os.path.join(config['bdir'], 'networks')
        super(frontendNetwork, self).__init__(config)

    # /!\ Will set the name of output_path.
    def load_data(self, corpus_name=None, randomize=False):
        """ Load data according to different scheme:
            * Corpus from file dataset
            * Corpus from random generator
            """
        if corpus_name is not None:
            self.corpus_name = corpus_name
        else:
            corpus_name = self.corpus_name

        self.get_corpus(corpus_name)

        np.fill_diagonal(self.data, 1)
        if randomize:
            self.shuffle_node()
        return self.data

    def shuffle_node(self):
        """ Shuffle rows and columns of data """
        N, M = self.data.shape
        nodes_list = [np.random.permutation(N), np.random.permutation(M)]
        self.reorder_node(nodes_list)

    def reorder_node(self, nodes_l):
        """ Subsample the data with reordoring of rows and columns """
        self.data = self.data[nodes_l[0], :][:, nodes_l[1]]
        # Track the original nodes
        self.nodes_list = [self.nodes_list[0][nodes_l[0]], self.nodes_list[1][nodes_l[1]]]

    def sample(self, N=None, symmetric=False, randomize=False):
        N = N or self.N
        n = self.data.shape[0]
        N_config = self.config['N']
        if not N_config or N_config == 'all':
            self.N = N
        else:
            # Can't get why modification inside self.nodes_list is not propagated ?
            N = int(N_config)
            if randomize is True:
                nodes_list = [np.random.permutation(N), np.random.permutation(N)]
                self.reorder_node(nodes_list)
            else:
                self.data = self.data[:N, :N]

            if hasattr(self, 'features') and self.features is not None:
                self.features = self.features[:N]

        return self.data

    def get_masked(self, percent_hole, diag_off=1):
        """ Construct a random mask """
        data = self.data
        if type(data) is np.ndarray:
            #self.data_mat = sp.sparse.csr_matrix(data)
            pass
        else:
            raise NotImplementedError('type %s unknow as corpus' % type(data))

        n = int(data.size * percent_hole)
        mask_index = np.unravel_index(np.random.permutation(data.size)[:n], data.shape)
        mask = np.zeros(data.shape)
        mask[mask_index] = 1

        if self.is_symmetric():
            mask = np.tril(mask) + np.tril(mask, -1).T

        data_ma = ma.array(data, mask=mask)
        if diag_off == 1:
            np.fill_diagonal(data_ma, ma.masked)

        return data_ma

    def get_masked_1(self, percent_hole, diag_off=1):
        """ Construct Mask nased on the proportion of 1/links """
        data = self.data
        if type(data) is np.ndarray:
            #self.data_mat = sp.sparse.csr_matrix(data)
            pass
        else:
            raise NotImplementedError('type %s unknow as corpus' % type(data))

        # Correponding Index
        _0 = np.array(zip(*np.where(data == 0)))
        _1 = np.array(zip(*np.where(data == 1)))
        n = int(len(_1) * percent_hole)
        # Choice of Index
        n_0 = _0[np.random.choice(len(_0), n, replace=False)]
        n_1 = _1[np.random.choice(len(_1), n, replace=False)]
        # Corresponding Mask
        mask_index = zip(*(np.concatenate((n_0, n_1))))
        mask = np.zeros(data.shape)
        mask[mask_index] = 1

        if self.is_symmetric():
            mask = np.tril(mask) + np.tril(mask, -1).T

        data_ma = ma.array(data, mask=mask)
        if diag_off == 1:
            np.fill_diagonal(data_ma, ma.masked)

        return data_ma

    def is_symmetric(self, update=False):
        if update or not hasattr(self, 'symmetric'):
            self.symmetric = (self.data == self.data.T).all()
        return self.symmetric

    def random(self, rnd):
        config = self.config
        # Generate Data
        if type(config.get('random')) is int:
            rvalue = _rvalue.get(config['random'])
            if rvalue == 'uniform':
                data = np.random.randint(0, 2, (N, N))
                np.fill_diagonal(data, 1)
            elif rvalue == 'clique':
                data = getClique(N, K=K)
                G = nx.from_numpy_matrix(data, nx.Graph())
                data = nx.adjacency_matrix(G, np.random.permutation(range(N))).A
            elif rvalue == 'barabasi-albert':
                data = nx.adjacency_matrix(nx.barabasi_albert_graph(N, m=13) ).A
            else:
                raise NotImplementedError()

        self.data = data
        return True

    ### Redirect to correct path depending on the corpus_name
    def get_corpus(self, corpus_name=None):
        config = self.config
        corpus_name = corpus_name or self.corpus_name
        self.make_output_path(corpus_name)
        self.corpus_name = corpus_name
        data_t = None
        try:
            N = int(self.N)
        except:
            # Catch later or elsewhere
            pass

        if corpus_name.startswith('generator'):
            # Reroot basedir
            K = int(corpus_name[len('generator'):])
            self.basedir = self.basedir[:-len(str(K))]
            corpus_name = self.corpus_name =  'Graph' + str(K)
            self.basedir = os.path.join(self.basedir, self.corpus_name)
            self.make_output_path()

            fn = os.path.join(self.basedir, 't0.graph')
            data = self.networkloader(fn)
        elif corpus_name in ('bench1'):
            raise NotImplementedError()
        elif corpus_name.startswith('clique'):
            # Reroot basedir
            K = int(corpus_name[len('clique'):])
            self.basedir = os.path.join(os.path.dirname(self.basedir), 'generator', os.path.basename(self.basedir[:-len(str(K))]))
            corpus_name = self.corpus_name = 'clique' + str(K)
            self.basedir = os.path.join(self.basedir, self.corpus_name)
            self.make_output_path()

            data = getClique(N, K=K)
            #G = nx.from_numpy_matrix(data, nx.Graph())
            #data = nx.adjacency_matrix(G, np.random.permutation(range(N))).A
        elif corpus_name.startswith('facebook'):
            bdir = self.basedir.split('/')[-1].split['_'][0]
            corpus_name = self.basedir.split('/')[-1].split['_'][1]
            self.basedir = os.path.join(os.path.dirname(self.basedir), bdir)
            self.make_output_path()

            fn = os.path.join(self.basedir, corpus_name, '0.edges')
            data = self.networkloader(fn)
        elif corpus_name == 'manufacturing':
            fn = os.path.join(self.basedir, 'manufacturing.csv')
            data = self.networkloader(fn)
        elif corpus_name == 'fb_uc':
            fn = os.path.join(self.basedir, 'graph.tnet')
            data = self.networkloader(fn)
        else:
            raise ValueError('Which corpus to Load; %s ?' % corpus_name)

        for a in ('features', 'clusters'):
            if not hasattr(self, a):
                setattr(self, a, None)

        self.data = data
        N, M = self.data.shape
        self.N = N
        self.nodes_list = [np.arange(N), np.arange(M)]
        return True

    def networkloader(self, fn):
        """ Load pickle or parse data """

        data = None
        if self._load_data and os.path.isfile(fn+'.pk'):
            try:
                data = self.load(fn+'.pk')
            except:
                data = None
        if data is None:
            ext = fn.split('.')[-1]
            if ext == 'graph':
                data = self.parse_graph(fn)
            elif ext == 'edges':
                data = self.parse_edges(fn)
            elif ext == 'tnet':
                data = self.parse_tnet(fn)
            elif os.path.basename(fn) == 'manufacturing.csv':
                data = self.parse_manufacturing(fn)
            else:
                raise ValueError('extension of network data unknown')

        if self._save_data:
            self.save(data, fn+'.pk')

        return data

    def parse_tnet(self, fn):
        sep = ' '
        with open(fn) as f:
            content = f.read()
        lines = filter(None, content.split('\n'))
        edges = [l.strip().split(sep)[-3:-1] for l in lines]
        edges = [ (int(e[0])-1, int(e[1])-1) for e in edges]
        N = max(list(chain(*edges))) + 1

        g = np.zeros((N,N))
        g[zip(*edges)] = 1
        return g

    def parse_manufacturing(self, fn):
        sep = ';'
        with open(fn) as f:
            content = f.read()
        lines = filter(None, content.split('\n'))[1:]
        edges = [l.strip().split(sep)[0:2] for l in lines]
        edges = [ (int(e[0])-1, int(e[1])-1) for e in edges]
        N = max(list(chain(*edges))) + 1

        g = np.zeros((N,N))
        g[zip(*edges)] = 1
        return g

    def parse_graph(self, fn):
        """ Parse Network data depending on type/extension """
        f = open(fn, 'r')
        data = []
        inside = {'vertices':False, 'edges':False }
        clusters = []
        features = []
        for line in f:
            if line.startswith('# Vertices') or inside['vertices']:
                if not inside['vertices']:
                    inside['vertices'] = True
                    N = 0
                    continue
                if line.startswith('#') or not line.strip() :
                    inside['vertices'] = False
                else:
                    # Parsing assignation
                    elements = line.strip().split(';')
                    clust = int(elements[-1])
                    feats = map(float, elements[-2].split('|'))
                    clusters.append(clust)
                    features.append(feats)
                    N += 1
            elif line.startswith('# Edges') or inside['edges']:
                if not inside['edges']:
                    inside['edges'] = True
                    continue
                if line.startswith('#') or not line.strip() :
                    inside['edges'] = False
                else:
                    # Parsing assignation
                    data.append( line.strip() )
        f.close()

        edges = [tuple(row.split(';')) for row in data]
        g = np.zeros((N,N))
        g[[e[0] for e in edges], [e[1] for e in edges]] = 1
        g[[e[1] for e in edges], [e[0] for e in edges]] = 1

        parameters = parse_file_conf(os.path.join(os.path.dirname(fn), 'parameters'))
        parameters['devs'] = map(float, parameters['devs'].split(';'))
        self.parameters_ = parameters

        self.clusters = clusters
        self.features = np.array(features)
        return g

    def communities_analysis(self):
        clusters = self.clusters
        if clusters is None:
            return None
        data = self.data
        symmetric = self.is_symmetric()
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

    def density(self):
        G = self.GG()
        return nx.density(G)

    def diameter(self):
        G = self.GG()
        try:
            diameter = nx.diameter(G)
        except:
            diameter = None
        return diameter

    def degree(self):
        G = self.GG()
        return nx.degree(G)

    def degree_histogram(self):
        G = self.GG()
        return nx.degree_histogram(G)

    def clustering_coefficient(self):
        G = self.GG()
        try:
            cc = nx.average_clustering(G)
        except:
            cc = None
        return cc

    def GG(self):
        if not hasattr(self, 'G'):
            if self.is_symmetric():
                # Undirected Graph
                typeG = nx.Graph()
            else:
                # Directed Graph
                typeG = nx.DiGraph()
            self.G = nx.from_numpy_matrix(self.data, typeG)
            #self.G = nx.from_scipy_sparse_matrix(self.data, typeG)
        return self.G

    def get_nfeat(self):
        nfeat = self.data.max() + 1
        if nfeat == 1:
            print 'Warning, only zeros in adjacency matrix...'
            nfeat = 2
        return nfeat

    def get_clusters(self):
        return self.clusters

    def clusters_len(self):
        cluster = self.get_clusters()
        if not cluster:
            return None
        else:
            return max(self.clusters)+1

    def get_data_prop(self):
        prop =  super(frontendNetwork, self).get_data_prop()
        nnz = self.data.sum(),
        _nnz = self.data.sum(axis=1)
        d = {'instances': self.data.shape[1],
               'nnz': nnz,
               'nnz_mean': _nnz.mean(),
               'nnz_var': _nnz.var(),
               'density': self.density(),
               'diameter': self.diameter(),
               'clustering_coef': self.clustering_coefficient(),
               'communities': self.clusters_len(),
               'features': self.get_nfeat(),
               'directed': not self.is_symmetric()
              }
        prop.update(d)
        return prop

    def template(self, d):
        d['time'] = d.get('time', None)
        netw_templ = '''###### $corpus_name
        Building: $time minutes
        Nodes: $instances
        Links: $nnz
        Degree mean: $nnz_mean
        Degree var: $nnz_var
        Diameter: $diameter
        Clustering Coefficient: $clustering_coef
        Density: $density
        Communities: $communities
        Relations: $features
        Directed: $directed
        \n'''
        return super(frontendNetwork, self).template(d, netw_templ)

    def similarity_matrix(self, sim='cos'):
        features = self.features
        if features is None:
            return None

        if sim == 'dot':
            sim = np.dot(features, features.T)
        elif sim == 'cos':
            norm = np.linalg.norm(features, axis=1)[np.newaxis]
            sim = np.dot(features, features.T)/np.dot(norm.T, norm)
        elif sim == 'kmeans':
            cluster = kmeans(features, K=2)[np.newaxis]
            cluster[cluster == 0] = -1
            sim = np.dot(cluster.T,cluster)
        elif sim == 'comm':
            N = len(self.clusters)
            sim = np.repeat(np.array(self.clusters)[np.newaxis], N, 0)
            sim = (sim == sim.T)*1
            sim[sim < 1] = -1
        elif sim == 'euclide_old':
            from sklearn.metrics.pairwise import euclidean_distances as ed
            #from plot import kmeans_plus
            #kmeans_plus(features, K=4)
            dist = ed(features)
            K = self.parameters_['k']
            devs = self.parameters_['devs'][0]
            sim = np.zeros(dist.shape)
            sim[dist <= 2.0 * devs / K] = 1
            sim[dist > 2.0  * devs / K] = -1
        elif sim == 'euclide_abs':
            from sklearn.metrics.pairwise import euclidean_distances as ed
            #from plot import kmeans_plus
            #kmeans_plus(features, K=4)
            N = len(features)
            K = self.parameters_['k']
            devs = self.parameters_['devs'][0]

            a = np.repeat(features[:,0][None], N, 0).T
            b = np.repeat(features[:,0][None], N, 0)
            sim1 = np.abs( a-b )
            a = np.repeat(features[:,1][None], N, 0).T
            b = np.repeat(features[:,1][None], N, 0)
            sim2 = np.abs( a-b )

            sim3 = np.zeros((N,N))
            sim3[sim1 <= 2.0*  devs / K] = 1
            sim3[sim1 > 2.0 *  devs / K] = -1
            sim4 = np.zeros((N,N))
            sim4[sim2 <= 2.0*  devs / K] = 1
            sim4[sim2 > 2.0 *  devs / K] = -1
            sim = sim4 + sim3
            sim[sim >= 0] = 1
            sim[sim < 0] = -1

        elif sim == 'euclide_dist':
            from sklearn.metrics.pairwise import euclidean_distances as ed
            #from plot import kmeans_plus
            #kmeans_plus(features, K=4)
            N = len(features)
            K = self.parameters_['k']
            devs = self.parameters_['devs'][0]

            sim1 = ed(np.repeat(features[:,0][None], 2, 0).T)
            sim2 = ed(np.repeat(features[:,0][None], 2, 0).T)

            sim3 = np.zeros((N,N))
            sim3[sim1 <= 2.0*  devs / K] = 1
            sim3[sim1 > 2.0 *  devs / K] = -1
            sim4 = np.zeros((N,N))
            sim4[sim2 <= 2.0*  devs / K] = 1
            sim4[sim2 > 2.0 *  devs / K] = -1
            sim = sim4 + sim3
            sim[sim >= 0] = 1
            sim[sim < 0] = -1
        return sim

    def homophily(self, sim='cos', type='kleinberg'):
        data = self.data
        N = self.data.shape[0]
        card = N*(N-1)
        sim_source = self.similarity_matrix(sim=sim)
        if sim_source is None:
            return np.nan, np.nan

        connected = data.sum()
        unconnected = N - connected
        similar = (sim_source > 0).sum()
        unsimilar = (sim_source <= 0).sum()

        indic_source = ma.array(np.ones(sim_source.shape)*-1, mask=ma.masked)
        indic_source[(data == 1) & (sim_source > 0)] = 0
        indic_source[(data == 1) & (sim_source <= 0)] = 1
        indic_source[(data == 0) & (sim_source > 0)] = 2
        indic_source[(data == 0) & (sim_source <= 0)] = 3

        np.fill_diagonal(indic_source, ma.masked)
        indic_source[indic_source == -1] = ma.masked

        a = (indic_source==0).sum()
        b = (indic_source==1).sum()
        c = (indic_source==2).sum()
        d = (indic_source==3).sum()

        if type == 'kleinberg':
            homo_obs = 1.0 * a / connected # precision; homophily respected
            homo_exp = 1.0 * similar / card # rappel; strenght of homophily
        else:
            raise NotImplementedError

        #if sim == 'euclide' and type is None:
        #    homo_obs = 1.0 * (a + d - c - b) / card
        #    pr = 1.0 * (data == 1).sum() / card
        #    ps = 1.0 * (indic_source==0).sum() / card
        #    pnr = 1.0 - pr
        #    pns = 1.0 - ps
        #    a_ = pr*ps*card
        #    b_ = pnr*ps*card
        #    c_ = pr*pns*card
        #    d_ = pnr*pns*card
        #    homo_expect = (a_+b_-c_-d_) /card
        #    return homo_obs, homo_expect

        return homo_obs, homo_exp

    def assort(self, model):
        #if not source:
        #    data = self.data
        #    sim_source = self.similarity_matrix('cos')
        data = self.data
        N = self.data.shape[0]
        sim_source = self.similarity_matrix(sim='cos')

        y, _, _ = model.generate(N)
        #y = np.triu(y) + np.triu(y, 1).T
        sim_learn = model.similarity_matrix(sim='cos')

        np.fill_diagonal(indic_source, ma.masked)

        assert(N == y.shape[0])

        indic_source = ma.array(np.ones(sim_source.shape)*-1, mask=ma.masked)
        indic_source[(data == 1) & (sim_source > 0)] = 0
        indic_source[(data == 1) & (sim_source <= 0)] = 1
        indic_source[(data == 0) & (sim_source > 0)] = 2
        indic_source[(data == 0) & (sim_source <= 0)] = 3

        indic_learn = ma.array(np.ones(sim_learn.shape)*-1, mask=ma.masked)
        indic_learn[(y == 1) & (sim_learn > 0)] = 0
        indic_learn[(y == 1) & (sim_learn <= 0)] = 1
        indic_learn[(y == 0) & (sim_learn > 0)] = 2
        indic_learn[(y == 0) & (sim_learn <= 0)] = 3

        np.fill_diagonal(indic_learn, ma.masked)
        np.fill_diagonal(indic_source, ma.masked)
        indic_source[indic_source == -1] = ma.masked
        indic_learn[indic_learn == -1] = ma.masked

        ### Indicateur Homophily Christine
        homo_ind1_source = 1.0 * ( (indic_source==0).sum()+(indic_source==3).sum()-(indic_source==1).sum() - (indic_source==2).sum() ) / (N*(N-1))
        homo_ind1_learn = 1.0 * ( (indic_learn== 0).sum()+(indic_learn==3).sum()-(indic_learn==1).sum() - (indic_learn==2).sum() ) / (N*(N-1))

        # AMI / NMI
        from sklearn import metrics
        AMI = metrics.adjusted_mutual_info_score(indic_source.compressed(), indic_learn.compressed())
        NMI = metrics.normalized_mutual_info_score(indic_source.compressed(), indic_learn.compressed())

        print 'homo_ind1 source: %f' % (homo_ind1_source)
        print 'homo_ind1 learn: %f' % (homo_ind1_learn)
        print 'AMI: %f, NMI: %f' % (AMI, NMI)

        d = {'NMI' : NMI, 'homo_ind1_source' : homo_ind1_source, 'homo_ind1_learn' : homo_ind1_learn}
        return d


