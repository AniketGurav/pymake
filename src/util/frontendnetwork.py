import sys, os
import json
from itertools import chain
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
        config['bdir'] = os.path.join(config['bdir'], 'networks')
        super(frontendNetwork, self).__init__(config)

    def load_data(self, corpus_name=None, randomize=False):
        corpus_name = corpus_name or self.config.get('corpus_name')
        if corpus_name:
            self.get_corpus(corpus_name)
        elif self.config.get('random'):
            self.random()
        else:
            raise ValueError()

        np.fill_diagonal(self.data, 1)
        if randomize:
            self.shuffle_node()
        return self.data

    # Shuffle rows and columns of data
    def shuffle_node(self):
        N, M = self.data.shape
        nodes_list = [np.random.permutation(N), np.random.permutation(M)]
        self.reorder_node(nodes_list)

    # Subsample the data with reordoring of rows and columns
    def reorder_node(self, nodes_l):
        self.data = self.data[nodes_l[0], :][:, nodes_l[1]]
        # Track the original nodes
        self.nodes_list = [self.nodes_list[0][nodes_l[0]], self.nodes_list[1][nodes_l[1]]]

    def sample(self, N=None, symmetric=False, randomize=False):
        N = N or self.N
        n = self.data.shape[0]
        if not N or N == 'all':
            self.N = 'all'
        else:
            # Can't get why modification inside self.nodes_list is not propagated ?
            N = int(N)
            if randomize is True:
                nodes_list = [np.random.permutation(N), np.random.permutation(N)]
                self.reorder_node(nodes_list)
            else:
                self.data = self.data[:N, :N]

            if hasattr(self, 'features'):
                self.features = self.features[:N]

        return self.data

    def get_masked(self, percent_hole, diag_off=1):
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

    # Construct Mask nased on the proportion of 1/links
    def get_masked_1(self, percent_hole, diag_off=1):
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
        N = int(self.N)

        if corpus_name[:-1] in ('generator'):
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
        else:
            raise ValueError('Which corpus to Load ?')

        self.data = data
        N, M = self.data.shape
        self.nodes_list = [np.arange(N), np.arange(M)]
        return True

    ### Load pickle or parse data
    def networkloader(self, fn):

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
            else:
                raise ValueError('extension of network data unknown')

        if self._save_data:
            self.save(data, fn+'.pk')

        return data

    ### Parse Network data depending on type/extension
    def parse_graph(self, fn):
        max_n = 1100
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
                    if N >= max_n:
                        inside['vertices'] = False
                        continue
                    elements = line.strip().split(';')
                    clust = int(elements[-1])
                    feats = map(float, elements[-2].split('|'))
                    clusters.append(clust)
                    features.append(feats)
                    N += 1
            elif line.startswith('# Edges') or inside['edges']:
                if not inside['edges']:
                    inside['edges'] = True
                    N = 0
                    continue
                if line.startswith('#') or not line.strip() :
                    inside['edges'] = False
                else:
                    # Parsing assignation
                    if N >= max_n:
                        inside['edges'] = False
                        continue
                    data.append( line.strip() )
                    N += 1
        f.close()

        edges = [tuple(row.split(';')) for row in data]
        g = np.zeros((N,N))
        g[[e[0] for e in edges], [e[1] for e in edges]] = 1
        g[[e[1] for e in edges], [e[0] for e in edges]] = 1

        self.clusters = clusters
        self.features = np.array(features)
        return g

    def communities_analysis(self):
        clusters = self.clusters
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

    def similarity_matrix(self, sim='cos'):
        features = self.features
        if sim == 'dot':
            sim = np.dot(features, features.T)
        elif sim == 'cos':
            norm = np.linalg.norm(features, axis=1)
            sim = np.dot(features, features.T)/norm/norm.T
        return sim

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
        cc = None
        try:
            cc = nx.average_clustering(G)
        except:
            pass
        return cc

    def GG(self):
        if not hasattr(self, 'G'):
            if self.is_symmetric():
                typeG = nx.Graph()
            else:
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
               'communities': max(self.clusters)+1,
               'features': self.get_nfeat(),
              }
        prop.update(d)
        return prop

    def template(self, d):
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
        \n'''
        return super(frontendNetwork, self).template(d, netw_templ)

    def get_json(self):
        fn = self.config['output_path'][:-len('.out')] + '.json'
        d = json.load(open(fn,'r'))
        return d
    def update_json(self, d):
        fn = self.config['output_path'][:-len('.out')] + '.json'
        res = json.load(open(fn,'r'))
        res.update(d)
        print 'updating json: %s' % fn
        json.dump(res, open(fn,'w'))
        return fn

    def assort(self, model):
        #if not source:
        #    data = self.data
        #    sim_source = self.similarity_matrix('cos')
        data = self.data
        N = self.data.shape[0]
        sim_source = self.similarity_matrix(sim='cos')

        y, _, _ = model.generate(N)
        y = np.triu(y) + np.triu(y, 1).T
        sim_learn = model.similarity_matrix(sim='cos')

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
        homo_ind1_source = 2.0 * ( (indic_source==0).sum()+(indic_source==3).sum()-(indic_source==1).sum() - (indic_source==2).sum() ) / (N*(N-1))
        homo_ind1_learn = 2.0 * ( (indic_learn== 0).sum()+(indic_learn==3).sum()-(indic_learn==1).sum() - (indic_learn==2).sum() ) / (N*(N-1))

        # AMI / NMI
        from sklearn import metrics
        AMI = metrics.adjusted_mutual_info_score(indic_source.compressed(), indic_learn.compressed())
        NMI = metrics.normalized_mutual_info_score(indic_source.compressed(), indic_learn.compressed())

        print 'homo_ind1 source: %f' % (homo_ind1_source)
        print 'homo_ind1 learn: %f' % (homo_ind1_learn)
        print 'AMI: %f, NMI: %f' % (AMI, NMI)

        d = {'NMI' : NMI, 'homo_ind1_source' : homo_ind1_source, 'homo_ind1_learn' : homo_ind1_learn}
        return d

    @staticmethod
    def make_conf(spec):
        targets = []
        for base in spec['base']:
            for hook in spec['hook_dir']:
                for c in spec['corpus']:
                    p = os.path.join(base, c, hook)
                    for n in spec['Ns']:
                        for m in spec['models']:
                            for k in spec['Ks']:
                                for h in spec['hyper']:
                                    for hm in spec['homo']:
                                        # generate path trick wrong
                                        #t = 'inference-%s_%s_%s_%s_%s' % (m, k, h, hm,  n)
                                        #t = os.path.join(p, t)
                                        #filen = os.path.join(os.path.dirname(__file__), "../../data/", t)
                                        #if not os.path.isfile(filen) or os.stat(filen).st_size == 0:
                                        #    continue
                                        d = {'N' : n,
                                             'K' : k,
                                             'hyper': h,
                                             'homo': hm,
                                             'model': m,
                                             'corpus_name': c,
                                             'refdir': hook,
                                            }
                                        targets.append(d)
        return targets





