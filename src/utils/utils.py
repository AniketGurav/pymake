import sys, os
from os.path import dirname
from datetime import datetime
from collections import defaultdict
import logging

import numpy as np
import scipy as sp

# use https://github.com/kennethreitz/args

def argParse(usage="Usage ?"):
    argdict = defaultdict(lambda: False)
    for i, arg in enumerate(sys.argv):
        if arg in ('-np', '--no-print', '--printurl'):
            argdict['noprint'] = True
        elif arg in ('-w',):
            argdict.update(write = True)
        elif arg in ('-nv',):
            argdict.update(verbose = logging.WARN)
        elif arg in ('-v',):
            argdict.update(verbose = logging.DEBUG)
        elif arg in ('-vv',):
            argdict.update(verbose = logging.CRITICAL)
        elif arg in ('-p',):
            argdict.update(predict = 1)
        elif arg in ('-s',):
            argdict['simul'] = arg
        elif arg in ('-nld',):
            argdict['load_data'] = False
        elif arg in ('--seed',):
            argdict['seed'] = 42
        elif arg in ('-n', '--limit'):
            # no int() because could be all.
            _arg = sys.argv.pop(i+1)
            argdict['N'] = _arg
        elif arg in ('--alpha', '--hyper'):
            _arg = sys.argv.pop(i+1)
            argdict['hyper'] = _arg
        elif arg in ('--hyper_prior', ):
            _arg1 = int(sys.argv.pop(i+1))
            _arg2 = int(sys.argv.pop(i+1))
            _arg3 = int(sys.argv.pop(i+1))
            _arg4 = int(sys.argv.pop(i+1))
            argdict['hyper_prior'] = (_arg1, _arg2, _arg3, _arg4)
        elif arg in ('--refdir',):
            _arg = sys.argv.pop(i+1)
            argdict['refdir'] = _arg
        elif arg in ('--repeat',):
            repeat = int(sys.argv.pop(i+1))
            if repeat < 0:
                _arg = ''
            else:
                _arg = str(repeat)
            argdict['repeat'] = _arg
        elif arg in ('-k',):
            _arg = int(sys.argv.pop(i+1))
            argdict['K'] = _arg
        elif arg in ('--homo',):
            _arg = int(sys.argv.pop(i+1))
            argdict['homo'] = _arg
        elif arg in ('-i',):
            _arg = int(sys.argv.pop(i+1))
            argdict['iterations'] = _arg
        elif arg in ('-c',):
            _arg = sys.argv.pop(i+1)
            argdict['corpus_name'] = _arg
        elif arg in ('-m',):
            _arg = sys.argv.pop(i+1)
            argdict['model_name'] = _arg
        elif arg in ('-d',):
            _arg = sys.argv.pop(i+1)
            argdict['bdir'] = _arg+'/'
        elif arg in ('-lall', ):
            argdict['lall'] = True
        elif arg in ('-l', '-load', '--load'):
            try:
                _arg = sys.argv[i+1]
            except:
                _arg = 'tmp'
            if not os.path.exists(_arg):
                if _arg == 'corpus' or _arg == 'model':
                    argdict['load'] = sys.argv.pop(i+1)
                else:
                    argdict['load'] = False
            else:
                _arg = sys.argv.pop(i+1)
                argdict['load'] = _arg
        elif arg in ('-r', '--random', 'random'):
            try:
                int(sys.argv[i+1])
                _arg = int(sys.argv.pop(i+1))
            except:
                _arg = 1
            finally:
                argdict['random'] = _arg
        elif arg in ('-g'):
            argdict.update(random = False)
        elif arg in ('--help','-h'):
            print(usage)
            exit(0)
        else:
            if i == 0:
                argdict.setdefault('arg', False)
                #argdict.setdefault('arg', 'no args')
            else:
                argdict.update({arg:arg})

    return argdict

def ask_sure_exit(question):
    while True:
        a = raw_input(question+' ').lower()
        if a == 'yes':
            break
        elif a == 'no':
            exit()
        else:
            print("Enter either yes/no")

def setup_logger(name, fmt, verbose, file=None):
    #formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    #logging.basicConfig(format='Gensim : %(message)s', level=logging.DEBUG)
    logging.basicConfig(format=fmt, level=verbose)

    formatter = logging.Formatter(fmt=fmt)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(verbose)
    logger.addHandler(handler)
    return logger

def ellapsed_time(text, since):
    current = datetime.now()
    delta = current - since
    print(text + ' : %s' % (delta))
    return current

def jsondict(d):
    if isinstance(d, dict):
        return {str(k):v for k,v in d.items()}
    return d

def parse_file_conf(fn):
    with open(fn) as f:
        parameters = f.read()
    parameters = filter(None, parameters.split('\n'))
    parameters = dict((p[0], p[1]) for p  in (t.strip().split(':') for t in parameters))
    for k, v in parameters.items():
        if  '.' in v:
            try:
                parameters[k] = float(v)
            except:
                pass
        else:
            try:
                parameters[k] = int(v)
            except:
                pass
    return parameters


def getGraph(target='', **conf):
    basedir = conf.get('filen', dirname(__file__) + '/../../data/networks/')
    filen = basedir + target
    f = open(filen, 'r')

    data = []
    N = 0
    inside = [False, False]
    for line in f:
        if line.startswith('# Vertices') or inside[0]:
            if not inside[0]:
                inside[0] = True
                continue
            if line.startswith('#') or not line.strip() :
                inside[0] = False
            else:
                # Parsing assignation
                N += 1
        elif line.startswith('# Edges') or inside[1]:
            if not inside[1]:
                inside[1] = True
                continue
            if line.startswith('#') or not line.strip() :
                inside[1] = False
            else:
                # Parsing assignation
                data.append( line.strip() )
    f.close()
    edges = [tuple(row.split(';')) for row in data]
    g = np.zeros((N,N))
    g[[e[0] for e in edges],  [e[1] for e in edges]] = 1
    g[[e[1] for e in edges],  [e[0] for e in edges]] = 1
    return g

from scipy import ndimage
def dilate(y):
    mask = ndimage.generate_binary_structure(2, 2)
    y_f = ndimage.binary_dilation(y, structure=mask).astype(y.dtype)
    return y_f

def getClique(N=100, K=4):
    from scipy.linalg import block_diag
    b = []
    for k in range(K):
        n = N / K
        b.append(np.ones((n,n)))

    C = block_diag(*b)
    return C

try:
    from sklearn.cluster import KMeans
    from sklearn.datasets.samples_generator import make_blobs
except:
    pass
def kmeans(M, K=4):
    km = KMeans( init='k-means++',  n_clusters=K)
    km.fit(M)
    clusters = km.predict(M.astype(float))
    return clusters.astype(int)

from matplotlib import pyplot as plt
def kmeans_plus(X=None, K=4):
    if X is None:
        centers = [[1, 1], [-1, -1], [1, -1]]
        K = len(centers)
        X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)

	##############################################################################
	# Compute clustering with Means

    k_means = KMeans(init='k-means++', n_clusters=K, n_init=10)
    k_means.fit(X)
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    k_means_labels_unique = np.unique(k_means_labels)

	##############################################################################
	# Plot result

    colors = ['#4EACC5', '#FF9C34', '#4E9A06', '#4E9A97']
    plt.figure()
    plt.hold(True)
    for k, col in zip(range(K), colors):
    #for k in range(K):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        plt.plot(X[my_members, 0], X[my_members, 1], 'w',
                 markerfacecolor=col, marker='.')
        plt.plot(cluster_center[0], cluster_center[1], 'o',
                 markeredgecolor='k', markersize=6, markerfacecolor=col)
    plt.title('KMeans')
    plt.grid(True)
    plt.show()



from scipy.io.matlab import loadmat
from scipy.sparse import lil_matrix
try:
    from rescal import rescal_als
except:
    pass
def rescal(X, K):

    ## Set logging to INFO to see RESCAL information
    #logging.basicConfig(level=logging.INFO)

    ## Load Matlab data and convert it to dense tensor format
    #T = loadmat('data/alyawarra.mat')['Rs']
    #X = [lil_matrix(T[:, :, k]) for k in range(T.shape[2])]

    X = [sp.sparse.csr_matrix(X)]
    A, R, fit, itr, exectimes = rescal_als(X, K, init='nvecs', lambda_A=10, lambda_R=10)

    theta =  A.dot(R).dot(A.T)
    Y = 1 / (1 + np.exp(-theta))
    Y =  Y[:,0,:]
    Y[Y <= 0.5] = 0
    Y[Y > 0.5] = 1
    #Y = sp.stats.bernoulli.rvs(Y)
    return Y

# Assign new values to an array according to a map list
def set_v_to(a, map):
    new_a = a.copy()
    for k, c in dict(map).iteritems():
        new_a[a==k] = c

    return new_a


# Re-order the confusion matrix in order to map the cluster (columns) to the best (classes) according to purity
# One class by topics !
# It modify confu and map in-place
# Return: list of tuple that map topic -> class
import sys
sys.setrecursionlimit(10000)
def map_class2cluster_from_confusion(confu, map=None, cpt=0, minmax='max'):
    assert(confu.shape[0] == confu.shape[1])

    if minmax == 'max':
        obj_f = np.argmax
    else:
        obj_f = np.argmin

    if len(confu) -1  == cpt:
        # Recursive stop condition
        return sorted(map)
    if map is None:
        confu = confu.copy()
        map = [ (i,i) for i in range(len(confu)) ]
        #map = np.array(map)

    #K = confu.shape[0]
    #C = confu.shape[1]
    previous_assigned = [i[1] for i in map[:cpt]]
    c_l = obj_f(np.delete(confu[cpt], previous_assigned))
    # Get the right id of the class
    for j in sorted(previous_assigned):
        # rectify c_l depending on which class where already assigning
        if c_l >= j:
            c_l += 1
        else:
            break
    m_l = confu[cpt, c_l]
    # Get the right id of the topic
    c_c = obj_f(confu[cpt:,c_l]) + cpt
    m_c = confu[c_c, c_l]
    if m_c > m_l:
        # Move the line corresponding to the max for this class to the top
        confu[[cpt, c_c], :] = confu[[c_c, cpt], :]
        map[cpt], map[c_c] = map[c_c], map[cpt] # Doesn't work if it's a numpy array
        return map_class2cluster_from_confusion(confu, map, cpt)
    else:
        # Map topic 1 to class c_l and return confu - topic 1 and class c_l
        map[cpt] = (map[cpt][0], c_l)
        cpt += 1
        return map_class2cluster_from_confusion(confu, map, cpt)

def make_path(bdir):
    fn = os.path.basename(bdir)
    _bdir = os.path.dirname(bdir)
    if not os.path.exists(_bdir) and _bdir:
        os.makedirs(_bdir)
    if not os.path.exists(fn) and fn:
        #open(fn, 'a').close()
        pass # do i need it
    return bdir

