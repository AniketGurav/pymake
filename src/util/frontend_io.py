import re, os, sys, json, logging
import fnmatch
import numpy as np

LOCAL_BDIR = '../../data'
"""
    #### I/O
    Corpus are load/saved using Pickle format in:
    * bdir/corpus_name.pk
    Models are load/saved using cPickle/json/cvs in :
    * bdir/debug/rept/model_name_parameters.pk <--> ModelManager
    * bdir/debug/rept/model_name_parameters.json <--> DataBase
    * bdir/debug/rept/inference-model_name_parameters <--> ModelBase

    Filnemame is formatted as follow:
    fname_out = '%s_%s_%s_%s_%s' % (self.model_name,
                                        self.K,
                                        self.hyper_optimiztn,
                                        self.homo,
                                        self.N)
"""

# @DEBUG: Align topic config and frontend_io
# * refdir vs debug
# * plurial (s) vs whithout s...
# get output_path from here

def model_walker(bdir, fmt='list'):
    models_files = []
    if fmt == 'list':
        ### Easy formating
        for root, dirnames, filenames in os.walk(bdir):
            for filename in fnmatch.filter(filenames, '*.pk'):
                models_files.append(os.path.join(root, filename))
        return models_files
    else:
        ### More Complex formating
        tree = { 'json': [],
                'pk': [],
                'inference': [] }
        for filename in fnmatch.filter(filenames, '*.pk'):
            if filename.startswith(('dico.','vocab.')):
                dico_files.append(os.path.join(root, filename))
            else:
                corpus_files.append(os.path.join(root, filename))
        raise NotImplementedError()
    return tree


def make_forest_path(spec, _type, sep=None, status='f'):
    """ Make a list of path from a spec/dict
        @status: f finished,  e exist.
        @type: pk, json or inference.
    """
    targets = []
    for base in spec['data_type']:
        for hook in spec['debug']:
            for c in spec['corpus']:
                if c.startswith(('clique', 'Graph', 'generator')):
                    c = c.replace('generator', 'Graph')
                    _c = 'generator/' + c
                else:
                    _c = c
                if spec.get('repeat'):
                    _p = [os.path.join(base, _c, hook, rep) for rep in spec['repeat']]
                else:
                    _p = [ os.path.join(base, _c, hook) ]
                for p in _p:
                    for n in spec['N']:
                        for m in spec['model']:
                            for k in spec['K']:
                                for h in spec['hyper']:
                                    for hm in spec['homo']:
                                        t = '%s_%s_%s_%s_%s' % (m, k, h, hm,  n)
                                        filen = os.path.join(p, t)
                                        ext = ext_status(filen, _type)
                                        if ext:
                                            filen = ext
                                        else:
                                            pass

                                        _f = os.path.join(os.path.dirname(__file__), LOCAL_BDIR, filen)
                                        if status is 'f' and is_empty_file(_f):
                                            continue

                                        targets.append(filen)

    return targets

def make_output_path(spec, _type=None, sep=None, status=False):
    """ Make a single output path from a spec/dict
        @status: f finished,  e exist.
        @type: pk, json or inference.
    """
    filen = None
    base = spec['data_type']
    hook = spec.get('debug') or spec.get('refdir', '')
    c = spec.get('corpus') or spec['corpus_name']
    if c.startswith(('clique', 'Graph', 'generator')):
        c = c.replace('generator', 'Graph')
        c = 'generator/' + c
    if spec.get('repeat'):
        p = os.path.join(base, c, hook, spec['repeat'])
    else:
        p = os.path.join(base, c, hook)
    m  = spec.get('model') or spec['model_name']
    k = spec['K']
    h = spec['hyper']
    hm = spec['homo']
    n = spec['N']
    t = '%s_%s_%s_%s_%s' % (m, k, h, hm,  n)
    t = os.path.join(p, t)
    filen = os.path.join(os.path.dirname(__file__), LOCAL_BDIR, t)

    ext = ext_status(filen, _type)
    if ext:
        filen = ext
    else:
        basedir = os.path.join(os.path.dirname(__file__), LOCAL_BDIR,
                               base, c)
        filen = (basedir, filen)

    if status is 'f' and is_empty_file(filen):
        filen = None

    return filen

def is_empty_file(filen):
    if not os.path.isfile(filen) or os.stat(filen).st_size == 0:
        return True

    with open(filen, 'r') as f: first_line = f.readline()
    if first_line[0] in ('#', '%') and sum(1 for line in open(filen)) <= 1:
        # empy file
        return True
    else:
       return False

def ext_status(filen, _type):
    nf = None
    if _type == 'pk':
        nf = filen + '.pk'
    elif _type == 'json':
        nf = filen + '.json'
    elif _type in ('inf', 'inference'):
        nf = 'inference-' + filen
    elif _type == 'all':
        nf = dict(pk=filen + '.pk',
                 json=filen + '.json',
                 inference='inference-'+filen)
    return nf

def make_forest_conf(spec):
    """ Make a list of config/dict """
    targets = []
    for base in spec.get('data_type'):
        for hook in spec.get('refdir'):
            for c in spec.get('corpus'):
                p = os.path.join(base, c, hook)
                for n in spec.get('N'):
                    for m in spec.get('model'):
                        for k in spec.get('K'):
                            for h in spec.get('hyper'):
                                for hm in spec.get('homo'):
                                    for repeat in spec.get('repeat'):
                                        d = {'N' : n,
                                             'K' : k,
                                             'hyper': h,
                                             'homo': hm,
                                             'model': m,
                                             'corpus_name': c,
                                             'refdir': hook,
                                             'repeat': repeat,
                                            }
                                        targets.append(d)
    return targets

def get_conf_from_file(target):
    """ Return dictionary of property for an expe file.
    * format inference-model_K_hyper_N.
    * @template_file order important to align the dictionnary.
        """
    ##template = 'networks/generator/Graph13/debug11/inference-immsb_10_auto_0_all'
    template_file = [ 'data_type', 'corpus', 'debug',
        'model', 'K', 'hyper', 'homo', 'N', ]
    path = target.split('/')

    # Add repeat setting
    for i, v in enumerate(path):
        if str.isdigit(v):
            template_file.insert(i, 'repeat')
            break

    _prop = os.path.splitext(path.pop())[0]
    _prop = path + _prop.split('_')

    # Pseudo path ignore
    if len(_prop) > len(template_file):
        _prop.pop(1)

    prop = {k: _prop[i] for i, k in enumerate(template_file)}
    return prop

def get_conf_dim_from_files(targets):
    """ Return size of proportie in a list for expe files """
    template = 'networks/generator/Graph13/debug11/inference-immsb_10_auto_0_all'
    c = []
    for t in targets:
        c.append(get_conf_from_file(t))

    sets = {}
    keys_name = get_conf_from_file(template).keys()
    for p in keys_name:
        sets[p] = len(set([ _p[p] for _p in c ]))

    return sets

def get_json(fn):
    try:
        d = json.load(open(fn,'r'))
        return d
    except Exception, e:
        return None

def forest_tensor(target_files, map_parameters, verbose=False):
    """ @in target_files has to be orderedDict to align the the tensor access. """
    # Expe analyser / Tabulyze It

    # res shape ([expe], [model], [measure]
    # =================================================================================
    # Expe: [debug, corpus] -- from the dirname
    # Model: [name, K, hyper, homo] -- from the expe filename
    # measure:
    #   * 0: global precision,
    #   * 1: local precision,
    #   * 2: recall

    ### Output: rez.shape rez_map_l rez_map
    if not target_files:
        print 'Target Files empty'
        return None

    dim = get_conf_dim_from_files(target_files)
    map_parameters = map_parameters

    rez_map = map_parameters.keys() # order !
    # Expert knowledge value
    new_dims = [{'measure':4}]
    # Update Mapping
    [dim.update(d) for d in new_dims]
    [rez_map.append(n.keys()[0]) for n in new_dims]

    # Create the shape of the Ananisys/Resulst Tensor
    #rez_map = dict(zip(rez_map_l, range(len(rez_map_l))))
    shape = []
    for n in rez_map:
        shape.append(dim[n])

    # Create the numpy array to store all experience values, whith various setings
    rez = np.zeros(shape) * np.nan

    not_finished = []
    info_file = []
    for _f in target_files:
        prop = get_conf_from_file(_f)
        pt = np.empty(rez.ndim)

        #print prop
        assert(len(pt) - len(new_dims) == len(prop))
        for k, v in prop.items():
            try:
                v = int(v)
            except:
                pass
            try:
                idx = map_parameters[k].index(v)
            except Exception, e:
                print e
                print k, v
                raise ValueError
            pt[rez_map.index(k)] = idx

        f = os.path.join(os.path.dirname(__file__), LOCAL_BDIR, _f)
        d = get_json(f)
        if not d:
            not_finished.append( '%s not finish...\n' % _f)
            continue

        g_precision = d.get('g_precision')
        precision = d.get('Precision')
        recall = d.get('Recall')
        K = len(d['Local_Attachment'])
        #density = d['density_all']
        #mask_density = d['mask_density']
        #h_s = d.get('homo_ind1_source', np.inf)
        #h_l = d.get('homo_ind1_learn', np.inf)
        #nmi = d.get('NMI', np.inf)

        pt = list(pt.astype(int))
        pt[-1] = 0
        rez[zip(pt)] = g_precision
        pt[-1] = 1
        rez[zip(pt)] = precision
        pt[-1] = 2
        rez[zip(pt)] = recall
        pt[-1] = 3
        rez[zip(pt)] = K

        #info_file.append( '%s %s; \t K=%s\n' % (corpus_type, f, K) )

    if verbose:
        [ sys.stdout.write(m) for m in not_finished ]
        print
        #[ sys.stdout.write(m) for m in info_file]
    return rez



