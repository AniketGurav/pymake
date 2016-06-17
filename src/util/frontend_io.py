import os
import fnmatch

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


def make_forest_path(spec, _type, sep=None):
    """ Make a list of path from a spec/dict """
    targets = []
    for base in ('networks',):
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
                                        t = 'inference-%s_%s_%s_%s_%s' % (m, k, h, hm,  n)
                                        t = os.path.join(p, t)
                                        filen = os.path.join(os.path.dirname(__file__), LOCAL_BDIR, t)
                                        if not os.path.isfile(filen) or os.stat(filen).st_size == 0:
                                            continue
                                        if sum(1 for line in open(filen)) <= 1:
                                            # empy file
                                            continue
                                        targets.append(t)

    return targets

def make_output_path(spec, _type, sep=None):
    """ Make a single output path from a spec/dict """
    filen = None
    base = 'networks'
    hook = spec['debug']
    c = spec['corpus']
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
        n = spec['N']
        m  = spec['model']
        k = spec['K']
        h = spec['hyper']
        hm = spec['homo']
        t = '%s_%s_%s_%s_%s' % (m, k, h, hm,  n)
        t = os.path.join(p, t)
        filen = os.path.join(os.path.dirname(__file__), LOCAL_BDIR, t)
        if not os.path.isfile(filen) or os.stat(filen).st_size == 0:
            continue
        if sum(1 for line in open(filen)) <= 1:
            # empy file
            continue

    if _type == 'pk':
        filen = filen + '.pk'
    elif _type == 'json':
        filen = filen + '.json'
    elif _type in ('inf', 'inference'):
        filen = 'inference-' + filen
    elif _type == 'all':
        filen = dict(pk=filen + '.pk',
                 json=filen + '.json',
                 inference='inference-'+filen)
    else:
        raise NotImplementedError

    return filen

def make_forest_conf(spec):
    """ Make a list of config/dict """
    targets = []
    for base in spec.get('base'):
        for hook in spec.get('hook_dir'):
            for c in spec.get('corpus'):
                p = os.path.join(base, c, hook)
                for n in spec.get('Ns'):
                    for m in spec.get('models'):
                        for k in spec.get('Ks'):
                            for h in spec.get('hyper'):
                                for hm in spec.get('homo'):
                                    for repeat in spec.get('repeat'):
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
                                             'repeat': repeat,
                                            }
                                        targets.append(d)
    return targets

def get_conf_from_file(target):
    """ Return dictionary of property for an expe file. (format
    inference-model_K_hyper_N) """
    _id = target.split('_')
    model = ''
    st = 0
    for s in _id:
        try:
            int(s)
            break
        except:
            st += 1
            model += s

    # @debug
    print target
    try:
        int(target.split('/')[-2])
        id_debug = -3
        id_corpus = -4
    except:
        id_debug = -2
        id_corpus = -3

    _id = _id[st:]
    prop = dict(
        model = model.split('-')[-1],
        repeat = target.split('/')[-2],
        debug = target.split('/')[id_debug],
        corpus = target.split('/')[id_corpus],
        K     = _id[0],
        hyper = _id[1],
        homo = _id[2],
        N     = _id[3],)

    # @debug
    try:
        int(target.split('/')[-2])
    except:
        del prop['repeat']

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


