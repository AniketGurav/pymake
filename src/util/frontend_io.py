import re, os, json, logging
from collections import defaultdict, OrderedDict
import fnmatch
import numpy as np
from expe.spec import _spec_

lgg = logging.getLogger('root')

LOCAL_BDIR = '../../data/' # Last slash(/) necessary.
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

### Command Line Reference
_OPTKEYS_ = OrderedDict((
    ('N'           , '-n'),
    ('K'           , '-k'),
    ('hyper'       , '--hyper'),
    ('homo'        , '--homo'),
    ('model'       , '-m'),
    ('model_name'  , '-m'),
    ('corpus_name' , '-c'),
    ('corpus'      , '-c'),
    ('refdir'      , '--refdir'),
    ('debug'       , '--refdir'),
    ('repeat'      , '--repeat'),
    ('iterations'  , '-i'),
    ('data_type'   , 'null'),
))

### directory/file tree reference
_MASTERKEYS_ = OrderedDict((
    ('data_type'   , None),
    ('corpus'      , None),
    ('debug'       , None),
    ('repeat'      , None),
    ('model'       , None),
    ('K'           , 5),
    ('hyper'       , None),
    ('homo'        , None),
    ('N'           , 'all'),
))

_New_Dims = [{'measure':4}]


# Factorize the io; One argument / One options.
# ALign file and args generation
# support opt: list of str or int

# @DEBUG: Align topic config and frontend_io
# * plurial (s) vs whithout s...
# * Make_output_path from materskeys with default value and handle complex tree logic!

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

def make_forest_path(dol_spec, _type, sep=None, status='f', full_path=False):
    """ Make a list of path from a spec/dict, the filename are
        oredered need to be align with teh get_from_conf_file.

        *args -> make_output_path
    """
    targets = []
    check_spec(dol_spec)
    lod_spec = make_forest_conf(dol_spec)
    for spec in lod_spec:
        filen = make_output_path(spec, _type, status=status)
        if filen:
            s = filen.find(LOCAL_BDIR)
            pt = 0
            if not full_path and s >= 0:
                pt = s + len(LOCAL_BDIR)
            targets.append(filen[pt:])
    return targets

def make_output_path(spec, _type=None, sep=None, status=False):
    """ Make a single output path from a spec/dict
        @status: f finished
        @type: pk, json or inference.
    """
    filen = None
    base = spec['data_type']
    hook = spec.get('debug') or spec.get('refdir', '')
    c = spec.get('corpus') or spec['corpus_name']
    if c.startswith(('clique', 'Graph', 'generator')):
        c = c.replace('generator', 'Graph')
        c = 'generator/' + c
    if 'repeat' in spec and ( spec['repeat'] is not None and spec['repeat'] is not False):
        p = os.path.join(base, c, hook, str(spec['repeat']))
    else:
        p = os.path.join(base, c, hook)
    m  = spec.get('model') or spec['model_name']
    k = spec['K']
    h = spec['hyper']
    hm = spec['homo']
    n = spec['N']
    t = '%s_%s_%s_%s_%s' % (m, k, h, hm,  n)
    filen = os.path.join(p, t)
    filen = os.path.join(os.path.dirname(__file__), LOCAL_BDIR, filen)

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

from operator import mul
from itertools import product
def make_forest_conf(dol_spec):
    """ Make a list of config/dict.
        Convert a dict of list to a list of dict.
    """
    check_spec(dol_spec)
    len_l = [len(l) for l in dol_spec.values()]
    _len = reduce(mul, len_l )
    keys = sorted(dol_spec)
    lod = [dict(zip(keys, prod)) for prod in product(*(dol_spec[key] for key in keys))]

    targets = []
    for d in lod:
        _d = defaultdict(lambda: False)
        _d.update(d)
        targets.append(_d)

    return targets

def check_spec(dol_spec):
    for k, v in dol_spec.items():
        if not isinstance(v, (list, tuple, set)):
            dol_spec[k] = [v]
        else:
            pass
    return True

def make_forest_runcmd(spec):
    optkeys = _OPTKEYS_
    opts = []
    confs = make_forest_conf(spec)
    for expe in confs:
        if '' in expe:
            expe.pop('')
        opt = [' '.join(map(str, (optkeys[i], j))) for i, j in expe.items() if optkeys[i] != 'null']
        opt = ' '.join(opt)
        opts.append(opt)
    return opts

def tree_hook(key, value):
    hook = False
    if key == 'corpus':
        if value in ('generator', ):
            hook = True
    return hook


def get_conf_from_file(target, mp):
    """ Return dictionary of property for an expe file.
        @mp: map parameters
        format inference-model_K_hyper_N.
        @template_file order important to align the dictionnary.
        """
    masterkeys = _MASTERKEYS_.copy()
    template_file = masterkeys.keys()
    ##template_file = 'networks/generator/Graph13/debug11/inference-immsb_10_auto_0_all'

    # Relative path ignore
    if target.startswith(LOCAL_BDIR):
        target.replace(LOCAL_BDIR, '')

    path = target.lstrip('/').split('/')

    _prop = os.path.splitext(path.pop())[0]
    _prop = path + _prop.split('_')

    prop = {}
    cpt_hook_master = 0
    cpt_hook_user = 0
    # @Debug/Improve the nasty Hook here
    def update_pt(cur, master, user):
        return cur - master + user

    #prop = {k: _prop[i] for i, k in enumerate(template_file) if k in mp}
    for i, k in enumerate(template_file):
        if not k in mp:
            cpt_hook_master += 1
            continue
        pt = update_pt(i, cpt_hook_master, cpt_hook_user)
        hook = tree_hook(k, _prop[pt])
        if hook:
            cpt_hook_user += 1
            pt = update_pt(i, cpt_hook_master, cpt_hook_user)
        prop[k] = _prop[pt]

    return prop

def get_conf_dim_from_files(targets, mp):
    """ Return the sizes of proporties in a list for expe files
        @mp: map parameters """
    c = []
    for t in targets:
        c.append(get_conf_from_file(t, mp))

    sets = {}
    keys_name = mp.keys()
    for p in keys_name:
        sets[p] = len(set([ _p[p] for _p in c ]))

    return sets

def get_json(fn):
    try:
        d = json.load(open(fn,'r'))
        return d
    except Exception, e:
        return None

def forest_tensor(target_files, map_parameters):
    """ It has to be ordered the same way than the file properties.
        Fuze directory to find available files then construct the tensor
        according the set space fomed by object found.
        @in target_files has to be orderedDict to align the the tensor access.
    """
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
        lgg.info('Target Files empty')
        return None

    #dim = get_conf_dim_from_files(target_files, map_parameters) # Rely on Spec...
    dim = dict( (k, len(v)) if isinstance(v, (list, tuple)) else (k, len([v])) for k, v in map_parameters.items() )

    rez_map = map_parameters.keys() # order !
    # Expert knowledge value
    new_dims = _New_Dims
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
        prop = get_conf_from_file(_f, map_parameters)
        pt = np.empty(rez.ndim)

        assert(len(pt) - len(new_dims) == len(prop))
        for k, v in prop.items():
            try:
                v = int(v)
            except:
                pass
            try:
                idx = map_parameters[k].index(v)
            except Exception, e:
                lgg.error(prop)
                lgg.error('key:value error --  %s, %s'% (k, v))
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

        try:
            pt = list(pt.astype(int))
            pt[-1] = 0
            rez[zip(pt)] = g_precision
            pt[-1] = 1
            rez[zip(pt)] = precision
            pt[-1] = 2
            rez[zip(pt)] = recall
            pt[-1] = 3
            rez[zip(pt)] = K
        except IndexError, e:
            lgg.error(e)
            lgg.error('Index Error: Files are probably missing here to complete the results...\n')

        #info_file.append( '%s %s; \t K=%s\n' % (corpus_type, f, K) )

    lgg.debug(''.join(not_finished))
    #lgg.debug(''.join(info_file))
    rez = np.ma.masked_array(rez, np.isnan(rez))
    return rez

def clean_extra_expe(expe, map_parameters):
    for k in expe:
        if k not in map_parameters and k not in [ k for d in _New_Dims for k in d.keys() ] :
            del expe[k]
    return expe

def make_tensor_expe_index(expe, map_parameters):
    ptx = []
    expe = clean_extra_expe(expe, map_parameters)
    for i, o in enumerate(expe.items()):
        k, v = o[0], o[1]
        print i, k, v
        if v in ( '*', ':'): #wildcar / indexing ...
            ptx.append(slice(None))
        elif k in map_parameters:
            ptx.append(map_parameters[k].index(v))
        elif type(v) is int:
            ptx.append(v)
        else:
            raise ValueError('Unknow data type for tensor forest')

    ptx = tuple(ptx)
    return ptx


