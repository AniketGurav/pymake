#!/usr/bin/python -u
# -*- coding: utf-8 -*-

from joblib import Parallel, delayed
import multiprocessing
import args

from local_utils import *
from util.frontend_io import *
from expe.spec import *
from expe.run import *

### Parsing CLI
if args.flags.contains('---help') or args.flags.contains('-h'):
    print '''Usage:
    zymake.py runcmd SPEC
    or
    zymake path SPEC Filetype(pk|json|inf)
    ''';  exit()

gargs = args.grouped['_']
OUT_TYPE = gargs.get(0) or 'path'
if OUT_TYPE == 'runcmd' and len(gargs) == 2:
    SPEC = globals()[gargs.get(1)]
elif OUT_TYPE == 'path' and len(gargs) == 3:
    SPEC = globals()[gargs.get(1)]
    FTYPE = gargs.get(2)
    STATUS = None
else:
    print 'Debug Specification /!\ '
    SPEC = RUN_DD
    FTYPE = 'pk'
    STATUS = None

### Makes OUT Files
if OUT_TYPE == 'runcmd':
    source_files = make_forest_runcmd(SPEC)
elif OUT_TYPE == 'path':
    source_files = make_forest_path(SPEC, FTYPE, status=STATUS)
else:
    raise NotImplementedError


### Makes figures on remote / parallelize
#num_cores = int(multiprocessing.cpu_count() / 4)
#results_files = Parallel(n_jobs=num_cores)(delayed(expe_figures)(i) for i in source_files)
### ...and Retrieve the figure


print '\n'.join(source_files)

