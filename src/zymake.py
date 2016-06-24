#!/usr/bin/python -u
# -*- coding: utf-8 -*-

from joblib import Parallel, delayed
import multiprocessing
import args

from local_utils import *
from util.frontend_io import *
from expe.spec import *
from expe.run import *


args = args.grouped['_']
if len(args) == 2:
    OUT_TYPE = args.get(0)
    SPEC = globals()[args.get(1)]
else:
    OUT_TYPE = 'path'
    SPEC = RUN_DD

### Makes target files
if OUT_TYPE in ('path', 'files'):
    source_files = make_forest_path(SPEC, 'pk', status=None)
elif OUT_TYPE == 'runcmd':
    source_files = make_forest_runcmd(SPEC)
else:
    raise NotImplementedError


### Makes figures on remote / parallelize
#num_cores = int(multiprocessing.cpu_count() / 4)
#results_files = Parallel(n_jobs=num_cores)(delayed(expe_figures)(i) for i in source_files)
### ...and Retrieve the figure


print '\n'.join(source_files)

