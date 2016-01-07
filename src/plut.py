#!/usr/bin/env python

import numpy as np
import scipy as sp
import  random, datetime
from util.plot import *
import matplotlib.pyplot as plt

#g = getGraph()

#p = ('K_test_diag', 'K_test_W')
#for t in p:
#    plot_K_fix(target_dir=t)

p = ('consistency_test_anop',
     'consistency_test_aprior1',
     'consistency_test_aprior2',
    )

for t in p:
    plot_K_dyn(target_dir=t)

#p = ('K_test_diag',
#     'K_test_W',
#    )
#
#for t in p:
#    plot_K_fix(target_dir=t)

plt.show()

