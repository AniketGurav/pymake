#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import os
from multiprocessing import Process
from itertools import cycle

from local_utils import *

def display(block=False):
    #p = Process(target=_display)
    #p.start()
    plt.show(block=block)

def _display():
    os.setsid()
    plt.show()

def tag_from_csv(c):
    ## loglikelihood_Y, loglikelihood_Z, alpha, sigma, _K, Z_sum, ratio_MH_F, ratio_MH_W
    if c == 0:
        ylabel = 'Iteration'
        label = 'Iteration'
    elif c == 1:
        ylabel = 'loglikelihood'
        label = 'loglikelihood'

    return ylabel, label

def csv_row(s):
    #csv_typo = '# mean_eta, var_eta, mean_alpha, var_alpha, log_perplexity'
    if s == 'Iteration':
        row = 0
    elif s in ('loglikelihood', 'likelihood', 'perplexity'):
        row = 1
    elif s == 'K':
        row = 2
    else:
        row = s
    return row

def plot_csv(target_dirs='', columns=0, sep=' ', separate=False, title=None):
    if type(columns) is not list:
        columns = [columns]

    if type(target_dirs) is not list:
        target_dirs = [target_dirs]

    title = title or 'Inference'
    xlabel = 'Iterations'

    prop_set = get_expe_file_set_prop(target_dirs)
    id_plot = 0
    old_prop = 0
    fig = None
    _Ks = None
    markers = cycle([ '+', '*', ',', 'o', '.', '1', 'p', ])
    if separate is False:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel(xlabel)
        ax1.set_title(title)

    for i, target_dir in enumerate(target_dirs):

        filen = os.path.join(os.path.dirname(__file__), "../data/", target_dir)
        with open(filen) as f:
            data = f.read()

        data = filter(None, data.split('\n'))
        data = [x.strip() for x in data if not x.startswith(('#', '%'))]

        prop = get_expe_file_prop(target_dir)
        for column in columns:
            if type(column) is str:
                column = csv_row(column)

			### Optionnal row...?%
            if separate is True:
                fig = plt.figure()
                plt.title(title)
                plt.xlabel(xlabel)
                ax1 = plt.gca()
            elif type(separate) is int:
                if i % (separate*2) == 0:
                    fig = plt.figure()
                    plt.subplot(1,2,1)
                if i % (separate*2) == 2:
                    plt.subplot(1,2,2)
                plt.title(title)
                plt.xlabel(xlabel)
                ax1 = plt.gca()
            elif type(separate) is str:
                if fig is None:
                    fig = plt.figure()
                if prop['N'] != old_prop:
                    id_plot += 1
                old_prop = prop['N']
                plt.subplot(1,prop_set['N'],id_plot)
                plt.title(title)
                plt.xlabel(xlabel)
                ax1 = plt.gca()

            ll_y = [row.split(sep)[column] for row in data]
            ll_y = np.ma.masked_invalid(np.array(ll_y, dtype='float'))

            Ks = [int(float(row.split(sep)[csv_row('K')])) for row in data]
            Ks = np.ma.masked_invalid(np.array(Ks, dtype='int'))

            print 'plotting in %s ' % (filen, )

            ylabel, label = tag_from_csv(column)
            ax1.set_ylabel(ylabel)

            #label = target_dir + ' ' + label
            label = target_dir.split('/')[-3] +' '+ target_dir.split('/')[-1][len('inference-'):]
            if 'ilda' in label:
                label += ' K -> %d' % (float(Ks[-1]))

            ax1.plot(ll_y, marker=next(markers), label=label)
            leg = ax1.legend(loc=1,prop={'size':10})

            if prop['model'] == 'ilda' or _Ks is None:
                _Ks = Ks
            Ks = _Ks
            ax2 = ax1.twinx()
            ax2.plot(Ks, marker='*')

        #plt.savefig('../results/debug1/%s.pdf' % (prop['corpus']))
    plt.draw()

def basic_plot():

    columns = 1
    targets = ['text/nips12/debug/inference-ilda_10_auto_100',
               'text/nips12/debug/inference-lda_cgs_1_auto_100',
               'text/nips12/debug/inference-lda_cgs_2_auto_100',
               'text/nips12/debug/inference-lda_cgs_5_auto_100',
               'text/nips12/debug/inference-lda_cgs_10_auto_100000000', ]
    plot_csv(targets, columns, separate=False)
    return

def make_path(spec, sep=None):
    targets = []
    base = 'text'
    hook = spec['hook_dir']
    if sep:
        tt = []
    for c in spec['corpus']:
        p = os.path.join(base, c, hook)
        for n in spec['Ns']:
            for m in spec['models']:
                for k in spec['Ks']:
                    for h in spec['hyper']:
                        t = 'inference-%s_%s_%s_%s' % (m, k, h, n)
                        t = os.path.join(p, t)
                        filen = os.path.join(os.path.dirname(__file__), "../data/", t)
                        if not os.path.isfile(filen) or os.stat(filen).st_size == 0:
                            continue

                        targets.append(t)

        if sep == 'corpus' and targets:
            tt.append(targets)
            targets = []

    if sep:
        return tt
    else:
        return targets

# Return dictionary of property for an expe file. (format inference-model_K_hyper_N)
def get_expe_file_prop(target):
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

    _id = _id[st:]
    prop = dict(
        corpus = target.split('/')[-3],
        model = model.split('-')[-1],
        K     = _id[0],
        hyper = _id[1],
        N     = _id[2],)
    return prop

# Return size of proportie in a list if expe files
def get_expe_file_set_prop(targets):
    c = []
    for t in targets:
        c.append(get_expe_file_prop(t))

    sets = {}
    for p in ('N', 'K'):
        sets[p] = len(set([ _p[p] for _p in c ]))

    return sets


if __name__ ==  '__main__':
    block = True
    conf = argParse()

    spec = dict(
        hook_dir = 'debug1/',
        corpus   = ['kos', 'nips12', 'nips', 'reuter50', '20ngroups'],
        columns  = ['perplexity'],
        models   = ['ilda', 'lda_cgs', ],
        #models   = ['lda_cgs', ],
        Ns       = [1000, 2000, 10000],
        Ks       = [1, 5, 10, 15, 20],
        hyper    = ['fix'],
    )

    sep = 'corpus'
    separate = 'N'
    targets = make_path(spec, sep=sep)
    if sep:
        for t in targets:
            plot_csv(t, spec['columns'], separate=separate)
    else:
        plot_csv(targets, spec['columns'], separate=True)

    ### Basic Plots
    #basic_plot()

    display(block)

