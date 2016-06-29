import inspect
from functools import wraps
import args as clargs
from util.frontend_io import *
from expe.spec import _spec_
from expe.run import *

class askhelp(object):

    def __init__(self, view_func, help=False):
        self.view_func = view_func
        self.help = help
        wraps(view_func)(self)

    def __call__(self, *args, **kwargs):
        try:
            usage = args[0]
        except:
            usage = self.view_func.__name__ + ' ?'
        # function name
        #(inspect.currentframe().f_code.co_name
        if clargs.flags.contains('---help') or clargs.flags.contains('-h') or self.help:
            print usage
            exit()

        response = self.view_func(*args, **kwargs)
        return response

class argparser(object):
    """ Utility class for parsing arguments of various script of @project.
        Each method has the same name of the function/scrpit for which it is used.
        @return dict of variables used by function/scritpts
    """

    @staticmethod
    @askhelp
    def zymake(USAGE=''):
        """ Generates output (files or line arguments) according to the SPEC
            @return OUT_TYPE: runcmd or path
                    SPEC: expe spec
                    FTYPE: filetype targeted
                    STATUS: status of file required  on the filesystem
        """
        # Default request
        req = dict(
            OUT_TYPE = 'path',
            SPEC = RUN_DD,
            FTYPE = 'pk',
            STATUS = None )

        ontologies = dict(
            out_type = ('runcmd', 'path'),
            spec = map(str.lower, vars(_spec_).keys()),
            ftype = ('json', 'pk', 'inf') )

        gargs = clargs.grouped['_'].all
        checksum = len(gargs)
        for v in gargs:
            v = v.lower()
            for ont, words in ontologies.items():
                if v in words:
                    if ont == 'spec':
                        v = getattr(_spec_, v.upper())
                    req[ont.upper()] = v
                    checksum -= 1
                    break

        if checksum != 0:
            raise ValueError('unknow argument: %s' % gargs)
        return req

    @staticmethod
    @askhelp
    def generate(USAGE=''):
        conf = {}
        write_keys = ('-w', 'write', '--write')
        # Write plot
        for k in write_keys:
            if k in clargs.all:
                conf['write_to_file'] = True
        # K setting
        if '-k' in clargs.grouped:
            conf['K'] = clargs.grouped['-k'].get(0)
        return conf


