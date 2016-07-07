import inspect
from functools import wraps
import args as clargs
from util.frontend_io import *
from expe.spec import _spec_
from expe.run import *

#########
# @TODO:
#   * wraps cannot handle the decorator chain :(, why ?

class askhelp(object):

    def __init__(self, func, help=False):
        self.func = func
        self.help = help
        #wraps(func)(self)
        #functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        try:
            usage = args[0]
        except:
            usage = self.func.__name__ + ' ?'
        # function name
        #(inspect.currentframe().f_code.co_name
        if clargs.flags.contains('--help') or clargs.flags.contains('-h') or self.help:
            print(usage)
            exit()

        response = self.func(*args, **kwargs)
        return response

class askverbose(object):

    def __init__(self, func):
        self.func = func
        #wraps(func)(self)
        #functools.update_wrapper(self, func)
        pass

    def __call__(self, *args, **kwargs):

        if clargs.flags.contains('-v'):
            self.logger = self.setup_logger('root','%(message)s', logging.DEBUG)
        else:
            self.logger = self.setup_logger('root','%(message)s', logging.INFO)

        response = self.func(*args, **kwargs)
        return response

    def setup_logger(self, name, fmt, level):
        #formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

        # Get logger
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Format handler
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt=fmt))
        logger.addHandler(handler)

        return logger


class argparser(object):
    """ Utility class for parsing arguments of various script of @project.
        Each method has the same name of the function/scrpit for which it is used.
        @return dict of variables used by function/scritpts
    """

    @staticmethod
    @askhelp
    @askverbose
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

        ### Making ontologie based argument attribution
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


        if '-status' in clargs.grouped:
            req['STATUS'] = clargs.grouped['-status'].get(0)

        if checksum != 0:
            raise ValueError('unknow argument: %s' % gargs)
        return req

    @staticmethod
    @askhelp
    @askverbose
    def generate(USAGE=''):
        conf = {}
        write_keys = ('-w', 'write', '--write')
        # Write plot
        for write_key in write_keys:
            if write_key in clargs.all:
                conf['write_to_file'] = True
        # K setting
        if '-k' in clargs.grouped:
            conf['K'] = clargs.grouped['-k'].get(0)
        return conf


    @staticmethod
    @askverbose
    @askhelp
    def expe_tabulate(USAGE=''):
        conf = dict( model = None,
                    K = None)

        gargs = clargs.grouped['_'].all
        for arg in gargs:
            try:
                conf['K'] = int(arg)
            except:
                conf['model'] = arg
        return conf


