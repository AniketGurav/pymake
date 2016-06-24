#############################################################################
### CORPUSES
#############################################################################
"""
=================
=== Networks
================= """

### Bursty
CORPUS_BURST_1     = ( 'generator3', 'generator11', 'generator12', 'generator7', 'generator14',)

### Non Bursty
CORPUS_NBURST_1    = ( 'generator4', 'generator5', 'generator6', 'generator9', 'generator10',)

CORPUS_SYN_ICDM_1  = ( 'generator4', 'generator10', 'generator12', 'generator7')
CORPUS_REAL_ICDM_1 = ( 'manufacturing', 'fb_uc',)

"""
=================
=== Text
================= """

CORPUS_TEXT_ALL = ['kos', 'nips12', 'nips', 'reuter50', '20ngroups'],


#############################################################################
### Experimentation / Specification
#############################################################################

EXPE_ICDM = dict((
    ('data_type', ('networks',)),
    ('debug'  , ('debug10',)),
    #('corpus' , ('fb_uc', 'manufacturing')),
    ('corpus' , ('Graph7', 'Graph12', 'Graph10', 'Graph4')),
    ('model'  , ('immsb', 'ibp')),
    ('K'      , (10,)),
    ('N'      , ('all',)),
    ('hyper'  , ('fix', 'auto')),
    ('homo'   , (0, 1)),
    #('repeat'   , (0, 1, 2, 4, 5)),
))

MODEL_FOR_CLUSTER_IBP = dict ((
    ('data_type'    , 'networks'),
    ('debug'        , 'debug11') ,
    ('model'        , 'ibp')   ,
    ('K'            , 20)        ,
    ('N'            , 'all')     ,
    ('hyper'        , 'fix')     ,
    ('homo'         , 0)         ,
    #('repeat'      , '*')       ,
))

SPEC_TO_PARSE = dict(
    data_type = ['networks'],
    refdir = ['debug5'],
    corpus   = [ 'generator6'],
    model   = ['ibp_cgs', 'mmsb_cgs'],
    N       = [1000,],
    K       = [5, 10, 30],
    homo     = [0,1,2],
    hyper    = ['fix', 'auto'],
)


NETWORKS_DD        = ('generator10', )
MODELS_DD = [ dict ((
('data_type'    , 'networks'),
('debug'        , 'debug10') ,
('model'        , 'ibp')   ,
('K'            , 10)        ,
('N'            , 'all')     ,
('hyper'        , 'auto')     ,
('homo'         , 0)         ,
#('repeat'      , '*')       ,
))]











