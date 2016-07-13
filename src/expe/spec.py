
class _spec_(object):
    #############################################################################
    ### Corpuses
    #############################################################################
    """
    =================
    === Networks
    ================= """

    ### Bursty
    CORPUS_BURST_1     = ( 'generator3', 'generator11', 'generator12', 'generator7', 'generator14',)

    ### Non Bursty
    CORPUS_NBURST_1    = ( 'generator4', 'generator5', 'generator6', 'generator9', 'generator10',)

    ### Expe ICDM
    CORPUS_SYN_ICDM_1  = ( 'generator4', 'generator10', 'generator12', 'generator7')
    CORPUS_REAL_ICDM_1 = ( 'manufacturing', 'fb_uc',)

    CORPUS_ALL_3 = CORPUS_SYN_ICDM_1 + CORPUS_REAL_ICDM_1

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
        ('debug'  , ('debug10', 'debug11')),
        #('corpus' , ('fb_uc', 'manufacturing')),
        ('corpus' , CORPUS_ALL_3),
        ('model'  , ('immsb', 'ibp')),
        ('K'      , (5,10,15, 20)),
        ('N'      , ('all',)),
        ('hyper'  , ('fix', 'auto')),
        ('homo'   , (0,)),
        #('repeat'   , (0, 1, 2, 4, 5)),
    ))
    ICDM = EXPE_ICDM
    EXPE_DD = ICDM

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
        data_type     = ['networks'],
        refdir        = ['debug5'],
        corpus        = ['generator6'],
        model         = ['ibp_cgs', 'mmsb_cgs'],
        N             = [1000,],
        K             = [5, 10, 30],
        homo          = [0,1,2],
        hyper         = ['fix', 'auto'],
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

    MODELS_GENERATE_IBP = [dict ((
        ('data_type'    , 'networks'),
        ('debug'        , 'debug10') ,
        ('model'        , 'ibp')   ,
        ('K'            , 10)        ,
        ('N'            , 'all')     ,
        ('hyper'        , 'fix')     ,
        ('homo'         , 0)         ,
        #('repeat'      , '*')       ,
    ))]
    MODELS_GENERATE_IMMSB = [dict ((
        ('data_type'    , 'networks'),
        ('debug'        , 'debug10') ,
        ('model'        , 'immsb')   ,
        ('K'            , 10)        ,
        ('N'            , 'all')     ,
        ('hyper'        , 'auto')     ,
        ('homo'         , 0)         ,
        #('repeat'      , '*')       ,
    ))]
    MODELS_GENERATE = MODELS_GENERATE_IMMSB +  MODELS_GENERATE_IBP


#### Temp

    EXPE_ALL_3_IBP = dict((
        ('data_type', ('networks',)),
        ('debug'  , ('debug111111', 'debug101010')),
        ('corpus' , CORPUS_ALL_3),
        ('model'  , ('ibp')),
        ('K'      , (5, 10, 15, 20)),
        ('N'      , ('all',)),
        ('hyper'  , ('fix',)),
        ('homo'   , (0,)),
        ('repeat'   , (6, 7, 8, 9)),
    ))
    EXPE_ALL_3_IMMSB = dict((
        ('data_type', ('networks',)),
        ('debug'  , ('debug111111', 'debug101010')),
        ('corpus' , CORPUS_ALL_3),
        ('model'  , ('immsb')),
        ('K'      , (5, 10, 15, 20)),
        ('N'      , ('all',)),
        ('hyper'  , ('auto',)),
        ('homo'   , (0,)),
        ('repeat'   , (6, 7, 8, 9)),
    ))



    RUN_DD = dict((
        ('data_type', ('networks',)),
        #('corpus' , ('fb_uc', 'manufacturing')),
        ('corpus' , ('generator1')),
        ('model'  , ('immsb', 'ibp')),
        ('K'      , (5,)),
        ('N'      , ('all',)),
        ('hyper'  , ('auto')),
        ('homo'   , (0)),
        ('hyper_prior', ('1 2 3 4', '10 2')),
        ('repeat'   , (0, 1, 2, 4, 5)),
    ))
