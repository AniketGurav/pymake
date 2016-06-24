
RUN_1 = dict(
    data_type = ['networks'],
    refdir = ['debug0'],
    corpus   = [ 'generator6'],
    model   = ['ibp_cgs', 'mmsb_cgs'],
    N       = [1000,],
    K       = [5, 10, 30],
    homo     = [0,1,2],
    hyper    = ['fix', 'auto'],
    ### Global
    iterations = 200,
)


RUN_DD = dict((
    ('data_type', ('networks',)),
    #('corpus' , ('fb_uc', 'manufacturing')),
    ('corpus' , ('generator1')),
    ('model'  , ('immsb', 'ibp')),
    ('K'      , (5,)),
    ('N'      , ('all',)),
    ('hyper'  , ('auto')),
    ('homo'   , (0)),
    #('repeat'   , (0, 1, 2, 4, 5)),
))
