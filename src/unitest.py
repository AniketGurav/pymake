#!/usr/bin/env python
import subprocess


tests = ( 'topics',
         'expe_icdm',
         'expe_k',
         'check_networks',
        )

for t in tests:

    cmd = 'python ' + t + '.py'

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    result = out.split('\n')

    ### Output
    if False:
        for lin in result:
            if not lin.startswith('#'):
                print(lin)

    ### Error
#    print '### exec: %s' % (t)
#    if err:
#        print err
#    else:
#        print '...ok'
