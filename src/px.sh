#!/bin/bash

#############
### GNU Parallel parameters
JOBS="2"

COMMAND="python ./topics.py -w -i 50 --refdir debug_local"

#############
### parameters
#CORPUS="nips12 kos nips reuter50 20ngroups"  
CORPUS="kos"  
MODELS="lda_vb lda_cgs"
Ks="10"
ALPHAS="10 1000 10000"
Ns="all"
RUNS=""

for corpus in $CORPUS; do
    for N in $Ns; do
        for K in $Ks; do
            for alpha in $ALPHAS; do
                for model in $MODELS; do
                    RUNS="${RUNS} -m $model -k $K --alpha $alpha -c $corpus -n $N\n"
                done
            done
        done
    done
done
# Remove last breakline
RUNS=${RUNS::-2}

###--- Gnu Parallel ---###
#parallel --no-notice -k -j$JOBS  $RUN ::: {1..4}
echo -e  "$RUNS" | parallel --progress -k -j$JOBS --colsep ' ' "$COMMAND {}"
