#!/bin/bash

#############
### GNU Parallel parameters
JOBS="12"

COMMAND="python ./topics.py -w -i 1000 --refdir debug3"

#############
### parameters
#CORPUS="clique7 generator1" 
CORPUS="generator1 generator2 clique3" 
MODELS="immsb mmsb_cgs"
Ks="10 20 30"
ALPHAS="auto fix"
Ns="100"
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
echo -e  "$RUNS" | parallel --eta -k -j$JOBS --colsep ' ' "$COMMAND {}"
