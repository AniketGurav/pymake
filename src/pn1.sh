#!/bin/bash

#############
### GNU Parallel parameters
JOBS="20"

COMMAND="python ./topics.py -w -i 1000 --refdir kkk"

#############
### parameters
#CORPUS="clique7 generator1" 
CORPUS="generator1 generator2 clique3"
MODELS="ibp ibp_cgs"
Ks="10 20 30"
ALPHAS="auto"
Ns="100"
homo="0 1"
RUNS=""

for corpus in $CORPUS; do
    for N in $Ns; do
        for K in $Ks; do
            for hom in $homo; do
                for alpha in $ALPHAS; do
                    for model in $MODELS; do
                        RUNS="${RUNS} -m $model --homo $hom -k $K --alpha $alpha -c $corpus -n $N\n"
                    done
                done
            done
        done
    done
done
# Remove last breakline
RUNS=${RUNS::-2}

###--- Gnu Parallel ---###
#parallel --no-notice -k -j$JOBS  $RUN ::: {1..4}
echo -e "$RUNS" | parallel --eta -k -j$JOBS --colsep ' ' "$COMMAND {}"
