#!/bin/bash

TESTING="topics.py -i 10 -m ilda -k 10 -n 100"
TESTING="topics.py -m immsb -c generator1 -n 100 -i 10"
TESTING="./topics.py -m immsb -c clique10 -n 1000 -i 2"
TESTING="./topics.py -m ibp --homo 0 -c clique10 -k 5  -n 500 -i 5"

TESTING="./assortt.py -n 1000 -k 10 --alpha auto --homo 0 -m ibp_cgs -c generator3 -l model --refdir debug5"

python -m cProfile -o profile.out $TESTING
./ptime.py profile.out > t.out

#python -OO -m cProfile -s cumulative -o profile_data.pyprof $TESTING
#pyprof2calltree -i profile_data.pyprof -k
