python -m cProfile -o profile.out topics.py -i 2 -m lda_cgs
./ptime.py profile.out > t.out
