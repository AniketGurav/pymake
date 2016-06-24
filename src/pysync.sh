#!/bin/bash

REMOTE_LOC='/home/ama/adulac/workInProgress/networkofgraphs/process/PyNPB/data/'

LOCAL_LOC='../data/'


SPEC='EXPE_ICDM'
FTYPE='pk'

./zymake.py path $SPEC $FTYPE > pysync.out

#DR="--dry-run"
OPTS_RSYNC='-av -u --modify-window=1 --stats --prune-empty-dirs -e ssh '
/usr/bin/rsync $DR \
    $OPTS_RSYNC \
    --include='*/' \
    --include-from=pysync.out \
    --exclude '*' \
    adulac@racer:"${REMOTE_LOC}" \
    "${LOCAL_LOC}"

rm -f pysync.out

