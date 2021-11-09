#!/bin/bash

quarter=$1

# project directory
WORKDIR=$(dirname `pwd`)

info="${WORKDIR}/data/support/kepler_batch_info_quarter${quarter}.dat"
totallines=`cat "$info" | wc -l | sed 's/^ *//g'`

cd "$WORKDIR/kepler_workflow"
echo `pwd`

echo "Quarter $quarter"
echo "Total batches in quarter $totallines"

echo "Will run the following command:"
echo "seq 1 ${totallines} | xargs -n 1 -I {} -P 10 python make_lightcurves.py --quarter ${quarter} --batch-index {} --tar-tpfs --tar-lcs --fit-va --log 20 --dry-run"
seq 1 40 | xargs -n 1 -I {} -P 1 python make_lightcurves.py --quarter ${quarter} --batch-index {} --tar-tpfs --tar-lcs --fit-va --log 20 --dry-run

exit 0
