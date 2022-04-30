#!/bin/bash

quarter=$1
WORKDIR=$(dirname `pwd`)

# get batch info from quarter file
info="${WORKDIR}/data/support/kepler_batch_info_quarter${quarter}.dat"
totallines=`cat "$info" | wc -l | sed 's/^ *//g'`

echo "Total batch index $totallines"

start=1
finish=201

while [ $start -le $totallines ]
do
  echo "from $start to $finish"
  echo "qsub -v 'quarter=$1,batch_start=$start,batch_finish=$finish' pbs_quarter_batch.sh"
  qsub -v "quarter=$1,batch_start=$start,batch_finish=$finish" pbs_quarter_batch.sh

  start=$(($start + 200))
  finish=$(($finish + 200))
  if [ "$finish" -ge $totallines ]
  then
    finish=$totallines
  fi
done

exit 0
