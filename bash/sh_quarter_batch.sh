#!/bin/bash

quarter=$1
WORKDIR=$(dirname `pwd`)

# get batch info from quarter file
info="${WORKDIR}/data/support/kepler_batch_info_quarter${quarter}.dat"
totallines=`cat "$info" | wc -l | sed 's/^ *//g'`

echo "Quarter $quarter"
echo "Total batch index $totallines"

start=1
end=201

while [ $start -le $totallines ]
do
  echo "qsub -v 'quarter=$1,batch_start=$start,batch_end=$end' pbs_quarter_batch.sh"
  qsub -v "quarter=$1,batch_start=$start,batch_end=$end" pbs_quarter_batch.sh

  start=$(($start + 200))
  end=$(($end + 200))
  if [ "$end" -ge $totallines ]
  then
    end=$totallines
  fi
done

exit 0
