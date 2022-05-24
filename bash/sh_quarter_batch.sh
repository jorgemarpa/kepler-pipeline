#!/bin/bash

quarter=$1
WORKDIR=$(dirname `pwd`)

# get batch info from quarter file
info="${WORKDIR}/data/support/kepler_batch_info_quarter${quarter}.dat"
totallines=`cat "$info" | wc -l | sed 's/^ *//g'`
totallines=$(($totallines - 1))

echo "Quarter $quarter"
echo "Total batch index $totallines"

start=1
end=101

while [ $start -le $totallines ]
do
  echo "qsub -v 'quarter=$1,batch_start=$start,batch_end=$end' pbs_quarter_batch.sh"
  qsub -v "quarter=$1,batch_start=$start,batch_end=$end" pbs_quarter_batch.sh

  start=$(($start + 100))
  end=$(($end + 100))
  if [ "$end" -ge $totallines ]
  then
    end=$totallines
  fi
done

exit 0
