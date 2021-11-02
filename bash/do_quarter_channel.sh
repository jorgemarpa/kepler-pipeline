#!/bin/bash

quarter=$1
channel=$2

echo "Channel $channel Quarter $quarter"

quarter=$((quarter + 2))
channel=$((channel + 1))

WORKDIR=$(dirname `pwd`)

tpftot=`head -"$quarter" "$WORKDIR"/data/support/kepler_ntpf_qch.csv | tail -1 | cut -f "$channel" -d","`
bsize=`head -"$quarter" "$WORKDIR"/data/support/kepler_batchsize_qch.csv | tail -1 | cut -f "$channel" -d","`
bntot=`head -"$quarter" "$WORKDIR"/data/support/kepler_nbatches_qch.csv | tail -1 | cut -f "$channel" -d","`

echo "Total TPFs $tpftot"
echo "$bntot batches of size $bsize"

hh="00"
mm="30"

if [[ $bntot -gt 10 ]]
then
  hh="01"
  mm="00"
fi

hh="00"
mm="01"

echo qsub -J 0-${bntot} -lwalltime=${hh}:${mm}:00 ${WORKDIR}/kepler-workflow/do_quarter_channel.sh ${quarter} ${channel}
