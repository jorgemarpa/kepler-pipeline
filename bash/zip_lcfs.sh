#!/bin/bash

prefix=$1
sufix=$2

files=`ls $prefix*$sufix*`

echo $files
echo $prefix $sufix

n=`ls -1 $prefix*$sufix* | wc -l | sed 's/^ *//g'`

echo $n

if [[ $n -lt 5 ]]
then
  jobs=$n
fi

echo "$jobs"

echo $files | xargs -n 1 -I {} -P $jobs tar xzvf {}

lcs=`ls -1 hlsp_kbonus*`
tarname="${prefix}_${sufix}.tar.gz"

tar cvf $tarname hlsp_kbonus*

rm $lcs
