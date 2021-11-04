#!/bin/bash

prefix=$1
sufix=$2
tarname="${prefix}_${sufix}.tar.gz"

rm $tarname

files=`ls $prefix*$sufix*`

echo $files
echo $prefix $sufix

n=`ls -1 $prefix*$sufix* | wc -l | sed 's/^ *//g'`

echo $n

if [[ $n -lt 5 ]]
then
  jobs=$n
else
  jobs=5
fi

echo "$jobs"
echo " Uncompressing files..."
ls -1 $prefix*$sufix* | xargs -n 1 -I {} -P $jobs tar xzf {}

lcs=`ls -1 hlsp_kbonus*`
echo "Compressing again..."
tar czvf $tarname hlsp_kbonus*

echo "Removing old files..."
rm $lcs
rm $files
