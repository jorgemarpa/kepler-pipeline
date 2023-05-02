#!/bin/bash

export dirname=$1

cd "/Users/jorgemarpa/Work/BAERI/ADAP/kepler-workflow/data/lcs/rename"
echo `pwd`

tar xzvf "$dirname.tar.gz"

mv -v "$dirname"_s "$dirname"

echo "Working on directory"
python /Users/jorgemarpa/Work/BAERI/ADAP/kepler-workflow/kepler_workflow/fix_hlsp_lcf_header.py --dirname $dirname
exit_status=$?

if [ "${exit_status}" -ne 0 ];
then
    echo "Got Failure Exit Code: $exit_status"
    exit 1
fi

echo "Tarballing"
tar czvf tmp/$dirname.tar.gz $dirname/
exit_status=$?

if [ "${exit_status}" -ne 0 ];
then
    echo "Got Failure Exit Code: $exit_status"
    exit 1
fi

rm -rv $dirname/
rm "$dirname.tar.gz"
