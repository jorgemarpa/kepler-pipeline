#!/bin/bash

export quarter=$1

cd "/nobackupp19/jimartin/ADAP/kbonus/lcs/kepler/"
mkdir q"$quarter"

cat dirnames.txt | xargs -n 1 -i -P 4 sh -c ' tar -xv --directory /nobackupp19/jimartin/ADAP/kbonus/lcs/kepler/q$quarter/ -f {}.tar --wildcards "*-q$quarter*" &&
cd q$quarter &&
tar cvf {}.tar {} &&
rm -r {} &&
cd ../'

# ls -1 00*.tar | tail -45 | xargs -n 1 -I {} -P 4 tar -xv --directory /nobackupp19/jimartin/ADAP/kbonus/lcs/kepler/$quarter/ -f {} --wildcards "*-$quarter*"
# cd "/nobackupp19/jimartin/ADAP/kbonus/lcs/kepler/$quarter"
# ls -1 | xargs -i -P 4 -n 1 sh -c 'tar cvf {}.tar {} && rm -r {}'
# cd "/nobackupp19/jimartin/ADAP/kbonus/lcs/kepler"

cd "/nobackupp19/jimartin/ADAP/kbonus/lcs/kepler/"
tar cvf kbonus_kepler_lcf_q$quarter.tar q$quarter/
rm q"$quarter"/*.tar
