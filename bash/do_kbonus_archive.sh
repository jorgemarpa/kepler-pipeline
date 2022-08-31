#!/bin/bash

quarter=$1
kwpath="/home4/jimartin/ADAP/kepler-workflow"

echo "Creating tarball archive..."
seq 1 10 | xargs -n 1 -P 5 -I {} python $kwpath/kepler_workflow/kbonus_lcf_archive.py --quarter $quarter --channel {}
cat $kwpath/data/support/kbonus_archive_id4.dat | xargs -n 1 -P 5 -I {} python $kwpath/kepler_workflow/kbonus_lcf_archive.py --dir {} --delete
cat $kwpath/data/support/kbonus_archive_id4.dat | xargs -P 5 -l bash -c 'rm -r /nobackupp19/jimartin/ADAP/kbonus/lcs/kepler/$0'

seq 11 20 | xargs -n 1 -P 5 -I {} python $kwpath/kepler_workflow/kbonus_lcf_archive.py --quarter $quarter --channel {}
cat $kwpath/data/support/kbonus_archive_id4.dat | xargs -n 1 -P 5 -I {} python $kwpath/kepler_workflow/kbonus_lcf_archive.py --dir {} --delete
cat $kwpath/data/support/kbonus_archive_id4.dat | xargs -P 5 -l bash -c 'rm -r /nobackupp19/jimartin/ADAP/kbonus/lcs/kepler/$0'

seq 21 30 | xargs -n 1 -P 5 -I {} python $kwpath/kepler_workflow/kbonus_lcf_archive.py --quarter $quarter --channel {}
cat $kwpath/data/support/kbonus_archive_id4.dat | xargs -n 1 -P 5 -I {} python $kwpath/kepler_workflow/kbonus_lcf_archive.py --dir {} --delete
cat $kwpath/data/support/kbonus_archive_id4.dat | xargs -P 5 -l bash -c 'rm -r /nobackupp19/jimartin/ADAP/kbonus/lcs/kepler/$0'

seq 31 40 | xargs -n 1 -P 5 -I {} python $kwpath/kepler_workflow/kbonus_lcf_archive.py --quarter $quarter --channel {}
cat $kwpath/data/support/kbonus_archive_id4.dat | xargs -n 1 -P 5 -I {} python $kwpath/kepler_workflow/kbonus_lcf_archive.py --dir {} --delete
cat $kwpath/data/support/kbonus_archive_id4.dat | xargs -P 5 -l bash -c 'rm -r /nobackupp19/jimartin/ADAP/kbonus/lcs/kepler/$0'

seq 41 50 | xargs -n 1 -P 5 -I {} python $kwpath/kepler_workflow/kbonus_lcf_archive.py --quarter $quarter --channel {}
cat $kwpath/data/support/kbonus_archive_id4.dat | xargs -n 1 -P 5 -I {} python $kwpath/kepler_workflow/kbonus_lcf_archive.py --dir {} --delete
cat $kwpath/data/support/kbonus_archive_id4.dat | xargs -P 5 -l bash -c 'rm -r /nobackupp19/jimartin/ADAP/kbonus/lcs/kepler/$0'

seq 51 60 | xargs -n 1 -P 5 -I {} python $kwpath/kepler_workflow/kbonus_lcf_archive.py --quarter $quarter --channel {}
cat $kwpath/data/support/kbonus_archive_id4.dat | xargs -n 1 -P 5 -I {} python $kwpath/kepler_workflow/kbonus_lcf_archive.py --dir {} --delete
cat $kwpath/data/support/kbonus_archive_id4.dat | xargs -P 5 -l bash -c 'rm -r /nobackupp19/jimartin/ADAP/kbonus/lcs/kepler/$0'

seq 61 70 | xargs -n 1 -P 5 -I {} python $kwpath/kepler_workflow/kbonus_lcf_archive.py --quarter $quarter --channel {}
cat $kwpath/data/support/kbonus_archive_id4.dat | xargs -n 1 -P 5 -I {} python $kwpath/kepler_workflow/kbonus_lcf_archive.py --dir {} --delete
cat $kwpath/data/support/kbonus_archive_id4.dat | xargs -P 5 -l bash -c 'rm -r /nobackupp19/jimartin/ADAP/kbonus/lcs/kepler/$0'

seq 71 84 | xargs -n 1 -P 5 -I {} python $kwpath/kepler_workflow/kbonus_lcf_archive.py --quarter $quarter --channel {}
cat $kwpath/data/support/kbonus_archive_id4.dat | xargs -n 1 -P 5 -I {} python $kwpath/kepler_workflow/kbonus_lcf_archive.py --dir {} --delete
cat $kwpath/data/support/kbonus_archive_id4.dat | xargs -P 5 -l bash -c 'rm -r /nobackupp19/jimartin/ADAP/kbonus/lcs/kepler/$0'

echo "Creating source catalog"
cat $kwpath/data/support/kbonus_archive_id4.dat | xargs -n 1 -P 5 -I {} python $kwpath/kepler_workflow/do_catalog.py --quarter $quarter --dir {} --tar

python $kwpath/kepler_workflow/do_catalog.py --quarter $quarter --concat
