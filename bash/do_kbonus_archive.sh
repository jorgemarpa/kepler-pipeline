#!/bin/bash

quarter=$1
kwpath="/home4/jimartin/ADAP/kepler-workflow/kepler_workflow"

echo "Creating tarball archive..."
seq 1 10 | xargs -n 1 -P 5 -I {} python $kwpath/kbonus_lcf_archive.py --quarter $quarter --channel {}
cat ../data/support/kbonus_archive_id4.dat | xargs -n 1 -P 5 -I {} python $kwpath/kbonus_lcf_archive.py --dir {} --delete

seq 11 20 | xargs -n 1 -P 5 -I {} python $kwpath/kbonus_lcf_archive.py --quarter $quarter --channel {}
cat ../data/support/kbonus_archive_id4.dat | xargs -n 1 -P 5 -I {} python $kwpath/kbonus_lcf_archive.py --dir {} --delete

seq 21 30 | xargs -n 1 -P 5 -I {} python $kwpath/kbonus_lcf_archive.py --quarter $quarter --channel {}
cat ../data/support/kbonus_archive_id4.dat | xargs -n 1 -P 5 -I {} python $kwpath/kbonus_lcf_archive.py --dir {} --delete

seq 31 40 | xargs -n 1 -P 5 -I {} python $kwpath/kbonus_lcf_archive.py --quarter $quarter --channel {}
cat ../data/support/kbonus_archive_id4.dat | xargs -n 1 -P 5 -I {} python $kwpath/kbonus_lcf_archive.py --dir {} --delete

seq 41 50 | xargs -n 1 -P 5 -I {} python $kwpath/kbonus_lcf_archive.py --quarter $quarter --channel {}
cat ../data/support/kbonus_archive_id4.dat | xargs -n 1 -P 5 -I {} python $kwpath/kbonus_lcf_archive.py --dir {} --delete

seq 51 60 | xargs -n 1 -P 5 -I {} python $kwpath/kbonus_lcf_archive.py --quarter $quarter --channel {}
cat ../data/support/kbonus_archive_id4.dat | xargs -n 1 -P 5 -I {} python $kwpath/kbonus_lcf_archive.py --dir {} --delete

seq 61 70 | xargs -n 1 -P 5 -I {} python $kwpath/kbonus_lcf_archive.py --quarter $quarter --channel {}
cat ../data/support/kbonus_archive_id4.dat | xargs -n 1 -P 5 -I {} python $kwpath/kbonus_lcf_archive.py --dir {} --delete

seq 71 84 | xargs -n 1 -P 5 -I {} python $kwpath/kbonus_lcf_archive.py --quarter $quarter --channel {}
cat ../data/support/kbonus_archive_id4.dat | xargs -n 1 -P 5 -I {} python $kwpath/kbonus_lcf_archive.py --dir {} --delete

echo "Creating source catalog"
cat ../data/support/kbonus_archive_id4.dat | xargs -n 1 -P 5 -I {} python $kwpath/do_catalog.py --quarter $quarter --dir {}

python $kwpath/do_catalog.py --quarter 5 --concat
