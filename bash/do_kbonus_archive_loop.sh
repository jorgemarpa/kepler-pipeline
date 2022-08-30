#!/bin/bash

quarter=$1
kwpath="/home4/jimartin/ADAP/kepler-workflow"

echo "Creating tarball archive..."
start=1
end=4
for (( i = 1; i < 22; i++ )); do
  echo $start $end

  seq $start $end | xargs -n 1 -P 4 -I {} python $kwpath/kepler_workflow/kbonus_lcf_archive.py --quarter $quarter --channel {}
  cat $kwpath/data/support/kbonus_archive_id4.dat | xargs -n 1 -P 4 -I {} python $kwpath/kepler_workflow/kbonus_lcf_archive.py --dir {} --delete
  cat $kwpath/data/support/kbonus_archive_id4.dat | xargs -P 4 -l bash -c 'rm -r /nobackupp19/jimartin/ADAP/kbonus/lcs/kepler/$0'
  start=$(($start + 4))
  end=$(($end + 4))
done

echo "Creating source catalog"
cat $kwpath/data/support/kbonus_archive_id4.dat | xargs -n 1 -P 4 -I {} python $kwpath/kepler_workflow/do_catalog.py --quarter $quarter --dir {}

python $kwpath/kepler_workflow/do_catalog.py --quarter 5 --concat
