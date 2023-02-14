#!/bin/bash

export dir=$1

cd "/nobackupp19/jimartin/ADAP/kbonus/lcs/kepler/"
# cd "/Users/jorgemarpa/Work/BAERI/ADAP/kepler-workflow/data/lcs/kepler"
echo `pwd`
tar xvf $dir.tar

cd "/home4/jimartin/ADAP/kepler-workflow/kepler_workflow"
# cd "/Users/jorgemarpa/Work/BAERI/ADAP/kepler-workflow/kepler_workflow"
echo `pwd`
python do_fits_stitch_nas.py --dir $dir

cd "/nobackupp19/jimartin/ADAP/kbonus/lcs/kepler/"
# cd "/Users/jorgemarpa/Work/BAERI/ADAP/kepler-workflow/data/lcs/kepler"
echo `pwd`

stitch_dir="${dir}_s"
tar cvf $stitch_dir.tar $stitch_dir/
rm -rv $stitch_dir/
rm -rv $dir/
