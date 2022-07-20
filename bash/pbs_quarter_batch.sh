#PBS -S /bin/sh
#PBS -N make-LCs
#PBS -q normal
#PBS -l select=1:ncpus=10:mem=63G:model=ivy
#PBS -l walltime=6:00:00
#PBS -j oe
#PBS -m e
#PBS -V
#PBS -o pfe21:/home4/jimartin/ADAP/kepler-workflow/logs/

# activate conda env
source /nasa/jupyter/4.4/miniconda/etc/profile.d/conda.sh
conda activate kepler-workflow

echo `which python`

# project directory
WORKDIR=$(dirname `pwd`)

# get batch info from quarter file
# info="${WORKDIR}/data/support/kepler_batch_info_quarter${quarter}.dat"
# totallines=`cat "$info" | wc -l | sed 's/^ *//g'`

# change directory
cd "$WORKDIR/kepler_workflow"
echo `pwd`

echo "Quarter $quarter"
echo "Batches in quarter $batch_start to $batch_end"

# lunch parallel jobs
echo "Will run the following command:"
echo "seq $batch_start ${batch_end} | xargs -n 1 -I {} -P 10 python make_lightcurves_new.py --quarter ${quarter} --batch-index {} --tar-tpfs --tar-lcs --fit-va --use-cbv --augment-bkg --iter-neg --save-arrays feather --log 20"
seq $batch_start ${batch_end} | xargs -n 1 -I {} -P 10 python make_lightcurves_new.py --quarter ${quarter} --batch-index {} --tar-tpfs --tar-lcs --fit-va --use-cbv --augment-bkg --iter-neg --save-arrays feather --log 20

exit 0
