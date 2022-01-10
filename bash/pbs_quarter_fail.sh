#PBS -S /bin/sh
#PBS -N make-LCs
#PBS -q normal
#PBS -l select=1:ncpus=15:mem=60G:model=ivy
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -m e
#PBS -V
#PBS -o pfe21:/home4/jimartin/ADAP/kepler-workflow/logs/

# activate conda env
source /nasa/jupyter/4.4/miniconda/etc/profile.d/conda.sh
conda activate kepler-workflow

# project directory
WORKDIR=$(dirname `pwd`)
quarter=5

# get batch info from quarter file
info="${WORKDIR}/data/support/fail_batch_index_quarter${quarter}.dat"
totallines=`cat "$info" | wc -l | sed 's/^ *//g'`

# change directory
cd "$WORKDIR/kepler_workflow"
echo `pwd`

echo "Quarter $quarter"
echo "Total batches in quarter $totallines"

# lunch parallel jobs
echo "Will run the following command:"
echo "cat ${info} | xargs -n 1 -I {} -P 15 python make_lightcurves.py --quarter ${quarter} --batch-index {} --tar-tpfs --tar-lcs --fit-va --augment-bkg --log 20"
cat ${info} | xargs -n 1 -I {} -P 15 python make_lightcurves.py --quarter ${quarter} --batch-index {} --tar-tpfs --tar-lcs --fit-va --augment-bkg --log 20

exit 0
