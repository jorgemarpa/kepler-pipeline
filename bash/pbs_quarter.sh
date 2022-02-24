#PBS -S /bin/sh
#PBS -N make-LCs
#PBS -q long
#PBS -l select=1:ncpus=12:mem=62G:model=ivy
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -m e
#PBS -V
#PBS -o pfe21:/home4/jimartin/ADAP/kepler-workflow/logs/

# activate conda env
source /nasa/jupyter/4.4/miniconda/etc/profile.d/conda.sh
conda activate kepler-workflow

# project directory
WORKDIR=$(dirname `pwd`)

# get batch info from quarter file
info="${WORKDIR}/data/support/kepler_batch_info_quarter${quarter}.dat"
totallines=`cat "$info" | wc -l | sed 's/^ *//g'`

# change directory
cd "$WORKDIR/kepler_workflow"
echo `pwd`

echo "Quarter $quarter"
echo "Total batches in quarter $totallines"

# lunch parallel jobs
echo "Will run the following command:"
echo "seq 1 ${totallines} | xargs -n 1 -I {} -P 12 python make_lightcurves_new.py --quarter ${quarter} --batch-index {} --tar-tpfs --tar-lcs --fit-va --augment-bkg --save-npy --log 20"
seq 1 ${totallines} | xargs -n 1 -I {} -P 12 python make_lightcurves_new.py --quarter ${quarter} --batch-index {} --tar-tpfs --tar-lcs --fit-va --augment-bkg --save-npy --log 20

exit 0
