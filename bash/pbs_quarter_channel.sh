#PBS -S /bin/sh
#PBS -N make-LCs
#PBS -q normal
#PBS -l select=1:ncpus=5:mem=60G:model=ivy
#PBS -l walltime=07:00:00
#PBS -j oe
#PBS -m e
#PBS -V
#PBS -o pfe21:/home4/jimartin/ADAP/kepler-workflow/logs/

# activate conda env
source /nasa/jupyter/4.4/miniconda/etc/profile.d/conda.sh
conda activate kepler-workflow

# project directory
WORKDIR=$(dirname `pwd`)

# change directory
cd "$WORKDIR/kepler_workflow"
echo `pwd`



echo "Channel $ch Quarter $qu"
echo "$bn batches"

# lunch parallel jobs
echo "Will run the following command:"
echo "seq 1 ${bn} | xargs -n 1 -I {} -P 5 python make_lightcurves.py --quarter ${qu} --channel ${ch} --batch-number {} --tar-tpfs --tar-lcs --fit-va --use-cbv --augment-bkg --iter-neg --save-arrays feather --log 20"
seq 1 ${bn} | xargs -n 1 -I {} -P 5 python make_lightcurves.py --quarter ${qu} --channel ${ch} --batch-number {} --tar-tpfs --tar-lcs --fit-va --use-cbv --augment-bkg --iter-neg --save-arrays feather --log 20

exit 0
