#PBS -S /bin/sh
#PBS -N Compute Corr Metric
#PBS -q normal
#PBS -l select=1:ncpus=5:mem=60G:model=ivy
#PBS -l walltime=6:00:00
#PBS -j oe
#PBS -m e
#PBS -V
#PBS -o pfe:/home4/jimartin/ADAP/kepler-workflow/logs/

# change directory
cd /home4/jimartin/ADAP/kepler-workflow/kepler_workflow
echo `pwd`

# activate conda env
source /nasa/jupyter/4.4/miniconda/etc/profile.d/conda.sh
conda activate kepler-workflow

echo "Will compute correlation metric for quarters..."

# lunch parallel jobs
echo "seq 0 4 | xargs -n 1 -I {} -P 5 python compute_lc_corr.py --quarter {}"
seq 0 4 | xargs -n 1 -I {} -P 5 python compute_lc_corr.py --quarter {}

exit 0
