#PBS -S /bin/sh
#PBS -N make-LCs
#PBS -q devel
#PBS -l select=1:ncpus=10:mem=60G:model=ivy
#PBS -j oe
#PBS -m e
#PBS -V
#PBS -o pfe21:/home4/jimartin/ADAP/kepler-workflow/logs/

# change directory
cd /home4/jimartin/ADAP/kepler-workflow/kepler_workflow

# activate conda env
source /nasa/jupyter/4.4/miniconda/etc/profile.d/conda.sh
conda activate kepler-workflow

echo "PBS Job Id PBS_JOBID is ${PBS_JOBID}"
echo "PBS job array index PBS_ARRAY_INDEX value is ${PBS_ARRAY_INDEX}"

#
#  To isolate the job id number, cut on the character "[" instead of
#  ".".  PBS_JOBID might look like "48274[].server" rather "48274.server"
#  in job arrays
#

JOBID=`echo ${PBS_JOBID} | cut -d'[' -f1`

# lunch parallel jobs
python make_lightcurves.py --quarter ${1} --channel ${2} --batch-size 201 --batch-number ${PBS_ARRAY_INDEX} --tar-tpfs --tar-lcs --fit-va --log 20 --dry-run
