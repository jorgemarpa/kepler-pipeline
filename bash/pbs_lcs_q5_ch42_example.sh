#PBS -S /bin/sh
#PBS -N make-LCs
#PBS -q normal
#PBS -l select=1:ncpus=10:mem=60G:model=ivy
#PBS -l walltime=02:30:00
#PBS -j oe
#PBS -m e
#PBS -o pfe21:/home4/jimartin/ADAP/kepler-workflow/logs/

# change directory
cd /home4/jimartin/ADAP/kepler-workflow/kepler_workflow
echo `pwd`

# activate conda env
source /nasa/jupyter/4.4/miniconda/etc/profile.d/conda.sh
conda activate kepler-workflow

# lunch parallel jobs
# seq 1 3 | xargs -n 1 -I {} -P 3 python make_lightcurves.py --quarter 0 --channel 41 --batch-size 212 --batch-number {} --tar-tpfs --tar-lcs --fit-va --log 20

seq 1 10 | xargs -n 1 -I {} -P 10 python make_lightcurves.py --quarter 1 --channel 42 --batch-size 201 --batch-number {} --tar-tpfs --tar-lcs --fit-va --log 20
seq 1 10 | xargs -n 1 -I {} -P 10 python make_lightcurves.py --quarter 2 --channel 43 --batch-size 214 --batch-number {} --tar-tpfs --tar-lcs --fit-va --log 20
seq 1 10 | xargs -n 1 -I {} -P 10 python make_lightcurves.py --quarter 3 --channel 44 --batch-size 212 --batch-number {} --tar-tpfs --tar-lcs --fit-va --log 20
seq 1 10 | xargs -n 1 -I {} -P 10 python make_lightcurves.py --quarter 4 --channel 41 --batch-size 217 --batch-number {} --tar-tpfs --tar-lcs --fit-va --log 20

seq 1 10 | xargs -n 1 -I {} -P 10 python make_lightcurves.py --quarter 5 --channel 42 --batch-size 215 --batch-number {} --tar-tpfs --tar-lcs --fit-va --log 20
seq 1 10 | xargs -n 1 -I {} -P 10 python make_lightcurves.py --quarter 6 --channel 43 --batch-size 236 --batch-number {} --tar-tpfs --tar-lcs --fit-va --log 20
seq 1 10 | xargs -n 1 -I {} -P 10 python make_lightcurves.py --quarter 7 --channel 44 --batch-size 231 --batch-number {} --tar-tpfs --tar-lcs --fit-va --log 20
seq 1 10 | xargs -n 1 -I {} -P 10 python make_lightcurves.py --quarter 8 --channel 41 --batch-size 229 --batch-number {} --tar-tpfs --tar-lcs --fit-va --log 20

seq 1 10 | xargs -n 1 -I {} -P 10 python make_lightcurves.py --quarter 9 --channel 42 --batch-size 228 --batch-number {} --tar-tpfs --tar-lcs --fit-va --log 20
seq 1 10 | xargs -n 1 -I {} -P 10 python make_lightcurves.py --quarter 10 --channel 43 --batch-size 234 --batch-number {} --tar-tpfs --tar-lcs --fit-va --log 20
seq 1 10 | xargs -n 1 -I {} -P 10 python make_lightcurves.py --quarter 11 --channel 44 --batch-size 228 --batch-number {} --tar-tpfs --tar-lcs --fit-va --log 20
seq 1 10 | xargs -n 1 -I {} -P 10 python make_lightcurves.py --quarter 12 --channel 41 --batch-size 224 --batch-number {} --tar-tpfs --tar-lcs --fit-va --log 20

seq 1 10 | xargs -n 1 -I {} -P 10 python make_lightcurves.py --quarter 13 --channel 42 --batch-size 223 --batch-number {} --tar-tpfs --tar-lcs --fit-va --log 20
seq 1 10 | xargs -n 1 -I {} -P 10 python make_lightcurves.py --quarter 14 --channel 43 --batch-size 229 --batch-number {} --tar-tpfs --tar-lcs --fit-va --log 20
seq 1 10 | xargs -n 1 -I {} -P 10 python make_lightcurves.py --quarter 15 --channel 44 --batch-size 225 --batch-number {} --tar-tpfs --tar-lcs --fit-va --log 20
seq 1 10 | xargs -n 1 -I {} -P 10 python make_lightcurves.py --quarter 16 --channel 41 --batch-size 224 --batch-number {} --tar-tpfs --tar-lcs --fit-va --log 20

seq 1 10 | xargs -n 1 -I {} -P 10 python make_lightcurves.py --quarter 17 --channel 42 --batch-size 223 --batch-number {} --tar-tpfs --tar-lcs --fit-va --log 20

exit 0
