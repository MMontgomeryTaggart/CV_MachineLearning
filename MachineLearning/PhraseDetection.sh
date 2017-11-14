#!/bin/bash

#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH -o slurmjob-%j.out
#SBATCH --ntasks=1
#SBATCH --account=owner-guest
#SBATCH --partition=kingspeak-guest
#SBATCH --mem=0


pyvenv --system-site-packages ~/VENV3.5.2
source VENV3.5.2/bin/activate

pip install n

OUTFILE=$HOME/test.out
module load python/3.5.2

source /uufs/chpc.utah.edu/common/home/u0585767/VENV3.5.2/bin/activate
python runStuff.py > $OUTFILE