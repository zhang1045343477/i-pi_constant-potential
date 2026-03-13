#!/bin/bash
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -N 1
#SBATCH -n 1

ulimit -s unlimited

export PYTHONPATH=""
source path_to_ipi/env.sh

i-pi csvr.xml > log.ipi 2>&1 &
