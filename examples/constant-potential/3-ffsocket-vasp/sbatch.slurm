#!/bin/bash
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -N 1
#SBATCH -n 128


ulimit -s unlimited

export PYTHONPATH=""
source path_to_ipi/env.sh

i-pi constant_potential.xml > log.ipi 2>&1 &

sleep 5

mpirun -np $SLURM_NTASKS path_to_vasp/vasp_gam > log.vasp 2>&1

wait

