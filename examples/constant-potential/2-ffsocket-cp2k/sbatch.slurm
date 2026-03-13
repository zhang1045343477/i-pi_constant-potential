#!/bin/bash
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -N 1
#SBATCH -n 100


source path_to_ipi/env.sh
 
i-pi constant_potential.xml > ipi.log 2>&1
#There is no need to start cp2k, because the path of the executable file has been written, and i-pi will start automatically.