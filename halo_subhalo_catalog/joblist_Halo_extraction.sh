#!/bin/bash
#PBS -q SQUID
#PBS -N Halo_extraction
#PBS --group=hp240141
#PBS -l elapstim_req=1:00:00
#PBS -b 1
#PBS -T intmpi
#PBS -v OMP_NUM_THREADS=56

cd $PBS_O_WORKDIR
source ~/.bashrc
#source activate zhangzhao-env
conda activate zhangzhao-env
#module load BaseCPU
#module load BasePy
#emodule load mpi/latest


python3 $PBS_O_WORKDIR/FoF_halo_catalog_extraction.py > Halo_extraction.txt 2>&1 &
