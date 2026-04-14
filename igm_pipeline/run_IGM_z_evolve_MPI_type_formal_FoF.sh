#!/bin/bash
#PBS -N IGM_FoF_type_z_P_20
#PBS --group=hp240141
#PBS -q SQUID
#PBS -l elapstim_req=24:00:00,cpunum_job=60,memsz_job=248GB
#PBS -b 2
#PBS -T intmpi
#PBS -v OMP_NUM_THREADS=60
#PBS -M zzhang@astro-osaka.jp
#PBS -mabe
cd $PBS_O_WORKDIR
source ~/.bashrc
conda activate zhangzhao-env
#module load BaseGCC
#module load BaseCPU
#module load BasePy

#mpirun ${NQSV_MPIOPTS} -genv I_MPI_DEBUG 5 -np 2 -genv I_MPI_PIN_DOMAIN=node  python3 $PBS_O_WORKDIR/f_igm_z_halo_paralell_MPI_type_formal_FoF.py 1 13.1 15.5 4.0 3.5  > logfile_figm_z_P_type1_13_15_FoF
#wait
mpirun ${NQSV_MPIOPTS} -genv I_MPI_DEBUG 5 -np 2 -genv I_MPI_PIN_DOMAIN=node  python3 $PBS_O_WORKDIR/f_igm_z_halo_paralell_MPI_type_formal_FoF.py 1 10 15.5 4.0 3.5  > logfile_figm_z_P_type1_10_15

#mpirun ${NQSV_MPIOPTS} -genv I_MPI_DEBUG 5 -np 2 -genv I_MPI_PIN_DOMAIN=node  python3 $PBS_O_WORKDIR/f_igm_z_halo_paralell_MPI_type_formal_FoF.py 2 10.0 13.1 4.0 3.5 > logfile_figm_z_P_type2
#Dwait

#mpirun ${NQSV_MPIOPTS} -genv I_MPI_DEBUG 5 -np 2 -genv I_MPI_PIN_DOMAIN=node  python3 $PBS_O_WORKDIR/f_igm_z_halo_paralell_MPI_type_formal_FoF.py 3 10.0 13.1 4.0 3.5 > logfile_figm_z_P_type3
