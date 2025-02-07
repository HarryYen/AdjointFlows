#!/bin/bash
#PBS -S /bin/bash

## job name and output file
#PBS -N adjointflows_run
#PBS -j oe
#PBS -o adjointflows.log
#PBS -e adjointflows.err

###########################################################
# USER PARAMETERS FOR PARALLELIZATION

## request 512 CPUs from four nodes
#PBS -l nodes=node01:ppn=48+node02:ppn=48+node03:ppn=48+node04:ppn=48+node05:ppn=48+node06:ppn=48+node07:ppn=48+node08:ppn=48+node09:ppn=48+node10:ppn=48+node11:ppn=32
###########################################################
# PREPROCESS

cd $PBS_O_WORKDIR
# source ~/.bashrc
source activate adjflows
module load gcc630/compiler gcc630/openmpi-1.10.5
cat $PBS_NODEFILE > nodefile


python main.py
