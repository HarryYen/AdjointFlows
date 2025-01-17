#!/bin/bash
#PBS -S /bin/bash

## job name and output file
#PBS -N sem_mod
#PBS -j oe
#PBS -o database_generate.log
#PBS -e database_generate.err

###########################################################
# USER PARAMETERS

### GPU MODE ###
### Queue name (default)
#PBS -q qgpu
### Number of nodes (select:nodes, ncpus: process per node)
#PBS -l select=1:ncpus=1:ngpus=1

###########################################################

cd $PBS_O_WORKDIR

# number of cores for the job
NPROC=`grep NPROC DATA/Par_file | head -1 | cut -d = -f 2 `
numnodes=$NPROC
nslice=`expr $NPROC - 1`
echo $numnodes

# obtain pbs job information
cat $PBS_NODEFILE > OUTPUT_FILES/compute_nodes
echo "$PBS_JOBID" > OUTPUT_FILES/jobid

echo starting MPI database generating on $numnodes processors
echo " "

sleep 2

./utils/change_model_type.pl -g

if [ $NPROC -eq 1 ]; then
    ./bin/xgenerate_databases
else 
    mpirun -np $numnodes ./bin/xgenerate_databases
fi

# cp OUTPUT_FILES/values_from_mesher.h ../iterate_inv/shared
echo "done database generating"
echo " " > OUTPUT_FILES/model_database_ready


# combine all slices to a complete model output
# ./bin/xcombine_vol_data_vtk 0 $nslice vp DATABASES_MPI/ . 0
# ./bin/xcombine_vol_data_vtk 0 $nslice vs DATABASES_MPI/ . 0
# echo "model collected and constructed!"

