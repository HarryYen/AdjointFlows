#!/bin/bash
#PBS -S /bin/bash

## job name and output file
#PBS -N sem_mod
#PBS -j oe
#PBS -o model.log
#PBS -e model.err

###########################################################
# USER PARAMETERS

## request 512 CPUs from four nodes
##PBS -l nodes=green5:ppn=64+green6:ppn=64+green7:ppn=64+green8:ppn=64+green9:ppn=64+green10:ppn=64+green11:ppn=64+green12:ppn=64
##PBS -l nodes=node01:ppn=48+node02:ppn=48+node03:ppn=48+node04:ppn=48+node05:ppn=48+node06:ppn=48+node07:ppn=48+node08:ppn=48+node09:ppn=48+node10:ppn=48+node11:ppn=32
## request 512 CPUs from idle nodes
##PBS -l process=512
##PBS -q default
##PBS -l nodes=10:ppn=48
##PBS -l nodes=1:ppn=32

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

echo starting MPI internal mesher on $numnodes processors
echo " "

sleep 2


if [ $NPROC -eq 1 ]; then
    ./bin/xmeshfem3D
else
    mpirun -np $numnodes ./bin/xmeshfem3D
fi
echo "done meshing"
sleep 2
./utils/change_model_type.pl -t

if [ $NPROC -eq 1 ]; then
    ./bin/xgenerate_databases
else
    mpirun -np $numnodes ./bin/xgenerate_databases
fi

cp OUTPUT_FILES/values_from_mesher.h ../iterate_inv/shared
echo "done database generating"
echo " " > OUTPUT_FILES/model_database_ready


# combine all slices to a complete model output
# ./bin/xcombine_vol_data_vtk 0 $nslice vp DATABASES_MPI/ . 0
# ./bin/xcombine_vol_data_vtk 0 $nslice vs DATABASES_MPI/ . 0
echo "model collected and constructed!"

