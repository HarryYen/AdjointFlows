#!/bin/bash


# number of cores for the job
NPROC=`grep NPROC DATA/Par_file | head -1 | cut -d = -f 2 `
numnodes=$NPROC
nslice=`expr $NPROC - 1`
echo $numnodes

echo starting MPI internal mesher on $numnodes processors
echo " "

sleep 2

./bin/xmeshfem3D


echo "done meshing"
sleep 2
./utils/change_model_type.pl -t

./bin/xgenerate_databases


cp OUTPUT_FILES/values_from_mesher.h ../iterate_inv/shared
echo "done database generating"
echo " " > OUTPUT_FILES/model_database_ready


# combine all slices to a complete model output
# ./bin/xcombine_vol_data_vtk 0 $nslice vp DATABASES_MPI/ . 0
# ./bin/xcombine_vol_data_vtk 0 $nslice vs DATABASES_MPI/ . 0
echo "model collected and constructed!"

