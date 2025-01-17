#!/bin/bash
#PBS -S /bin/bash

## job name and output file
#PBS -N sum_smooth_kernel
#PBS -j oe
#PBS -o sum_smooth.log
#PBS -e sum_smooth.err

###########################################################
# USER PARAMETERS FOR PARALLELIZATION

## request 512 CPUs from four nodes
##PBS -l nodes=8:ppn=64
##PBS -l nodes=green5:ppn=64+green6:ppn=64+green7:ppn=64+green8:ppn=64+green9:ppn=64+green10:ppn=64+green11:ppn=64+green12:ppn=64
##PBS -l nodes=node01:ppn=48+node02:ppn=48+node03:ppn=48+node04:ppn=48+node05:ppn=48+node06:ppn=48+node07:ppn=48+node08:ppn=48+node09:ppn=48+node10:ppn=48+node11:ppn=32

## request 512 CPUs from idle nodes
##PBS -l process=512
##PBS -q default

### GPU MODE ###
### Queue name (default)
#PBS -q qgpu
### Number of nodes (select:nodes, ncpus: process per node)
#PBS -l select=1:ncpus=1:ngpus=1

###########################################################
# PREPROCESS

cd $PBS_O_WORKDIR

# GPU MODE or not
gpu_flag=`grep GPU_MODE DATA/Par_file | awk '{print $3}'`

# number of cores for the job
NPROC=`grep NPROC DATA/Par_file | head -1 | cut -d = -f 2 `
numnodes=$NPROC
nslice=`expr $NPROC - 1`
echo $numnodes

###########################################################
# PARAMETER SETTING

# set path to event and station list
mbeg=`grep MBEG ../tomo.par | gawk '{print $3}'`    # the iteration currently running
evlst=`grep EVLST ../tomo.par | gawk '{print $3}'`  # path to event list
stlst=`grep STLST ../tomo.par | gawk '{print $3}'`  # path to station list
ichk=`grep ICHK ../tomo.par | gawk '{print $3}'`    # continue to run or start over


# sum up event kernels
rm INPUT_KERNELS
ls KERNEL/DATABASE > kernels_list.txt
ln -s KERNEL/DATABASE/ INPUT_KERNELS
mkdir -p OUTPUT_SUM
rm -f OUTPUT_SUM/*

if [ $NPROC -eq 1 ]; then
    # Get Sum(Hess)
    ./bin/xsum_preconditioned_kernels
    # Get Sum(event_kernel)
    ./bin/xsum_kernels
else
    mpirun -np $numnodes ./bin/xsum_preconditioned_kernels
    mpirun -np $numnodes ./bin/xsum_kernels
fi

mv OUTPUT_SUM/* KERNEL/SUM/

# kernel smoothing
mkdir -p KERNEL/SMOOTH/
ismooth=`grep ISMOOTH ../tomo.par | gawk '{print $3}'`
sigma_h=`grep SIGMA_H ../tomo.par | gawk '{print $3}'`
sigma_v=`grep SIGMA_V ../tomo.par | gawk '{print $3}'`
if [ $ismooth -eq 1 ]; then
    if [ $NPROC -eq 1 ]; then
        ./bin/xsmooth_sem $sigma_h $sigma_v alpha_kernel KERNEL/SUM/ KERNEL/SMOOTH/ $gpu_flag
        ./bin/xsmooth_sem $sigma_h $sigma_v beta_kernel KERNEL/SUM/ KERNEL/SMOOTH/ $gpu_flag
        ./bin/xsmooth_sem $sigma_h $sigma_v rho_kernel KERNEL/SUM/ KERNEL/SMOOTH/ $gpu_flag
        ./bin/xsmooth_sem $sigma_h $sigma_v hess_inv_kernel KERNEL/SUM/ KERNEL/SMOOTH/ $gpu_flag
    else
        mpirun -np $numnodes ./bin/xsmooth_sem $sigma_h $sigma_v alpha_kernel KERNEL/SUM/ KERNEL/SMOOTH/ $gpu_flag
        mpirun -np $numnodes ./bin/xsmooth_sem $sigma_h $sigma_v beta_kernel KERNEL/SUM/ KERNEL/SMOOTH/ $gpu_flag
        mpirun -np $numnodes ./bin/xsmooth_sem $sigma_h $sigma_v rho_kernel KERNEL/SUM/ KERNEL/SMOOTH/ $gpu_flag
        mpirun -np $numnodes ./bin/xsmooth_sem $sigma_h $sigma_v hess_inv_kernel KERNEL/SUM/ KERNEL/SMOOTH/ $gpu_flag
    fi
fi

# combine summed kernels for VTK visualization
ivtkout=`grep IVTKOUT ../tomo.par | gawk '{print $3}'`
if [ $ivtkout -eq 1 ]; then
if [ $ismooth -eq 1 ]; then
./bin/xcombine_vol_data_vtk 0 $nslice alpha_kernel_smooth KERNEL/SMOOTH/ KERNEL/VTK/ 0
./bin/xcombine_vol_data_vtk 0 $nslice beta_kernel_smooth KERNEL/SMOOTH/ KERNEL/VTK/ 0
./bin/xcombine_vol_data_vtk 0 $nslice rho_kernel_smooth KERNEL/SMOOTH/ KERNEL/VTK/ 0
else
./bin/xcombine_vol_data_vtk 0 $nslice alpha_kernel KERNEL/SUM/ KERNEL/VTK/ 0
./bin/xcombine_vol_data_vtk 0 $nslice beta_kernel KERNEL/SUM/ KERNEL/VTK/ 0
./bin/xcombine_vol_data_vtk 0 $nslice rho_kernel KERNEL/SUM/ KERNEL/VTK/ 0
fi
fi

echo " "  > ../iterate_inv/model_gradient_ready