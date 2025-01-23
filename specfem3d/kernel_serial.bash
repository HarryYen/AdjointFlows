#!/bin/bash
#PBS -S /bin/bash

## job name and output file
#PBS -N sem_kernel
#PBS -j oe
#PBS -o kernel.log
#PBS -e kernel.err

###########################################################
# USER PARAMETERS FOR PARALLELIZATION

## request 512 CPUs from four nodes
##PBS -l nodes=8:ppn=64
##PBS -l nodes=green5:ppn=64+green6:ppn=64+green7:ppn=64+green8:ppn=64+green9:ppn=64+green10:ppn=64+green11:ppn=64+green12:ppn=64
#PBS -l nodes=node01:ppn=48+node02:ppn=48+node03:ppn=48+node04:ppn=48+node05:ppn=48+node06:ppn=48+node07:ppn=48+node08:ppn=48+node09:ppn=48+node10:ppn=48+node11:ppn=32

## request 512 CPUs from idle nodes
##PBS -l process=512
##PBS -q default

### GPU MODE ###
### Queue name (default)
##PBS -q qgpu
### Number of nodes (select:nodes, ncpus: process per node)
#PBS -l select=1:ncpus=1:ngpus=1

###########################################################
# PREPROCESS

cd $PBS_O_WORKDIR

# # GPU MODE or not
# gpu_flag=`grep GPU_MODE DATA/Par_file | awk '{print $3}'`

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
#cd bin

###########################################################
# PARAMETER SETTING
par_file=../adjointflows/config.yaml
tmp_var_file=env_vars.txt

# set path to event and station list
mbeg=`grep current_model_num $par_file | gawk '{print $2}'`    # the iteration currently running
evlst=`grep evlst $par_file | gawk '{print $2}'`  # path to event list
stlst=`grep stlst $par_file | gawk '{print $2}'`  # path to station list
ichk=`grep ICHK $par_file | gawk '{print $2}'`    # continue to run or start over
flexwin_flag=`grep FLEXWIN_FLAG $par_file | gawk '{print $2}'`    # re-pick flexwin window or not


stage_initial_model=`grep stage_initial_model $tmp_var_file | awk '{print $3}'`
mrun=`grep mrun $tmp_var_file | awk '{print $3}'`
echo "Get stage_initial_model: ${stage_initial_model} and current model num: ${mrun} from env!"

# check where the last time stopped
if [ $ichk -eq 1 ]; then
    set kevt=0
    declare -i kevt=0
    for dir in `gawk '{print $1}' ../DATA/evlst/$evlst`
    do
        if [ -d KERNEL/DATABASE/$dir ]; then
            kevt=$kevt+1
        fi
    done
    echo "last time stop at the event "$kevt
fi

# set output directories
mkdir -p KERNEL
mkdir -p KERNEL/DATABASE
mkdir -p KERNEL/SMOOTH
mkdir -p KERNEL/SUM
mkdir -p KERNEL/VTK
#mkdir -p ../MOD
if [ $ichk -eq 0 ]; then
    rm -rf KERNEL/DATABASE/*
fi
rm -rf KERNEL/SMOOTH/*
rm -rf KERNEL/VTK/*
rm -rf KERNEL/SUM/*
#rm -rf MOD/*
rm -f kernels_list.txt


# START WAVEFORM SIMULATION AND KERNEL CALCULATION
set nevt=0
declare -i nevt=0

for cmt in `gawk '{print $1"/"$2"/"$3"/"$4"/"$5"/"$6"/"$7"/"$8"/"$9"/"$13"/"$14"/"$15"/"$16"/"$17"/"$18"/"$19"/"$20"/"$21}' ../DATA/evlst/$evlst`
do

    nevt=$nevt+1

    dir=`echo $cmt | gawk -F[:/] '{print $1}'`

    if [ $ichk -eq 1 ]; then
        if [ $nevt -lt $kevt ]; then
            echo "We have done the ${dir}, so it will be skipped."
            continue
        fi
    fi

    # estimate empirical half duration of source rupture
    #mg=`echo $cmt | gawk -F[:/] '{print $14}'`
    #hd=`echo $mg  | gawk '{print 1.1*(10**(-8))*((10**(($1+10.7)*1.5))**(1/3)) }'`


    # make event file
    echo $cmt | gawk -F[:/] '{printf "%3s  %4d %2d %2d %2d %2d %5.2f  %6.3f %6.3f %4.1f %3.1f %3.1f %12d\n","PDE",$2,$3,$4,$5,$6,$7,$9,$8,$10,$14,$14,$1}' > DATA/CMTSOLUTION
    echo $cmt | gawk -F[:/] '{printf "%11s       %12d\n","event name:",$1}' >> DATA/CMTSOLUTION
    echo $cmt | gawk -F[:/] '{printf "%11s       %6.4f\n","time shift:",0.0}' >> DATA/CMTSOLUTION
    #echo $cmt | gawk -F[:/] '{printf "%14s    %3.1f\n","half duration:",hd}' hd=$hd  >> ../inputs/CMTSOLUTION
    echo $cmt | gawk -F[:/] '{printf "%14s    %3.1f\n","half duration:",0.0}' >> DATA/CMTSOLUTION
    echo $cmt | gawk -F[:/] '{printf "%9s        %6.3f\n","latitude:",$9}' >> DATA/CMTSOLUTION
    echo $cmt | gawk -F[:/] '{printf "%10s      %7.3f\n","longitude:",$8}' >> DATA/CMTSOLUTION
    echo $cmt | gawk -F[:/] '{printf "%6s          %6.2f\n","depth:",$10}' >> DATA/CMTSOLUTION
    #echo $cmt | gawk -F[:/] '{print "../utils/dc2mt",$11,$12,$13,$14}' | sh | gawk 'NR>=11 && NR<=16 {print $0}' >> ../inputs/CMTSOLUTION
    echo $cmt | gawk -F[:/] '{printf "%4s  %7.4e\n","Mrr:",$16*1.e+24}' >> DATA/CMTSOLUTION
    echo $cmt | gawk -F[:/] '{printf "%4s  %7.4e\n","Mtt:",$17*1.e+24}' >> DATA/CMTSOLUTION
    echo $cmt | gawk -F[:/] '{printf "%4s  %7.4e\n","Mpp:",$18*1.e+24}' >> DATA/CMTSOLUTION
    echo $cmt | gawk -F[:/] '{printf "%4s  %7.4e\n","Mrt:",$19*1.e+24}' >> DATA/CMTSOLUTION
    echo $cmt | gawk -F[:/] '{printf "%4s  %7.4e\n","Mrp:",$20*1.e+24}' >> DATA/CMTSOLUTION
    echo $cmt | gawk -F[:/] '{printf "%4s  %7.4e\n","Mtp:",$21*1.e+24}' >> DATA/CMTSOLUTION
    cp DATA/CMTSOLUTION ../measure_adj


    # make station file
    gawk '{printf "%4s %2s %6.3f %7.3f %3.1f %3.1f\n",$1,"TW",$3,$2,0.0,0.0}' ../DATA/stlst/$stlst > DATA/STATIONS


    # run forward simulation
    sleep 5
    ./utils/change_simulation_type.pl -F
    rm -f OUTPUT_FILES/*.sem?



    if [ $NPROC -eq 1 ]; then
        ./bin/xspecfem3D
    else
        mpirun -np $numnodes ./bin/xspecfem3D
    fi


    echo "event "$nevt" forward simulation finished"

    mkdir -p ../SYN/$dir 
    cp DATA/STATIONS_FILTERED ../measure_adj/PLOTS
    rm -f ../SYN/$dir/*
    mv OUTPUT_FILES/*.sem? ../SYN/$dir
    rm -f ../DATA/wav/$dir/*.sac.tomo
    rm -f DATA/STATIONS_ADJOINT

    # shift begin time and mark the origin time (shift 1.66667 seconds in consistent with RMT process)
    #/opt/pub/bin/ascii2sac.csh ../../SYN/$dir/*
    #ls ../../SYN/$dir/*.sac | gawk '{print "saclst b f",$0}' | sh | gawk '{print "r "$1"; ch b",$2+1.66667"; ch o 0; w over"}END{print "q"}' | sac

    # preprocess waveforms and measure adjoint sources
    cd ../flexwin
    if [[ $stage_initial_model -eq $mrun ]] || [[ $flexwin_flag -eq 1 ]]; then 
	bash run_win.bash $dir
    else
        bash ini_proc.bash $dir
        m_last=`printf "%s%03d" "m" $stage_initial_model`
        cp ../TOMO/$m_last/MEASURE/adjoints/$dir/MEASUREMENT.WINDOWS ../measure_adj
        cd ../measure_adj
        bash run_adj.bash $dir
    fi


    # check if the adjoint sources is ready to launch
    while [ 1 ];
    do
    if [ -f ../specfem3d/DATA/STATIONS_ADJOINT ]; then
    cd ../specfem3d
    break
    fi
    sleep 3
    done

    # if ready, run adjoint simulation
    ./utils/change_simulation_type.pl -b

    if [ $NPROC -eq 1 ]; then
        ./bin/xspecfem3D
    else
        mpirun -np $numnodes ./bin/xspecfem3D
    fi

    # mkdir -p OUTPUT_FILES/waveform_adj/
    # mv OUTPUT_FILES/*.sem? OUTPUT_FILES/waveform_adj/
    echo "event",$nevt," adjoint simulation finished"
    mkdir -p KERNEL/
    mkdir -p KERNEL/DATABASE/
    mkdir -p KERNEL/DATABASE/$dir
    mv DATABASES_MPI/proc*kernel.bin KERNEL/DATABASE/$dir

    
    echo "kernel constructed and collected!"

    continue

    # combine all slices to one for VTK visualization
    mkdir -p KERNEL/VTK/
    mkdir -p KERNEL/VTK/$dir
    ivtkout=`grep IVTKOUT $par_file | gawk '{print $3}'`
    if [ $ivtkout -eq 1 ]; then
        ./bin/xcombine_vol_data_vtk 0 $nslice alpha_kernel KERNEL/DATABASE/$dir/ KERNEL/VTK/$dir 0
        ./bin/xcombine_vol_data_vtk 0 $nslice beta_kernel KERNEL/DATABASE/$dir/ KERNEL/VTK/$dir 0
        ./bin/xcombine_vol_data_vtk 0 $nslice rho_kernel KERNEL/DATABASE/$dir/ KERNEL/VTK/$dir 0
    fi


    #if [ $nevt -eq 2 ]; then
    #break
    #fi

done

echo "all events finished" > kernel_databases_ready
