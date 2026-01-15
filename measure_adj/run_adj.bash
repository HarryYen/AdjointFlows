#!/bin/bash
# --------------------------------------------------------------------------------------------
# This script is to calculate the adjoint sources for each matched windows and plot results
# --------------------------------------------------------------------------------------------
# check arguments and print usage
if [ "$#" -ne 1 ]; then
   echo "USAGE: run_adj [output_dir]"
   echo "Note that output_dir will be placed in PACK/"
   exit
fi
echo $1    # output_dir
mkdir -p PLOTS/DATA
mkdir -p PLOTS/SYN
mkdir -p PLOTS/ADJOINT_SOURCES
mkdir -p PLOTS/RECON
mkdir -p PACK
rm -rf PACK/$1
mkdir PACK/$1
par_file=../adjointflows/config.yaml
p1=`grep P1 $par_file | gawk '{print $2}'`        # min period for bandpass
p2=`grep P2 $par_file | gawk '{print $2}'`        # max period for bandpass
en2rt=`grep EN2RT $par_file | gawk '{print $2}'`  # rotate the E & N component to R & T
tmin=`grep tbeg $par_file | gawk '{print $2}'`      # tbeg of simulated waveforms for plotting
tmax=`grep tend $par_file | gawk '{print $2}'`      # tend of simulated waveforms for plotting
# measuring the adjoint sources using flexwin output
mkdir -p OUTPUT_FILES
rm -f OUTPUT_FILES/*

rm window_index window_chi window_chi_sum
./measure_adj


# collecting files for plotting
rm -f PLOTS/SYN/*
grep SYN MEASUREMENT.WINDOWS | gawk -F. '{print "cp .."$3"."$4"*tomo PLOTS/SYN"}' | sh
rm -f PLOTS/DATA/*
grep DATA MEASUREMENT.WINDOWS | gawk -F. '{print "cp .."$3"."$4"*tomo PLOTS/DATA"}' | sh


rm -f PLOTS/ADJOINT_SOURCES/*
cp OUTPUT_FILES/*adj PLOTS/ADJOINT_SOURCES
rm -f PLOTS/RECON/*
ls OUTPUT_FILES/*recon.sac | gawk -F/ '{print "r "$1"/"$2"; ch o 0; w PLOTS/RECON/"$2}END{print "q"}' | sac
#cp OUTPUT_FILES/*recon.sac PLOTS/RECON
cp window_index window_chi window_chi_sum PLOTS

# plotting figures
rm -f STATIONS_ADJOINT
rm -f ADJOINT_SOURCES/*adj
if [ $en2rt -eq 1 ]; then
./prepare_adj_src.pl -m CMTSOLUTION -s PLOTS/STATIONS_FILTERED -z BH -o ADJOINT_SOURCES -i 1 OUTPUT_FILES/*adj -r
else
./prepare_adj_src.pl -m CMTSOLUTION -s PLOTS/STATIONS_FILTERED -z BH -o ADJOINT_SOURCES -i 1 OUTPUT_FILES/*adj
fi

rm -f PLOTS/STATIONS_ADJOINT
cp STATIONS_ADJOINT PLOTS
rm -f PLOTS/MEASUREMENT.WINDOWS
cp MEASUREMENT.WINDOWS PLOTS
mkdir -p ../specfem3d/SEM
rm -f ../specfem3d/SEM/*
ls ADJOINT_SOURCES/*adj | gawk -F[./] '{print "cp "$0,"../specfem3d/SEM/"$3"."$2"."$4".adj"}' | sh
rename BH BX ../specfem3d/SEM/*
cd PLOTS
#ls ADJOINT_SOURCES/* | gawk -F[/.] '{print "grep "$2" STATIONS_FILTERED"}' | sh | uniq | wc -l > STATIONS_ADJOINT
#ls ADJOINT_SOURCES/* | gawk -F[/.] '{print "grep "$2" STATIONS_FILTERED"}' | sh | uniq >> STATIONS_ADJOINT
#cp STATIONS_ADJOINT ../../specfem3d/inputs

## PLOT (close temporarily) ##
if [ $en2rt -eq 1 ]; then
./plot_win_adj_all.pl -l $tmin/$tmax -m ../CMTSOLUTION -n BH -b 0 -k 7/1 -a STATIONS_ADJOINT -d DATA -s SYN -c RECON -w MEASUREMENT.WINDOWS -i m00 -j $p1/$p2 -p
else
./plot_win_adj_all.pl -l $tmin/$tmax -m ../CMTSOLUTION -n BH -b 0 -k 7/1 -a STATIONS_ADJOINT -d DATA -s SYN -c RECON -w MEASUREMENT.WINDOWS -i m00 -j $p1/$p2
fi
cd ..

# trasfer ps into png
cd PLOTS
for psfile in `ls *.ps`;
do
   gs -sDEVICE=pngalpha -o $psfile.png $psfile
   rm $psfile
done

cd ..


# pack results by event
cp CMTSOLUTION PACK/$1
cp PLOTS/*.ps PACK/$1
cp PLOTS/*.pdf PACK/$1
cp PLOTS/*.png PACK/$1
cp PLOTS/MEASUREMENT.WINDOWS PACK/$1
cp PLOTS/window_* PACK/$1
cp PLOTS/STATIONS* PACK/$1

rm PLOTS/*.ps
rm PLOTS/*.png

# triggering the next forward simulation
rm -f ../specfem3d/DATA/STATIONS_ADJOINT
cat STATIONS_ADJOINT | gawk 'NR>=2 {print $0}' > ../specfem3d/DATA/STATIONS_ADJOINT
