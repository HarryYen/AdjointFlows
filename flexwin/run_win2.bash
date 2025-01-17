#!/bin/bash
# --------------------------------------------------------------------------------------------
# This script is to pair the observations and synthetics and make the input file for flexwin
# --------------------------------------------------------------------------------------------
# check arguments and print usage
if [ "$#" -ne 1 ]; then
   echo "USAGE: run_win [evt_dir]"
   echo "Note that one can put evt_dir 0 to do all the events on the list"
   exit
fi
echo $1    # evt_dir

# how many cores for distributed computing
nproc=20

# set path to event list
evlst=`grep EVLST ../tomo.par | gawk '{print $3}'`  # path to event list
comp=`grep COMP ../tomo.par | gawk '{print $3}'`    # synthetic component to use
en2rt=`grep EN2RT ../tomo.par | gawk '{print $3}'`  # rotate the E & N component to R & T
echo $evlst $comp

# make observations and synthetics under the same conditions prior to running flexwin
#bash ini_proc.bash $1

mkdir -p PACK
rm -rf PACK/$1
mkdir PACK/$1
rm -f MEASURE/*
for dir in `gawk '{print $1}' $evlst`
do
   if [ $1 -ne 0 ]; then
      if [ $1 -ne $dir ]; then
         continue
      fi
   fi

   if [ -d ../DATA/$dir -a -d ../SYN/$dir ]; then
      echo $dir" match!"
      rm -rf MEASURE/*
      ls ../DATA/$dir/*.tomo | gawk -F[./] '{print $6"."$7"."$8}' > pairs.tmp
      np=`cat pairs.tmp | wc -l`
      ip=0
      for pair in `cat pairs.tmp`
      do
         ip=`expr $ip + 1`
         echo 1 > input
         echo ../DATA/$dir/$pair.sac.tomo >> input
         echo ../SYN/$dir/$pair.$comp.sac.tomo >> input
         echo $pair | gawk '{print "MEASURE/"$1}' >> input
         if [ $ip -lt $np ]; then
            flexwin < input &
         else
            flexwin < input
         fi
      done
      while [ 1 ];
      do
         if [ `ls MEASURE/*_input | wc -l` -eq $np ]; then
            ip=0
            for pair in `cat pairs.tmp`
            do
               ip=`expr $ip + 1`
               if [ $ip -lt $np ]; then
                  sh plot_seismos_gmt.sh MEASURE/$pair &
               else
                  sh plot_seismos_gmt.sh MEASURE/$pair
               fi
            done
         fi
      done
      let "npair+=`ls MEASURE/*_input | wc -l`"

   else
      echo "event directory "$dir" does not exist!"
   fi

   while [ 1 ];
   do
      if [ `ls MEASURE/*ps | wc -l` -eq $np ]; then
         echo "measured pairs:"$npair
         echo $npair > MEASUREMENT.WINDOWS
         cat MEASURE/*_input >> MEASUREMENT.WINDOWS
         rm -f ../measure_adj/MEASUREMENT.WINDOWS
         cp MEASUREMENT.WINDOWS ../measure_adj/
         extract_event_windowing_stats.sh MEASURE $en2rt
      fi
   done

   # pack results by event
   mv MEASURE/*.ps PACK/$1
   mv MEASURE/*.eps PACK/$1

   # export matched windows to measure_adj
   cd ../measure_adj
#   bash run_adj.bash $1

done

