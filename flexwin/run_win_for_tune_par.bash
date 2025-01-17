#!/bin/bash
# --------------------------------------------------------------------------------------------
# This script is only for the test of tuning the parameters of flexwin.
# Hung-Yu Yen
# --------------------------------------------------------------------------------------------
# check arguments and print usage
if [ "$#" -ne 1 ]; then
   echo "USAGE: run_win [evt_dir]"
   echo "Note that one can put evt_dir 0 to do all the events on the list"
   exit
fi
echo $1    # evt_dir

# set path to event list
evlst=`grep EVLST ../tomo_tune.par | gawk '{print $3}'`  # path to event list
comp=`grep COMP ../tomo_tune.par | gawk '{print $3}'`    # synthetic component to use
en2rt=`grep EN2RT ../tomo_tune.par | gawk '{print $3}'`  # rotate the E & N component to R & T
echo $evlst $comp

evlst=../DATA/$evlst

# make observations and synthetics under the same conditions prior to running flexwin
bash ini_proc.bash $1

mkdir -p PACK_test
rm -rf PACK_test/$1
mkdir PACK_test/$1
rm -f MEASURE/*
for cmt in `gawk '{print $1"/"$2"/"$3"/"$4"/"$5"/"$6"/"$7"/"$8"/"$9"/"$13"/"$14}' $evlst`
do
   dir=`echo $cmt | gawk -F[:/] '{print $1}'`
   yy=`echo $cmt | gawk -F[:/] '{print $2}'`
   mm=`echo $cmt | gawk -F[:/] '{print $3}'`
   dd=`echo $cmt | gawk -F[:/] '{print $4}'`
   jd=`date +%j --date=$yy"/"$mm"/"$dd`   # julian day
   hr=`echo $cmt | gawk -F[:/] '{print $5}'`
   mi=`echo $cmt | gawk -F[:/] '{print $6}'`

   if [ $1 -ne 0 ]; then
      if [ $1 -ne $dir ]; then
         continue
      fi
   fi

   if [ -d ../DATA/$dir -a -d ../SYN/$dir ]; then
      echo $dir" match!"
#      np=`ls ../DATA/$dir | gawk -F. '{print $1"."$2}' | uniq | wc -l`
      ls ../DATA/$dir/*.tomo | gawk -F[./] '{print $6"."$7"."$8}' > pairs.tmp

#      echo $np > input
      for pair in `cat pairs.tmp`
      do
         echo 1 > input
         echo ../DATA/$dir/$pair.sac.tomo >> input
         echo ../SYN/$dir/$pair.$comp.sac.tomo >> input
         echo $pair | gawk '{print "MEASURE/"$1}' >> input
         ./flexwin < input
         echo $pair | gawk -F. '{print "sh plot_seismos_gmt.sh MEASURE/"$1"."$2"."$3}' | sh
      done
      let "npair+=`ls MEASURE/*_input | wc -l`"

   else
      echo "event directory"$dir" does not exist!"
   fi

   echo "measured pairs:"$npair
   echo $npair > MEASUREMENT.WINDOWS
   cat MEASURE/*_input >> MEASUREMENT.WINDOWS
   rm -f ../measure_adj/MEASUREMENT.WINDOWS
   cp MEASUREMENT.WINDOWS ../measure_adj/
   ./extract_event_windowing_stats.sh MEASURE $en2rt

   # pack results by event
   mv MEASURE/*.ps PACK_test/$1
   mv MEASURE/*.eps PACK_test/$1
   cp PAR_FILE PACK_test/$1
   cp MEASUREMENT.WINDOWS PACK_test/$1

done

