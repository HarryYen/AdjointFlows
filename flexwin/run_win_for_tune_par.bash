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

# set path to event list
par_file=../adjointflows/config.yaml
evlst=`grep evlst $par_file | gawk '{print $2}'`  # path to event list
evlst=`echo ../DATA/evlst/$evlst`
source_type=`awk '/^source:/{flag=1;next} flag && /^[[:space:]]+type:/{print $2; exit}' $par_file`
if [ "$source_type" = "force" ]; then
force_depth=`awk '/^source:/{flag=1;next} flag && /depth_km:/{print $2; exit}' $par_file`
if [ -z "$force_depth" ]; then
force_depth="0.0"
fi
flexwin_evlst="${evlst}.flexwin"
if [ ! -f "$flexwin_evlst" ] || [ "$evlst" -nt "$flexwin_evlst" ] || [ "$par_file" -nt "$flexwin_evlst" ]; then
# build a CMT-like event list for FLEXWIN using the virtual station list
awk -v date="2000/01/01" -v time="00:00:00" -v dep="$force_depth" -v mag="1.0" 'BEGIN {OFS=" "} $1 ~ /^#/ || NF < 3 {next} {print $1, date, time, $2, $3, dep, 0, 0, 0, 0, 0, 0, mag, mag}' "$evlst" > "$flexwin_evlst"
fi
evlst="$flexwin_evlst"
fi
comp=`grep COMP $par_file  | gawk '{print $2}'`    # synthetic component to use
en2rt=`grep EN2RT $par_file  | gawk '{print $2}'`  # rotate the E & N comp onent to R & T
echo $evlst $comp


# make observations and synthetics under the same conditions prior to running flexwin
bash ini_proc.bash $1

mkdir -p PACK
rm -rf PACK/$1
mkdir PACK/$1
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

   if [ "$1" != "0" ]; then
      if [ "$1" != "$dir" ]; then
         continue
      fi
   fi

   if [ -d ../DATA/wav/$dir -a -d ../SYN/$dir ]; then
      echo $dir" match!"
   #   np=`ls ../DATA/$dir | gawk -F. '{print $1"."$2}' | uniq | wc -l`
      shopt -s nullglob
      tomo_files=(../DATA/wav/$dir/*.tomo)
      shopt -u nullglob
      if [ ${#tomo_files[@]} -eq 0 ]; then
         echo "No .tomo files for $dir, skip."
         continue
      fi
      printf "%s\n" "${tomo_files[@]}" | gawk -F[./] '{print $7"."$8"."$9}' > pairs.tmp

   #   echo $np > input

      for pair in `cat pairs.tmp`
      do
         echo 1 > input
         echo ../DATA/wav/$dir/$pair.sac.tomo >> input
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
   cd MEASURE
   for psfile in `ls *ps`;
   do
      echo $psfile
      /usr/bin/gs -sDEVICE=pngalpha -o $psfile.png $psfile
   done
   cd ..
   # mv MEASURE/*.ps PACK/$1
   # mv MEASURE/*.eps PACK/$1
   # rm MEASURE/*ps
   mv MEASURE/*.png PACK/$1

done
