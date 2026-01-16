#!/bin/bash
# --------------------------------------------------------------------------------------------
# This script is to apply the same waveform process to all observations and synthetics
# --------------------------------------------------------------------------------------------
# check arguments and print usage
if [ "$#" -ne 1 ]; then
   echo "USAGE: ini_proc [evt_dir]"
   echo "Note that one can put evt_dir 0 to do all the events on the list"
   exit
fi
echo $1    # evt_dir

# set path and parameters
pwd=`pwd`
par_file=../adjointflows/config.yaml
evlst=`grep evlst $par_file | gawk '{print $2}'`  # path to event list
stlst=`grep stlst $par_file | gawk '{print $2}'`  # path to station list
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
stlst=`echo ../DATA/stlst/$stlst`
tbeg=`grep tbeg $par_file | gawk '{print $2}'`    # begin time to cut (adding 30 sec prior to origin time for STA/LTA)
tend=`grep tend $par_file | gawk '{print $2}'`    # ending time to cut (make sure long enough to include desired phases)
tcor=`grep tcor $par_file | gawk '{print $2}'`    # systematic time shift to correct (if any), e.g. 1.66667 is the RMT time shift.
dt=`grep dt $par_file | gawk '{print $2}'`        # sampling rate to reset
p1=`grep P1 $par_file | gawk '{print $2}'`        # min period for bandpass
p2=`grep P2 $par_file | gawk '{print $2}'`        # max period for bandpass
comp=`grep COMP $par_file | gawk '{print $2}'`    # synthetic component to use
en2rt=`grep EN2RT $par_file | gawk '{print $2}'`  # rotate the E & N component to R & T

normalize_sac() {
local sac_file=$1
local max_amp
if [ ! -f "$sac_file" ]; then
return
fi
depmax=$(saclst depmax f "$sac_file" | awk '{print $2}')
depmin=$(saclst depmin f "$sac_file" | awk '{print $2}')
max_amp=$(awk -v mx="$depmax" -v mn="$depmin" 'BEGIN {if (mx=="") exit; if (mn=="") mn=0; if (mx<0) mx=-mx; if (mn<0) mn=-mn; if (mx<mn) mx=mn; print mx}')
if [ -z "$max_amp" ]; then
return
fi
awk -v x="$max_amp" 'BEGIN {exit !(x > 0)}'
if [ $? -eq 0 ]; then
sac<<EOF
r $sac_file
div $max_amp
w over
q
EOF
fi
}


f1=$(echo "scale=3; 1 / $p2"|bc)
f2=$(echo "scale=3; 1 / $p1"|bc)
for cmt in `gawk '{print $1"/"$2"/"$3"/"$4"/"$5"/"$6"/"$7"/"$8"/"$9"/"$13"/"$14}' $evlst`
do
dir=`echo $cmt | gawk -F[:/] '{print $1}'`
if [ "$1" != "0" ]; then
if [ "$1" != "$dir" ]; then
continue
fi
fi

if [ -d ../DATA/wav/$dir -a -d ../SYN/$dir ]; then

rm -f ../DATA/wav/$dir/*.tomo
rm -f ../SYN/$dir/*.tomo
yy=`echo $cmt | gawk -F[:/] '{print $2}'`
mm=`echo $cmt | gawk -F[:/] '{print $3}'`
dd=`echo $cmt | gawk -F[:/] '{print $4}'`
jd=`date +%j --date=$yy"/"$mm"/"$dd`   # julian day
hr=`echo $cmt | gawk -F[:/] '{print $5}'`
mi=`echo $cmt | gawk -F[:/] '{print $6}'`
sc=`echo $cmt | gawk -F[:/] '{print $7}'`
ns=`echo $sc  | gawk -F[.]  '{print $1}'`
ms=`echo $sc  | gawk -F[.]  '{print $2*100}'`
lo=`echo $cmt | gawk -F[:/] '{print $8}'`
la=`echo $cmt | gawk -F[:/] '{print $9}'`
dp=`echo $cmt | gawk -F[:/] '{print $10}'`
mg=`echo $cmt | gawk -F[:/] '{print $14}'`
# empirical half duration of source rupture
hd=`echo $mg  | gawk '{print 1.1*(10**(-8))*((10**(($1+10.7)*1.5))**(1/3)) }'`
echo $hd > ../specfem3d/half_duration.out


# convolve source time function scaled by local magnitude (non-force sources only)
if [ "$source_type" == "cmt" ]; then
csh ../specfem3d/utils/convolve_source_timefunction.csh ../SYN/$dir/*.$comp
else
for sac_file in ../SYN/$dir/*.$comp
do
    [ -e "$sac_file" ] || continue
    if [ ! -e "${sac_file}.convolved" ]; then
        cp "$sac_file" "${sac_file}.convolved"
    fi
done
fi

# shift begin time and mark the origin time (shift 1.66667 seconds in consistent with RMT process)
# the file need to be 'REALPATH' !!!
csh ../specfem3d/utils/seis_process/ascii2sac.csh $pwd/../SYN/$dir/*.$comp.convolved

ls ../SYN/$dir/*.sac | gawk '{print "saclst b f",$0}' | sh | gawk '{print "r "$1"; ch b",$2+tcor"; ch o 0; w over"}END{print "q"}' tcor=$tcor | sac

echo $stlst

for sta in `ls ../DATA/wav/$dir | gawk -F. '{print $1}' | uniq | gawk '{print "grep "$1" "stlst}' stlst="$stlst" | sh | gawk '{print $1"_"$2"_"$3"_"$4}'`
do
echo $sta
stnm=`echo $sta | gawk -F_ '{print $1}'`
stlo=`echo $sta | gawk -F_ '{print $2}'`
stla=`echo $sta | gawk -F_ '{print $3}'`
stel=`echo $sta | gawk -F_ '{print $4}'`


# write event/station info into sac header
sac<<EOF
r ../DATA/wav/$dir/$stnm*.sac ../SYN/$dir/*$stnm*.sac
ch nzyear $yy
ch nzjday $jd
ch nzhour $hr
ch nzmin  $mi
ch nzsec  $ns
ch nzmsec $ms
ch evlo   $lo
ch evla   $la
ch evdp   $dp
ch mag    $mg
ch knetwk TW
ch kstnm  $stnm
ch stlo   $stlo
ch stla   $stla
ch stel   $stel
ch o 0
interp delta $dt
w over
q
EOF

if [ $en2rt -eq 1 ]; then
# rotate recorded N/E traces to filtered R/T components to better identify phase windows
if [ -f ../DATA/wav/$dir/$stnm*N.*sac -a -f ../DATA/wav/$dir/$stnm*E.*sac -a -f ../SYN/$dir/*$stnm*N.$comp.*sac -a -f ../SYN/$dir/*$stnm*E.$comp.*sac]; then

sac<<EOF
cuterr fillz
cut $tbeg $tend
r ../DATA/wav/$dir/$stnm*N.*sac ../DATA/wav/$dir/$stnm*E.*sac
rotate to GCP
rmean
rtr; rtr; rtr
taper
bp p 1 n 4 c $f1 $f2
w ../DATA/wav/$dir/$stnm.TW.BHR.sac.tomo ../DATA/wav/$dir/$stnm.TW.BHT.sac.tomo
cut off
q
EOF
# rotate synthetic N/E traces to filtered R/T components to better identify phase windows
sac<<EOF
cuterr fillz
cut $tbeg $tend
r ../SYN/$dir/$stnm*N.$comp.*sac ../SYN/$dir/*$stnm*E.$comp.*sac
rotate to GCP
rmean
rtr; rtr; rtr
taper
bp p 1 n 4 c $f1 $f2
w ../SYN/$dir/*$stnm.TW.BHR.$comp.sac.tomo ../SYN/$dir/*$stnm.TW.BHT.$comp.sac.tomo
cut off
q
EOF
if [ "$source_type" = "force" ]; then
normalize_sac ../DATA/wav/$dir/$stnm.TW.BHR.sac.tomo
normalize_sac ../DATA/wav/$dir/$stnm.TW.BHT.sac.tomo
normalize_sac ../SYN/$dir/*$stnm.TW.BHR.$comp.sac.tomo
normalize_sac ../SYN/$dir/*$stnm.TW.BHT.$comp.sac.tomo
fi
fi

else

# filtered E & N components
echo $dir $stnm $comp
if [ -f ../DATA/wav/$dir/$stnm*E.*sac -a -f ../SYN/$dir/*$stnm*E.$comp.*sac ]; then
sac<<EOF
cuterr fillz
cut $tbeg $tend
r ../DATA/wav/$dir/*$stnm*E.*sac ../SYN/$dir/*$stnm*E.$comp.*sac
rmean
rtr; rtr; rtr
taper
bp p 1 n 4 c $f1 $f2
w ../DATA/wav/$dir/$stnm.TW.BHE.sac.tomo ../SYN/$dir/$stnm.TW.BHE.$comp.sac.tomo
cut off
q
EOF
if [ "$source_type" = "force" ]; then
normalize_sac ../DATA/wav/$dir/$stnm.TW.BHE.sac.tomo
normalize_sac ../SYN/$dir/$stnm.TW.BHE.$comp.sac.tomo
fi
fi
if [ -f ../DATA/wav/$dir/$stnm*N.*sac -a -f ../SYN/$dir/*$stnm*N.$comp.*sac ]; then
sac<<EOF
cuterr fillz
cut $tbeg $tend
r ../DATA/wav/$dir/$stnm*N.*sac ../SYN/$dir/*$stnm*N.$comp.*sac
rmean
rtr; rtr; rtr
taper
bp p 1 n 4 c $f1 $f2
w ../DATA/wav/$dir/$stnm.TW.BHN.sac.tomo ../SYN/$dir/$stnm.TW.BHN.$comp.sac.tomo
cut off
q
EOF
if [ "$source_type" = "force" ]; then
normalize_sac ../DATA/wav/$dir/$stnm.TW.BHN.sac.tomo
normalize_sac ../SYN/$dir/$stnm.TW.BHN.$comp.sac.tomo
fi
fi

fi

# filtered Z components as well
if [ -f ../DATA/wav/$dir/$stnm*Z.*sac -a -f ../SYN/$dir/*$stnm*Z.$comp.*sac ]; then
sac<<EOF
cuterr fillz
cut $tbeg $tend
r ../DATA/wav/$dir/$stnm*Z.*sac ../SYN/$dir/*$stnm*Z.$comp.*sac
rmean
rtr; rtr; rtr
taper
bp p 1 n 4 c $f1 $f2
w ../DATA/wav/$dir/$stnm.TW.BHZ.sac.tomo ../SYN/$dir/$stnm.TW.BHZ.$comp.sac.tomo
cut off
q
EOF
if [ "$source_type" = "force" ]; then
normalize_sac ../DATA/wav/$dir/$stnm.TW.BHZ.sac.tomo
normalize_sac ../SYN/$dir/$stnm.TW.BHZ.$comp.sac.tomo
fi
fi

done

# specify the component in header
if [ $en2rt -eq 1 ]; then
ls ../DATA/wav/$dir/*BHR*tomo | gawk '{print "r "$0"; ch kcmpnm BHR; w over"}END{print "q"}' | sac
ls ../DATA/wav/$dir/*BHT*tomo | gawk '{print "r "$0"; ch kcmpnm BHT; w over"}END{print "q"}' | sac
ls ../DATA/wav/$dir/*BHZ*tomo | gawk '{print "r "$0"; ch kcmpnm BHZ; w over"}END{print "q"}' | sac
ls ../SYN/$dir/*BHR*tomo  | gawk '{print "r "$0"; ch kcmpnm BHR; w over"}END{print "q"}' | sac
ls ../SYN/$dir/*BHT*tomo  | gawk '{print "r "$0"; ch kcmpnm BHT; w over"}END{print "q"}' | sac
ls ../SYN/$dir/*BHZ*tomo  | gawk '{print "r "$0"; ch kcmpnm BHZ; w over"}END{print "q"}' | sac
else
ls ../DATA/wav/$dir/*BHE*tomo | gawk '{print "r "$0"; ch kcmpnm BHE; w over"}END{print "q"}' | sac
ls ../DATA/wav/$dir/*BHN*tomo | gawk '{print "r "$0"; ch kcmpnm BHN; w over"}END{print "q"}' | sac
ls ../DATA/wav/$dir/*BHZ*tomo | gawk '{print "r "$0"; ch kcmpnm BHZ; w over"}END{print "q"}' | sac
ls ../SYN/$dir/*BHE*tomo  | gawk '{print "r "$0"; ch kcmpnm BHE; w over"}END{print "q"}' | sac
ls ../SYN/$dir/*BHN*tomo  | gawk '{print "r "$0"; ch kcmpnm BHN; w over"}END{print "q"}' | sac
ls ../SYN/$dir/*BHZ*tomo  | gawk '{print "r "$0"; ch kcmpnm BHZ; w over"}END{print "q"}' | sac
fi

fi
done
#rm half_duration.out
