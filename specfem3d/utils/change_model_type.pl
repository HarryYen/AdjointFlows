#!/usr/bin/perl

# this script changes the model_type in inputs/Par_file
# Hsin-Hua Huang, 5 May 2014, IES

use Time::Local;
use Getopt::Std;
use POSIX;

sub Usage{
print STDERR <<END;

Usage:   change_model_type.pl  [-d|-a|-e|-g|-t]
         Changes SIMULATION_TYPE in inputs/Par_file
         -d -- change model type to default
         -a -- change model type to aniso
         -e -- change model type to external
         -g -- change model type to gll
         -t -- change model type to tomo
END
exit(1);
}

@ARGV == 1 or Usage();
if(!getopts('daegt')) {die(" check input arguments\n");}

open(IN,"DATA/Par_file");
@vfm=<IN>;
close(IN);

foreach $vfm (@vfm){
  if($vfm=~/MODEL/){
    if(${opt_d}){
      print "Changed model type to default in Par_file \n";
      $vfm=~s/= aniso/= default/;
      $vfm=~s/= external/= default/;
      $vfm=~s/= gll/= default/;
      $vfm=~s/= tomo/= default/;
    }
    elsif(${opt_a}){
      print "Changed model type to aniso in Par_file \n";
      $vfm=~s/= default/= aniso/;
      $vfm=~s/= external/= aniso/;
      $vfm=~s/= gll/= aniso/;
      $vfm=~s/= tomo/= aniso/;
    }
    elsif(${opt_e}){
      print "Changed model type to external in Par_file \n";
      $vfm=~s/= default/= external/;
      $vfm=~s/= aniso/= external/;
      $vfm=~s/= gll/= external/;
      $vfm=~s/= tomo/= external/;
    }
    elsif(${opt_g}){
      print "Changed model type to gll in Par_file \n";
      $vfm=~s/= default/= gll/;
      $vfm=~s/= aniso/= gll/;
      $vfm=~s/= external/= gll/;
      $vfm=~s/= tomo/= gll/;
    }
    elsif(${opt_t}){
      print "Changed model type to tomo in Par_file \n";
      $vfm=~s/= default/= tomo/;
      $vfm=~s/= aniso/= tomo/;
      $vfm=~s/= external/= tomo/;
      $vfm=~s/= gll/= tomo/;
    }
  }
}

open(OUT,">DATA/Par_file");
foreach $vfm (@vfm){
  print OUT "$vfm";
}
close(OUT);
