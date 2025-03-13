#!/bin/csh

set script_dir = `dirname $0`

# checks arguments
if( $1 =~ "" ) then
  echo "Usage: ascii2sac.csh filenames"
  exit 1
endif

foreach file ($*)
  $script_dir/asc2sac $file
end
