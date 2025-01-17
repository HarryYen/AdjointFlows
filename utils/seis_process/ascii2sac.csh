#!/bin/csh

# checks arguments
if( $1 =~ "" ) then
  echo "Usage: ascii2sac.csh filenames"
  exit 1
endif

set SCRIPT_PATH = `realpath "$0"`
set SCRIPT_DIR = `dirname "$SCRIPT_PATH"`

foreach file ($*)
   $SCRIPT_DIR/asc2sac $file
end
