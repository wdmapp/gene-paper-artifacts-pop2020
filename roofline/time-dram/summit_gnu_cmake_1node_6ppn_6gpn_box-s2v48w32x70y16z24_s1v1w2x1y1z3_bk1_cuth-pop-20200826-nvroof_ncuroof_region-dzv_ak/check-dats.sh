#!/bin/bash

# Check .dat files for obvious signs of a bad run.

rundir="$1"

rval=0

if [ ! -d "$rundir"/out ]; then
    echo "ERROR: out dir not found at '$rundir/out'"
    exit 1
fi

grep -q NaN "$rundir"/out/*.dat
if [ $? -eq 0 ]; then
    echo "ERROR: NaN found in output file"
    rval=1
fi

grep -q '^ *-' "$rundir"/out/profiles*.dat
if [ $? -eq 0 ]; then
    echo "ERROR: negative x value in profile data"
    rval=1
fi

exit $rval
