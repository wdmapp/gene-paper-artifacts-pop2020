#!/bin/bash

set -e
source ../run-config.sh
set +e

COMPRESS=xz

extra_args=""
# for summit, see https://www.olcf.ornl.gov/for-users/system-user-guides/summit/summit-user-guide/#cuda-aware-mpi
if [ $GPN -gt 0 ]; then
    extra_args+=' --smpiargs="-gpu"'
fi

WRAPPER=""
if [ -n "$PROFILE_RANK_START" -o -n "$PROFILE_RANK_MOD" ]; then
    WRAPPER=../profile-wrapper.sh
fi

export MEMORY_PER_CORE GPU_MEMORY_PER_CORE

echo "Starting js_task_info $(date)"
set -x
jsrun -n $NODES -r 1 -a $PPN -c $PPN -g $GPN \
    -l gpu-cpu -d packed $extra_args \
    js_task_info 2>&1 | sort >task_info.txt
set +x
echo "End js_task_info $(date)"

echo "Start jsrun $(date)"
set -x
jsrun -n $NODES -r 1 -a $PPN -c $PPN -g $GPN \
    -l gpu-cpu -d packed $extra_args \
    $WRAPPER ./gene >stdout.txt 2>&1
set +x
echo "End jsrun $(date)"

$COMPRESS stdout.txt
for f in out/perfout.*.txt; do
  $COMPRESS $f
done

if [ $KEEP_OUTPUT = 'False' ]; then
    outsize=$(du -sh out)
    echo -n "Cleaning output (size $outsize)..."
    # delete output files bigger than 1 MB
    find out -size +1M -type f -exec rm \{\} \;
    echo "DONE"
fi
