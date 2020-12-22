#!/bin/bash

set -e

rundir=$(dirname $0)
cd "$rundir"

source submit-config.sh

module purge >/dev/null 2>&1
for m in $(cat build/modules.txt); do
    module load $m >/dev/null 2>&1
done

jobid=$(date +%Y%m%d_%H%M%S)
mkdir "$jobid"

module -t list >"$jobid"/modules.txt 2>&1

ln -s "$EXE_PATH" "$jobid"/gene

cd "$jobid"
ln -s ../job-$MPIRUN.sh .
cp ../parameters .
for f in ../files/*; do
  ln -s "$f" .
done
mkdir out

if [ $GPN -gt 0 -a $PPN -gt 6 ]; then
    # TODO: refactor summit specific setup
    alloc_flags="smt1 gpumps"
else
    alloc_flags="smt1"
fi

EXTRA_BSUB_ARGS=""
if [ -n "$QUEUE" ]; then
    EXTRA_BSUB_ARGS+=" -q $QUEUE"
fi

(set -x;
bsub -P $PROJECT -J ${NAME}-$jobid -o job-output.txt -nnodes $NODES \
    -W $WALLTIME -alloc_flags "$alloc_flags" \
    $EXTRA_BSUB_ARGS \
    job-$MPIRUN.sh 2>&1
) 2>bsub-command.txt >bsub-output.txt

cat bsub-command.txt
cat bsub-output.txt

jobid=$(grep '^Job' bsub-output.txt | sed -r 's/Job <([0-9]*)>.*/\1/'  2>/dev/null)
echo $jobid > jobid.txt
cat jobid.txt
