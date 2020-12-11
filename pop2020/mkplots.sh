#!/bin/bash

cd $(dirname $0)

min=8
max=512

plotname="summit 2020pop scaling $min:$max"

detail=""
dpi="160"

node_pattern="$min"
if [ $min -eq 1 ]; then
  i=8
else
  i=$((2*min));
fi
while [ $i -lt $max ]; do
  node_pattern="$node_pattern,$i"
  i=$((2*i));
done
node_pattern="$node_pattern,$max"

if [ -n "$detail" ]; then
  plotname+=" detail-$detail"
fi

outdir="plots/$plotname"
mkdir -p "$outdir"

RUNS="$(eval ls -d summit_gnu_*_{${node_pattern}}node_* | sort -n -t_ -k4 | tr '\n' ' ')"

if [ -n "$detail" ]; then
  ./ht_perf.py $RUNS -n --show-speedups \
    --legend-loc 'upper right' --dpi $dpi -r 2560x1440 \
    --title "GENE $plotname" -o "$outdir"/scaling.pdf -c "$outdir"/scaling.csv \
    -u $detail --per-ts
else
  ./ht_perf.py $RUNS -n \
    -o "$outdir"/scaling.pdf -c "$outdir"/scaling.csv -l "$outdir"/scaling.tex \
    --legend-loc 'upper right' --dpi $dpi -r 2560x1440 --per-ts --mpi \
    --annotate-total --font-size 12
  #--title "GENE $plotname" 
  # --show-speedups --annotate-total --annotate-bars
fi

i=$min
while [ $i -le $max ]; do
  ./ht_perf.py summit_*${i}node* > "$outdir"/compare_${i}.txt
  i=$((i*2))
done

for run in $RUNS; do
  [[ $run =~ _([0-9]+)node_ ]]
  nodes=${BASH_REMATCH[1]}
  gpu="_gpu"
  [[ $run =~ _0gpn_ ]] && gpu=""
  cp -v $run/parameters "$outdir"/parameters_${nodes}${gpu}
done

./parabox_table.py -n "$outdir" > "$outdir"/table.txt
./parabox_table.py -n -g "$outdir" > "$outdir"/table-gpu.txt
