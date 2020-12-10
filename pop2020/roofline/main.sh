#!/bin/bash

set -e
set -x

cd $(dirname $0)

DEBUG="echo "

# Note: calculate roofline for rank 1 only
ranks="1"
#ranks=$(seq 0 5)

# export ncu-rep files to CSV
for f in time-{flops,dram}/summit*/profile/*.ncu-rep; do
  csv_out="${f%.ncu-rep}.csv"
  if [ ! -f "$csv_out" ]; then
    $DEBUG ncu --import "$f" --units base --csv "$csv_out"
  fi
done

runtags="nvroof_ncuroof_region-dgdxy nvroof_ncuroof_region-dzv_ak fused-roof2_ncuroof_region-dgdxy"

out_dir="out"
mkdir -p "$out_dir"

# for each rank, join the CSV files with flops and dram into one csv
for runtag in $runtags; do
  out_csv="$out_dir/$runtag.csv"
  head -n1 time-dram/*$runtag/2020*/profile/*-0.csv > "$out_csv"
  for i in $ranks; do
    tail -n +2 time-dram/*$runtag/2020*/profile/*-$i.csv >> "$out_csv"
    tail -n +2 time-flops/*$runtag/2020*/profile/*-$i.csv \
       | grep -v cycles >> "$out_csv"
  done
done

./gene_rooflines.py out/*.csv
