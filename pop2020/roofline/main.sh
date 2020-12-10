#!/bin/bash

cd $(dirname $0)

# Note: calculate roofline for rank 1 only
ranks="1"

# export ncu-rep files to CSV
for f in time-{flops,dram}/summit*/2020*/profile/*.ncu-rep; do
  csv_out="${f%.ncu-rep}.csv"
  if [ ! -f "$csv_out" ]; then
    ncu --import "$f" --units base --csv > "$csv_out"
  fi
done

runtags="nvroof_ncuroof_region-dgdxy nvroof_ncuroof_region-dzv_ak fused-roof2_ncuroof_region-dgdxy"

out_dir="out"
mkdir -p "$out_dir"

# For each run tag, join the CSV files with flops and dram for each
# desired ranks into one CSV
for runtag in $runtags; do
  # Note: can get header from any for the ranks
  for i in $ranks; do
    out_csv="$out_dir/$runtag-$i.csv"
    cp time-dram/*$runtag/2020*/profile/*-$i.csv "$out_csv"
    tail -n +2 time-flops/*$runtag/2020*/profile/*-$i.csv \
       | grep -v cycles >> "$out_csv"
  done
done

./gene_rooflines.py out/*.csv
