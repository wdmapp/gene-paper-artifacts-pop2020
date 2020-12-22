# GENE Roofline analysis

For each kernel (dgdxy, dzv, fused dgdxy), nsight-compute was run twice, once
to collect dram transfers and cycles (to calculate elapsed time), and once to
collect instruction counts. This is because of a bug that results in cycle
skew when doing instruction counts.

Scripts were used to export the ncu profiles as CSV, and combine them into a
single set of CSV files for each MPI rank. These are then run through a python
script, `gene_roofline.py`, to produce the roofline plot used in the paper. For
the analysis, only rank 1 was used.

To reproduce the results based on ncu-rep files, the `main.sh` script can be
run.
