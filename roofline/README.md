# GENE Roofline analysis

For each kernel (dgdxy, dzv, fused dgdxy), nsight-compute was run twice, once
to collect dram transfers and cycles (to calculate elapsed time), and once to
collect instruction counts. This is because of a bug that results in cycle
skew when doing instruction counts.

Scripts were used to export the ncu profiles as CSV, and combine them into a
single set of CSV files for each MPI rank. These are then run through a python
script, [gene\_roofline.py](gene_roofline.py), to produce the roofline plot
used in the paper. For the analysis, only rank 1 was used.

To reproduce the results based on ncu-rep files, the [main.sh](main.sh) script
can be run.

The methology and code is build on work by Charlene Yang et al.  See
[roofline-on-nvidia-gpus](https://gitlab.com/NERSC/roofline-on-nvidia-gpus/-/tree/roofline-hackathon-2020)
for the code and for methodology, see [C. Yang, T. Kurth, and S. Williams,
Hierarchical Roofline analysis for GPUs: Accelerating performance optimization
for the NERSC‐9 Perlmutter system, Concurrency and Computation: Practice and
Experience, e5547, 2019](https://doi.org/10.1002/cpe.5547).
