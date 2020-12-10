# GENE scaling analysis

To re-generate the results, run mkplots.sh. The plots and related data
files will be generated under the plots subdirectory.

# GENE roofline analysis

See `roofline/README.md`.

# Structure of run directories

Run directories have the following naming structure:
`{machine_name}_{compiler}_{make}_{nodes}node_{procs_per_node}ppn_{gpus_per_node}gpn_box-{box}_{para_str}_bk{nblocks}{version}`
For runs with profiling or roofline enabled, additional values may be appended,
to indicate the type of profiler, and for nsight compute / roofline, the kernel
being profiled. The version string is a tag suplied by the performance engineer
to help keep track of what the run represents.

Both `{box}` and `{para_str}` are of the form `s{S}v{V}w{W}x{X}y{Y}z{Z}`, where
the values represent the number of points (`{box}`) or MPI parallelization
(`{para_str}`) in that direction of the 6 dimensional domain.

Within the run directory, you will see the following:

- build: information about the options and environment used to build GENE
- {datetime}: output from a specific run
  - out: GENE output files and performance output
  - profile: Profiler output files
- submit.sh: submit script for the target machine
- submit-config.sh: sourced by submit.sh
- job-\*.sh: machine specific script running gene with mpi
- submit-config.sh: sourced by job-\*.sh
- profiler-wrapper.sh: script used to execute profiler, based on values in run-config.sh

The job can be re-run by uploading to the target machine, re-building the
executable based on the description in `build/`, and updating the `EXE_PATH`
and `RUN_PATH` variables in submit-config.sh and run-config.sh based on the new
upload location and exe location. Note that while GENE is open source, it
requires first accepting the license and requesting access, so we are not able
to directly distribute executables or source code here.
