#!/bin/bash

# Script to facilitate selectively profiling certain MPI ranks. PROFILER must
# be set to one of "nvprof", "tau", "gprof", or "perf". For "gprof", profiling
# will be enabled for all ranks (since it's compiled in and each process will
# dump the info anyway - this script insures each process will use a different
# output file). For the other profilers, either PROFILE_RANK_MOD or
# PROFILE_RANK_START/END env vars must be set to enable profiling, otherwise
# all ranks will be run without profiling. For perf, PPN must be set to the
# number of processes per node (used to set per process memory limits on
# profiling data).

#echo PWD=$(pwd)
mkdir -p profile

SPACK_HOME=${SPACK_HOME:-$HOME/fusion/spack}

if [ -n $PMIX_RANK ]; then
    # works on summit, generic linux w/ openmpi
    RANK=$PMIX_RANK
elif [ -n $ALPS_APP_PE ]; then
    # for titan
    RANK=$ALPS_APP_PE
elif [ -n $OMPI_COMM_WORLD_RANK ]; then
    RANK=$OMPI_COMM_WORLD_RANK
else
    echo "Can't determine rank from ENV:"
    env
    exit 1
fi

export PROFILE_WRAPPER_RANK=$RANK

# NOTE: bash doesn't do short circuit evaluation
profile_rank=0
if [ -z "$PROFILE_RANK_MOD" ]; then
    PROFILE_RANK_MOD=0
fi
if [ "$PROFILER" = "gprof" ]; then
    profile_rank=1
elif [ "$PROFILE_RANK_MOD" -gt 0 ]; then
    if [ $((RANK % PROFILE_RANK_MOD)) = 0 ]; then
        profile_rank=1
    fi
elif [ -n "$PROFILE_RANK_START" ]; then
    if [ $RANK -ge $PROFILE_RANK_START -a $RANK -le $PROFILE_RANK_END ]; then
        profile_rank=1
    fi
fi

if [ $profile_rank = 1 ]; then
    echo profile wrapper rank=$RANK profile ON $(date) profiler=$PROFILER
    if [ $PROFILER = "nvprof" ]; then
        set -x
        nvprof --log-file profile/nvprof-log.%q{PROFILE_WRAPPER_RANK} \
          -o profile/nvprof-timeline.%q{PROFILE_WRAPPER_RANK}.nvvp \
          --profile-from-start off \
          "$@"
        set +x
    elif [ $PROFILER = "nvcpu" ]; then
        set -x
        nvprof --log-file profile/nvcpu-log.%q{PROFILE_WRAPPER_RANK} \
          -o profile/nvcpu-timeline.%q{PROFILE_WRAPPER_RANK}.nvvp \
          --profile-from-start on \
          "$@"
        set +x
    elif [ $PROFILER = "nsys" ]; then
        which nsys
        if [ $? -ne 0 ]; then
            module load nsight-systems
        fi
        # -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
        # --export=sqlite \
        #-c cudaProfilerApi \
        # --stats=true \
        # --trace=mpi \
        # -c nvtx -p t_loop \
        # --trace=cuda,nvtx -s none \
        nsys --version
        set -x
        nsys profile -c cudaProfilerApi \
          --output=profile/nsys-timeline.%q{PROFILE_WRAPPER_RANK} \
          --trace=cuda,nvtx -s none \
          --kill=none \
          "$@"
        set +x
    elif [ $PROFILER = "ncu" ]; then
        which ncu
        if [ $? -ne 0 ]; then
            module load nsight-compute
        fi
        EXTRA_ARGS=""
        if [ -n "$PROFILE_KERNEL" ]; then
            EXTRA_ARGS+=" -k $PROFILE_KERNEL"
        fi
        if [ -n "$PROFILE_NVTX_REGION" ]; then
            EXTRA_ARGS+=" --nvtx-include $PROFILE_NVTX_REGION/"
        else
            EXTRA_ARGS+=" --nvtx-include t_loop/"
        fi
        set -x
        #ncu --nvtx --profile-from-start=off $EXTRA_ARGS \
        #  -o profile/ncu "$@"

        ncu --nvtx $EXTRA_ARGS \
          --profile-from-start off --csv --units base \
          --metrics "sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,gpu__time_duration.sum" \
          -o profile/ncutime-${RANK} \
          "$@"

        set +x
    elif [ $PROFILER = "nvroof" ]; then
        EXTRA_ARGS=""
        if [ -n "$PROFILE_KERNEL" ]; then
            EXTRA_ARGS+=" --kernels $PROFILE_KERNEL"
        fi

        nvprof --log-file profile/nvroof-log.%q{PROFILE_WRAPPER_RANK} \
          $EXTRA_ARGS \
          -o profile/nvroof-timeline.%q{PROFILE_WRAPPER_RANK} \
          --metrics flop_count_dp \
          --metrics dram_read_transactions \
          --metrics dram_write_transactions \
          --profile-from-start off \
          "$@"
        set +x
    elif [ $PROFILER = "ncuroof" ]; then
        which ncu
        #which nv-nsight-cu-cli
        if [ $? -ne 0 ]; then
            module load nsight-compute/2020.1.2
            #module load nsight-compute/2019.5.0
        fi
        EXTRA_ARGS=""
        if [ -n "$PROFILE_KERNEL" ]; then
            EXTRA_ARGS+=" -k $PROFILE_KERNEL"
        fi
        if [ -n "$PROFILE_NVTX_REGION" ]; then
            EXTRA_ARGS+=" --nvtx-include $PROFILE_NVTX_REGION/"
        else
            EXTRA_ARGS+=" --nvtx-include t_loop/"
        fi

        # Time
        metrics="sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,"

        # DP
        metrics+="sm__sass_thread_inst_executed_op_dadd_pred_on.sum,sm__sass_thread_inst_executed_op_dfma_pred_on.sum,sm__sass_thread_inst_executed_op_dmul_pred_on.sum,"
        metrics+="dram__bytes.sum"

        # SP
        #metrics+="sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
        #sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
        #sm__sass_thread_inst_executed_op_fmul_pred_on.sum,"

        # HP
        #metrics+="sm__sass_thread_inst_executed_op_hadd_pred_on.sum,\
        #sm__sass_thread_inst_executed_op_hfma_pred_on.sum,\
        #sm__sass_thread_inst_executed_op_hmul_pred_on.sum,"

        # Tensor Core
        #metrics+="sm__inst_executed_pipe_tensor.sum,"

        # DRAM, L2 and L1
        #metrics+="dram__bytes.sum"
        #lts__t_bytes.sum,\
        #l1tex__t_bytes.sum"

        set -x
        ncu --nvtx $EXTRA_ARGS \
          --profile-from-start off --csv --units base \
          --metrics "sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,dram__bytes.sum" \
          -o profile/ncuroof-time-${RANK} \
          "$@"
        #  --metrics "sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,sm__sass_thread_inst_executed_op_dadd_pred_on.sum,sm__sass_thread_inst_executed_op_dfma_pred_on.sum,sm__sass_thread_inst_executed_op_dmul_pred_on.sum" \
        #--metrics "sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second" \
        #--metrics "sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,dram__bytes.sum" \

        #ncu --nvtx $EXTRA_ARGS \
        #  --profile-from-start off --csv --units base \
        #  --metrics "sm__sass_thread_inst_executed_op_dadd_pred_on.sum,sm__sass_thread_inst_executed_op_dfma_pred_on.sum,sm__sass_thread_inst_executed_op_dmul_pred_on.sum,dram__bytes.sum" \
        #  -o profile/ncuroof-dops-mem-${RANK} \
        #  "$@"

        #  --section SpeedOfLight_RooflineChart \
        #  --section LaunchStats \
        #nv-nsight-cu-cli --nvtx $EXTRA_ARGS \
        #   --profile-from-start off --csv --units base \
        #   -o profile/ncuroof-${RANK} \
        #   --metrics $metrics \
        #   "$@"

        #  --profile-from-start off --csv --units base \
        #--metrics $metrics \
        #  --section-folder "$OLCF_NSIGHT_SYSTEMS_ROOT/sections" \
        #  --section-folder $HOME/fusion/roofline-on-nvidia-gpus/ncu-sections \
        #  --set default \
        #  --section SpeedOfLight_RooflineChart \
        #  --section InstructionStats \
        #  --section MemoryWorkloadAnalysis \
        #  --section SpeedOfLight_HierarchicalDoubleRooflineChart \
        set +x
    elif [ $PROFILER = "gprof" ]; then
        set -x
        GMON_OUT_PREFIX="gprof.$(hostname).$RANK" "$@"
        set +x
    elif [ $PROFILER = "tau" ]; then
        which tau_exec
        if [ $? -ne 0 ]; then
            if [ -f "${SPACK_HOME}/share/spack/setup-env.sh" ]; then
              source "${SPACK_HOME}/share/spack/setup-env.sh"
              spack load -r tau
            fi
        fi
        which tau_exec
        set -x
        tau_exec "$@"
        set +x
    elif [ $PROFILER = "perf" ]; then
        MLOCK_KB=$(cat /proc/sys/kernel/perf_event_mlock_kb)
        MMAP_PAGES=$((MLOCK_KB / PPN / 2))
        set -x
        perf record -g -o "perf.$(hostname).$RANK" -m ${MMAP_PAGES}K "$@"
        set +x
    elif [ $PROFILER = "scalasca" ]; then
        set -x
        export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--cuda --mpp=mpi"
        scalasca -analyze "$@"
        set +x
    else
        echo "ERROR: unknown profiler '$PROFILER'"
        exit 1
    fi
    echo profile wrapper rank=$RANK profile ON $(date) DONE
else
    echo profile wrapper rank=$RANK profile OFF $(date)
    set -x
    "$@"
    set +x
    echo profile wrapper rank=$RANK profile OFF $(date) DONE
fi
