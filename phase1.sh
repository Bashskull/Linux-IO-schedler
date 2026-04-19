#!/bin/bash
# Phase 1: Benchmark 3 schedulers across 2 workloads
# Run as: sudo bash phase1.sh
# Takes ~15 minutes total

DEVICE="/dev/sda"          # Your VM disk — change to /dev/nvme0n1 if needed
OUTPUT_DIR="/home/sarthak/results"
mkdir -p "$OUTPUT_DIR"

SCHEDULERS=("none" "bfq" "kyber" "mq-deadline")

run_bench() {
    local WORKLOAD=$1   # name
    local FIO_ARGS=$2   # fio parameters

    for SCHED in "${SCHEDULERS[@]}"; do
        echo "Running $WORKLOAD with scheduler: $SCHED ..."

        # Switch scheduler
        echo "$SCHED" | sudo tee /sys/block/$(basename $DEVICE)/queue/scheduler > /dev/null

        # Flush cache for clean results
        sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

        # Run benchmark, save JSON
        sudo fio \
            --filename=$DEVICE \
            --direct=1 \
            --ioengine=libaio \
            --time_based=1 \
            --runtime=20 \
            --group_reporting \
            --output-format=json \
            --output="$OUTPUT_DIR/${WORKLOAD}_${SCHED}.json" \
            $FIO_ARGS

        echo "  Done. Saved: ${WORKLOAD}_${SCHED}.json"
    done
}

echo "=== Phase 1: I/O Scheduler Benchmarks ==="
echo ""

# Workload 1: Latency-sensitive (like a DB query — small random reads, low queue depth)
run_bench "W1_latency" "--name=W1 --rw=randread --bs=4k --iodepth=1 --numjobs=1 --lat_percentiles=1 --percentile_list=50:99:99.9"

# Workload 2: Interference (foreground app + heavy background I/O happening simultaneously)
# This is where BFQ/Kyber shine vs None — expect big P99 differences here
run_bench "W2_interference" "--name=fg --rw=randread --bs=4k --iodepth=1 --numjobs=1 --lat_percentiles=1 --percentile_list=50:99:99.9 \
                             --name=bg --rw=randwrite --bs=4k --iodepth=64 --numjobs=2 --lat_percentiles=1 --percentile_list=50:99:99.9"
echo ""
echo "=== All done! Now run: python3 phase1_parse.py ==="
