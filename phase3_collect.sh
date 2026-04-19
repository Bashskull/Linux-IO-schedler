#!/bin/bash
# Phase 3a: Collect 3 more workloads for ML training data
# Run as: sudo bash phase3_collect.sh
# Takes ~15 minutes

DEVICE="/dev/sda"
OUTPUT_DIR="$HOME/results/phase3"
mkdir -p "$OUTPUT_DIR"

run_bench() {
    local LABEL=$1
    local SCHED=$2
    local FIO_ARGS=$3

    echo "  $LABEL ($SCHED)..."
    echo "$SCHED" | sudo tee /sys/block/$(basename $DEVICE)/queue/scheduler > /dev/null
    sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

    sudo fio \
        --filename=$DEVICE \
        --direct=1 \
        --ioengine=libaio \
        --time_based=1 \
        --runtime=20 \
        --group_reporting \
        --output-format=json \
        --output="$OUTPUT_DIR/${LABEL}_${SCHED}.json" \
        $FIO_ARGS

    echo "    Done."
}

SCHEDULERS=("none" "bfq" "kyber" "mq-deadline")

echo "=== Phase 3a: Collecting Additional Workloads ==="
echo ""

# W3: Sequential reads — like streaming a large file, no competition
# Schedulers should all perform similarly here (SSD-bound, not CPU-bound)
echo "── W3: Sequential reads ──"
for SCHED in "${SCHEDULERS[@]}"; do
    run_bench "W3_sequential" "$SCHED" \
        "--name=W3 --rw=read --bs=128k --iodepth=32 --numjobs=1
         --lat_percentiles=1 --percentile_list=50:99:99.9"
done

# W4: Heavy random writes only — write-dominated workload
# Tests schedulers under pure write pressure (no reads to protect)
echo ""
echo "── W4: Heavy random writes ──"
for SCHED in "${SCHEDULERS[@]}"; do
    run_bench "W4_randwrite" "$SCHED" \
        "--name=W4 --rw=randwrite --bs=4k --iodepth=64 --numjobs=2
         --lat_percentiles=1 --percentile_list=50:99:99.9"
done

# W5: Many concurrent readers — high queue depth, throughput-focused
# Like a database serving many simultaneous queries
echo ""
echo "── W5: Concurrent readers (QD=32) ──"
for SCHED in "${SCHEDULERS[@]}"; do
    run_bench "W5_concurrent" "$SCHED" \
        "--name=W5 --rw=randread --bs=4k --iodepth=32 --numjobs=4
         --lat_percentiles=1 --percentile_list=50:99:99.9"
done

echo ""
echo "=== Done! Now run: python3 phase3_ml.py ==="
