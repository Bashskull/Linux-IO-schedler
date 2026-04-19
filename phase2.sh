#!/bin/bash
# Phase 2: Parameter Tuning
# Tests 8 scheduler parameters across multiple values
# Run as: sudo bash phase2.sh
# Takes ~25-30 minutes

DEVICE="/dev/sda"
OUTPUT_DIR="$HOME/results/phase2"
mkdir -p "$OUTPUT_DIR"

# We always use the W2 interference workload (most interesting from Phase 1)
# Foreground: latency-sensitive reads | Background: heavy writes
run_bench() {
    local LABEL=$1    # e.g. "bfq_slice_idle_8"
    local SCHED=$2    # scheduler name

    echo "  Running: $LABEL ..."

    # Switch scheduler
    echo "$SCHED" | sudo tee /sys/block/$(basename $DEVICE)/queue/scheduler > /dev/null

    # Flush cache
    sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

    sudo fio \
        --filename=$DEVICE \
        --direct=1 \
        --ioengine=libaio \
        --time_based=1 \
        --runtime=20 \
        --group_reporting=0 \
        --output-format=json \
        --output="$OUTPUT_DIR/${LABEL}.json" \
        --name=fg --rw=randread --bs=4k --iodepth=1 --numjobs=1 \
        --name=bg --rw=randwrite --bs=4k --iodepth=64 --numjobs=2 \
        --lat_percentiles=1 --percentile_list=50:99:99.9

    echo "    Done."
}

set_param() {
    local PATH=$1
    local VALUE=$2
    echo "$VALUE" | sudo tee "$PATH" > /dev/null 2>&1
}

echo "=== Phase 2: Parameter Tuning ==="
echo ""

# ── BFQ Parameters ────────────────────────────────────────────────────────────
echo "── BFQ Parameters ──"
BFQ_PATH="/sys/block/$(basename $DEVICE)/queue/iosched"

# 1. slice_idle: how long BFQ waits for more I/O from same process (ms)
# Lower = less waiting = more throughput. Higher = better latency isolation.
for VAL in 0 4 8 16; do
    echo "$BFQ" | sudo tee /sys/block/$(basename $DEVICE)/queue/scheduler > /dev/null 2>&1
    echo "bfq" | sudo tee /sys/block/$(basename $DEVICE)/queue/scheduler > /dev/null
    set_param "$BFQ_PATH/slice_idle" "$VAL"
    run_bench "bfq_slice_idle_${VAL}" "bfq"
done

# 2. low_latency: BFQ's special mode that detects and protects latency-sensitive apps
# 0 = off (pure throughput), 1 = on (protects interactive/latency apps)
for VAL in 0 1; do
    echo "bfq" | sudo tee /sys/block/$(basename $DEVICE)/queue/scheduler > /dev/null
    set_param "$BFQ_PATH/low_latency" "$VAL"
    run_bench "bfq_low_latency_${VAL}" "bfq"
done

# 3. timeout_sync: max time (ms) a process can hold the disk before being preempted
for VAL in 64 124 250; do
    echo "bfq" | sudo tee /sys/block/$(basename $DEVICE)/queue/scheduler > /dev/null
    set_param "$BFQ_PATH/timeout_sync" "$VAL"
    run_bench "bfq_timeout_sync_${VAL}" "bfq"
done

# ── Kyber Parameters ──────────────────────────────────────────────────────────
echo ""
echo "── Kyber Parameters ──"
KYBER_PATH="/sys/block/$(basename $DEVICE)/queue/iosched"

# 4. read_lat_nsec: target read latency in nanoseconds
# Lower = Kyber prioritizes reads more aggressively
# Default is 2000000 (2ms). Try lower to protect reads under interference.
for VAL in 500000 1000000 2000000 4000000; do
    echo "kyber" | sudo tee /sys/block/$(basename $DEVICE)/queue/scheduler > /dev/null
    set_param "$KYBER_PATH/read_lat_nsec" "$VAL"
    run_bench "kyber_read_lat_$(( VAL / 1000000 ))ms" "kyber"
done

# 5. write_lat_nsec: target write latency in nanoseconds
# Higher = writes get less priority = reads protected more
# Default is 10000000 (10ms).
for VAL in 2000000 10000000 100000000; do
    echo "kyber" | sudo tee /sys/block/$(basename $DEVICE)/queue/scheduler > /dev/null
    set_param "$KYBER_PATH/write_lat_nsec" "$VAL"
    run_bench "kyber_write_lat_$(( VAL / 1000000 ))ms" "kyber"
done

# ── MQ-Deadline Parameters ────────────────────────────────────────────────────
echo ""
echo "── MQ-Deadline Parameters ──"
MQD_PATH="/sys/block/$(basename $DEVICE)/queue/iosched"

# 6. read_expire: deadline (ms) before a read MUST be served, no matter what
# Lower = reads served faster but less throughput
for VAL in 100 500 2000; do
    echo "mq-deadline" | sudo tee /sys/block/$(basename $DEVICE)/queue/scheduler > /dev/null
    set_param "$MQD_PATH/read_expire" "$VAL"
    run_bench "mqd_read_expire_${VAL}ms" "mq-deadline"
done

# 7. write_expire: deadline (ms) before a write MUST be served
for VAL in 1000 5000 10000; do
    echo "mq-deadline" | sudo tee /sys/block/$(basename $DEVICE)/queue/scheduler > /dev/null
    set_param "$MQD_PATH/write_expire" "$VAL"
    run_bench "mqd_write_expire_${VAL}ms" "mq-deadline"
done

# 8. writes_starved: how many read batches before writes get a turn
# Higher = reads more protected, writes more starved
for VAL in 1 2 4; do
    echo "mq-deadline" | sudo tee /sys/block/$(basename $DEVICE)/queue/scheduler > /dev/null
    set_param "$MQD_PATH/writes_starved" "$VAL"
    run_bench "mqd_writes_starved_${VAL}" "mq-deadline"
done

echo ""
echo "=== Phase 2 Done! Run: python3 phase2_parse.py ==="
