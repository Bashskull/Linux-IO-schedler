#!/usr/bin/env python3
"""
Phase 4: Adaptive I/O Scheduler Daemon
=======================================
Runs in the background, samples disk stats every 5 seconds,
extracts workload features, predicts optimal scheduler,
and switches it automatically.

Run as: sudo python3 phase4_daemon.py
Stop with: Ctrl+C
"""

import time
import subprocess
import os
import sys
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
import numpy as np

DEVICE      = "sda"
SCHED_FILE  = f"/sys/block/{DEVICE}/queue/scheduler"
STAT_FILE   = f"/sys/block/{DEVICE}/stat"
INTERVAL    = 5       # seconds between checks
LOG_FILE    = Path.home() / "results" / "phase4_daemon.log"

# ── Retrain the model from Phase 3 data ───────────────────────────────────────
# (Same training data as phase3_ml.py — we just inline it here)

TRAINING_DATA = [
    # read_ratio, queue_depth, block_size_kb, num_jobs, has_interference, label
    (1.0,  1,   4,  1, 0, "kyber"),        # W1: latency-sensitive
    (0.3, 33,   4,  3, 1, "bfq"),          # W2: interference
    (1.0, 32, 128,  1, 0, "kyber"),        # W3: sequential
    (0.0, 64,   4,  2, 0, "mq-deadline"), # W4: heavy writes
    (1.0, 32,   4,  4, 0, "none"),         # W5: concurrent readers
]

X = np.array([row[:5] for row in TRAINING_DATA])
y = np.array([row[5]  for row in TRAINING_DATA])

clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)

# ── Helpers ───────────────────────────────────────────────────────────────────

def read_stat():
    """
    Read /sys/block/sda/stat — kernel updates this every second.
    Fields (from kernel docs):
      0: reads completed
      1: reads merged
      2: sectors read
      3: ms spent reading
      4: writes completed
      5: writes merged
      6: sectors written
      7: ms spent writing
      8: I/Os in progress (queue depth proxy)
      9: ms spent doing I/Os
    """
    fields = Path(STAT_FILE).read_text().split()
    return [int(f) for f in fields]

def get_current_scheduler():
    content = Path(SCHED_FILE).read_text().strip()
    # Format: "none mq-deadline [bfq] kyber" — active one is in brackets
    for part in content.split():
        if part.startswith("[") and part.endswith("]"):
            return part[1:-1]
    return content.split()[0]

def switch_scheduler(sched):
    try:
        with open(SCHED_FILE, "w") as f:
            f.write(sched)
        return True
    except PermissionError:
        print("ERROR: Need root. Run with: sudo python3 phase4_daemon.py")
        sys.exit(1)

def log(msg):
    timestamp = time.strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ── Feature extraction from live disk stats ───────────────────────────────────

def extract_features(stat_before, stat_after, elapsed):
    """
    Derive the 5 model features from two consecutive /sys/block/sda/stat reads.
    This is what makes the daemon 'intelligent' — it observes real I/O behavior
    and maps it to the same feature space the model was trained on.
    """
    reads_delta   = stat_after[0]  - stat_before[0]   # completed reads
    writes_delta  = stat_after[4]  - stat_before[4]   # completed writes
    sectors_read  = stat_after[2]  - stat_before[2]   # sectors read (512B each)
    queue_depth   = stat_after[8]                      # I/Os in progress right now

    total_ios = reads_delta + writes_delta
    if total_ios == 0:
        # No I/O happening — default to kyber (lightweight, good default)
        return None

    # read_ratio: what fraction of I/O is reads
    read_ratio = reads_delta / total_ios

    # block_size_kb: average request size in KB
    if reads_delta > 0:
        avg_block_kb = (sectors_read * 512) / reads_delta / 1024
    else:
        avg_block_kb = 4  # default

    # num_jobs proxy: estimate concurrency from queue depth
    # Low QD = 1 job, high QD = many jobs
    if queue_depth <= 1:
        num_jobs = 1
    elif queue_depth <= 8:
        num_jobs = 2
    elif queue_depth <= 32:
        num_jobs = 3
    else:
        num_jobs = 4

    # has_interference: reads AND writes happening simultaneously
    has_interference = 1 if (reads_delta > 10 and writes_delta > 10) else 0

    return [read_ratio, queue_depth, avg_block_kb, num_jobs, has_interference]

# ── Main daemon loop ──────────────────────────────────────────────────────────

def run():
    log("=" * 55)
    log("Adaptive I/O Scheduler Daemon starting")
    log(f"Device: /dev/{DEVICE} | Check interval: {INTERVAL}s")
    log(f"Model: Decision Tree ({len(TRAINING_DATA)} training workloads)")
    log("=" * 55)

    current_sched = get_current_scheduler()
    log(f"Current scheduler: {current_sched}")

    prev_stat    = read_stat()
    switch_count = 0
    check_count  = 0

    try:
        while True:
            time.sleep(INTERVAL)
            check_count += 1

            curr_stat = read_stat()
            features  = extract_features(prev_stat, curr_stat, INTERVAL)
            prev_stat = curr_stat

            if features is None:
                log(f"Check #{check_count}: No I/O detected — keeping {current_sched}")
                continue

            read_ratio, queue_depth, block_kb, num_jobs, has_interf = features

            # Model prediction
            X_live = np.array(features).reshape(1, -1)
            predicted = clf.predict(X_live)[0]

            feature_str = (f"read%={read_ratio:.0%} QD={queue_depth:.0f} "
                          f"BS={block_kb:.0f}KB jobs={num_jobs} interf={has_interf}")

            if predicted != current_sched:
                log(f"Check #{check_count}: [{feature_str}]")
                log(f"  → Switching: {current_sched} → {predicted}")
                switch_scheduler(predicted)
                current_sched = predicted
                switch_count += 1
            else:
                log(f"Check #{check_count}: [{feature_str}] → keeping {current_sched}")

    except KeyboardInterrupt:
        log(f"\nDaemon stopped. Made {switch_count} switches in {check_count} checks.")
        log(f"Log saved: {LOG_FILE}")

if __name__ == "__main__":
    if os.geteuid() != 0:
        print("ERROR: Run as root: sudo python3 phase4_daemon.py")
        sys.exit(1)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    run()
