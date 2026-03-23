#!/bin/bash
# Task 2a: Distributed training with gather/scatter (4 nodes, 1 epoch)
# Run locally — SSHes into all 4 CloudLab nodes in parallel.
# Usage: bash scripts/run_task2a.sh

set -e

NODES=(
  "rl6004@ms1008.utah.cloudlab.us"
  "rl6004@ms1033.utah.cloudlab.us"
  "rl6004@ms1026.utah.cloudlab.us"
  "rl6004@ms1005.utah.cloudlab.us"
)

SSH_KEY="/Users/luorunpeng/.ssh/id_ed25519_cloudlab"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10"
MASTER_IP="10.10.1.2"
MASTER_PORT="12345"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Copy code to all nodes
echo "=== Syncing code to all nodes ==="
for node in "${NODES[@]}"; do
  scp $SSH_OPTS "$SCRIPT_DIR/run_glue.py" "$SCRIPT_DIR/utils_glue.py" "${node}:~/" &
done
wait

# Launch all 4 ranks in parallel
echo "=== Launching Task 2a (gather/scatter) on 4 nodes ==="
for i in 0 1 2 3; do
  ssh $SSH_OPTS "${NODES[$i]}" "cd ~ && export PATH=\$HOME/.local/bin:\$PATH && python3 run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-cased \
    --task_name RTE \
    --do_train \
    --do_eval \
    --data_dir \$HOME/glue_data/RTE \
    --max_seq_length 128 \
    --per_device_train_batch_size 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --output_dir /tmp/RTE/ \
    --overwrite_output_dir \
    --master_ip $MASTER_IP \
    --master_port $MASTER_PORT \
    --world_size 4 \
    --local_rank $i \
    --comm_method gather_scatter 2>&1 | tee /tmp/RTE/rank_\${i}.log" 2>&1 | sed "s/^/[Rank $i] /" &
done
wait

# Fetch results from rank 0
RESULTS_DIR="$SCRIPT_DIR/results/task2a"
mkdir -p "$RESULTS_DIR"
echo "=== Fetching results to $RESULTS_DIR ==="
scp $SSH_OPTS "${NODES[0]}:/tmp/RTE/*" "$RESULTS_DIR/"
for i in 0 1 2 3; do
  scp $SSH_OPTS "${NODES[$i]}:/tmp/RTE/rank_${i}.log" "$RESULTS_DIR/" 2>/dev/null || true
done

echo "=== Task 2a complete ==="
