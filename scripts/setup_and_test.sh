#!/bin/bash
# Setup and test 4 CloudLab Utah nodes for distributed training.
# Usage: bash scripts/setup_and_test.sh

set -e

NODES=(
  "rl6004@ms1008.utah.cloudlab.us"
  "rl6004@ms1033.utah.cloudlab.us"
  "rl6004@ms1026.utah.cloudlab.us"
  "rl6004@ms1005.utah.cloudlab.us"
)

SSH_KEY="/Users/luorunpeng/.ssh/id_ed25519_cloudlab"
MASTER_IP="10.10.1.2"
MASTER_PORT="23456"
REPO_URL="git@github.com:RunpengLuo/COS568-DistLM-SP26.git"
REPO_DIR="COS568-DistLM-SP26"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10"

# Step 1: Install dependencies on all nodes (skip if already installed)
echo "=== Installing dependencies on all 4 nodes (parallel) ==="
for i in 0 1 2 3; do
  (
    echo "[Node $i] Checking ${NODES[$i]}..."
    ssh $SSH_OPTS "${NODES[$i]}" bash -s <<'INSTALL_EOF'
      export PATH=$HOME/.local/bin:$PATH
      if python3 -c "import torch; import pytorch_transformers" 2>/dev/null; then
        echo "Dependencies already installed, skipping."
      else
        echo "Installing dependencies..."
        sudo apt-get update -qq
        sudo apt-get install -y -qq htop dstat python3-pip
        grep -q 'HOME/.local/bin' ~/.bashrc || echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
        pip install -q torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu
        pip install -q numpy scipy scikit-learn tqdm pytorch_transformers apex
      fi
INSTALL_EOF
    echo "[Node $i] Done."
  ) &
done
wait
echo "=== All nodes ready ==="

# Step 1.5: Clone repo and download GLUE data on all nodes (skip if exists)
echo ""
echo "=== Cloning repo and downloading GLUE data (parallel) ==="
for i in 0 1 2 3; do
  (
    echo "[Node $i] Checking repo and data..."
    ssh $SSH_OPTS "${NODES[$i]}" bash -s <<REPO_EOF
      export PATH=\$HOME/.local/bin:\$PATH
      # Clone repo if not exists
      if [ ! -d ~/$REPO_DIR ]; then
        echo "Cloning repo..."
        git clone $REPO_URL ~/$REPO_DIR
      else
        echo "Repo already exists, pulling latest..."
        cd ~/$REPO_DIR && git pull
      fi
      # Download GLUE data if not exists
      if [ ! -d ~/glue_data/RTE ]; then
        echo "Downloading GLUE data..."
        cd ~/$REPO_DIR && python3 download_glue_data.py --data_dir ~/glue_data --tasks RTE
      else
        echo "GLUE data already exists, skipping."
      fi
REPO_EOF
    echo "[Node $i] Done."
  ) &
done
wait
echo "=== Repo and data ready on all nodes ==="

# Step 2: Check experimental network IPs
echo ""
echo "=== Checking experimental network IPs ==="
for i in 0 1 2 3; do
  echo -n "[Node $i] "
  ssh $SSH_OPTS "${NODES[$i]}" "ip addr show | grep '10.10.1' | awk '{print \$2}'"
done

# Step 3: Test network connectivity from node-0
echo ""
echo "=== Testing network connectivity (ping from node-0) ==="
ssh $SSH_OPTS "${NODES[0]}" bash -s <<'PING_EOF'
  for ip in 10.10.1.1 10.10.1.2 10.10.1.3 10.10.1.4; do
    ping -c 2 -W 2 $ip > /dev/null 2>&1 && echo "  $ip reachable" || echo "  $ip UNREACHABLE"
  done
PING_EOF

# Step 4: Test torch.distributed all_reduce
echo ""
echo "=== Testing torch.distributed (all_reduce on 4 nodes) ==="
for i in 0 1 2 3; do
  ssh $SSH_OPTS "${NODES[$i]}" "export PATH=\$HOME/.local/bin:\$PATH && python3 -c \"
import os, subprocess
# Auto-detect the interface with 10.10.1.* IP
iface = subprocess.check_output(\\\"ip -o addr show | grep '10.10.1'\\\", shell=True).decode().split()[1]
os.environ['GLOO_SOCKET_IFNAME'] = iface
import torch, torch.distributed as dist
dist.init_process_group(backend='gloo', init_method='tcp://${MASTER_IP}:${MASTER_PORT}', world_size=4, rank=${i}, timeout=__import__('datetime').timedelta(seconds=60))
t = torch.tensor([${i}], dtype=torch.float32)
dist.all_reduce(t, op=dist.ReduceOp.SUM)
print(f'Rank ${i}: all_reduce = {t.item()} (expected 6.0)', flush=True)
dist.destroy_process_group()
\"" &
done
wait

echo ""
echo "=== Done. If all ranks printed 6.0, your cluster is ready. ==="
