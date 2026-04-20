# Prime Pod Setup

## Purpose

Fast, repeatable setup for a Prime Intellect training pod without burning time on environment drift.

Use this for world-model runs that need:
- repo code from `world-model`
- replay manifests and `.npz` data synced manually
- local checkpoint warm-starts synced manually

## Recommended Hardware

- Preferred single-GPU remote run: `A100 80GB`, non-spot
- Do not use spot for multi-hour training runs
- Do not use multi-GPU unless the run already works on one GPU

## Pod Creation

From local machine:

```bash
prime pods create --id <availability-id>
prime pods status <pod-id>
```

Wait until:
- `Status: ACTIVE`
- SSH target is populated

## SSH

From local WSL shell:

```bash
ssh -i ~/.ssh/prime_pi_key.pem ubuntu@<pod-ip>
```

## Minimal Pod Environment

Run on the pod:

```bash
cd ~
git clone <repo-url>
cd Racing_Gym_RL
git checkout world-model
sudo apt update
sudo apt install -y python3.10-venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "numpy>=1.26,<2.0"
pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python matplotlib pyyaml wandb tensorboard
mkdir -p /tmp/mplconfig
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvidia-smi
```

Required success condition:

```text
torch.cuda.is_available() == True
```

Known-good torch result on the A100 pod:

```text
2.5.1+cu121 12.1 True
```

## Sync Required Non-Git Files

From local machine:

Replay data:

```bash
rsync -avz -e "ssh -i ~/.ssh/prime_pi_key.pem" /mnt/c/Users/yuv05/OneDrive/Documents/GitHub/Racing_Gym_RL/results/world_model/replay/ ubuntu@<pod-ip>:~/Racing_Gym_RL/results/world_model/replay/
```

Warm-start checkpoint directory:

```bash
rsync -avz -e "ssh -i ~/.ssh/prime_pi_key.pem" /mnt/c/Users/yuv05/OneDrive/Documents/GitHub/Racing_Gym_RL/models/world_model/P4_d4_main_telemetry_horizon/ ubuntu@<pod-ip>:~/Racing_Gym_RL/models/world_model/P4_d4_main_telemetry_horizon/
```

## Verify Required Files On Pod

Run on pod:

```bash
ls ~/Racing_Gym_RL/results/world_model/replay/d4_main_train_manifest.json
ls ~/Racing_Gym_RL/results/world_model/replay/d4_main_val_manifest.json
ls ~/Racing_Gym_RL/models/world_model/P4_d4_main_telemetry_horizon/rssm_sequence_epoch_005.pt
```

## Launch Pattern

Run on pod:

```bash
cd ~/Racing_Gym_RL
source .venv/bin/activate
MPLCONFIGDIR=/tmp/mplconfig nohup python world_model_train.py \
  --config config/world_model_cluster_e3_diverse_horizon.yaml \
  --train-manifest results/world_model/replay/d4_main_train_manifest.json \
  --val-manifest results/world_model/replay/d4_main_val_manifest.json \
  --init-checkpoint models/world_model/P4_d4_main_telemetry_horizon/rssm_sequence_epoch_005.pt \
  --run-name P5_d4_main_telemetry_a100 \
  --epochs 5 \
  --batch-log-every 50 \
  > p5.log 2>&1 &
```

Monitor:

```bash
tail -f ~/Racing_Gym_RL/p5.log
```

## Pull Results Back

From local machine:

```bash
rsync -avz -e "ssh -i ~/.ssh/prime_pi_key.pem" ubuntu@<pod-ip>:~/Racing_Gym_RL/models/world_model/P5_d4_main_telemetry_a100/ /mnt/c/Users/yuv05/OneDrive/Documents/GitHub/Racing_Gym_RL/models/world_model/P5_d4_main_telemetry_a100/
rsync -avz -e "ssh -i ~/.ssh/prime_pi_key.pem" ubuntu@<pod-ip>:~/Racing_Gym_RL/results/world_model/artifacts/hallucination/P5_d4_main_telemetry_a100/ /mnt/c/Users/yuv05/OneDrive/Documents/GitHub/Racing_Gym_RL/results/world_model/artifacts/hallucination/P5_d4_main_telemetry_a100/
```

## Terminate Pod

From local machine:

```bash
prime pods terminate <pod-id>
```
