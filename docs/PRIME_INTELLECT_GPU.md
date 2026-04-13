# Prime Intellect and cloud GPU training (2-car MARL)

This repo’s Torch trainer already enables **cuDNN benchmark**, **TF32**, and **AMP on CUDA** (`train.py`). **MultiCarRacing-v0 cannot use `num_envs > 1`** (pyglet); `train.py` forces a single vectorized env. Throughput is from **GPU PPO updates**, not parallel rollouts.

- Platform: [Prime Intellect](https://www.primeintellect.ai/) — Lab, on-demand GPUs, billing in the [dashboard](https://app.primeintellect.ai/dashboard/home).

## 0. Prime Lab instance checklist (on-demand A100 80GB)

Do these in the Prime web app before or while SSH is open (you cannot complete billing or instance creation from this repo alone):

1. **Price check**: Open [on-demand GPUs](https://app.primeintellect.ai/dashboard/on-demand-gpus) and note **$/hr** for **A100 80GB**. Compute `max_hours ≈ budget_usd / price_per_hour` and reserve ~0.5–1 h for clone, [`scripts/prime_install_deps.sh`](../scripts/prime_install_deps.sh), and smoke.
2. **Launch**: Create a **Linux** instance with **A100 80GB**, CUDA driver, **Python 3.9+**.
3. **Persistent disk**: Attach or mount storage so `./artifacts` (or the whole repo directory) survives restarts; configs write under `./artifacts/...` (see table in §2).
4. **SSH**: Add your **public** SSH key in Prime account settings if needed; use the `ssh user@host …` command from the instance panel.
5. **After training**: Run the **rsync** examples from [`scripts/prime_sync_artifacts_example.sh`](../scripts/prime_sync_artifacts_example.sh) from your **laptop**, then **stop or delete** the instance and confirm spend on the [dashboard home](https://app.primeintellect.ai/dashboard/home).

## 1. Budget and GPU choice

- **Estimate hours**: `hours ≈ budget_usd / price_per_gpu_hour` (use the price shown for your selected GPU in Prime’s UI).
- **Recommendation**: One **mid-range** GPU (e.g. L4 / RTX 4090 class) is enough for this workload; **A100 80GB** is optional (extra headroom for larger `ppo.batch_size`). Multi-GPU DDP is not wired from `train.py`’s CLI today.
- **Artifacts**: Checkpoints are large; ensure enough **disk** for `save_freq` intervals over the full run.

## 2. Config files

| File | Purpose |
|------|---------|
| [`config/prime_marl_2car_long.yaml`](../config/prime_marl_2car_long.yaml) | Long production run: **32M** stream timesteps, `eval_freq` 100k, `save_freq` 200k, `fail_fast` off, same reward as `multi_car_marl_config.yaml`. |
| [`config/prime_marl_2car_budget.yaml`](../config/prime_marl_2car_budget.yaml) | **8M** stream timesteps with eval/save scaled like the long run; default for **~$10–20** runs after you calibrate (see §1 and example sequence below). |
| [`config/prime_marl_2car_budget_fast.yaml`](../config/prime_marl_2car_budget_fast.yaml) | Same **8M** stream goal as budget, with **fewer evals/saves**, `batch_size: 2048`, `n_epochs: 2`; artifacts under `./artifacts/prime_marl_2car_budget_fast/`. Launcher: [`scripts/prime_train_budget_fast.sh`](../scripts/prime_train_budget_fast.sh). |
| [`config/prime_marl_2car_smoke.yaml`](../config/prime_marl_2car_smoke.yaml) | **~12k** stream steps + one mid-run eval; sanity check before spending credits. |
| [`config/prime_marl_2car_budget_fast_smoke.yaml`](../config/prime_marl_2car_budget_fast_smoke.yaml) | Short smoke using the **fast** PPO/eval settings; artifacts under `./artifacts/prime_marl_budget_fast_smoke/`. |

Paths default to `./artifacts/prime_marl_2car/`, `./artifacts/prime_marl_2car_budget/`, and `./artifacts/prime_marl_smoke/`. On Prime, mount a **persistent volume** at that path (or at repo root so `./artifacts` is durable).

### Shell helpers (`scripts/`)

After `git clone` and `cd` into the repo on your GPU instance, you can run these in order (all assume **Linux** with a driver; activate nothing manually except where noted):

| Script | Purpose |
|--------|---------|
| [`scripts/prime_verify_env.sh`](../scripts/prime_verify_env.sh) | `nvidia-smi` + `torch.cuda.is_available()`; **exits 1** if PyTorch cannot see CUDA. |
| [`scripts/prime_install_deps.sh`](../scripts/prime_install_deps.sh) | Creates `venv/`, `pip install -r requirements.txt`, `pip install -r requirements_sb3.txt --no-deps`, checks CUDA, then `pip install git+https://github.com/igilitschenski/multi_car_racing.git --no-deps`. |
| [`scripts/prime_train_smoke.sh`](../scripts/prime_train_smoke.sh) | Sets `RACING_HEADLESS_PYGLET=1`, runs [`config/prime_marl_2car_smoke.yaml`](../config/prime_marl_2car_smoke.yaml) (override seed with `PRIME_SEED`). |
| [`scripts/prime_train_long.sh`](../scripts/prime_train_long.sh) | Same headless default, runs [`config/prime_marl_2car_long.yaml`](../config/prime_marl_2car_long.yaml); pass extra CLI args through (e.g. `--resume ... --timesteps_add N`). |
| [`scripts/prime_train_budget.sh`](../scripts/prime_train_budget.sh) | Same headless default, runs [`config/prime_marl_2car_budget.yaml`](../config/prime_marl_2car_budget.yaml); pass extra CLI args through. |
| [`scripts/prime_train_budget_fast.sh`](../scripts/prime_train_budget_fast.sh) | Same as budget script but runs [`config/prime_marl_2car_budget_fast.yaml`](../config/prime_marl_2car_budget_fast.yaml). |
| [`scripts/prime_sync_artifacts_example.sh`](../scripts/prime_sync_artifacts_example.sh) | Prints **rsync** examples for pulling `artifacts/prime_marl_2car/`, smoke, and budget dirs to your laptop before teardown. |

Example sequence from repo root:

```bash
bash scripts/prime_verify_env.sh
bash scripts/prime_install_deps.sh
bash scripts/prime_train_smoke.sh
# Calibrate (optional): time a short run, then edit total_timesteps in prime_marl_2car_budget.yaml if needed.
bash scripts/prime_train_budget.sh
# Or full long run:
# bash scripts/prime_train_long.sh
# On your laptop (see prime_sync_artifacts_example.sh):
# rsync -avz USER@HOST:~/Racing_Gym_RL/artifacts/prime_marl_2car_budget/ ./backup/
```

## 3. Environment variables

- **`RACING_HEADLESS_PYGLET=1`**: set **before** launching Python on **Linux** (Prime, Docker) so pyglet uses headless EGL (see `train.py` header). On **macOS**, this flag often fails with `Library "EGL" not found`; run **without** it for local smoke tests (Cocoa display path).
- If Linux headless EGL is missing: `xvfb-run -a python train.py ...`

## 4. Docker (optional)

From repo root:

```bash
docker build -f Dockerfile.prime -t racing-gym-rl:prime .
docker run --gpus all \
  -e RACING_HEADLESS_PYGLET=1 \
  -v "$(pwd)/artifacts:/workspace/Racing_Gym_RL/artifacts" \
  racing-gym-rl:prime
```

Override the training command:

```bash
docker run --gpus all -e RACING_HEADLESS_PYGLET=1 \
  -v "$(pwd)/artifacts:/workspace/Racing_Gym_RL/artifacts" \
  racing-gym-rl:prime \
  python train.py --config config/prime_marl_2car_long.yaml --seed 42
# Budget run: use config/prime_marl_2car_budget.yaml instead of long.
```

The image installs [`requirements.txt`](../requirements.txt), [`requirements_sb3.txt`](../requirements_sb3.txt) (with `--no-deps`), and `multi_car_racing` from `https://github.com/igilitschenski/multi_car_racing.git` (see [README.md](../README.md)).

## 5. Prime job without Docker (typical steps)

1. Start a **CUDA** GPU instance with Python 3.9+ (match `requirements.txt`).
2. Clone this repository and `cd` into it.
3. Create a venv, `pip install -r requirements.txt`, then `pip install -r requirements_sb3.txt --no-deps`, then install `multi_car_racing` per README (`pip install git+https://github.com/igilitschenski/multi_car_racing.git --no-deps`).
4. Install a **CUDA build of PyTorch** matching the instance if `pip` did not install GPU wheels.
5. Verify: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"`.
6. Run smoke, then **budget** or long job (see §6 and scripts above).
7. Before the instance terminates, **copy** `artifacts/` (see [`scripts/prime_sync_artifacts_example.sh`](../scripts/prime_sync_artifacts_example.sh)) to durable storage, then **stop** the instance and check billing on the dashboard.

## 6. Commands

**Smoke (~12k stream steps, one eval mid-run):**

Linux / Prime:

```bash
export RACING_HEADLESS_PYGLET=1
python train.py --config config/prime_marl_2car_smoke.yaml --seed 0
```

macOS (no EGL): omit `RACING_HEADLESS_PYGLET` or use a normal display.

Smoke success criteria: run completes; `artifacts/prime_marl_smoke/models/best_model_torch.pt` and `torch_ppo_step_6144.pt` exist; subprocess eval runs without pyglet crash. A local smoke run on 2026-04-12 completed on **Apple MPS** (CUDA unavailable on that host) with checkpoints written under `artifacts/prime_marl_smoke/models/`.

**Long run:**

```bash
export RACING_HEADLESS_PYGLET=1
python train.py --config config/prime_marl_2car_long.yaml --seed 42
```

**Budget run (default 8M stream steps, artifacts under `artifacts/prime_marl_2car_budget/`):**

```bash
export RACING_HEADLESS_PYGLET=1
python train.py --config config/prime_marl_2car_budget.yaml --seed 42
```

**Resume** (after copying checkpoints back):

```bash
python train.py --config config/prime_marl_2car_long.yaml \
  --resume artifacts/prime_marl_2car/models/best_model_torch.pt \
  --seed 42
```

Optional continuation without editing YAML total: `--timesteps_add 8000000` (adds stream steps on top of restored counter; see `train.py` help).

**Resume (budget config)** — use the same config file path as the run you started:

```bash
python train.py --config config/prime_marl_2car_budget.yaml \
  --resume artifacts/prime_marl_2car_budget/models/best_model_torch.pt \
  --seed 42
```

## 7. Monitoring and evaluation

- TensorBoard (long): `tensorboard --logdir artifacts/prime_marl_2car/logs`
- TensorBoard (budget): `tensorboard --logdir artifacts/prime_marl_2car_budget/logs`
- After training: `python evaluate.py --model artifacts/prime_marl_2car/models/best_model_torch.pt --config config/prime_marl_2car_long.yaml --episodes 10 --no-video --seed 42` (swap paths/config for the budget run if you trained with `prime_marl_2car_budget.yaml`).

## 8. Optional throughput tuning (CUDA only)

If GPU utilization is low, try increasing `ppo.batch_size` (and optionally `n_steps`) in a **copy** of the config until you approach VRAM limits. Do **not** increase `training.num_envs` for `MultiCarRacing-v0`.
