# MARL search `my_run` — results summary and follow-up training

## Verification (no auto-promotion)

- There is **no** `promoted/` directory: no candidate passed `gate_candidate` with a valid checkpoint in any generation.
- `branch_state.json` still points `parent` at `bootstrap/` with `metrics: null`.
- `candidates.jsonl` contains no `"promoted": true`.

## Best completed experiment (manual pick)

**`generation_001/01_baseline_continuation`** (baseline, no LLM config changes):

- `error`: null; train and eval return codes 0.
- Eval: `contact_rate` 0, `hook_contact_rate` 0, `mean_progress` ~0.49, `mean_reward` ~137.
- Failed **search gates** only on `offtrack_rate` 1.0 (`smoke_control_offtrack` / `offtrack_near_total`) — metric definition vs threshold, not a training crash.
- **Checkpoint:** `generation_001/01_baseline_continuation/screen/models/final_model_torch.pt`  
  (`metrics_screen.json` also references `best_model_torch.pt` when present; this run lists `final_model_torch.pt`.)

## Why later generations failed

`screen/train_stderr.log` (e.g. gen 5 baseline) shows **pyglet** `IndexError: list index out of range` in `display.get_default_screen()` — typical of **headless SSH** (no X11 display) when the env renders `state_pixels`.

**Mitigation (in repo):** set `RACING_HEADLESS_PYGLET=1` before `train.py` / `evaluate.py` so pyglet uses its headless EGL path on **Linux** (requires EGL libraries). On **macOS**, EGL is typically missing—leave the variable **unset** and use a normal desktop session, or use `xvfb-run -a python train.py ...` if you install XQuartz/Xvfb. If headless EGL fails, prefer **`xvfb-run -a python train.py ...`** on batch nodes.

## Long training (recommended)

From repository root, with fresh paths in [`config/marl_followup_my_run.yaml`](../../../../config/marl_followup_my_run.yaml) (`torch_policy_variant_source` is relative to the `config/` directory).

```bash
# Linux headless with EGL: export RACING_HEADLESS_PYGLET=1
# macOS desktop: omit (EGL not available). Linux without EGL: xvfb-run -a python ...

python evaluate.py \
  --model autoresearch/results/research_branches/my_run/generation_001/01_baseline_continuation/screen/models/final_model_torch.pt \
  --config config/marl_followup_my_run.yaml \
  --episodes 5 --no-video \
  --output-json results/marl_followup_my_run/sanity_eval.json

python train.py \
  --config config/marl_followup_my_run.yaml \
  --trainer_backend torch \
  --resume autoresearch/results/research_branches/my_run/generation_001/01_baseline_continuation/screen/models/final_model_torch.pt \
  --resume_mode policy_only \
  --timesteps_add 2000000 \
  --seed 42
```

Increase `--timesteps_add` or adjust `training.total_timesteps` as needed. Ensure `./models/marl_followup_my_run`, `./logs/marl_followup_my_run`, and `./results/marl_followup_my_run` are writable.

## Re-running autoresearch on a headless machine

Pass `RACING_HEADLESS_PYGLET=1` into the environment for subprocesses that invoke `train.py`, or use `xvfb-run -a python -m autoresearch.marl_search_loop ...`.
