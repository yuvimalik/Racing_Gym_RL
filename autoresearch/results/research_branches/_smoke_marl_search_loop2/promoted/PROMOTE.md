# Promoted MARL experiment

Warm-start checkpoint used during search: `/private/tmp/dummy_marl_warm.pt`
Promoted post-search checkpoint: `promoted/checkpoint.pt`

## Continue training (example)

```bash
python train.py --config promoted/effective_config.yaml --trainer_backend torch \
  --resume promoted/checkpoint.pt --resume_mode policy_only --timesteps_add 2000000
```

Adjust `--timesteps_add` or use your full `training.total_timesteps` workflow as needed.
