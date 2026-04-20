#!/usr/bin/env bash
# Example only: copy checkpoints/logs FROM the Prime instance TO your laptop (run rsync on your laptop).
# Replace USER, HOST, and local DEST with your values.
#
#   mkdir -p ~/prime_racing_backup
#   rsync -avz -e ssh USER@HOST:~/Racing_Gym_RL/artifacts/prime_marl_2car/ ~/prime_racing_backup/
#
# Or from the instance, push to S3 / Hugging Face / scp to another host — do this BEFORE deleting the instance.

cat <<'TXT'
=== Artifact sync (run from your laptop) ===

  mkdir -p ~/prime_racing_backup
  rsync -avz -e ssh USER@PRIME_HOST:~/Racing_Gym_RL/artifacts/prime_marl_2car/ ~/prime_racing_backup/

Smoke run artifacts (if you ran smoke only):

  rsync -avz -e ssh USER@PRIME_HOST:~/Racing_Gym_RL/artifacts/prime_marl_smoke/ ~/prime_racing_backup_smoke/

Budget run (config/prime_marl_2car_budget.yaml):

  rsync -avz -e ssh USER@PRIME_HOST:~/Racing_Gym_RL/artifacts/prime_marl_2car_budget/ ~/prime_racing_backup_budget/

10-car preflight (config/prime_marl_10car_preflight.yaml):

  rsync -avz -e ssh USER@PRIME_HOST:~/Racing_Gym_RL/artifacts/prime_marl_10car_preflight/ ~/prime_racing_backup_10car_preflight/

10-car long run (config/prime_marl_10car_16m.yaml):

  rsync -avz -e ssh USER@PRIME_HOST:~/Racing_Gym_RL/artifacts/prime_marl_10car_16m/ ~/prime_racing_backup_10car_16m/

10-car 5-loop videos only (tiled + broadcast exports):

  rsync -avz -e ssh USER@PRIME_HOST:~/Racing_Gym_RL/artifacts/prime_marl_10car_16m/results/5loop_videos/*.mp4 ~/prime_racing_backup_10car_5loop_videos/
  rsync -avz -e ssh USER@PRIME_HOST:~/Racing_Gym_RL/artifacts/prime_marl_10car_16m/results/5loop_videos/*.json ~/prime_racing_backup_10car_5loop_videos/

~16h A100 run (config/prime_marl_2car_budget_fast.yaml, launcher scripts/prime_launch_16h_nohup.sh):

  rsync -avz -e ssh USER@PRIME_HOST:~/Racing_Gym_RL/artifacts/prime_marl_2car_budget_fast/ ~/prime_racing_backup_budget_fast/

Before deleting the instance: confirm copies, then stop the machine in the Prime dashboard and note usage cost on the billing / home page.

TXT
