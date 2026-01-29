#!/bin/bash
#SBATCH --job-name=eval_stage1_watch
#SBATCH --output=/scratch/%u/eval_logs/stage1/eval_watcher_%j.out
#SBATCH --error=/scratch/%u/eval_logs/stage1/eval_watcher_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --partition=boost_usr_prod

module purge
module load python/3.11
module load gcc/12
source /leonardo_scratch/large/userexternal/mpeppas0/venvs/nanotron_env/bin/activate

export EVAL_DATA_DIR=${EVAL_DATA_DIR:-/scratch/$USER/eval_data/stage1}
export EVAL_RESULTS_DIR=${EVAL_RESULTS_DIR:-/scratch/$USER/eval_results/stage1}
export EVAL_LOG_DIR=${EVAL_LOG_DIR:-/scratch/$USER/eval_logs/stage1}

mkdir -p "$EVAL_LOG_DIR"

CONFIG_PATH=${CONFIG_PATH:-config/config_stage1.yaml}

read -r CKPT_PATH CKPT_INTERVAL <<< "$(python - <<'PY'
import os
import yaml
cfg_path = os.environ.get('CONFIG_PATH', 'config/config_stage1.yaml')
with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
ckpt_path = os.path.expandvars(cfg['checkpoints']['checkpoints_path'])
ckpt_interval = cfg['checkpoints']['checkpoint_interval']
print(ckpt_path, ckpt_interval)
PY
)"

CKPT_PATH=${CHECKPOINTS_PATH:-$CKPT_PATH}
CKPT_INTERVAL=${CKPT_INTERVAL:-$CKPT_INTERVAL}

export CKPT_PATH

if [ -z "$CKPT_PATH" ] || [ -z "$CKPT_INTERVAL" ]; then
  echo "Missing checkpoints_path or checkpoint_interval"
  exit 1
fi

EVAL_INTERVAL=$((2 * CKPT_INTERVAL))
POLL_INTERVAL=${POLL_INTERVAL:-300}
STATE_FILE="$EVAL_LOG_DIR/submitted_steps_stage1.txt"

: > "$STATE_FILE"

echo "Watching $CKPT_PATH (interval=$EVAL_INTERVAL, poll=${POLL_INTERVAL}s)"

while true; do
  mapfile -t ENTRIES < <(python - <<'PY'
import os
import re
ckpt_dir = os.environ.get('CKPT_PATH')
if not ckpt_dir or not os.path.isdir(ckpt_dir):
    raise SystemExit
pattern = re.compile(r"(?:step|iter|ckpt|checkpoint)[-_]?(\d+)")
steps = {}
for name in os.listdir(ckpt_dir):
    path = os.path.join(ckpt_dir, name)
    if not (os.path.isdir(path) or os.path.isfile(path)):
        continue
    match = pattern.search(name)
    if match:
        step = int(match.group(1))
    else:
        nums = re.findall(r"\d+", name)
        if not nums:
            continue
        step = int(nums[-1])
    steps[step] = path
for step in sorted(steps):
    print(f"{step}|{steps[step]}")
PY
  )

  for entry in "${ENTRIES[@]}"; do
    step=${entry%%|*}
    path=${entry#*|}

    if (( step % EVAL_INTERVAL != 0 )); then
      continue
    fi

    result_dir="$EVAL_RESULTS_DIR/by_checkpoint/step_${step}"
    if [ -f "$result_dir/tier1_health.json" ] && [ -f "$result_dir/tier2_ppl.json" ] && [ -f "$result_dir/tier3_cf.json" ]; then
      continue
    fi

    if grep -qx "$step" "$STATE_FILE"; then
      continue
    fi

    if [ ! -e "$path" ]; then
      continue
    fi
    if [ -d "$path" ] && [ -z "$(ls -A "$path" 2>/dev/null)" ]; then
      continue
    fi
    if [ -f "$path" ] && [ ! -s "$path" ]; then
      continue
    fi

    sbatch --export=ALL,CHECKPOINT_PATH="$path",STEP="$step" slurm/eval_stage1_every2ckpt.sbatch
    echo "$step" >> "$STATE_FILE"
  done

  sleep "$POLL_INTERVAL"
done
