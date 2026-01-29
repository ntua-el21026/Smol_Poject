# Stage 1 Evaluation (Minimal)

Stage 1 evaluation runs **Tier1 + Tier2 + Tier3 together** for a given checkpoint. All outputs live on Leonardo shared FS (not committed).

## Structure

- `evaluation/run_eval_stage1.py` — single entrypoint for all tiers.
- `evaluation/task_suites/` — minimal configs for Tier2 + Tier3.

## What we evaluate (3 tiers)

### Tier 1 — Health & Integrity

- numerical stability (NaN/Inf events, loss variance, gradient spikes if available)
- performance/throughput (tokens/sec, step time, dataloader stalls)
- packing sanity (sequence length ~2048, low padding rate, EOS statistics)

Output: `tier1_health.json`

### Tier 2 — Held-out LM Metrics (NLL / Perplexity)

- `general_web`: broad web / Q&A style text
- `code`: code distribution
- `books_newslike`: curated long-form + factual/technical text

Output: `tier2_ppl.json` (per-slice + macro average)

### Tier 3 — Downstream Early-Signal Suite (CF-only)

- HellaSwag (CF), PIQA (CF), WinoGrande (CF), ARC-Easy (CF), OpenBookQA (CF)
- Scoring: choose option with highest normalized log-likelihood

Output: `tier3_cf.json` (per-task + macro average)

## Cadence (every 2 checkpoints)

Cadence rule:
```
Evaluate when step % (2 * checkpoint_interval) == 0
```
With `checkpoint_interval = 2000`, evaluate at **4000, 8000, 12000, 16000, 20000, ...**

## Outputs (shared FS only)

Defaults (override with env vars):
- `EVAL_DATA_DIR=/scratch/$USER/eval_data/stage1`
- `EVAL_RESULTS_DIR=/scratch/$USER/eval_results/stage1`
- `EVAL_LOG_DIR=/scratch/$USER/eval_logs/stage1`

Per-checkpoint layout:
```
$EVAL_RESULTS_DIR/by_checkpoint/step_<STEP>/
  tier1_health.json
  tier2_ppl.json
  tier3_cf.json
```

## Leonardo quick start

1) Set shared FS defaults (optional if already exported):
```
export EVAL_DATA_DIR=/scratch/$USER/eval_data/stage1
export EVAL_RESULTS_DIR=/scratch/$USER/eval_results/stage1
export EVAL_LOG_DIR=/scratch/$USER/eval_logs/stage1
```

2) Run eval for a single checkpoint (also prepares cache if missing):
```
python evaluation/run_eval_stage1.py \
  --checkpoint_path /scratch/$USER/ckpts/stage1/step_00004000 \
  --step 4000
```

3) Start the watcher (polls checkpoints and submits GPU eval jobs):
```
sbatch slurm/eval_stage1_watcher.sbatch
```

Notes:
- Eval data is cached in `EVAL_DATA_DIR` on first run (no repo artifacts).
- Training data is on `/tmp`; eval data/results/logs stay on shared FS.
- The watcher reads `config/config_stage1.yaml` for `checkpoints_path` and `checkpoint_interval` (override with `CHECKPOINTS_PATH`, `CKPT_INTERVAL`, `POLL_INTERVAL`).
- If your checkpoints are not HF-compatible, convert them or set `LIGHTEVAL_CMD` to run a Nanotron-capable LightEval and emit `tier2_ppl.json` + `tier3_cf.json` into the results directory.
