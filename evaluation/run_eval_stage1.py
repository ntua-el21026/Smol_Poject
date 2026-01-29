#!/usr/bin/env python
import argparse
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import yaml
except ImportError as exc:  # pragma: no cover - env-level dependency
    raise RuntimeError("PyYAML is required for evaluation configs") from exc

try:
    from datasets import load_dataset, load_from_disk
except ImportError as exc:  # pragma: no cover - env-level dependency
    raise RuntimeError("datasets is required for evaluation") from exc

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover - env-level dependency
    raise RuntimeError("transformers is required for evaluation") from exc

DEFAULT_TOKENIZER = "meta-llama/Meta-Llama-3-8B"
DEFAULT_SEQ_LEN = 2048


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def resolve_shared_dir(env_var: str, default_root: str) -> Path:
    if env_var in os.environ and os.environ[env_var]:
        return Path(os.path.expandvars(os.environ[env_var])).expanduser()
    user = os.environ.get("USER", "unknown")
    return Path(f"/scratch/{user}/{default_root}/stage1")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def is_hf_checkpoint(path: Path) -> bool:
    if path.is_file():
        return path.suffix in {".safetensors", ".bin"}
    if not path.is_dir():
        return False
    config = path / "config.json"
    if not config.exists():
        return False
    for pattern in ("*.safetensors", "pytorch_model.bin", "model.safetensors"):
        if list(path.glob(pattern)):
            return True
    return False


def load_tokenizer(tokenizer_name_or_path: str, checkpoint_path: Path) -> AutoTokenizer:
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    except Exception:
        if checkpoint_path.is_dir():
            tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
        else:
            raise
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_hf_model(checkpoint_path: Path) -> AutoModelForCausalLM:
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_path),
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return model


def _iter_dataset(dataset: Iterable, max_docs: Optional[int]) -> Iterable:
    count = 0
    for example in dataset:
        yield example
        count += 1
        if max_docs is not None and count >= max_docs:
            break


def _select_text(example: dict, text_field: Optional[str]) -> Optional[str]:
    if "input_ids" in example:
        return None
    if text_field and text_field in example:
        return example[text_field]
    for key in ("text", "content", "document"):
        if key in example:
            return example[key]
    return None


def _tokens_from_example(example: dict, tokenizer: AutoTokenizer, text_field: Optional[str]) -> Optional[List[int]]:
    if "input_ids" in example and isinstance(example["input_ids"], list):
        return example["input_ids"]
    text = _select_text(example, text_field)
    if not text or not isinstance(text, str):
        return None
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _load_dataset(cfg: dict):
    if cfg.get("dataset_path"):
        dataset = load_from_disk(cfg["dataset_path"])
        split = cfg.get("split")
        if split and hasattr(dataset, "__getitem__") and hasattr(dataset, "keys") and split in dataset.keys():
            return dataset[split]
        return dataset
    streaming = cfg.get("streaming", True)
    try:
        return load_dataset(
            cfg["hf_dataset"],
            cfg.get("hf_config"),
            split=cfg.get("split", "train"),
            streaming=streaming,
        )
    except Exception:
        if streaming:
            return load_dataset(
                cfg["hf_dataset"],
                cfg.get("hf_config"),
                split=cfg.get("split", "train"),
                streaming=False,
            )
        raise


def prepare_ppl_slice(
    slice_cfg: dict,
    tokenizer: AutoTokenizer,
    seq_len: int,
    out_dir: Path,
) -> Tuple[torch.Tensor, dict]:
    ensure_dir(out_dir)
    tokens_path = out_dir / "packed_tokens.pt"
    meta_path = out_dir / "meta.json"

    if tokens_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("tokenizer_name_or_path") == tokenizer.name_or_path
            and meta.get("sequence_length") == seq_len
        ):
            tokens = torch.load(tokens_path, map_location="cpu")
            return tokens, meta

    dataset = _load_dataset(slice_cfg)
    if not slice_cfg.get("streaming", True) and slice_cfg.get("shuffle", True):
        dataset = dataset.shuffle(seed=slice_cfg.get("seed", 42))
    elif slice_cfg.get("streaming", True) and slice_cfg.get("shuffle", True):
        dataset = dataset.shuffle(seed=slice_cfg.get("seed", 42), buffer_size=10_000)

    target_tokens = int(slice_cfg.get("target_tokens", seq_len * 400))
    target_sequences = max(1, target_tokens // seq_len)
    max_docs = slice_cfg.get("max_docs")

    buffer: List[int] = []
    sequences: List[List[int]] = []
    docs_used = 0

    for example in _iter_dataset(dataset, max_docs=max_docs):
        token_ids = _tokens_from_example(example, tokenizer, slice_cfg.get("text_field"))
        if not token_ids:
            continue
        token_ids = list(token_ids)
        if tokenizer.eos_token_id is not None:
            token_ids.append(tokenizer.eos_token_id)
        buffer.extend(token_ids)
        docs_used += 1
        while len(buffer) >= seq_len:
            sequences.append(buffer[:seq_len])
            buffer = buffer[seq_len:]
            if len(sequences) >= target_sequences:
                break
        if len(sequences) >= target_sequences:
            break

    if not sequences:
        raise RuntimeError(f"No data collected for slice {slice_cfg['name']}")

    tokens = torch.tensor(sequences, dtype=torch.int32)
    meta = {
        "slice_name": slice_cfg["name"],
        "tokenizer_name_or_path": tokenizer.name_or_path,
        "sequence_length": seq_len,
        "target_tokens": target_tokens,
        "actual_tokens": int(tokens.numel()),
        "docs_used": docs_used,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(tokens, tokens_path)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return tokens, meta


def prepare_ppl_slices(cfg: dict, tokenizer: AutoTokenizer, seq_len: int, eval_data_dir: Path):
    slices_dir = eval_data_dir / "ppl_slices"
    ensure_dir(slices_dir)
    output = {}
    for slice_cfg in cfg.get("slices", []):
        slice_dir = slices_dir / slice_cfg["name"]
        tokens, meta = prepare_ppl_slice(slice_cfg, tokenizer, seq_len, slice_dir)
        output[slice_cfg["name"]] = (tokens, meta)
    return output


def _format_prompt_candidate(prompt: str, candidate: str) -> Tuple[str, str]:
    prompt = prompt.strip()
    candidate = candidate.strip()
    if prompt and candidate:
        if not prompt.endswith(tuple(" \n\t")) and not candidate.startswith(tuple(" \n\t,.!?;:")):
            candidate = " " + candidate
    return prompt, candidate


def _build_cf_example(task: str, example: dict) -> Optional[dict]:
    task_lower = task.lower()

    if task_lower == "hellaswag":
        ctx = example.get("ctx")
        if not ctx and "ctx_a" in example:
            ctx = f"{example.get('ctx_a', '')} {example.get('ctx_b', '')}".strip()
        endings = example.get("endings") or example.get("choices")
        label = example.get("label")
        if ctx is None or endings is None or label is None:
            return None
        prompt = ctx
        candidates = list(endings)
        label_idx = int(label)

    elif task_lower == "piqa":
        goal = example.get("goal")
        sol1 = example.get("sol1")
        sol2 = example.get("sol2")
        label = example.get("label")
        if goal is None or sol1 is None or sol2 is None or label is None:
            return None
        prompt = f"Goal: {goal}\nAnswer:"
        candidates = [sol1, sol2]
        label_idx = int(label)

    elif task_lower == "winogrande":
        sentence = example.get("sentence")
        option1 = example.get("option1")
        option2 = example.get("option2")
        answer = example.get("answer")
        if sentence is None or option1 is None or option2 is None or answer is None:
            return None
        if "_" in sentence:
            prefix, suffix = sentence.split("_", 1)
            prompt = prefix
            candidates = [option1 + suffix, option2 + suffix]
        else:
            prompt = sentence
            candidates = [option1, option2]
        label_idx = int(answer) - 1 if str(answer).isdigit() else (0 if answer == "1" else 1)

    elif task_lower in {"arc_easy", "arc-easy", "arc"}:
        stem = None
        choices = None
        if isinstance(example.get("question"), dict):
            stem = example["question"].get("stem")
            choices = example["question"].get("choices")
        if stem is None:
            stem = example.get("question_stem") or example.get("stem")
        if choices is None:
            choices = example.get("choices")
        answer_key = example.get("answerKey") or example.get("answer")
        if stem is None or choices is None or answer_key is None:
            return None
        labels, texts = _normalize_choice_fields(choices)
        prompt = _format_mc_prompt(stem, labels, texts)
        candidates = texts
        label_idx = _label_to_index(answer_key, labels)

    elif task_lower in {"openbookqa", "openbook"}:
        stem = example.get("question_stem") or example.get("stem")
        choices = example.get("choices")
        answer_key = example.get("answerKey") or example.get("answer")
        if stem is None or choices is None or answer_key is None:
            return None
        labels, texts = _normalize_choice_fields(choices)
        prompt = _format_mc_prompt(stem, labels, texts)
        candidates = texts
        label_idx = _label_to_index(answer_key, labels)

    else:
        return None

    prompt, candidates = _sanitize_prompt_candidates(prompt, candidates)
    return {
        "prompt": prompt,
        "candidates": candidates,
        "label": label_idx,
    }


def _normalize_choice_fields(choices) -> Tuple[List[str], List[str]]:
    labels: List[str] = []
    texts: List[str] = []
    if isinstance(choices, dict):
        labels = list(choices.get("label", []))
        texts = list(choices.get("text", []))
    elif isinstance(choices, list):
        for choice in choices:
            labels.append(str(choice.get("label")))
            texts.append(str(choice.get("text")))
    return labels, texts


def _label_to_index(answer_key: str, labels: List[str]) -> int:
    if answer_key in labels:
        return labels.index(answer_key)
    if str(answer_key).isdigit():
        idx = int(answer_key) - 1
        if 0 <= idx < len(labels):
            return idx
    raise ValueError(f"Unable to map answer key {answer_key} to labels {labels}")


def _format_mc_prompt(stem: str, labels: List[str], texts: List[str]) -> str:
    lines = [f"Question: {stem}", "Choices:"]
    for label, text in zip(labels, texts):
        lines.append(f"{label}. {text}")
    lines.append("Answer:")
    return "\n".join(lines)


def _sanitize_prompt_candidates(prompt: str, candidates: List[str]) -> Tuple[str, List[str]]:
    prompt = prompt.strip()
    cleaned = [str(c).strip() for c in candidates]
    return prompt, cleaned


def prepare_cf_task(
    task_cfg: dict,
    eval_data_dir: Path,
) -> Tuple[List[dict], dict]:
    task_name = task_cfg["name"]
    task_dir = eval_data_dir / "cf_tasks" / task_name
    ensure_dir(task_dir)
    data_path = task_dir / "examples.jsonl"
    meta_path = task_dir / "meta.json"

    if data_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        examples = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line]
        return examples, meta

    dataset = _load_dataset(task_cfg)
    if not task_cfg.get("streaming", False):
        dataset = dataset.shuffle(seed=task_cfg.get("seed", 42))

    subset_size = int(task_cfg.get("subset_size", 1000))
    examples: List[dict] = []
    for example in _iter_dataset(dataset, max_docs=task_cfg.get("max_docs")):
        built = _build_cf_example(task_name, example)
        if built is None:
            continue
        examples.append(built)
        if len(examples) >= subset_size:
            break

    if not examples:
        raise RuntimeError(f"No CF examples collected for {task_name}")

    meta = {
        "task": task_name,
        "subset_size": subset_size,
        "actual_size": len(examples),
        "seed": task_cfg.get("seed", 42),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    data_path.write_text("\n".join(json.dumps(ex) for ex in examples), encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return examples, meta


def prepare_cf_tasks(cfg: dict, eval_data_dir: Path):
    output = {}
    defaults = {
        "subset_size": cfg.get("subset_size", 1000),
        "seed": cfg.get("seed", 42),
    }
    for task_cfg in cfg.get("tasks", []):
        merged_cfg = {**defaults, **task_cfg}
        examples, meta = prepare_cf_task(merged_cfg, eval_data_dir)
        output[merged_cfg["name"]] = (examples, meta)
    return output


def _batch_iter(tensor: torch.Tensor, batch_size: int) -> Iterable[torch.Tensor]:
    for start in range(0, tensor.size(0), batch_size):
        yield tensor[start : start + batch_size]


def compute_ppl(
    model: AutoModelForCausalLM,
    tokens: torch.Tensor,
    pad_token_id: int,
    batch_size: int,
    device: torch.device,
    nan_tracker: Dict[str, int],
) -> Tuple[float, int]:
    total_nll = 0.0
    total_tokens = 0

    for batch in _batch_iter(tokens, batch_size=batch_size):
        input_ids = batch.to(device=device, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)  # packed slices: no padding

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:]

        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        )
        loss = loss * shift_mask.reshape(-1)
        batch_nll = loss.sum().item()
        batch_tokens = int(shift_mask.sum().item())

        if not math.isfinite(batch_nll):
            nan_tracker["nan_inf_count"] += 1
            continue

        total_nll += batch_nll
        total_tokens += batch_tokens

    nll = total_nll / max(1, total_tokens)
    return nll, total_tokens


def _prepare_scoring_inputs(
    tokenizer: AutoTokenizer,
    prompt: str,
    candidates: List[str],
    seq_len: int,
) -> Tuple[torch.Tensor, List[Tuple[int, int, int]], List[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    sequences: List[List[int]] = []
    spans: List[Tuple[int, int, int]] = []
    lengths: List[int] = []
    pad_id = tokenizer.pad_token_id

    for candidate in candidates:
        _, candidate_text = _format_prompt_candidate(prompt, candidate)
        candidate_ids = tokenizer(candidate_text, add_special_tokens=False)["input_ids"]

        if len(candidate_ids) >= seq_len:
            candidate_ids = candidate_ids[-seq_len:]
            context_ids = []
        else:
            max_prompt = seq_len - len(candidate_ids)
            context_ids = prompt_ids[-max_prompt:]

        if not context_ids and tokenizer.bos_token_id is not None:
            context_ids = [tokenizer.bos_token_id]
        elif not context_ids and tokenizer.eos_token_id is not None:
            context_ids = [tokenizer.eos_token_id]

        input_ids = context_ids + candidate_ids
        candidate_start = len(context_ids)
        sequences.append(input_ids)
        lengths.append(len(input_ids))
        spans.append((candidate_start, len(candidate_ids), len(candidate_text)))

    max_len = max(len(seq) for seq in sequences)
    input_tensor = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        input_tensor[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

    return input_tensor, spans, lengths


def score_candidates(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    candidates: List[str],
    seq_len: int,
    device: torch.device,
    nan_tracker: Dict[str, int],
) -> List[float]:
    input_ids, spans, lengths = _prepare_scoring_inputs(tokenizer, prompt, candidates, seq_len)
    attention_mask = torch.zeros_like(input_ids, dtype=torch.long)
    for i, length in enumerate(lengths):
        attention_mask[i, :length] = 1  # mask by true length
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    log_probs = F.log_softmax(logits, dim=-1)
    scores: List[float] = []

    for idx, (start, length, char_len) in enumerate(spans):
        if start <= 0 or length == 0:
            scores.append(float("-inf"))
            continue
        token_log_probs = []
        for offset in range(length):
            token_id = input_ids[idx, start + offset]
            token_lp = log_probs[idx, start + offset - 1, token_id].item()
            token_log_probs.append(token_lp)
        total_lp = float(sum(token_log_probs))
        if not math.isfinite(total_lp):
            nan_tracker["nan_inf_count"] += 1
        norm = char_len if char_len > 0 else length
        scores.append(total_lp / max(1, norm))

    return scores


def parse_log_metrics() -> Dict[str, Optional[float]]:
    log_path = os.environ.get("TRAIN_LOG_PATH")
    log_glob = os.environ.get("TRAIN_LOG_GLOB")
    log_dir = os.environ.get("TRAIN_LOG_DIR")

    paths: List[Path] = []
    if log_path:
        paths.append(Path(log_path))
    if log_glob:
        paths.extend(Path(p) for p in Path(".").glob(log_glob))
    if log_dir:
        paths.extend(Path(log_dir).glob("*.out"))
        paths.extend(Path(log_dir).glob("*.log"))

    metrics = {
        "loss_mean": None,
        "loss_std": None,
        "grad_norm_max": None,
        "tokens_per_sec": None,
        "step_time_sec": None,
        "dataloader_wait_pct": None,
    }
    if not paths:
        return metrics

    loss_vals = []
    grad_vals = []
    tps_vals = []
    step_vals = []
    wait_vals = []

    loss_re = re.compile(r"(?:loss|train_loss|lm_loss)[=: ]+([0-9.]+)")
    grad_re = re.compile(r"(?:grad_norm|grad_norm_max)[=: ]+([0-9.]+)")
    tps_re = re.compile(r"(?:tokens_per_sec|tokens/sec)[=: ]+([0-9.]+)")
    step_re = re.compile(r"(?:step_time|step_time_sec|iter_time)[=: ]+([0-9.]+)")
    wait_re = re.compile(r"(?:dataloader_wait_pct|data_wait_pct)[=: ]+([0-9.]+)")

    for path in paths:
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in lines[-2000:]:
            if match := loss_re.search(line):
                loss_vals.append(float(match.group(1)))
            if match := grad_re.search(line):
                grad_vals.append(float(match.group(1)))
            if match := tps_re.search(line):
                tps_vals.append(float(match.group(1)))
            if match := step_re.search(line):
                step_vals.append(float(match.group(1)))
            if match := wait_re.search(line):
                wait_vals.append(float(match.group(1)))

    if loss_vals:
        mean = sum(loss_vals) / len(loss_vals)
        var = sum((x - mean) ** 2 for x in loss_vals) / len(loss_vals)
        metrics["loss_mean"] = mean
        metrics["loss_std"] = math.sqrt(var)
    if grad_vals:
        metrics["grad_norm_max"] = max(grad_vals)
    if tps_vals:
        metrics["tokens_per_sec"] = sum(tps_vals) / len(tps_vals)
    if step_vals:
        metrics["step_time_sec"] = sum(step_vals) / len(step_vals)
    if wait_vals:
        metrics["dataloader_wait_pct"] = sum(wait_vals) / len(wait_vals)

    return metrics


def compute_packing_stats(tokens: torch.Tensor, tokenizer: AutoTokenizer) -> Dict[str, float]:
    sample = tokens[: min(64, tokens.size(0))]
    eos_id = tokenizer.eos_token_id

    seq_len_mean = float(sample.shape[1])  # packed slices: fixed length
    pad_rate = 0.0

    flat = sample.reshape(-1)
    eos_rate = (flat == eos_id).float().mean().item() if eos_id is not None else 0.0

    return {
        "seq_len_mean": seq_len_mean,
        "pad_rate": pad_rate,
        "eos_rate": eos_rate,
    }


def write_json(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def try_lighteval(
    checkpoint_path: Path,
    step: int,
    results_dir: Path,
    tokenizer_name_or_path: str,
    seq_len: int,
) -> Optional[Tuple[dict, dict]]:
    cmd = os.environ.get("LIGHTEVAL_CMD")
    if not cmd:
        return None
    env = os.environ.copy()
    env.update(
        {
            "LIGHTEVAL_CHECKPOINT_PATH": str(checkpoint_path),
            "LIGHTEVAL_RESULTS_DIR": str(results_dir),
            "LIGHTEVAL_STEP": str(step),
            "LIGHTEVAL_TOKENIZER": tokenizer_name_or_path,
            "LIGHTEVAL_SEQ_LEN": str(seq_len),
        }
    )
    log("Running LightEval via LIGHTEVAL_CMD")
    completed = subprocess.run(cmd, shell=True, env=env)
    if completed.returncode != 0:
        raise RuntimeError("LightEval command failed")
    tier2_path = results_dir / "tier2_ppl.json"
    tier3_path = results_dir / "tier3_cf.json"
    if not tier2_path.exists() or not tier3_path.exists():
        raise RuntimeError("LightEval did not produce tier2/tier3 outputs")
    tier2 = json.loads(tier2_path.read_text(encoding="utf-8"))
    tier3 = json.loads(tier3_path.read_text(encoding="utf-8"))
    return tier2, tier3


def run_eval(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint_path)
    step = int(args.step)

    eval_data_dir = resolve_shared_dir("EVAL_DATA_DIR", "eval_data") if args.eval_data_dir is None else Path(args.eval_data_dir)
    eval_results_dir = resolve_shared_dir("EVAL_RESULTS_DIR", "eval_results") if args.eval_results_dir is None else Path(args.eval_results_dir)
    eval_log_dir = resolve_shared_dir("EVAL_LOG_DIR", "eval_logs")

    ensure_dir(eval_data_dir)
    ensure_dir(eval_results_dir)
    ensure_dir(eval_log_dir)

    results_dir = eval_results_dir / "by_checkpoint" / f"step_{step}"
    ensure_dir(results_dir)

    tokenizer = load_tokenizer(args.tokenizer_name_or_path, checkpoint_path)

    ppl_cfg_path = Path(__file__).resolve().parent / "task_suites" / "stage1_ppl_slices.yaml"
    cf_cfg_path = Path(__file__).resolve().parent / "task_suites" / "stage1_cf_core.yaml"

    ppl_cfg = load_yaml(ppl_cfg_path)
    cf_cfg = load_yaml(cf_cfg_path)

    log("Preparing PPL slices")
    ppl_data = prepare_ppl_slices(ppl_cfg, tokenizer, args.seq_len, eval_data_dir)
    log("Preparing CF tasks")
    cf_data = prepare_cf_tasks(cf_cfg, eval_data_dir)

    nan_tracker = {"nan_inf_count": 0}

    lighteval_outputs = try_lighteval(
        checkpoint_path, step, results_dir, args.tokenizer_name_or_path, args.seq_len
    )

    if lighteval_outputs is None:
        if not is_hf_checkpoint(checkpoint_path):
            raise RuntimeError(
                "Checkpoint is not HF-compatible. Provide a converted HF checkpoint or enable LightEval/Nanotron backend."
            )

        model = load_hf_model(checkpoint_path)
        device = next(model.parameters()).device

        batch_size = int(os.environ.get("EVAL_BATCH_SIZE", "4"))

        log("Running Tier2 (PPL)")
        tier2 = {"slices": {}, "ppl_macro_avg": None}
        for name, (tokens, meta) in ppl_data.items():
            nll, token_count = compute_ppl(
                model,
                tokens,
                tokenizer.pad_token_id,
                batch_size,
                device,
                nan_tracker,
            )
            ppl = math.exp(nll) if math.isfinite(nll) else float("inf")
            tier2["slices"][name] = {
                "nll": nll,
                "ppl": ppl,
                "tokens": token_count,
            }

        if tier2["slices"]:
            tier2["ppl_macro_avg"] = sum(v["ppl"] for v in tier2["slices"].values()) / len(tier2["slices"])

        log("Running Tier3 (CF)")
        tier3 = {"tasks": {}, "cf_macro_avg": None}
        for task, (examples, meta) in cf_data.items():
            correct = 0
            for example in examples:
                scores = score_candidates(
                    model,
                    tokenizer,
                    example["prompt"],
                    example["candidates"],
                    args.seq_len,
                    device,
                    nan_tracker,
                )
                pred = int(max(range(len(scores)), key=lambda i: scores[i]))
                if pred == example["label"]:
                    correct += 1
            acc = correct / len(examples)
            tier3["tasks"][task] = {"acc": acc, "n": len(examples)}

        if tier3["tasks"]:
            tier3["cf_macro_avg"] = sum(v["acc"] for v in tier3["tasks"].values()) / len(tier3["tasks"])
    else:
        tier2, tier3 = lighteval_outputs

    log("Computing Tier1 stats")
    sample_tokens = next(iter(ppl_data.values()))[0]
    packing_stats = compute_packing_stats(sample_tokens, tokenizer)
    log_metrics = parse_log_metrics()

    tier1 = {
        "nan_inf_count": nan_tracker["nan_inf_count"],
        **log_metrics,
        **packing_stats,
    }

    missing = [key for key, value in tier1.items() if value is None]
    if missing:
        tier1["missing_fields"] = missing

    write_json(results_dir / "tier1_health.json", tier1)
    write_json(results_dir / "tier2_ppl.json", tier2)
    write_json(results_dir / "tier3_cf.json", tier3)

    log(f"Evaluation complete for step {step} -> {results_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 evaluation runner")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--eval_data_dir", default=None)
    parser.add_argument("--eval_results_dir", default=None)
    parser.add_argument("--tokenizer_name_or_path", default=DEFAULT_TOKENIZER)
    parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    return parser.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())
