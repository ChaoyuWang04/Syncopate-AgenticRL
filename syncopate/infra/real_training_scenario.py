"""Validation and input preparation for registered real-training scenarios."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import random
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


ALLOWED_RL_FRAMEWORKS = {"verl", "areal", "roll", "nemo-rl", "miles", "slime"}
ALLOWED_TRAINERS = {"fsdp2", "megatron"}
ALLOWED_ROLLOUTS = {"vllm", "sglang", "tensorrt-llm"}
INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'
INSPECTION_SCRIPT = "python3 -m pip list"


def build_inspection_command(tag: str) -> list[str]:
    if not tag or tag.startswith("-"):
        raise ValueError("a registry image tag is required")
    return ["modal", "shell", "--image", tag, "--cmd", INSPECTION_SCRIPT, "--no-pty"]


def check_inspection_result(returncode: int, stdout: str, stderr: str) -> None:
    if returncode:
        detail = stderr.strip() or stdout.strip() or f"exit code {returncode}"
        raise RuntimeError(f"Modal image inspection failed: {detail}")


def _require_sha(value: object, field: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(f"{field} must be an exact 40-character git revision")


def validate_scenario(value: dict[str, Any]) -> dict[str, int | float]:
    """Reject a scenario that can silently drift or exceed the project boundary."""
    if value.get("schema_version") != 1 or value.get("experiment") != "B14":
        raise ValueError("Q21 must use schema version 1 and experiment B14")
    source = value["source"]
    _require_sha(source.get("verl_git_sha"), "source.verl_git_sha")
    if not source.get("image"):
        raise ValueError("source.image is required")

    stack = value["stack"]
    if stack.get("rl_framework") not in ALLOWED_RL_FRAMEWORKS:
        raise ValueError("RL framework is outside the frozen project boundary")
    if stack.get("trainer") not in ALLOWED_TRAINERS:
        raise ValueError("trainer must be fsdp2 or megatron")
    if stack.get("rollout") not in ALLOWED_ROLLOUTS:
        raise ValueError("rollout engine is outside the frozen project boundary")

    model = value["model"]
    _require_sha(model.get("revision"), "model.revision")
    if model.get("mtp_enabled") is not False:
        raise ValueError("MTP must be disabled for the first scene")
    dataset = value["input"]
    _require_sha(dataset.get("revision"), "input.revision")
    if dataset.get("subset_size", 0) < value["workload"].get("train_batch_size", 0):
        raise ValueError("input subset must cover at least one training batch")

    execution = value["execution"]
    gpu_count = execution.get("gpu_count")
    if execution.get("gpu_type") != "B200" or not isinstance(gpu_count, int) or not 1 <= gpu_count <= 8:
        raise ValueError("the scene may use at most 8 B200 GPUs")
    if execution.get("target_seconds", 0) > execution.get("timeout_seconds", 0):
        raise ValueError("target duration must fit within timeout")
    computed_cost = gpu_count * execution["timeout_seconds"] * execution["gpu_usd_per_second"]
    if computed_cost > execution.get("max_gpu_cost_usd", -1):
        raise ValueError("declared cost cap is below the timeout upper bound")

    parallel = value["parallelism"]
    if parallel.get("allocation") != "colocated":
        raise ValueError("the first scene uses colocated trainer and rollout roles")
    trainer = parallel["trainer"]
    for name in ("tp", "pp", "cp", "ep", "etp"):
        width = trainer.get(name)
        if not isinstance(width, int) or width < 1 or gpu_count % width:
            raise ValueError(f"trainer {name} must divide the GPU count")
    rollout_tp = parallel["rollout"].get("tp")
    if not isinstance(rollout_tp, int) or rollout_tp < 1 or gpu_count % rollout_tp:
        raise ValueError("rollout tp must divide the GPU count")

    behavior = value["behavior"]
    if behavior.get("use_remove_padding") is not False or behavior.get("dynamic_batching") is not False:
        raise ValueError("the first Megatron GDN scene must use padded BSHD batches")
    if not value.get("observations") or not str(value.get("evidence_root", "")).startswith("/vol/_audit/infra-probes/B14"):
        raise ValueError("observations and the B14 evidence root are required")

    workload = value["workload"]
    if workload.get("real_optimizer_updates_required") is not True or workload.get("global_updates", 0) < 2:
        raise ValueError("multiple real optimizer updates are required")
    token_upper_bound = (
        workload["train_batch_size"]
        * workload["rollout_n"]
        * workload["max_response_tokens"]
        * workload["global_updates"]
    )
    return {"max_gpu_cost_usd": computed_cost, "generated_tokens_upper_bound": token_upper_bound}


def validate_environment_inventory(
    scenario: dict[str, Any], inventory: dict[str, Any], *, required_modules: Sequence[str]
) -> None:
    installed = inventory.get("packages", {})
    for name, expected in scenario["source"].get("required_packages", {}).items():
        actual = installed.get(name)
        if actual != expected:
            raise ValueError(f"{name} must be {expected}, found {actual}")
    modules = inventory.get("modules", {})
    missing = [name for name in required_modules if modules.get(name) is not True]
    if missing:
        raise ValueError(f"required modules are unavailable: {', '.join(missing)}")


def verify_weight_index(model_dir: Path, index: dict[str, Any]) -> dict[str, Any]:
    shards = sorted(set(index.get("weight_map", {}).values()))
    if not shards:
        raise ValueError("model weight index contains no shards")
    missing = [name for name in shards if not (model_dir / name).is_file() or (model_dir / name).stat().st_size == 0]
    if missing:
        raise ValueError(f"missing or empty model shards: {', '.join(missing)}")
    return {
        "shard_count": len(shards),
        "total_bytes": sum((model_dir / name).stat().st_size for name in shards),
        "missing": [],
    }


def _distribution(values: Sequence[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "mean": sum(ordered) / len(ordered),
        "p50": ordered[math.ceil(0.50 * len(ordered)) - 1],
        "p90": ordered[math.ceil(0.90 * len(ordered)) - 1],
        "max": ordered[-1],
    }


def build_gsm8k_subset(
    rows: Sequence[dict[str, str]],
    tokenizer: Callable[[str], Sequence[object]],
    *,
    count: int,
    seed: int,
    split: str = "train",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the official verl GSM8K row contract from a deterministic sample."""
    if count < 1 or len(rows) < count:
        raise ValueError("dataset must contain enough rows for the requested subset")
    indices = random.Random(seed).sample(range(len(rows)), count)
    records: list[dict[str, Any]] = []
    prompt_lengths: list[int] = []
    for index in indices:
        row = rows[index]
        match = re.search(r"#### (\-?[0-9\.\,]+)", row["answer"])
        if match is None:
            raise ValueError("GSM8K answer is missing the #### numeric solution")
        prompt = f'{row["question"]} {INSTRUCTION}'
        prompt_lengths.append(len(tokenizer(prompt)))
        records.append(
            {
                "data_source": "openai/gsm8k",
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": match.group(1).replace(",", "")},
                "extra_info": {
                    "split": split,
                    "index": index,
                    "answer": row["answer"],
                    "question": row["question"],
                },
            }
        )
    encoded = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    manifest = {
        "source_row_indices": indices,
        "prompt_tokens": _distribution(prompt_lengths),
        "records_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    return records, manifest


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_reserved_output(out_dir: Path) -> None:
    if not out_dir.is_dir():
        raise FileNotFoundError(f"reserved output directory is missing: {out_dir}")
    if any(out_dir.iterdir()):
        raise FileExistsError(f"reserved output directory is not empty: {out_dir}")


def execute_q21_cpu_preflight(config_path: Path, out_dir: Path) -> dict[str, Any]:
    """Prepare B14 input and prove that the frozen CPU-side stack is coherent."""
    scenario = json.loads(config_path.read_text())
    derived = validate_scenario(scenario)
    initialize_reserved_output(out_dir)
    result: dict[str, Any] = {
        "experiment": "infra-probes/B14",
        "phase": "q21-cpu-preflight",
        "ok": False,
        "config": str(config_path),
        "derived": derived,
    }
    result_path = out_dir / "run.json"

    def persist() -> None:
        temporary = result_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        temporary.replace(result_path)

    persist()
    try:
        package_names = list(scenario["source"]["required_packages"])
        modules = ["megatron.core", "megatron.bridge", "vllm", "mbridge", "transformer_engine"]
        inventory = {
            "python": __import__("sys").version,
            "packages": {
                name: importlib.metadata.version(name) if _package_exists(name) else None for name in package_names
            },
            "modules": {name: importlib.util.find_spec(name) is not None for name in modules},
            "modal_image_id": __import__("os").environ.get("MODAL_IMAGE_ID"),
        }
        validate_environment_inventory(scenario, inventory, required_modules=modules)
        result["environment"] = inventory

        model_dir = Path(scenario["model"]["path"])
        model_config = json.loads((model_dir / "config.json").read_text())
        if model_config.get("architectures") != [scenario["model"]["architecture"]]:
            raise ValueError("model architecture does not match the frozen scenario")
        if model_config.get("model_type") != scenario["model"]["model_type"]:
            raise ValueError("model type does not match the frozen scenario")
        index_path = model_dir / "model.safetensors.index.json"
        model_files = verify_weight_index(model_dir, json.loads(index_path.read_text()))
        verified_evidence = Path(scenario["model"]["verified_files_evidence"])
        if not verified_evidence.exists():
            raise ValueError("the registered full-file verification evidence is missing")
        from verl.models.mcore.registry import supported_vlm

        if scenario["model"]["architecture"] not in supported_vlm:
            raise ValueError("the architecture is absent from the active verl Megatron registry")
        result["model"] = {
            **model_files,
            "config_sha256": _file_sha256(model_dir / "config.json"),
            "index_sha256": _file_sha256(index_path),
            "architecture_registered": True,
            "full_file_verification_evidence": str(verified_evidence),
        }

        import os

        os.environ["HF_HOME"] = "/vol/cache/infra-probes/B14/huggingface"
        from datasets import Dataset, load_dataset
        from transformers import AutoTokenizer
        from syncopate.train.rollout_loop import chat_template_ids

        source = load_dataset(
            scenario["input"]["dataset"],
            scenario["input"]["config"],
            revision=scenario["input"]["revision"],
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

        def tokenize_prompt(text: str) -> Sequence[int]:
            return chat_template_ids(
                tokenizer, [{"role": "user", "content": text}], add_generation_prompt=True
            )

        train_records, train_manifest = build_gsm8k_subset(
            list(source["train"]),
            tokenize_prompt,
            count=scenario["input"]["subset_size"],
            seed=scenario["input"]["seed"],
            split="train",
        )
        validation_records, validation_manifest = build_gsm8k_subset(
            list(source["test"]),
            tokenize_prompt,
            count=scenario["input"]["validation_subset_size"],
            seed=scenario["input"]["seed"] + 1,
            split="test",
        )
        if train_manifest["prompt_tokens"]["max"] > scenario["input"]["max_prompt_tokens"]:
            raise ValueError("frozen train subset exceeds max_prompt_tokens")
        if validation_manifest["prompt_tokens"]["max"] > scenario["input"]["max_prompt_tokens"]:
            raise ValueError("frozen validation subset exceeds max_prompt_tokens")

        data_dir = Path(scenario["input"]["volume_dir"])
        if data_dir.exists():
            raise FileExistsError(f"refusing to overwrite frozen input: {data_dir}")
        data_dir.mkdir(parents=True)
        train_path = data_dir / "train.parquet"
        validation_path = data_dir / "validation.parquet"
        Dataset.from_list(train_records).to_parquet(train_path)
        Dataset.from_list(validation_records).to_parquet(validation_path)
        input_manifest = {
            "dataset": scenario["input"]["dataset"],
            "revision": scenario["input"]["revision"],
            "config": scenario["input"]["config"],
            "train": {**train_manifest, "path": str(train_path), "sha256": _file_sha256(train_path)},
            "validation": {
                **validation_manifest,
                "path": str(validation_path),
                "sha256": _file_sha256(validation_path),
            },
            "tokenizer_config_sha256": _file_sha256(model_dir / "tokenizer_config.json"),
        }
        manifest_path = data_dir / "manifest.json"
        manifest_path.write_text(json.dumps(input_manifest, ensure_ascii=False, indent=2, sort_keys=True))
        result["input"] = input_manifest
        result["input_manifest"] = str(manifest_path)
        result["ok"] = True
    except BaseException as exc:
        result["exception"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        persist()
    return result


def _package_exists(name: str) -> bool:
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--q21-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = execute_q21_cpu_preflight(args.q21_config, args.out)
    if not result["ok"]:
        raise RuntimeError(f"Q21 CPU preflight failed; evidence: {args.out / 'run.json'}")


if __name__ == "__main__":
    _main()
