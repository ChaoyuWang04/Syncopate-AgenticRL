"""Bounded observability and launch helpers for the B14 Q22 GPU run."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


NVIDIA_SMI_QUERY = (
    "timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,"
    "memory.total,power.draw,clocks.sm,clocks.mem,temperature.gpu"
)

FATAL_TRAINING_MARKERS = (
    "RuntimeError: Worker failed with error",
    "ray.exceptions.RayTaskError(",
)


def find_fatal_training_error(log_tail: str) -> str | None:
    """Return the first terminal distributed-worker line from a bounded log tail."""
    for line in log_tail.splitlines():
        if any(marker in line for marker in FATAL_TRAINING_MARKERS):
            return line.strip()
    return None


def build_nvidia_smi_command(scenario: dict[str, Any]) -> list[str]:
    interval = scenario["observability"]["telemetry_interval_seconds"]
    if not isinstance(interval, int) or interval < 1:
        raise ValueError("telemetry interval must be an integer of at least one second")
    return [
        "nvidia-smi",
        f"--query-gpu={NVIDIA_SMI_QUERY}",
        "--format=csv,noheader,nounits",
        "-l",
        str(interval),
    ]


def build_capability_commands() -> dict[str, list[str]]:
    return {
        "nvidia_smi": ["nvidia-smi", "-L"],
        "topology": ["nvidia-smi", "topo", "-m"],
        "nsys": ["nsys", "--version"],
        "dcgm": ["dcgmi", "profile", "--list", "--entity-id", "gpu:0"],
    }


def estimate_b14_model_state(
    *,
    total_weight_bytes: int,
    hidden_size: int,
    expert_intermediate_size: int,
    num_layers: int,
    num_experts: int,
    tensor_parallel_size: int,
    data_parallel_size: int,
    expert_parallel_size: int,
    expert_tensor_parallel_size: int,
    expert_data_parallel_size: int,
) -> dict[str, float | int]:
    """Estimate BF16 Adam state using Megatron's distributed-optimizer byte model."""
    total_parameters = total_weight_bytes / 2
    routed_expert_parameters = (
        num_layers * num_experts * 3 * hidden_size * expert_intermediate_size
    )
    other_parameters = total_parameters - routed_expert_parameters
    if other_parameters <= 0:
        raise ValueError("routed expert parameters exceed the registered model size")
    local_expert_parameters = routed_expert_parameters / (
        expert_parallel_size * expert_tensor_parallel_size
    )
    local_other_parameters = other_parameters / tensor_parallel_size
    local_bf16_weight_gb = (local_expert_parameters + local_other_parameters) * 2 / 1e9
    expert_bytes_per_parameter = 6 + 12 / expert_data_parallel_size
    other_bytes_per_parameter = 6 + 12 / data_parallel_size
    model_state_gb = (
        local_expert_parameters * expert_bytes_per_parameter
        + local_other_parameters * other_bytes_per_parameter
    ) / 1e9
    return {
        "total_parameters": total_parameters,
        "routed_expert_parameters": routed_expert_parameters,
        "local_expert_parameters": local_expert_parameters,
        "local_other_parameters": local_other_parameters,
        "local_bf16_weight_gb": local_bf16_weight_gb,
        "model_state_gb_per_gpu": model_state_gb,
    }


def validate_q22_observability(scenario: dict[str, Any]) -> dict[str, Any]:
    """Reject profiling or memory settings that would invalidate the first smoke run."""
    observability = scenario["observability"]
    interval = observability["telemetry_interval_seconds"]
    if interval < 1:
        raise ValueError("telemetry interval must be at least one second")
    profiler = observability["torch_profiler"]
    steps = profiler["global_steps"]
    ranks = profiler["ranks"]
    contents = profiler["contents"]
    if not profiler.get("enabled") or len(steps) != 1:
        raise ValueError("Torch profiler must capture exactly one global step")
    if ranks != [0]:
        raise ValueError("Torch profiler must capture one rank, rank 0")
    if "stack" in contents or "memory" in contents:
        raise ValueError("stack and memory tracing are outside the low-overhead profile")
    updates = scenario["workload"]["global_updates"]
    if steps[0] <= 1 or steps[0] >= updates:
        raise ValueError("profiled step must leave warmup and unprofiled baseline steps")

    memory = scenario["memory_plan"]
    expected_model_len = (
        scenario["input"]["max_prompt_tokens"]
        + scenario["workload"]["max_response_tokens"]
    )
    if memory["rollout_max_model_len"] != expected_model_len:
        raise ValueError("rollout max model length must equal prompt plus response budget")
    expected_override = f"actor_rollout_ref.rollout.max_model_len={expected_model_len}"
    if expected_override not in scenario["entry_command"]:
        raise ValueError("entry command must pin the registered rollout max model length")
    rollout_parallelism = scenario["parallelism"]["rollout"]
    if rollout_parallelism["tp"] * rollout_parallelism["dp"] != scenario["execution"]["gpu_count"]:
        raise ValueError("rollout TP times DP must use exactly the registered GPU count")
    if rollout_parallelism["ep"] != rollout_parallelism["tp"] * rollout_parallelism["dp"]:
        raise ValueError("vLLM expert parallel size must equal rollout TP times DP")
    local_expert_intermediate = scenario["model"]["expert_intermediate_size"]
    if local_expert_intermediate % 128:
        raise ValueError("rollout TP creates an unsupported local expert intermediate size")
    utilization = memory["vllm_gpu_memory_utilization"]
    if utilization > 0.40:
        raise ValueError("vLLM memory target exceeds the first-run safety boundary")
    vllm_target = memory["gpu_memory_gb_per_gpu"] * utilization
    headroom = (
        memory["gpu_memory_gb_per_gpu"]
        - memory["theoretical_model_state_gb_per_gpu"]
        - vllm_target
    )
    if headroom < memory["minimum_static_headroom_gb_per_gpu"]:
        raise ValueError("registered model and vLLM targets leave insufficient static headroom")
    baseline_steps = [step for step in range(steps[0] + 1, updates + 1)]
    return {
        "telemetry_interval_seconds": interval,
        "profiled_steps": steps,
        "profiled_ranks": ranks,
        "baseline_steps": baseline_steps,
        "vllm_target_gb_per_gpu": vllm_target,
        "static_headroom_gb_per_gpu": headroom,
    }


def build_training_command(scenario: dict[str, Any], out_dir: Path) -> list[str]:
    """Materialize the one registered command without maintaining a second parameter list."""
    validate_q22_observability(scenario)
    replacements = {
        "{train_parquet}": f"{scenario['input']['volume_dir']}/train.parquet",
        "{validation_parquet}": f"{scenario['input']['volume_dir']}/validation.parquet",
        "{checkpoint_dir}": str(out_dir / "checkpoints"),
    }
    command: list[str] = []
    for item in scenario["entry_command"]:
        rendered = item
        for marker, value in replacements.items():
            rendered = rendered.replace(marker, value)
        command.append(rendered)

    profiler = scenario["observability"]["torch_profiler"]
    profile_dir = out_dir / "profiles"
    contents = ",".join(profiler["contents"])
    command.extend(
        [
            "global_profiler.tool=torch",
            f"global_profiler.steps={profiler['global_steps']}",
            f"global_profiler.save_path={profile_dir}",
            "global_profiler.profile_continuous_steps=False",
            "actor_rollout_ref.actor.profiler.enable=True",
            "actor_rollout_ref.actor.profiler.all_ranks=False",
            f"actor_rollout_ref.actor.profiler.ranks={profiler['ranks']}",
            f"actor_rollout_ref.actor.profiler.tool_config.torch.discrete={str(profiler['discrete'])}",
            f"actor_rollout_ref.actor.profiler.tool_config.torch.contents=[{contents}]",
            f"+actor_rollout_ref.actor.profiler.tool_config.torch.profile_token_start={profiler['profile_token_start']}",
            f"+actor_rollout_ref.actor.profiler.tool_config.torch.profile_token_end={profiler['profile_token_end']}",
        ]
    )
    return command


def parse_nvidia_smi_rows(rows: Iterable[str]) -> dict[str, Any]:
    """Summarize low-rate telemetry while preserving malformed-row counts."""
    by_gpu: dict[str, list[list[float]]] = defaultdict(list)
    invalid = 0
    for row in rows:
        fields = [field.strip() for field in row.split(",")]
        if len(fields) != 11:
            invalid += 1
            continue
        try:
            values = [float(value) for value in fields[3:]]
        except ValueError:
            invalid += 1
            continue
        by_gpu[fields[1]].append(values)
    result: dict[str, Any] = {"valid_samples": sum(map(len, by_gpu.values())), "invalid_samples": invalid, "gpus": {}}
    for gpu, samples in sorted(by_gpu.items(), key=lambda item: int(item[0])):
        result["gpus"][gpu] = {
            "sample_count": len(samples),
            "gpu_utilization_mean": mean(sample[0] for sample in samples),
            "gpu_utilization_max": max(sample[0] for sample in samples),
            "memory_controller_utilization_mean": mean(sample[1] for sample in samples),
            "memory_used_max_mib": max(sample[2] for sample in samples),
            "memory_total_mib": max(sample[3] for sample in samples),
            "power_mean_w": mean(sample[4] for sample in samples),
            "power_max_w": max(sample[4] for sample in samples),
            "sm_clock_mean_mhz": mean(sample[5] for sample in samples),
            "memory_clock_mean_mhz": mean(sample[6] for sample in samples),
            "temperature_max_c": max(sample[7] for sample in samples),
        }
    return result
