import json
from pathlib import Path

import pytest

from syncopate.infra.q22_training_probe import (
    build_capability_commands,
    build_nvidia_smi_command,
    build_training_command,
    estimate_b14_model_state,
    find_fatal_training_error,
    parse_nvidia_smi_rows,
    validate_q22_observability,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/infra/q21-b14-verl-qwen36-megatron-vllm.json"


def registered_scenario():
    return json.loads(CONFIG.read_text())


def test_b14_memory_estimate_uses_folded_dense_and_expert_grids():
    estimate = estimate_b14_model_state(
        total_weight_bytes=71_903_776_776,
        hidden_size=2048,
        expert_intermediate_size=512,
        num_layers=40,
        num_experts=256,
        tensor_parallel_size=2,
        data_parallel_size=4,
        expert_parallel_size=8,
        expert_tensor_parallel_size=1,
        expert_data_parallel_size=1,
    )

    assert estimate["routed_expert_parameters"] == 32_212_254_720
    assert estimate["local_bf16_weight_gb"] == pytest.approx(11.792697348)
    assert estimate["model_state_gb_per_gpu"] == pytest.approx(89.305924626)


def test_registered_observability_is_bounded_and_leaves_static_headroom():
    plan = validate_q22_observability(registered_scenario())

    assert plan["telemetry_interval_seconds"] == 1
    assert plan["profiled_steps"] == [2]
    assert plan["profiled_ranks"] == [0]
    assert plan["baseline_steps"] == [3, 4]
    assert plan["vllm_target_gb_per_gpu"] == pytest.approx(63.0)
    assert plan["static_headroom_gb_per_gpu"] == pytest.approx(27.694075374)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda x: x["memory_plan"].update(vllm_gpu_memory_utilization=0.55), "vLLM memory target"),
        (lambda x: x["observability"].update(telemetry_interval_seconds=0.1), "telemetry interval"),
        (lambda x: x["observability"]["torch_profiler"].update(ranks=list(range(8))), "one rank"),
        (lambda x: x["observability"]["torch_profiler"].update(contents=["cpu", "cuda", "stack"]), "stack"),
        (lambda x: x["observability"]["torch_profiler"].update(global_steps=[3, 4]), "one global step"),
    ],
)
def test_observability_rejects_expensive_or_unsafe_settings(mutate, message):
    value = registered_scenario()
    mutate(value)
    with pytest.raises(ValueError, match=message):
        validate_q22_observability(value)


def test_training_command_comes_from_the_registered_entry_and_adds_one_profile_window(tmp_path):
    value = registered_scenario()
    command = build_training_command(value, tmp_path)
    joined = " ".join(command)

    assert f"data.train_files={value['input']['volume_dir']}/train.parquet" in command
    assert f"data.val_files={value['input']['volume_dir']}/validation.parquet" in command
    assert "actor_rollout_ref.rollout.gpu_memory_utilization=0.35" in command
    assert "actor_rollout_ref.rollout.max_model_len=1536" in command
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=4" in command
    assert "actor_rollout_ref.rollout.data_parallel_size=2" in command
    assert "actor_rollout_ref.rollout.expert_parallel_size=8" in command
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=8" not in command
    assert "actor_rollout_ref.actor.megatron.use_mbridge=True" in command
    assert "actor_rollout_ref.actor.megatron.vanilla_mbridge=False" in command
    assert "actor_rollout_ref.actor.megatron.vanilla_mbridge=True" not in command
    assert "global_profiler.tool=torch" in command
    assert "global_profiler.steps=[2]" in command
    assert "actor_rollout_ref.actor.profiler.ranks=[0]" in command
    assert "actor_rollout_ref.actor.profiler.tool_config.torch.contents=[cpu,cuda,shapes]" in command
    assert "+actor_rollout_ref.actor.profiler.tool_config.torch.profile_token_start=16" in command
    assert "+actor_rollout_ref.actor.profiler.tool_config.torch.profile_token_end=32" in command
    assert "actor_rollout_ref.actor.profiler.tool_config.torch.profile_token_start=16" not in command
    assert str(tmp_path / "profiles") in joined


def test_nvidia_smi_rows_are_summarized_per_gpu_and_reject_bad_rows():
    rows = [
        "2026/09/09 10:00:00.000, 0, GPU-a, 10, 20, 1000, 184320, 300.0, 1500, 3200, 42",
        "2026/09/09 10:00:01.000, 0, GPU-a, 90, 40, 2000, 184320, 800.0, 1800, 3200, 55",
        "2026/09/09 10:00:00.000, 1, GPU-b, 50, 30, 1500, 184320, 500.0, 1600, 3200, 45",
        "malformed",
    ]

    result = parse_nvidia_smi_rows(rows)

    assert result["valid_samples"] == 3
    assert result["invalid_samples"] == 1
    assert result["gpus"]["0"]["gpu_utilization_mean"] == 50
    assert result["gpus"]["0"]["memory_used_max_mib"] == 2000
    assert result["gpus"]["1"]["power_mean_w"] == 500


def test_fatal_training_error_only_matches_terminal_worker_failures():
    assert find_fatal_training_error("INFO warmup\nERROR retrying a kernel") is None
    assert find_fatal_training_error(
        "prefix\nRuntimeError: Worker failed with error 'bad expert weight'\n"
    ) == "RuntimeError: Worker failed with error 'bad expert weight'"
    assert find_fatal_training_error(
        "ray.exceptions.RayTaskError(ValueError): actor failed\n"
    ) == "ray.exceptions.RayTaskError(ValueError): actor failed"


def test_low_rate_monitor_and_profiler_capability_checks_are_bounded():
    value = registered_scenario()
    monitor = build_nvidia_smi_command(value)
    checks = build_capability_commands()

    assert monitor[-2:] == ["-l", "1"]
    assert "timestamp,index,uuid" in monitor[1]
    assert checks == {
        "nvidia_smi": ["nvidia-smi", "-L"],
        "topology": ["nvidia-smi", "topo", "-m"],
        "nsys": ["nsys", "--version"],
        "dcgm": ["dcgmi", "profile", "--list", "--entity-id", "gpu:0"],
    }


def test_modal_q22_entry_requests_exactly_eight_b200s_and_never_enables_full_nsys_trace():
    source = (ROOT / "modal_app/q22_real_training.py").read_text()

    assert 'gpu="B200:8"' in source
    assert "fla-core==0.5.2" in source
    assert "flash-linear-attention==0.5.2" in source
    assert "from fla.modules import FusedRMSNormGated" in source
    assert "--python /opt/verl-q21/.venv/bin/python --no-deps" in source
    assert "build_training_command" in source
    assert "build_nvidia_smi_command" in source
    assert "training_trace_enabled" not in source
