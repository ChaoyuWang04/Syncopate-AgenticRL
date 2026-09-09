import importlib.util
import json
from pathlib import Path

import pytest

from syncopate.infra.real_training_scenario import (
    build_gsm8k_subset,
    initialize_reserved_output,
    validate_environment_inventory,
    validate_scenario,
    verify_weight_index,
)


def test_real_training_scenario_module_exists():
    assert importlib.util.find_spec("syncopate.infra.real_training_scenario") is not None


def scenario():
    return {
        "schema_version": 1,
        "experiment": "B14",
        "source": {
            "project_git_sha": "runtime",
            "verl_git_sha": "7cb65014d3a6c84f59458367df768999e4f36c67",
            "image": "verlai/verl:vllm024.latest",
        },
        "stack": {"rl_framework": "verl", "trainer": "megatron", "rollout": "vllm"},
        "model": {
            "path": "/vol/models/Qwen3.6-35B-A3B",
            "revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
            "architecture": "Qwen3_5MoeForConditionalGeneration",
            "mtp_enabled": False,
        },
        "input": {
            "dataset": "openai/gsm8k",
            "revision": "740312add88f781978c0658806c59bc2815b9866",
            "split": "train",
            "subset_size": 128,
            "seed": 20260909,
            "max_prompt_tokens": 512,
        },
        "workload": {
            "global_updates": 4,
            "train_batch_size": 32,
            "rollout_n": 4,
            "max_response_tokens": 1024,
            "real_optimizer_updates_required": True,
        },
        "parallelism": {
            "trainer": {"tp": 2, "pp": 1, "cp": 1, "ep": 8, "etp": 1},
            "rollout": {"tp": 8},
            "allocation": "colocated",
        },
        "execution": {
            "gpu_type": "B200",
            "gpu_count": 8,
            "target_seconds": 1800,
            "timeout_seconds": 2700,
            "gpu_usd_per_second": 0.001736,
            "max_gpu_cost_usd": 40.0,
        },
        "behavior": {
            "use_remove_padding": False,
            "dynamic_batching": False,
            "actor_offload": False,
            "rollout_offload": False,
        },
        "observations": ["timing_s/generation", "timing_s/update_actor", "gpu_utilization"],
        "evidence_root": "/vol/_audit/infra-probes/B14",
    }


def test_scenario_accepts_coherent_eight_b200_colocated_layout():
    result = validate_scenario(scenario())
    assert result["max_gpu_cost_usd"] == pytest.approx(37.4976)
    assert result["generated_tokens_upper_bound"] == 32 * 4 * 1024 * 4


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda x: x["execution"].update(gpu_count=9), "at most 8 B200"),
        (lambda x: x["stack"].update(trainer="fsdp1"), "trainer must be fsdp2 or megatron"),
        (lambda x: x["model"].update(mtp_enabled=True), "MTP must be disabled"),
        (lambda x: x["behavior"].update(dynamic_batching=True), "padded BSHD"),
        (lambda x: x["parallelism"]["trainer"].update(ep=7), "divide the GPU count"),
        (lambda x: x["execution"].update(max_gpu_cost_usd=30.0), "cost cap"),
    ],
)
def test_scenario_rejects_unfrozen_or_unsupported_first_scene(mutate, message):
    value = scenario()
    mutate(value)
    with pytest.raises(ValueError, match=message):
        validate_scenario(value)


def test_gsm8k_subset_is_deterministic_and_records_token_distribution():
    rows = [
        {"question": f"question {i}" + " long" * i, "answer": f"work {i} #### {i}"}
        for i in range(5)
    ]

    records, manifest = build_gsm8k_subset(rows, lambda text: text.split(), count=3, seed=7)

    assert manifest["source_row_indices"] == [2, 1, 3]
    assert [record["extra_info"]["index"] for record in records] == [2, 1, 3]
    assert records[0]["data_source"] == "openai/gsm8k"
    assert records[0]["prompt"][0]["role"] == "user"
    assert records[0]["reward_model"] == {"style": "rule", "ground_truth": "2"}
    assert manifest["prompt_tokens"] == {"min": 15, "mean": 16.0, "p50": 16, "p90": 17, "max": 17}
    assert len(manifest["records_sha256"]) == 64
    again = build_gsm8k_subset(rows, lambda text: text.split(), count=3, seed=7)
    assert again == (records, manifest)


def test_gsm8k_subset_rejects_bad_contracts():
    with pytest.raises(ValueError, match="enough rows"):
        build_gsm8k_subset([], str.split, count=1, seed=0)
    with pytest.raises(ValueError, match="####"):
        build_gsm8k_subset([{"question": "q", "answer": "missing"}], str.split, count=1, seed=0)


def test_gsm8k_subset_preserves_the_requested_split():
    records, _ = build_gsm8k_subset(
        [{"question": "q", "answer": "work #### 1"}], str.split, count=1, seed=0, split="test"
    )
    assert records[0]["extra_info"]["split"] == "test"


def test_environment_inventory_must_match_the_frozen_lock():
    value = scenario()
    value["source"]["required_packages"] = {"torch": "2.11.0", "vllm": "0.24.0"}
    inventory = {"packages": {"torch": "2.11.0", "vllm": "0.24.0"}, "modules": {"megatron.core": True}}
    validate_environment_inventory(value, inventory, required_modules=["megatron.core"])
    inventory["packages"]["vllm"] = "0.23.0"
    with pytest.raises(ValueError, match="vllm"):
        validate_environment_inventory(value, inventory, required_modules=["megatron.core"])


def test_weight_index_requires_every_nonempty_shard(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"a")
    (tmp_path / "b.safetensors").write_bytes(b"bb")
    index = {"weight_map": {"layer.0": "a.safetensors", "layer.1": "b.safetensors"}}
    result = verify_weight_index(tmp_path, index)
    assert result == {"shard_count": 2, "total_bytes": 3, "missing": []}
    (tmp_path / "b.safetensors").unlink()
    with pytest.raises(ValueError, match="b.safetensors"):
        verify_weight_index(tmp_path, index)


def test_registered_q21_config_matches_its_command_and_resource_contract():
    root = Path(__file__).resolve().parents[2]
    value = json.loads((root / "configs/infra/q21-b14-verl-qwen36-megatron-vllm.json").read_text())
    validate_scenario(value)
    command = value["entry_command"]
    assert "trainer.total_training_steps=4" in command
    assert "trainer.n_gpus_per_node=8" in command
    assert "actor_rollout_ref.actor.megatron.expert_model_parallel_size=8" in command
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=4" in command
    assert "actor_rollout_ref.rollout.data_parallel_size=2" in command
    assert "actor_rollout_ref.rollout.expert_parallel_size=8" in command
    assert value["source"]["modal_image_id"].startswith("im-")


def test_q21_modal_image_records_the_exact_verl_commit_and_runtime_image():
    root = Path(__file__).resolve().parents[2]
    value = json.loads((root / "configs/infra/q21-b14-verl-qwen36-megatron-vllm.json").read_text())
    assert value["source"]["verl_git_sha"] == "7cb65014d3a6c84f59458367df768999e4f36c67"
    assert value["source"]["modal_image_id"] == "im-ddKz6toouSd10HPSoofnFY"


def test_q21_prompt_lengths_use_the_shared_transformers5_adapter():
    root = Path(__file__).resolve().parents[2]
    source = (root / "syncopate/infra/real_training_scenario.py").read_text()
    assert "from syncopate.train.rollout_loop import chat_template_ids" in source
    assert "return chat_template_ids(" in source


def test_q21_reuses_an_immutable_runtime_image_and_mounts_experiment_code():
    root = Path(__file__).resolve().parents[2]
    source = (root / "modal_app/q21_real_training.py").read_text()
    assert "modal.Image.from_id(RUNTIME_IMAGE_ID)" in source
    assert 'RUNTIME_IMAGE_ID = "im-' in source
    assert "copy=True" not in source


def test_preflight_accepts_only_an_empty_reserved_output_directory(tmp_path):
    reserved = tmp_path / "attempt-1"
    reserved.mkdir()
    initialize_reserved_output(reserved)
    (reserved / "partial.json").write_text("{}")
    with pytest.raises(FileExistsError, match="not empty"):
        initialize_reserved_output(reserved)
