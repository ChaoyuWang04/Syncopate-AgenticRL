"""Q21 CPU-only registration for the first real verl training scene."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import modal


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_IMAGE_ID = "im-ddKz6toouSd10HPSoofnFY"
CONFIG = ROOT / "configs/infra/q21-b14-verl-qwen36-megatron-vllm.json"
REMOTE_SOURCE = "/opt/syncopate-q21"
REMOTE_CONFIG = "/opt/syncopate-q21/config.json"

app = modal.App("syncopate-q21-real-training")
vol = modal.Volume.from_name("syncopate-home", create_if_missing=True)
image = (
    modal.Image.from_id(RUNTIME_IMAGE_ID)
    .add_local_dir(ROOT / "syncopate", f"{REMOTE_SOURCE}/syncopate")
    .add_local_file(CONFIG, REMOTE_CONFIG)
)


@app.function(image=image, volumes={"/vol": vol}, cpu=8, memory=32768, timeout=1800, retries=0)
def cpu_preflight(run_id: str) -> dict:
    sys.path.insert(0, REMOTE_SOURCE)
    from syncopate.infra.probe_run import reserve_attempt

    out = reserve_attempt(Path("/vol/_audit/infra-probes"), "B14", run_id)
    env = {
        **os.environ,
        "PYTHONPATH": REMOTE_SOURCE,
        "HF_HOME": "/vol/cache/infra-probes/B14/huggingface",
        "PYTHONUNBUFFERED": "1",
    }
    command = [
        "/opt/verl-q21/.venv/bin/python",
        "-m",
        "syncopate.infra.real_training_scenario",
        "--q21-config",
        REMOTE_CONFIG,
        "--out",
        str(out),
    ]
    completed = subprocess.run(command, cwd="/opt/verl-q21", env=env, text=True, timeout=1650)
    vol.commit()
    result = json.loads((out / "run.json").read_text())
    result["evidence_path"] = str(out)
    result["command"] = command
    result["modal_image_id"] = os.environ.get("MODAL_IMAGE_ID")
    if completed.returncode or result.get("ok") is not True:
        raise RuntimeError(f"Q21 CPU preflight failed; evidence: {out}")
    return result


@app.local_entrypoint()
def main(run_id: str):
    result = cpu_preflight.remote(run_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
