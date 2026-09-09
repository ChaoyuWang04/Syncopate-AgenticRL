"""Q22: run the first real eight-B200 verl training profile."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import modal


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_IMAGE_ID = "im-ddKz6toouSd10HPSoofnFY"
CONFIG = ROOT / "configs/infra/q21-b14-verl-qwen36-megatron-vllm.json"
REMOTE_SOURCE = "/opt/syncopate-q22"
REMOTE_CONFIG = "/opt/syncopate-q22/config.json"

app = modal.App("syncopate-q22-real-training")
vol = modal.Volume.from_name("syncopate-home", create_if_missing=True)
image = (
    modal.Image.from_id(RUNTIME_IMAGE_ID)
    .run_commands(
        "uv pip install --python /opt/verl-q21/.venv/bin/python --no-deps "
        "fla-core==0.5.2 flash-linear-attention==0.5.2",
        "/opt/verl-q21/.venv/bin/python -c "
        "'from fla.modules import FusedRMSNormGated; "
        "from fla.ops.gated_delta_rule import chunk_gated_delta_rule'",
    )
    .add_local_dir(ROOT / "syncopate", f"{REMOTE_SOURCE}/syncopate")
    .add_local_file(CONFIG, REMOTE_CONFIG)
)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.function(
    image=image,
    volumes={"/vol": vol},
    gpu="B200:8",
    cpu=64,
    memory=524288,
    timeout=2700,
    retries=0,
)
def gpu_profile(run_id: str) -> dict:
    sys.path.insert(0, REMOTE_SOURCE)
    from syncopate.infra.probe_run import reserve_attempt
    from syncopate.infra.q22_training_probe import (
        build_capability_commands,
        build_nvidia_smi_command,
        build_training_command,
        find_fatal_training_error,
        parse_nvidia_smi_rows,
        validate_q22_observability,
    )

    scenario = json.loads(Path(REMOTE_CONFIG).read_text())
    plan = validate_q22_observability(scenario)
    out = reserve_attempt(Path("/vol/_audit/infra-probes"), "B14", run_id)
    command = build_training_command(scenario, out)
    result = {
        "experiment": "infra-probes/B14",
        "phase": "q22-gpu-profile",
        "run_id": run_id,
        "ok": False,
        "status": "starting",
        "evidence_path": str(out),
        "config_sha256": _sha256(Path(REMOTE_CONFIG)),
        "modal_image_id": os.environ.get("MODAL_IMAGE_ID"),
        "observability_plan": plan,
        "command": command,
        "started_unix": time.time(),
    }
    run_path = out / "run.json"
    _write_json(run_path, result)
    _write_json(out / "command.json", command)

    capabilities = {}
    for name, check in build_capability_commands().items():
        try:
            completed = subprocess.run(check, capture_output=True, text=True, timeout=20)
            capabilities[name] = {
                "available": completed.returncode == 0,
                "returncode": completed.returncode,
                "stdout": completed.stdout[-20000:],
                "stderr": completed.stderr[-10000:],
            }
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            capabilities[name] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    _write_json(out / "capabilities.json", capabilities)
    vol.commit()

    env = {
        **os.environ,
        "PYTHONPATH": REMOTE_SOURCE,
        "HF_HOME": "/vol/cache/infra-probes/B14/huggingface",
        "PYTHONUNBUFFERED": "1",
        "RAY_TMPDIR": f"/tmp/q22-ray-{run_id}",
        "TOKENIZERS_PARALLELISM": "false",
    }
    telemetry_path = out / "nvidia-smi.csv"
    training_log = out / "training.log"
    monitor = None
    process = None
    timed_out = False
    fatal_training_error = None
    returncode = None
    try:
        with telemetry_path.open("w") as telemetry:
            monitor = subprocess.Popen(
                build_nvidia_smi_command(scenario),
                stdout=telemetry,
                stderr=subprocess.STDOUT,
                text=True,
            )
            with training_log.open("w") as log:
                process = subprocess.Popen(
                    command,
                    cwd="/opt/verl-q21",
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                result["status"] = "running"
                _write_json(run_path, result)
                deadline = time.monotonic() + scenario["execution"]["timeout_seconds"] - 120
                next_commit = time.monotonic() + 120
                while process.poll() is None:
                    now = time.monotonic()
                    if training_log.exists():
                        with training_log.open("rb") as log_tail:
                            log_size = log_tail.seek(0, os.SEEK_END)
                            log_tail.seek(max(0, log_size - 131072))
                            fatal_training_error = find_fatal_training_error(
                                log_tail.read().decode(errors="replace")
                            )
                    if fatal_training_error is not None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                        break
                    if now >= deadline:
                        timed_out = True
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                        break
                    if now >= next_commit:
                        vol.commit()
                        next_commit = now + 120
                    time.sleep(5)
                returncode = process.wait()
    except BaseException as exc:
        result["exception"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if monitor is not None and monitor.poll() is None:
            monitor.terminate()
            try:
                monitor.wait(timeout=10)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)

    telemetry_rows = telemetry_path.read_text(errors="replace").splitlines() if telemetry_path.exists() else []
    result["telemetry"] = parse_nvidia_smi_rows(telemetry_rows)
    result["returncode"] = returncode
    result["timed_out"] = timed_out
    result["fatal_training_error"] = fatal_training_error
    result["finished_unix"] = time.time()
    result["elapsed_seconds"] = result["finished_unix"] - result["started_unix"]
    result["profile_files"] = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted((out / "profiles").rglob("*"))
        if path.is_file()
    ] if (out / "profiles").exists() else []
    result["status"] = "pass" if returncode == 0 and not timed_out else "failed"
    result["ok"] = result["status"] == "pass"
    _write_json(run_path, result)
    vol.commit()
    if not result["ok"]:
        raise RuntimeError(f"Q22 GPU profile failed; evidence: {out}")
    return result


@app.local_entrypoint()
def main(run_id: str):
    print(json.dumps(gpu_profile.remote(run_id), ensure_ascii=False, indent=2))
