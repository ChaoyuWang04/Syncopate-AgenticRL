"""看一个现成 docker 镜像里装了什么（给选基础镜像用）。  modal run modal_app/inspect_image.py --tag verlai/verl:uv.cu130.dev3"""
import modal
from syncopate.infra.real_training_scenario import build_inspection_command, check_inspection_result
app = modal.App("syncopate-inspect-image")


@app.function(cpu=2, memory=8192, timeout=900)
def noop(): pass


@app.local_entrypoint()
def main(tag: str = "verlai/verl:uv.cu130.dev3"):
    import subprocess
    r = subprocess.run(build_inspection_command(tag), capture_output=True, text=True, timeout=800)
    print(r.stdout[-6000:]); print(r.stderr[-2000:])
    check_inspection_result(r.returncode, r.stdout, r.stderr)


def inspect(): pass
