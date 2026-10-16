# ruff: noqa
"""Modal wrapper for the Celeste supplement tasks 2-4 (WorkspaceBench fork, branch celeste-supplement-tasks234).
Same images/pins as modal_of.py (torch/HF, `hf`) and modal_of_vllm.py (vLLM 0.21 + vllm-lens, `vllm`); app NAME reused (workspace app cap).
Volumes: /vol_c = wsb-celeste (READ-WRITE, everything of ours lands under /vol_c/tasks234), /vol = olens-flow READ-ONLY (lens ckpt, hf_extra),
/vol_data = olens-new-arch READ-ONLY (27B weights). Logs -> /vol_c/tasks234/logs.
    modal run --detach modal_celeste234.py --task vllm --script celeste234_gate.py --args "--task encoded ..." --tag c234_gate_enc
    modal run --detach modal_celeste234.py --task hf   --script capture_wsb_release.py --args "--cells /vol_c/tasks234/cells --out /vol_c/tasks234/evalcache" --tag c234_cap
"""

import os
import subprocess

import modal

APP_NAME = os.environ.get("OF_APP", "nla-glp")
GPU_TYPE = os.environ.get("OF_GPU", "B200")
HERE = os.path.dirname(os.path.abspath(__file__))
REMOTE = "/root/of"
ENV = {
    "HF_HOME": "/vol_data/hf_cache",
    "HF_HUB_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONUNBUFFERED": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "PYTHONPATH": f"{REMOTE}/src:{REMOTE}/vendor/go_src:{REMOTE}/vendor:{REMOTE}/wsbench_src",
}

hf_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.8.0",
        "transformers==5.5.4",
        "peft==0.19.1",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "pyarrow",
        "numpy",
        "pandas",
        "wandb",
        "einops",
        "scipy",
        "pyyaml",
        "huggingface_hub[hf_transfer]",
        "flash-linear-attention",
    )
    .env(ENV)
    .add_local_dir(
        os.path.join(HERE, "src"), f"{REMOTE}/src", copy=False, ignore=["__pycache__", "*.pyc"]
    )
    .add_local_dir(
        os.path.join(HERE, "vendor"),
        f"{REMOTE}/vendor",
        copy=False,
        ignore=["__pycache__", "*.pyc"],
    )
    .add_local_dir(
        os.path.expanduser("~/wsbench-celeste-tasks234/src"),
        f"{REMOTE}/wsbench_src",
        copy=False,
        ignore=["__pycache__", "*.pyc"],
    )
)
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system --python $(which python) "
        "'vllm==0.21.0' 'vllm-lens @ git+https://github.com/ceselder/vllm-metamodels@6a0a0e56a81d4c234346575238ee5fe8f580f332' 'transformers==5.5.4' "
        "peft==0.19.1 wandb accelerate pyarrow pandas numpy 'huggingface_hub[hf_transfer]' safetensors sentencepiece "
        "protobuf pyyaml tqdm flash-linear-attention scipy einops"
    )
    .env(
        {
            **ENV,
            "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "VLLM_USE_DEEP_GEMM": "0",
            "VLLM_DEEP_GEMM_WARMUP": "skip",
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            "VLLM_LENS_CUDA_GRAPHS": "1",
        }
    )
    .add_local_dir(
        os.path.join(HERE, "src"), f"{REMOTE}/src", copy=False, ignore=["__pycache__", "*.pyc"]
    )
    .add_local_dir(
        os.path.join(HERE, "vendor"),
        f"{REMOTE}/vendor",
        copy=False,
        ignore=["__pycache__", "*.pyc"],
    )
    .add_local_dir(
        os.path.expanduser("~/wsbench-celeste-tasks234/src"),
        f"{REMOTE}/wsbench_src",
        copy=False,
        ignore=["__pycache__", "*.pyc"],
    )
)

app = modal.App(APP_NAME)
volc = modal.Volume.from_name("wsb-celeste", create_if_missing=True)
VOLS = {
    "/vol_c": volc,
    "/vol": modal.Volume.from_name("olens-flow").read_only(),
    "/vol_data": modal.Volume.from_name("olens-new-arch").read_only(),
}
SECRETS = [modal.Secret.from_name("maemm-hf")]


def _script(script):
    for d in ("src", "vendor/go_src"):
        p = f"{REMOTE}/{d}/{script}"
        if os.path.exists(p):
            return p
    raise SystemExit(f"script not found: {script}")


def _run(cmd, tag):
    import shutil
    import time as _t

    name = f"{tag or 'run'}_{_t.strftime('%Y%m%d_%H%M%S')}.log"
    logf = f"/tmp/{name}"
    print("[run]", cmd, "| tee", logf, "-> /vol_c/tasks234/logs/", flush=True)
    p = subprocess.run(
        f"set -o pipefail; {cmd} 2>&1 | tee {logf}",
        shell=True,
        executable="/bin/bash",
        cwd=f"{REMOTE}/src",
        env=dict(os.environ),
    )
    os.makedirs("/vol_c/tasks234/logs", exist_ok=True)
    shutil.copy(logf, f"/vol_c/tasks234/logs/{name}")
    volc.commit()
    print(f"[run] exit {p.returncode}", flush=True)
    return p.returncode


@app.function(image=hf_image, gpu=GPU_TYPE, volumes=VOLS, timeout=12 * 60 * 60, secrets=SECRETS)
def hf(script: str, args: str = "", tag: str = ""):
    return _run(f"python {_script(script)} {args}", tag)


@app.function(image=vllm_image, gpu=GPU_TYPE, volumes=VOLS, timeout=12 * 60 * 60, secrets=SECRETS)
def vllm(script: str, args: str = "", tag: str = ""):
    return _run(f"python {_script(script)} {args}", tag)


@app.local_entrypoint()
def main(task: str, script: str = "", args: str = "", tag: str = ""):
    fn = {"hf": hf, "vllm": vllm}[task]
    h = fn.spawn(script, args, tag)
    print(f"SPAWNED {h.object_id}", flush=True)
