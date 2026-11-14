#!/usr/bin/env python3
"""Startup-only memory probe: launch the server per arm with a cold or warm
compile cache, wait for /health, record the memory-profile log lines, stop. No requests.

Each launch gets its own VLLM_CACHE_ROOT under the output directory (KV_PROBE_OUT, default ./kv-probe),
which must not exist yet for the cold launches to be cold. Writes kvprobe.json there."""
import json, os, re, signal, subprocess, sys, time, contextlib, urllib.request
from pathlib import Path

OUT = Path(os.environ.get("KV_PROBE_OUT", "kv-probe"))
MODEL = os.environ.get("KV_PROBE_MODEL", "Qwen/Qwen3.5-4B")
REV = os.environ.get("KV_PROBE_REVISION", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
PORT = int(os.environ.get("KV_PROBE_PORT", "8766"))
# (label, arm, cache root): a root that does not exist yet means a cold launch
PLAN = [
    ("none-cold", "none", "cacheA"), ("none-warm", "none", "cacheA"),
    ("fp8-warm", "fp8", "cacheA"), ("nvfp4-warm", "nvfp4", "cacheA"),
    ("nvfp4-cold", "nvfp4", "cacheB"), ("none-warm2", "none", "cacheB"),
]


def launch(label, arm, root):
    spec = {"method": "mtp", "num_speculative_tokens": 3}
    if arm != "none":
        spec["draft_lm_head_quantization"] = arm
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", MODEL,
           "--revision", REV, "--tokenizer-revision", REV, "--port", str(PORT), "--host", "127.0.0.1",
           "--max-model-len", "8192", "--gpu-memory-utilization", "0.85", "--max-num-seqs", "1",
           "--max-num-batched-tokens", "8192", "--seed", "0", "--no-enable-prefix-caching",
           "--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0}),
           "--speculative-config", json.dumps(spec)]
    env = dict(os.environ, VLLM_USE_V2_MODEL_RUNNER="1", VLLM_CACHE_ROOT=str((OUT / root).resolve()))
    logp = OUT / f"{label}.log"
    with open(logp, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        ok, deadline = False, time.time() + 420
        while time.time() < deadline and proc.poll() is None:
            with contextlib.suppress(Exception):
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                    if r.status == 200:
                        ok = True
                        break
            time.sleep(2)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=30)
    t = logp.read_text(errors="replace")
    g = lambda p: (re.search(p, t) or [None, None])[1]
    row = {"label": label, "ready": ok,
           "load_gib": g(r"Model loading took ([\d.]+) GiB"),
           "kv_gib": g(r"Available KV cache memory: ([\d.]+) GiB"),
           "kv_tokens": g(r"GPU KV cache size: ([\d,]+) tokens"),
           "consumed_gib": g(r"Actual usage is ([\d.]+) GiB for consumed"),
           "peak_act_gib": g(r"([\d.]+) GiB for peak activation"),
           "cudagraph_gib": g(r"and ([\d.]+) GiB for CUDAGraph"),
           "free": g(r"Free memory on device \(([\d./]+) GiB\)"),
           "compile_s": g(r"\(compilation: ([\d.]+) s\)"),
           "graphs_compiled": len(re.findall(r"Compiling a graph", t))}
    print(json.dumps(row), flush=True)
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [launch(*p) for p in PLAN]
    (OUT / "kvprobe.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
