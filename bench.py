#!/usr/bin/env python3
"""Draft LM-head precision A/B for vLLM speculative decoding.

The arms differ in one field, speculative_config.draft_lm_head_quantization:
"none" (the drafter reads the target's BF16 head), "fp8" or "nvfp4". The
model, prompts, K, sampling, full vocabulary and every other setting are fixed.

Modes (run in this order):
  prompts  build prompts.jsonl from SPEED-Bench "qualitative" at a pinned
           revision: chat template applied, token ids stored.
  track    target-only greedy continuations, no speculation: track.jsonl.
           This is the fixed text the teacher-forced mode scores on. Share
           the file between machines so both score identical contexts.
  tf       teacher-forced acceptance. Contexts are cut from the track every
           --stride tokens. Each context gets exactly one draft round of K
           tokens, scored against the target's own greedy pick. Contexts are
           identical across arms, so autoregressive drift cannot confound the
           comparison. Per-context accepted length is recorded for paired
           statistics.
  greedy   free-running greedy generation, one request in flight. Acceptance
           by draft position from vLLM's spec-decode counters (per category),
           and the output token ids for parity against the "none" arm.
  serve    one OpenAI server per arm and rep, a timed streaming client at
           concurrency --concurrency. Decode tok/s, TTFT, TPOT, served
           acceptance by position, model memory and KV cache size per arm.
  report   markdown tables from the results directory.

Each arm runs in a fresh subprocess. Results land in --out as JSON.
"""

from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import math
import os
import platform
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ARMS = ("none", "fp8", "nvfp4")
SPEED_BENCH = "nvidia/SPEED-Bench"
SPEED_BENCH_REV = "fa4e669bee2a4393c56aa8ed05a7c667897f9590"
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_MODEL_REV = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


# --------------------------------------------------------------------------
# helpers


def log(msg: str) -> None:
    print(f"[bench {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    tmp.replace(path)


def spec_config(args, arm: str) -> dict:
    spec = {"method": "mtp", "num_speculative_tokens": args.k}
    if arm != "none":
        spec["draft_lm_head_quantization"] = arm
    return spec


def env_info() -> dict:
    info = {"python": platform.python_version(), "host_os": platform.platform()}
    try:
        import torch
        import vllm

        info["vllm"] = vllm.__version__
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["capability"] = f"{p.major}.{p.minor}"
            info["gpu_mem_gib"] = round(p.total_memory / 2**30, 1)
    except Exception as e:  # report, do not fail
        info["env_error"] = repr(e)
    with contextlib.suppress(Exception):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
        info["driver"] = out.splitlines()[0] if out else None
    return info


def make_llm(args, arm: str | None, max_num_seqs: int):
    from vllm import LLM

    kw = dict(
        model=args.model,
        revision=args.revision,
        tokenizer_revision=args.revision,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        max_num_seqs=max_num_seqs,
        # vLLM's warm-up fails (illegal memory access) when the batch budget
        # exceeds max_num_seqs * max_model_len, e.g. at one sequence.
        max_num_batched_tokens=min(args.max_num_batched_tokens, max_num_seqs * args.max_model_len),
        seed=args.seed,
        enable_prefix_caching=False,
        disable_log_stats=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    if arm is not None:
        kw["speculative_config"] = spec_config(args, arm)
    return LLM(**kw)


def spec_counters(llm) -> dict:
    """Read vLLM's cumulative spec-decode counters."""
    out = {"drafts": 0, "draft_tokens": 0, "accepted": 0, "per_pos": []}
    for m in llm.get_metrics():
        if m.name == "vllm:spec_decode_num_drafts":
            out["drafts"] = int(m.value)
        elif m.name == "vllm:spec_decode_num_draft_tokens":
            out["draft_tokens"] = int(m.value)
        elif m.name == "vllm:spec_decode_num_accepted_tokens":
            out["accepted"] = int(m.value)
        elif m.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            out["per_pos"] = [int(v) for v in m.values]
    return out


def counter_delta(a: dict, b: dict) -> dict:
    n = max(len(a["per_pos"]), len(b["per_pos"]))
    pa = a["per_pos"] + [0] * (n - len(a["per_pos"]))
    pb = b["per_pos"] + [0] * (n - len(b["per_pos"]))
    return {
        "drafts": b["drafts"] - a["drafts"],
        "draft_tokens": b["draft_tokens"] - a["draft_tokens"],
        "accepted": b["accepted"] - a["accepted"],
        "per_pos": [y - x for x, y in zip(pa, pb)],
    }


def run_arms(args, mode: str) -> None:
    """Run one subprocess per arm so each gets a fresh engine."""
    for arm in args.arms.split(","):
        if arm not in ARMS:
            sys.exit(f"unknown arm {arm!r}; choose from {ARMS}")
        cmd = [sys.executable, __file__, mode, "--arm", arm] + passthrough(args)
        log(f"{mode} arm={arm}")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(f"{mode} arm={arm} failed with exit {r.returncode}")


def passthrough(args) -> list[str]:
    keep = [
        "model", "revision", "k", "max_model_len", "gpu_mem", "seed", "out",
        "prompts", "track", "stride", "max_tokens", "max_num_batched_tokens",
        "limit", "tf_max_num_seqs", "tag",
    ]
    flags = ["tf_counters_only"]
    out = []
    for name in keep:
        v = getattr(args, name, None)
        if v is not None:
            out += [f"--{name.replace('_', '-')}", str(v)]
    for name in flags:
        if getattr(args, name, False):
            out.append(f"--{name.replace('_', '-')}")
    return out


def out_name(args, mode: str, arm: str) -> Path:
    tag = f"-{args.tag}" if args.tag else ""
    return Path(args.out) / f"{mode}{tag}-k{args.k}-{arm}.json"


# --------------------------------------------------------------------------
# prompts


PLACEHOLDER = "FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH"
SPECDEC_BENCH_COMMIT = "e481d487570a1ad13092d88be900ba8d985ce850"  # NVIDIA/Model-Optimizer


def check_prompt_rows(rows: list[dict]) -> None:
    """Fail on the masked rows the HF copy ships."""
    masked = [r["question_id"] for r in rows if any(PLACEHOLDER in t for t in r["turns"])]
    if masked:
        sys.exit(
            f"{len(masked)} of {len(rows)} SPEED-Bench rows are the placeholder "
            f"{PLACEHOLDER!r}. The HF copy masks them; resolve the full data with NVIDIA's "
            "specdec_bench prepare_data.py (see README) and pass --speed-parquet."
        )


def cmd_prompts(args) -> None:
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    if not args.speed_parquet:
        sys.exit(
            "pass --speed-parquet: the qualitative split resolved by NVIDIA specdec_bench "
            f"prepare_data.py (Model-Optimizer {SPECDEC_BENCH_COMMIT[:10]}), see README"
        )
    rows = pq.read_table(args.speed_parquet).to_pylist()
    check_prompt_rows(rows)
    side_path = Path(str(args.speed_parquet) + ".dropped.json")
    side = None
    if side_path.exists():
        sd = json.loads(side_path.read_text())
        by_cat: dict[str, int] = {}
        for d in sd["dropped"]:
            by_cat[d["category"]] = by_cat.get(d["category"], 0) + 1
        side = {"skip_source": sd["skip_source"], "count": len(sd["dropped"]), "by_category": by_cat}
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    kept, dropped, seen, dups = [], 0, set(), []
    for r in rows:
        # Only the first turn is used; keep the first row of any repeated one.
        if r["turns"][0] in seen:
            dups.append(r["question_id"])
            continue
        seen.add(r["turns"][0])
        msgs = [{"role": "user", "content": r["turns"][0]}]
        ids = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, enable_thinking=False
        )
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        if len(ids) > args.max_prompt_tokens:
            dropped += 1
            continue
        kept.append({"id": r["question_id"], "category": r["category"], "prompt_ids": list(ids)})
    with open(args.prompts, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    import hashlib

    digest = hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest()
    write_json(Path(args.prompts + ".meta.json"), {
        "source": f"{SPEED_BENCH}@{SPEED_BENCH_REV} qualitative, resolved by specdec_bench "
                  f"prepare_data.py (NVIDIA/Model-Optimizer@{SPECDEC_BENCH_COMMIT})",
        "placeholders": 0, "rows": len(rows), "kept": len(kept), "dropped_long": dropped,
        "dropped_unresolvable": side, "dropped_duplicate_first_turn": dups,
        "max_prompt_tokens": args.max_prompt_tokens, "model": args.model,
        "revision": args.revision, "sha256": digest,
    })
    log(f"wrote {len(kept)} prompts to {args.prompts} (dropped {dropped} over "
        f"{args.max_prompt_tokens} tokens), sha256 {digest[:16]}")


def load_prompts(args) -> list[dict]:
    meta = Path(args.prompts + ".meta.json")
    if not meta.exists() or json.loads(meta.read_text()).get("placeholders") != 0:
        sys.exit(f"{args.prompts} was not built from the resolved SPEED-Bench data; rerun `prompts`")
    rows = read_jsonl(Path(args.prompts))
    if args.limit:
        by_cat: dict[str, list] = {}
        for r in rows:
            by_cat.setdefault(r["category"], []).append(r)
        rows = [r for cat in sorted(by_cat) for r in by_cat[cat][: args.limit]]
    return rows


# --------------------------------------------------------------------------
# track: target-only greedy continuations


def cmd_track(args) -> None:
    from vllm import SamplingParams, TokensPrompt

    prompts = load_prompts(args)
    llm = make_llm(args, None, max_num_seqs=256)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=args.seed)
    outs = llm.generate([TokensPrompt(prompt_token_ids=p["prompt_ids"]) for p in prompts], sp)
    with open(args.track, "w") as f:
        for p, o in zip(prompts, outs):
            f.write(json.dumps({"id": p["id"], "category": p["category"],
                                "track": list(o.outputs[0].token_ids)}) + "\n")
    pmeta = json.loads(Path(args.prompts + ".meta.json").read_text())
    meta = {"model": args.model, "revision": args.revision, "max_tokens": args.max_tokens,
            "n": len(prompts), "prompts_sha256": pmeta["sha256"], "env": env_info()}
    write_json(Path(args.track + ".meta.json"), meta)
    log(f"wrote {len(prompts)} tracks to {args.track}")


# --------------------------------------------------------------------------
# tf: teacher-forced, one draft round per context


def build_contexts(args) -> list[dict]:
    saved_limit, args.limit = args.limit, 0
    prompts = {p["id"]: p for p in load_prompts(args)}
    args.limit = saved_limit
    pmeta = json.loads(Path(args.prompts + ".meta.json").read_text())
    tmeta_path = Path(args.track + ".meta.json")
    tmeta = json.loads(tmeta_path.read_text()) if tmeta_path.exists() else {}
    if tmeta.get("prompts_sha256") != pmeta["sha256"]:
        sys.exit(f"{args.track} was built from a different prompt set "
                 f"({tmeta.get('prompts_sha256')} vs {pmeta['sha256']}); rebuild or re-pull it")
    ctxs = []
    for t in read_jsonl(Path(args.track)):
        p = prompts.get(t["id"])
        if p is None:
            continue
        track = t["track"]
        # Need the target pick at j plus K more tokens to score against.
        for j in range(0, len(track) - args.k - 1, args.stride):
            ctxs.append({
                "cid": f"{t['id']}:{j}",
                "category": t["category"],
                "ids": p["prompt_ids"] + track[:j],
                "expect": track[j : j + args.k + 1],
            })
    if args.limit:
        by_cat: dict[str, list] = {}
        for c in ctxs:
            by_cat.setdefault(c["category"], []).append(c)
        ctxs = [c for cat in sorted(by_cat) for c in by_cat[cat][: args.limit * 16]]
    return ctxs


def cmd_tf(args) -> None:
    if not args.arm:
        return run_arms(args, "tf")
    from vllm import SamplingParams, TokensPrompt

    ctxs = build_contexts(args)
    log(f"tf arm={args.arm}: {len(ctxs)} contexts, K={args.k}, stride={args.stride}")
    llm = make_llm(args, args.arm, max_num_seqs=args.tf_max_num_seqs)
    engine = llm.llm_engine
    # max_tokens = K + 2 leaves room for the prefill token plus a full round
    # (K accepted + 1 bonus), so the first verify step is never truncated.
    # --tf-counters-only uses max_tokens = 2 instead: every context then gets
    # exactly one verify round and vLLM's global counters alone measure it.
    # Use it once to cross-check the per-context numbers.
    max_tokens = 2 if args.tf_counters_only else args.k + 2
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True, seed=args.seed)
    before = spec_counters(llm)
    t0 = time.perf_counter()
    ids_map: dict[str, int] = {}
    for i, c in enumerate(ctxs):
        rid = engine.add_request(str(i), TokensPrompt(prompt_token_ids=c["ids"]), sp)
        ids_map[str(i)] = i
        ids_map[rid] = i
    # Record the cumulative output length at every step that adds tokens.
    # Event 0 is the prefill token; event 1 is the first verify step, which
    # adds accepted + 1 tokens.
    lens: dict[str, list[int]] = {}
    toks: dict[str, list[int]] = {}
    while engine.has_unfinished_requests():
        for o in engine.step():
            key = str(ids_map[o.request_id])
            ids = list(o.outputs[0].token_ids)
            ev = lens.setdefault(key, [])
            if not ev or len(ids) != ev[-1]:
                ev.append(len(ids))
            toks[key] = ids
    wall = time.perf_counter() - t0
    after = spec_counters(llm)
    if args.tf_counters_only:
        res = {"mode": "tf-counters", "arm": args.arm, "k": args.k, "stride": args.stride,
               "n_contexts": len(ctxs), "wall_s": wall, "counters": counter_delta(before, after),
               "env": env_info()}
        write_json(out_name(args, "tfcounters", args.arm), res)
        log(f"tf counters-only arm={args.arm}: {res['counters']}")
        return
    rows, bad = [], 0
    for i, c in enumerate(ctxs):
        ev = lens.get(str(i), [])
        ids = toks.get(str(i), [])
        if len(ev) < 2 or ev[0] != 1 or not (2 <= ev[1] <= args.k + 2):
            bad += 1
            rows.append({"cid": c["cid"], "cat": c["category"], "acc": None, "ev": ev})
            continue
        acc = ev[1] - 2
        rows.append({
            "cid": c["cid"],
            "cat": c["category"],
            "acc": acc,
            # target's first pick agrees with the shared track
            "t0": ids[0] == c["expect"][0],
            # accepted tokens equal the track (the target's own greedy text)
            "trk": ids[1 : 1 + acc] == c["expect"][1 : 1 + acc],
        })
    res = {
        "mode": "tf", "arm": args.arm, "k": args.k, "stride": args.stride,
        "model": args.model, "revision": args.revision, "n_contexts": len(ctxs),
        "bad_events": bad, "wall_s": wall, "counters": counter_delta(before, after),
        "rows": rows, "env": env_info(),
    }
    write_json(out_name(args, "tf", args.arm), res)
    log(f"tf arm={args.arm} done in {wall:.0f}s, {bad} contexts without a clean first round")


# --------------------------------------------------------------------------
# greedy: free-running, one request in flight


def cmd_greedy(args) -> None:
    if not args.arm:
        return run_arms(args, "greedy")
    from vllm import SamplingParams, TokensPrompt

    prompts = load_prompts(args)
    llm = make_llm(args, args.arm, max_num_seqs=1)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=args.seed)
    # warm up outside the measured counters
    llm.generate([TokensPrompt(prompt_token_ids=prompts[0]["prompt_ids"])],
                 SamplingParams(temperature=0.0, max_tokens=16), use_tqdm=False)
    cats = sorted({p["category"] for p in prompts})
    per_cat, outputs = {}, {}
    for cat in cats:
        ps = [p for p in prompts if p["category"] == cat]
        before = spec_counters(llm)
        t0 = time.perf_counter()
        outs = llm.generate([TokensPrompt(prompt_token_ids=p["prompt_ids"]) for p in ps], sp,
                            use_tqdm=False)
        wall = time.perf_counter() - t0
        d = counter_delta(before, spec_counters(llm))
        n_out = sum(len(o.outputs[0].token_ids) for o in outs)
        d.update({"n_prompts": len(ps), "out_tokens": n_out, "wall_s": wall})
        per_cat[cat] = d
        for p, o in zip(ps, outs):
            outputs[p["id"]] = list(o.outputs[0].token_ids)
        log(f"greedy arm={args.arm} {cat}: {n_out} tokens, "
            f"AL {1 + d['accepted'] / max(d['drafts'], 1):.3f}, {n_out / wall:.1f} tok/s")
    res = {
        "mode": "greedy", "arm": args.arm, "k": args.k, "model": args.model,
        "revision": args.revision, "max_tokens": args.max_tokens, "per_category": per_cat,
        "outputs": outputs, "env": env_info(),
    }
    write_json(out_name(args, "greedy", args.arm), res)


# --------------------------------------------------------------------------
# serve: OpenAI server per arm, timed streaming client


def scrape_metrics(port: int) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=30) as r:
        text = r.read().decode()
    out = {"drafts": 0, "draft_tokens": 0, "accepted": 0, "per_pos": {}}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r'^(vllm:spec_decode_[a-z_]+?)(?:_total)?(\{[^}]*\})?\s+([0-9.eE+-]+)$', line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", float(m.group(3))
        if name == "vllm:spec_decode_num_drafts":
            out["drafts"] += int(val)
        elif name == "vllm:spec_decode_num_draft_tokens":
            out["draft_tokens"] += int(val)
        elif name == "vllm:spec_decode_num_accepted_tokens":
            out["accepted"] += int(val)
        elif name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            pos = re.search(r'position="(\d+)"', labels)
            if pos:
                k = int(pos.group(1))
                out["per_pos"][k] = out["per_pos"].get(k, 0) + int(val)
    out["per_pos"] = [out["per_pos"].get(i, 0) for i in range(max(out["per_pos"], default=-1) + 1)]
    return out


def stream_one(port: int, model: str, prompt_ids: list[int], max_tokens: int) -> dict:
    body = json.dumps({
        "model": model, "prompt": prompt_ids, "max_tokens": max_tokens, "temperature": 0.0,
        "stream": True, "stream_options": {"include_usage": True},
    })
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:300]!r}")
    t_first = t_last = None
    n_tokens = None
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        ev = json.loads(data)
        if ev.get("usage"):
            n_tokens = ev["usage"]["completion_tokens"]
        if ev.get("choices") and ev["choices"][0].get("text"):
            now = time.perf_counter()
            if t_first is None:
                t_first = now
            t_last = now
    conn.close()
    t_end = time.perf_counter()
    if t_first is None or not n_tokens:
        raise RuntimeError("no tokens streamed")
    decode_s = t_last - t_first
    return {
        "ttft_s": t_first - t0,
        "e2e_s": t_end - t0,
        "out_tokens": n_tokens,
        "tpot_s": decode_s / (n_tokens - 1) if n_tokens > 1 else None,
        "decode_tok_s": (n_tokens - 1) / decode_s if n_tokens > 1 and decode_s > 0 else None,
    }


class GpuSampler:
    """Poll nvidia-smi once a second: SM clock, temperature, power and the
    thermal / power-brake throttle flags. A timed cell with a thermal flag set
    should be rested and retaken."""

    FIELDS = ("clocks.sm,temperature.gpu,power.draw,"
              "clocks_throttle_reasons.hw_thermal_slowdown,"
              "clocks_throttle_reasons.sw_thermal_slowdown,"
              "clocks_throttle_reasons.hw_power_brake_slowdown,fan.speed")

    @staticmethod
    def physical_index() -> str:
        """The physical GPU the run uses: first CUDA_VISIBLE_DEVICES entry."""
        first = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
        return first or "0"

    def __init__(self) -> None:
        self.index = self.physical_index()
        self.samples: list[list[str]] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            with contextlib.suppress(Exception):
                out = subprocess.run(
                    ["nvidia-smi", "-i", self.index, f"--query-gpu={self.FIELDS}",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout.strip()
                self.samples.append([x.strip() for x in out.split(",")])
            self._stop.wait(1.0)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._t.join(timeout=15)

    def summary(self) -> dict:
        def num(x):
            try:
                return float(x)
            except ValueError:
                return None
        clk = [num(r[0]) for r in self.samples if num(r[0]) is not None]
        tmp = [num(r[1]) for r in self.samples if num(r[1]) is not None]
        pw = [num(r[2]) for r in self.samples if num(r[2]) is not None]
        flags = [r[3:6] for r in self.samples if len(r) >= 6]
        fan = [num(r[6]) for r in self.samples if len(r) >= 7 and num(r[6]) is not None]
        return {
            "n": len(self.samples),
            "gpu_index": self.index,
            "fan_max": max(fan) if fan else None,
            "sm_clock_median": statistics.median(clk) if clk else None,
            "sm_clock_min": min(clk) if clk else None,
            "temp_max": max(tmp) if tmp else None,
            "power_median": statistics.median(pw) if pw else None,
            "thermal_throttled": any("Active" in f[0] or "Active" in f[1] for f in flags),
            "power_brake": any("Active" in f[2] for f in flags),
        }


def start_server(args, arm: str, port: int, logf):
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", args.model,
        "--revision", args.revision, "--tokenizer-revision", args.revision,
        "--port", str(port), "--host", "127.0.0.1",
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_mem),
        "--max-num-seqs", str(max(args.concurrency, 1)),
        "--max-num-batched-tokens",
        str(min(args.max_num_batched_tokens, max(args.concurrency, 1) * args.max_model_len)),
        "--seed", str(args.seed), "--no-enable-prefix-caching",
        "--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0}),
        "--speculative-config", json.dumps(spec_config(args, arm)),
    ]
    return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)


def wait_ready(port: int, proc, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        with contextlib.suppress(Exception):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return
        time.sleep(2)
    raise RuntimeError("server not ready in time")


def stop_server(proc) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=30)


def parse_server_log(text: str) -> dict:
    info = {}
    m = re.search(r"Model loading took ([0-9.]+) GiB", text)
    if m:
        info["model_load_gib"] = float(m.group(1))
    m = re.search(r"GPU KV cache size: ([0-9,]+) tokens", text)
    if m:
        info["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
    info["head_lines"] = [ln.strip()[-160:] for ln in text.splitlines()
                          if "copy of the lm_head" in ln or "NvFp4" in ln and "Using" in ln
                          or "Fp8" in ln and "Selected" in ln][:6]
    return info


def cmd_serve(args) -> None:
    prompts = load_prompts(args)
    arms = args.arms.split(",")
    env = env_info()
    for rep in range(args.reps):
        for arm in arms:
            path = Path(args.out) / f"serve{('-' + args.tag) if args.tag else ''}-k{args.k}-c{args.concurrency}-{arm}-r{rep}.json"
            if path.exists():
                log(f"skip existing {path}")
                continue
            logp = path.with_suffix(".log")
            with open(logp, "w") as logf:
                proc = start_server(args, arm, args.port, logf)
                try:
                    wait_ready(args.port, proc, 1800)
                    for p in prompts[: min(4, len(prompts))]:  # warm-up
                        stream_one(args.port, args.model, p["prompt_ids"], 32)
                    before = scrape_metrics(args.port)
                    t0 = time.perf_counter()
                    with GpuSampler() as gpu, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                        reqs = list(ex.map(lambda p: stream_one(args.port, args.model,
                                                                p["prompt_ids"], args.max_tokens),
                                           prompts))
                    wall = time.perf_counter() - t0
                    after = scrape_metrics(args.port)
                finally:
                    stop_server(proc)
            info = parse_server_log(logp.read_text(errors="replace"))
            res = {
                "mode": "serve", "arm": arm, "rep": rep, "k": args.k,
                "concurrency": args.concurrency, "model": args.model, "revision": args.revision,
                "max_tokens": args.max_tokens, "wall_s": wall,
                "out_tokens": sum(r["out_tokens"] for r in reqs),
                "requests": reqs, "counters": counter_delta(before, after),
                "server": info, "gpu": gpu.summary(), "env": env,
            }
            write_json(path, res)
            dt = [r["decode_tok_s"] for r in reqs if r["decode_tok_s"]]
            g = res["gpu"]
            log(f"serve arm={arm} rep={rep}: median decode {statistics.median(dt):.1f} tok/s/user, "
                f"{res['out_tokens'] / wall:.1f} tok/s total, SM clock {g['sm_clock_median']} MHz, "
                f"max {g['temp_max']} C, thermal throttle {g['thermal_throttled']}")


# --------------------------------------------------------------------------
# report


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def binom_two_sided(b: int, c: int) -> float:
    """Exact McNemar p-value: discordant pairs b vs c under p = 0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def load_results(out: Path, prefix: str) -> list[dict]:
    res = []
    for p in sorted(out.glob(f"{prefix}*.json")):
        with open(p) as f:
            res.append(json.load(f))
    return res


def report_tf(out: Path, lines: list[str]) -> None:
    runs = {r["arm"]: r for r in load_results(out, "tf") if r.get("mode") == "tf"}
    if not runs:
        return
    k = next(iter(runs.values()))["k"]
    lines += ["## Teacher-forced acceptance (one draft round per context)", ""]
    base = runs.get("none")
    any_run = next(iter(runs.values()))
    lines.append(f"K={k}, stride {any_run['stride']}, {any_run['n_contexts']} contexts per arm. "
                 "Rate at position j = contexts whose first j+1 drafts were all accepted, "
                 "divided by contexts.")
    lines.append("")
    hdr = "| arm | contexts | mean accepted | " + " | ".join(f"pos {j}" for j in range(k)) + \
          " | target pick = track | accepted = track |"
    lines += [hdr, "|" + "---|" * (k + 5)]
    for arm in ARMS:
        r = runs.get(arm)
        if not r:
            continue
        rows = [x for x in r["rows"] if x["acc"] is not None]
        n = len(rows)
        mean = sum(x["acc"] for x in rows) / n
        pos = [sum(1 for x in rows if x["acc"] > j) / n for j in range(k)]
        t0 = sum(1 for x in rows if x["t0"]) / n
        trk = sum(1 for x in rows if x["trk"]) / n
        lines.append(f"| {arm} | {n} | {mean:.3f} | " + " | ".join(pct(p) for p in pos) +
                     f" | {pct(t0)} | {pct(trk)} |")
    if base:
        lines += ["", "Paired against `none` on identical contexts. Position 0: b = none hit and "
                  "arm miss, c = none miss and arm hit; p is exact McNemar.", ""]
        lines += ["| arm | pairs | pos 0 delta (95% CI) | b | c | p | mean accepted delta (95% CI) |",
                  "|---|---|---|---|---|---|---|"]
        bmap = {x["cid"]: x["acc"] for x in base["rows"] if x["acc"] is not None}
        for arm in ARMS:
            if arm == "none" or arm not in runs:
                continue
            pairs = [(bmap[x["cid"]], x["acc"]) for x in runs[arm]["rows"]
                     if x["acc"] is not None and x["cid"] in bmap]
            n = len(pairs)
            b = sum(1 for a0, a1 in pairs if a0 > 0 and a1 == 0)
            c = sum(1 for a0, a1 in pairs if a0 == 0 and a1 > 0)
            d0 = (c - b) / n
            se0 = math.sqrt(max(b + c - (c - b) ** 2 / n, 0)) / n
            diffs = [a1 - a0 for a0, a1 in pairs]
            md = sum(diffs) / n
            sd = statistics.pstdev(diffs) / math.sqrt(n)
            lines.append(f"| {arm} | {n} | {100 * d0:+.2f} pts ({100 * (d0 - 1.96 * se0):+.2f}, "
                         f"{100 * (d0 + 1.96 * se0):+.2f}) | {b} | {c} | {binom_two_sided(b, c):.3g} | "
                         f"{md:+.4f} ({md - 1.96 * sd:+.4f}, {md + 1.96 * sd:+.4f}) |")
        chk = {r["arm"]: r for r in load_results(out, "tfcounters") if r.get("mode") == "tf-counters"}
        if chk:
            lines += ["", "Cross-check: same contexts with max_tokens=2, vLLM's own counters "
                      "(accepted at position / draft rounds):", "",
                      "| arm | draft rounds | " + " | ".join(f"pos {j}" for j in range(k)) + " |",
                      "|---|---|" + "---|" * k]
            for arm in ARMS:
                if arm in chk:
                    c = chk[arm]["counters"]
                    dr = max(c["drafts"], 1)
                    lines.append(f"| {arm} | {c['drafts']} | " +
                                 " | ".join(pct(c["per_pos"][j] / dr) if j < len(c["per_pos"]) else "-"
                                            for j in range(k)) + " |")
        cats = sorted({x["cat"] for x in base["rows"]})
        lines += ["", "Position-0 rate by category:", "",
                  "| category | " + " | ".join(a for a in ARMS if a in runs) + " |",
                  "|---|" + "---|" * len(runs)]
        for cat in cats:
            cells = []
            for arm in ARMS:
                if arm not in runs:
                    continue
                rows = [x for x in runs[arm]["rows"] if x["cat"] == cat and x["acc"] is not None]
                cells.append(pct(sum(1 for x in rows if x["acc"] > 0) / max(len(rows), 1)))
            lines.append(f"| {cat} | " + " | ".join(cells) + " |")
    lines.append("")


def report_greedy(out: Path, lines: list[str]) -> None:
    runs = {r["arm"]: r for r in load_results(out, "greedy") if r.get("mode") == "greedy"}
    if not runs:
        return
    k = next(iter(runs.values()))["k"]
    lines += ["## Free-running greedy, one request in flight", "",
              "From vLLM's spec-decode counters. Rate at position j = accepted at j / draft rounds.",
              ""]
    lines += ["| arm | accepted/draft | AL | " + " | ".join(f"pos {j}" for j in range(k)) +
              " | draft rounds | output tokens |", "|" + "---|" * (k + 5)]
    for arm in ARMS:
        r = runs.get(arm)
        if not r:
            continue
        tot = {"drafts": 0, "draft_tokens": 0, "accepted": 0, "per_pos": [0] * k, "out": 0}
        for d in r["per_category"].values():
            tot["drafts"] += d["drafts"]
            tot["draft_tokens"] += d["draft_tokens"]
            tot["accepted"] += d["accepted"]
            tot["out"] += d["out_tokens"]
            for j, v in enumerate(d["per_pos"][:k]):
                tot["per_pos"][j] += v
        dr = max(tot["drafts"], 1)
        lines.append(f"| {arm} | {tot['accepted']}/{tot['draft_tokens']} | "
                     f"{1 + tot['accepted'] / dr:.3f} | " +
                     " | ".join(pct(v / dr) for v in tot["per_pos"]) +
                     f" | {tot['drafts']} | {tot['out']} |")
    base = runs.get("none")
    if base:
        lines += ["", "Greedy output parity against `none`:", "",
                  "| arm | prompts | identical | mean common prefix | first divergence (median token) |",
                  "|---|---|---|---|---|"]
        for arm in ARMS:
            if arm == "none" or arm not in runs:
                continue
            same, fracs, divs = 0, [], []
            for pid, ref in base["outputs"].items():
                got = runs[arm]["outputs"].get(pid)
                if got is None:
                    continue
                n = 0
                while n < min(len(ref), len(got)) and ref[n] == got[n]:
                    n += 1
                if got == ref:
                    same += 1
                else:
                    divs.append(n)
                fracs.append(n / max(len(ref), len(got), 1))
            med = statistics.median(divs) if divs else "-"
            lines.append(f"| {arm} | {len(fracs)} | {same} | {sum(fracs) / len(fracs):.4f} | {med} |")
    lines.append("")


def report_serve(out: Path, lines: list[str]) -> None:
    runs = [r for r in load_results(out, "serve") if r.get("mode") == "serve"]
    if not runs:
        return
    groups: dict[tuple, list] = {}
    for r in runs:
        groups.setdefault((r["concurrency"], r["arm"]), []).append(r)
    k = runs[0]["k"]
    lines += ["## Served (OpenAI server, streaming client)", "",
              "Per rep: median over requests. Across reps: median (min..max).", ""]
    lines += ["| C | arm | reps | decode tok/s/user | total tok/s | TTFT ms | TPOT ms | AL | " +
              " | ".join(f"pos {j}" for j in range(k)) + " | model GiB | KV tokens | SM MHz | throttled reps |",
              "|" + "---|" * (k + 12)]

    def mmm(vals, fmt):
        return f"{fmt.format(statistics.median(vals))} ({fmt.format(min(vals))}..{fmt.format(max(vals))})"

    for (c, arm) in sorted(groups, key=lambda x: (x[0], ARMS.index(x[1]))):
        rs = groups[(c, arm)]
        dec = [statistics.median([q["decode_tok_s"] for q in r["requests"] if q["decode_tok_s"]]) for r in rs]
        tot = [r["out_tokens"] / r["wall_s"] for r in rs]
        ttft = [1000 * statistics.median([q["ttft_s"] for q in r["requests"]]) for r in rs]
        tpot = [1000 * statistics.median([q["tpot_s"] for q in r["requests"] if q["tpot_s"]]) for r in rs]
        cs = [r["counters"] for r in rs]
        dr = max(sum(x["drafts"] for x in cs), 1)
        al = 1 + sum(x["accepted"] for x in cs) / dr
        pp = [sum(x["per_pos"][j] if j < len(x["per_pos"]) else 0 for x in cs) / dr for j in range(k)]
        srv = rs[0]["server"]
        lines.append(f"| {c} | {arm} | {len(rs)} | {mmm(dec, '{:.1f}')} | {mmm(tot, '{:.1f}')} | "
                     f"{mmm(ttft, '{:.1f}')} | {mmm(tpot, '{:.2f}')} | {al:.3f} | " +
                     " | ".join(pct(v) for v in pp) +
                     f" | {srv.get('model_load_gib', '-')} | {srv.get('kv_cache_tokens', '-')} | "
                     f"{statistics.median([r['gpu']['sm_clock_median'] or 0 for r in rs]):.0f} | "
                     f"{sum(1 for r in rs if r['gpu']['thermal_throttled'])} |")
    lines.append("")


def cmd_report(args) -> None:
    out = Path(args.out)
    lines: list[str] = []
    envs = [r.get("env") for r in load_results(out, "") if isinstance(r, dict) and r.get("env")]
    if envs:
        e = envs[0]
        lines += [f"GPU {e.get('gpu')} (capability {e.get('capability')}, {e.get('gpu_mem_gib')} GiB), "
                  f"driver {e.get('driver')}, vLLM {e.get('vllm')}, torch {e.get('torch')} / CUDA {e.get('cuda')}", ""]
    report_tf(out, lines)
    report_greedy(out, lines)
    report_serve(out, lines)
    text = "\n".join(lines)
    print(text)
    (out / "REPORT.md").write_text(text + "\n")


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["prompts", "track", "tf", "greedy", "serve", "report"])
    ap.add_argument("--arm", choices=ARMS, help="run a single arm (internal; default runs --arms)")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=DEFAULT_MODEL_REV)
    ap.add_argument("--k", type=int, default=3, help="num_speculative_tokens")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-prompt-tokens", type=int, default=4096)
    ap.add_argument("--max-num-batched-tokens", type=int, default=16384)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--stride", type=int, default=16, help="tf: score every Nth track position")
    ap.add_argument("--tf-max-num-seqs", type=int, default=128)
    ap.add_argument("--tf-counters-only", action="store_true",
                    help="tf: one verify round per context, global counters only (cross-check)")
    ap.add_argument("--limit", type=int, default=0, help="first N prompts per category (0 = all)")
    ap.add_argument("--speed-parquet", help="prompts: resolved SPEED-Bench qualitative parquet")
    ap.add_argument("--prompts", default="data/prompts.jsonl")
    ap.add_argument("--track", default="data/track.jsonl")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")  # the feature is MRV2 only
    {"prompts": cmd_prompts, "track": cmd_track, "tf": cmd_tf, "greedy": cmd_greedy,
     "serve": cmd_serve, "report": cmd_report}[args.mode](args)


if __name__ == "__main__":
    main()
