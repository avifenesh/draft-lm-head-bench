# draft-lm-head-bench

A/B of the drafter's LM-head precision in vLLM speculative decoding
(`speculative_config.draft_lm_head_quantization`, vLLM #59973). Written for
vLLM issue #66101, so two machines (B200 and RTX 5090) can run the same
thing and compare.

Three arms. They differ in one field:

| arm | `draft_lm_head_quantization` |
|---|---|
| `none` | unset: the drafter reads the target's BF16 head |
| `fp8` | `"fp8"` |
| `nvfp4` | `"nvfp4"` |

The target always verifies with its own BF16 head, so greedy output should
not change. Only acceptance, and through it throughput, can move.

## Fixed settings

- Model `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, BF16, MTP, K=3.
  Its head is 248320 x 2560, the same shape as the Qwen3.8-Flash-Next head in #59973.
- Full vocabulary. No FR-Spec or sparse head.
- Prompts: all 880 of SPEED-Bench `qualitative` (11 categories x 80, first turn),
  at revision `fa4e669bee2a4393c56aa8ed05a7c667897f9590`, chat template with thinking off.
- Greedy, seed 0, 256 output tokens, prefix caching off, max model len 8192.
- vLLM 0.33.0 (Model Runner V2; the feature is MRV2 only).

## Run

```bash
pip install vllm==0.33.0 pyarrow

python bench.py prompts                 # data/prompts.jsonl
python bench.py track                   # data/track.jsonl (skip if you use a shared track)
python bench.py tf                      # teacher-forced, 3 arms
python bench.py greedy                  # free-running, one request in flight, 3 arms
python bench.py serve --concurrency 1 --limit 8 --reps 3   # timed, served
python bench.py report                  # results/REPORT.md
```

Each arm runs in its own process. `--arms nvfp4` runs one arm. `--limit N`
takes the first N prompts per category, useful for a smoke run.

## What each mode measures

**tf (teacher-forced).** The track is the target's own greedy continuation
of each prompt, generated once with speculation off. Contexts are cut from it
every 16 tokens (about 14k contexts). Each context gets exactly one draft
round of K tokens, scored against the target's greedy pick. Contexts are
identical across arms, so a difference comes from the head, not from the
generations drifting apart. The report gives the rate per draft position, the
mean accepted length, and a paired comparison against `none`: position-0
discordant pairs, exact McNemar p, and 95% intervals.

To compare two machines on identical contexts, generate `data/track.jsonl` on
one and copy it to the other. The report shows how often the target's own
first pick agrees with the shared track, so a cross-machine numerics gap is
visible rather than hidden.

`python bench.py tf --tf-counters-only` reruns the same contexts with
`max_tokens=2`, so vLLM's own counters see exactly one round per context. It
should match the per-context numbers. Use it once as a check.

**greedy.** Free-running generation at one request in flight. Acceptance by
position comes from vLLM's `spec_decode_num_*` counters, per category.
Contexts drift apart once acceptance differs, so this is the served view,
not the isolated one. Output token ids are kept for parity against `none`.

**serve.** One `vllm serve` per arm and rep, a streaming client at the given
concurrency. Per-user decode tok/s, TTFT, TPOT, served acceptance by position
from `/metrics`, the model memory and KV cache size the server logs, and the
SM clock and thermal throttle flags sampled during the timed window. Arms
interleave across reps. A rep with a thermal flag set should be rested and
retaken.

## Notes

- The NVFP4 kernel differs by GPU: Marlin W4A16 on SM12x, FlashInfer CuTe DSL
  W4A16 on SM100 by default. Check the `head_lines` field in the serve results
  or the engine log for what each arm resolved to.
- Throughput numbers are only comparable within one machine.
