GPU NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (capability 12.0, 95.1 GiB), driver 615.71.09, vLLM 0.33.0, torch 2.14.0+cu130 / CUDA 13.0

## Teacher-forced acceptance (one draft round per context)

K=3, stride 16, 8619 contexts per arm. Rate at position j = contexts whose first j+1 drafts were all accepted, divided by contexts.

| arm | contexts | mean accepted | pos 0 | pos 1 | pos 2 | target pick = track | accepted = track |
|---|---|---|---|---|---|---|---|
| none | 8619 | 2.105 | 86.5% | 70.1% | 53.9% | 99.5% | 99.4% |
| fp8 | 8619 | 2.104 | 86.4% | 70.0% | 53.9% | 99.5% | 99.4% |
| nvfp4 | 8619 | 2.104 | 86.4% | 69.9% | 54.1% | 99.5% | 99.3% |

Paired against `none` on identical contexts. Position 0: b = none hit and arm miss, c = none miss and arm hit; p is exact McNemar.

| arm | pairs | pos 0 delta (95% CI) | b | c | p | mean accepted delta (95% CI) |
|---|---|---|---|---|---|---|
| fp8 | 8619 | -0.07 pts (-0.30, +0.16) | 55 | 49 | 0.624 | -0.0007 (-0.0071, +0.0058) |
| nvfp4 | 8619 | -0.13 pts (-0.43, +0.18) | 96 | 85 | 0.457 | -0.0010 (-0.0096, +0.0076) |

Position-0 rate by category:

| category | none | fp8 | nvfp4 |
|---|---|---|---|
| coding | 96.4% | 96.5% | 96.5% |
| humanities | 87.5% | 86.7% | 86.7% |
| math | 96.6% | 97.0% | 97.0% |
| multilingual | 86.5% | 86.3% | 86.3% |
| qa | 83.2% | 83.7% | 83.7% |
| rag | 93.3% | 93.3% | 93.6% |
| reasoning | 90.6% | 90.7% | 90.2% |
| roleplay | 72.2% | 71.8% | 71.7% |
| stem | 88.5% | 88.5% | 88.5% |
| summarization | 88.0% | 88.2% | 88.7% |
| writing | 81.1% | 80.3% | 79.9% |

## Free-running greedy, one request in flight

From vLLM's spec-decode counters. Rate at position j = accepted at j / draft rounds.

| arm | accepted/draft | AL | pos 0 | pos 1 | pos 2 | draft rounds | output tokens |
|---|---|---|---|---|---|---|---|
| none | 26886/39930 | 3.020 | 84.6% | 66.3% | 51.1% | 13310 | 40145 |
| fp8 | 26875/39939 | 3.019 | 84.5% | 66.3% | 51.0% | 13313 | 40145 |
| nvfp4 | 26858/40011 | 3.014 | 84.3% | 66.3% | 50.7% | 13337 | 40145 |

Greedy output parity against `none`:

| arm | prompts | identical | mean common prefix | first divergence (median token) |
|---|---|---|---|---|
| fp8 | 192 | 192 | 1.0000 | - |
| nvfp4 | 192 | 192 | 1.0000 | - |

## Served (OpenAI server, streaming client)

Per rep: median over requests. Across reps: median (min..max).

| C | arm | reps | decode tok/s/user | total tok/s | TTFT ms | TPOT ms | AL | pos 0 | pos 1 | pos 2 | model GiB | KV tokens | SM MHz | max thermal-flag share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | none | 3 | 58.2 (58.2..58.2) | 56.7 (56.7..56.7) | 109.7 (109.5..109.8) | 17.19 (17.18..17.19) | 3.036 | 85.6% | 66.8% | 51.3% | 50.93 | 183785 | 1530 | 0% |
| 1 | fp8 | 3 | 61.0 (60.9..61.0) | 60.0 (59.9..60.0) | 103.8 (103.6..103.8) | 16.40 (16.39..16.42) | 3.034 | 85.5% | 66.8% | 51.1% | 52.11 | 186278 | 1537 | 0% |
| 1 | nvfp4 | 3 | 63.0 (62.9..63.0) | 61.3 (61.3..61.4) | 101.4 (101.2..101.5) | 15.88 (15.88..15.89) | 3.030 | 85.5% | 66.7% | 50.9% | 51.6 | 189484 | 1537 | 0% |

