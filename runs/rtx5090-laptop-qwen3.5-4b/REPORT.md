GPU NVIDIA GeForce RTX 5090 Laptop GPU (capability 12.0, 23.5 GiB), driver 615.71.09, vLLM 0.33.0, torch 2.14.0+cu130 / CUDA 13.0

## Teacher-forced acceptance (one draft round per context)

K=3, stride 16, 8730 contexts per arm. Rate at position j = contexts whose first j+1 drafts were all accepted, divided by contexts.

| arm | contexts | mean accepted | pos 0 | pos 1 | pos 2 | target pick = track | accepted = track |
|---|---|---|---|---|---|---|---|
| none | 8730 | 1.856 | 80.5% | 60.6% | 44.5% | 99.5% | 99.3% |
| fp8 | 8730 | 1.855 | 80.6% | 60.4% | 44.4% | 99.5% | 99.3% |
| nvfp4 | 8730 | 1.857 | 80.7% | 60.6% | 44.4% | 99.5% | 99.3% |

Paired against `none` on identical contexts. Position 0: b = none hit and arm miss, c = none miss and arm hit; p is exact McNemar.

| arm | pairs | pos 0 delta (95% CI) | b | c | p | mean accepted delta (95% CI) |
|---|---|---|---|---|---|---|
| fp8 | 8730 | +0.06 pts (-0.20, +0.31) | 63 | 68 | 0.727 | -0.0013 (-0.0084, +0.0059) |
| nvfp4 | 8730 | +0.18 pts (-0.20, +0.57) | 140 | 156 | 0.383 | +0.0016 (-0.0088, +0.0120) |

Position-0 rate by category:

| category | none | fp8 | nvfp4 |
|---|---|---|---|
| coding | 87.5% | 87.7% | 87.9% |
| humanities | 76.6% | 74.2% | 76.6% |
| math | 95.3% | 95.6% | 95.3% |
| multilingual | 81.1% | 80.9% | 80.7% |
| qa | 78.5% | 78.9% | 79.3% |
| rag | 88.2% | 88.4% | 87.7% |
| reasoning | 86.6% | 86.1% | 86.7% |
| roleplay | 62.0% | 61.7% | 63.2% |
| stem | 88.5% | 87.5% | 87.5% |
| summarization | 80.8% | 81.1% | 81.9% |
| writing | 75.3% | 75.8% | 74.3% |

## Free-running greedy, one request in flight

From vLLM's spec-decode counters. Rate at position j = accepted at j / draft rounds.

| arm | accepted/draft | AL | pos 0 | pos 1 | pos 2 | draft rounds | output tokens |
|---|---|---|---|---|---|---|---|
| none | 87900/153093 | 2.722 | 77.8% | 55.5% | 39.0% | 51031 | 138780 |
| fp8 | 87848/153303 | 2.719 | 77.8% | 55.3% | 38.9% | 51101 | 138780 |
| nvfp4 | 87680/153825 | 2.710 | 77.5% | 55.0% | 38.5% | 51275 | 138780 |

Greedy output parity against `none`:

| arm | prompts | identical | mean common prefix | first divergence (median token) |
|---|---|---|---|---|
| fp8 | 669 | 669 | 1.0000 | - |
| nvfp4 | 669 | 669 | 1.0000 | - |

## Served (OpenAI server, streaming client)

Per rep: median over requests. Across reps: median (min..max).

| C | arm | reps | decode tok/s/user | total tok/s | TTFT ms | TPOT ms | AL | pos 0 | pos 1 | pos 2 | model GiB | KV tokens | SM MHz | max thermal-flag share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | none | 3 | 156.1 (154.7..158.0) | 147.3 (146.8..149.1) | 43.8 (43.6..46.7) | 6.41 (6.33..6.46) | 2.815 | 80.0% | 58.8% | 42.8% | 8.09 | 171446 | 2100 | 3% |
| 1 | fp8 | 3 | 164.8 (162.0..181.0) | 160.7 (154.7..170.2) | 43.7 (43.4..46.0) | 6.07 (5.52..6.17) | 2.823 | 80.1% | 59.3% | 42.9% | 8.68 | 161792 | 2234 | 2% |
| 1 | nvfp4 | 3 | 191.5 (188.6..191.5) | 180.4 (177.1..184.5) | 40.8 (37.7..41.8) | 5.22 (5.22..5.30) | 2.794 | 79.3% | 58.4% | 41.7% | 8.42 | 164717 | 2238 | 0% |

## Served takes: conditions and void takes

Each served take is one server launch on its own GPU lease: warm-up, cool-down, then a timed window of
44 prompts x 256 tokens (`--limit 4 --cool-to 55`). Arm order rotated across the three reps. Files are
`serve-rep{1,2,3}-k3-c1-{arm}-r0.json`; each carries the raw GPU samples.

Rule set before the runs: a take is void and retaken when the software thermal flag is on for more than
5% of samples, the start temperature is 86 C or more, or mains power drops.

`void-takes/` holds every take that was not used:

| file | prompts | decode tok/s/user | thermal-flag share | why void |
|---|---|---|---|---|
| serve-rep1-k3-c1-none-r0-try0..try3 | 86 | 152.1, 143.9, 154.7, 149.0 | 48%, 71%, 28%, 55% | old 2-minute window with no cool-down; all throttled. The window was shortened after these. |
| serve-rep2-k3-c1-fp8-r0-try0 | 44 | 174.6 | 5.6% | over the 5% rule |
| serve-rep2-k3-c1-none-r0-try0 | 44 | 137.9 | 38% | over the 5% rule |

SM clock median per valid take (rep 1, 2, 3): BF16 2156, 2100, 2002 MHz; fp8 2234, 2325, 2212; nvfp4 2250, 2137, 2238.
The host ran other CPU work during all takes.
