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

