# Apple Silicon allocator optimization

Measured on a 24 GB Apple M4 Pro MacBook Pro (`Mac16,8`), macOS 15.4.1,
Chrome 152.0.7977.64. The prediction used 291 residues, A3M depth 10,332,
508 clustered rows, 1,024 extra rows, three configured recycles (four passes),
and the macOS compact-memory/chunked-transition path.

## Result

| Metric | Exact-size compact pool | Bounded best-fit compact pool | Change |
| --- | ---: | ---: | ---: |
| Recycle times | 104.65, 116.29, 128.06, 123.76 s | 104.60, 106.19, 110.74, 115.36 s | -0.05, -10.10, -17.32, -8.40 s |
| Total | 472.79 s | 436.94 s | -35.85 s (-7.6%) |
| Peak resident allocation | 4,041 MiB | 3,878 MiB | -163 MiB (-4.0%) |
| GPU buffers created | 21,623 | 21,078 | -545 (-2.5%) |
| Final pLDDT | 63.6 | 63.7 | within observed repeated-run variation |
| Final pTM | 0.457 | 0.457 | unchanged at reported precision |

The exact final scores were pLDDT 63.69584714096436 and pTM
0.4569765568787891. A native Metal differential forced a 64-byte physical
buffer to serve a 16-byte logical tensor and reproduced the independent GPU
sum exactly. The unbounded allocator continues to require exact-size matches.

## Initial block profile

Captured before the allocator change with
`?profile=1&profileRecycle=0&profileExtraBlock=0&profileMainBlock=0`.
Times below are GPU timestamp-query durations in milliseconds. The extra-MSA
block totaled 4,224.238 ms GPU / 4,410.900 ms wall. The main Evoformer block
totaled 1,798.882 ms GPU / 14,508.400 ms wall; its wall time includes first-use
pipeline compilation.

### Extra-MSA block 0

| Dispatch | ms |
| --- | ---: |
| `extra.msa-row-attention.normalize` | 2.171 |
| `extra.msa-row-attention.pair-normalize` | 0.696 |
| `extra.msa-row-attention.pair-bias` | 1.730 |
| `extra.msa-row-attention.project` | 4.110 |
| `extra.msa-row-attention.flash` | 1,201.528 |
| `extra.msa-row-attention.output` | 1.963 |
| `extra.msa-column-global-attention.normalize` | 3.876 |
| `extra.msa-column-global-attention.kv` | 1.622 |
| `extra.msa-column-global-attention.query` | 19.574 |
| `extra.msa-column-global-attention.flash` | 9.405 |
| `extra.msa-column-global-attention.output` | 369.276 |
| `extra.msa-transition.normalize-0` | 0.782 |
| `extra.msa-transition.first-0` | 1.534 |
| `extra.msa-transition.second-0` | 1.517 |
| `extra.msa-transition.normalize-98304` | 0.801 |
| `extra.msa-transition.first-98304` | 1.539 |
| `extra.msa-transition.second-98304` | 1.516 |
| `extra.msa-transition.normalize-196608` | 0.701 |
| `extra.msa-transition.first-196608` | 1.537 |
| `extra.msa-transition.second-196608` | 1.594 |
| `extra.msa-transition.normalize-294912` | 0.027 |
| `extra.msa-transition.first-294912` | 0.063 |
| `extra.msa-transition.second-294912` | 0.082 |
| `opm.normalize` | 2.131 |
| `opm.project` | 3.731 |

Each OPM row is `sequence offset: intermediate / accumulate` in ms; both
individual dispatches are reported.

| Offset | Intermediate | Accumulate |
| ---: | ---: | ---: |
| 0 | 7.430 | 66.166 |
| 32 | 7.488 | 65.775 |
| 64 | 7.527 | 66.296 |
| 96 | 7.429 | 66.383 |
| 128 | 7.430 | 66.163 |
| 160 | 7.488 | 65.954 |
| 192 | 7.485 | 66.088 |
| 224 | 7.550 | 66.226 |
| 256 | 7.430 | 65.959 |
| 288 | 7.440 | 66.181 |
| 320 | 7.431 | 66.394 |
| 352 | 7.430 | 65.995 |
| 384 | 7.488 | 66.227 |
| 416 | 7.504 | 66.062 |
| 448 | 7.431 | 66.375 |
| 480 | 7.551 | 66.082 |
| 512 | 7.553 | 66.186 |
| 544 | 7.549 | 66.085 |
| 576 | 7.430 | 65.977 |
| 608 | 7.429 | 66.011 |
| 640 | 7.517 | 66.026 |
| 672 | 7.431 | 66.042 |
| 704 | 7.497 | 66.156 |
| 736 | 7.429 | 67.764 |
| 768 | 7.434 | 66.836 |
| 800 | 7.429 | 65.966 |
| 832 | 7.489 | 66.647 |
| 864 | 7.429 | 66.250 |
| 896 | 7.538 | 65.856 |
| 928 | 7.503 | 67.608 |
| 960 | 9.085 | 66.341 |
| 992 | 7.513 | 65.975 |

| Dispatch | ms |
| --- | ---: |
| `opm.finalize` | 57.516 |
| `extra.outer-product-mean.residual` | 0.528 |
| `triangle.outgoing.normalize-input` | 0.537 |
| `triangle.outgoing.project` | 4.447 |
| `triangle.outgoing.contract` | 35.529 |
| `triangle.outgoing.normalize-hidden` | 0.531 |
| `triangle.outgoing.output` | 3.138 |
| `triangle.incoming.normalize-input` | 0.554 |
| `triangle.incoming.project` | 4.416 |
| `triangle.incoming.contract` | 35.265 |
| `triangle.incoming.normalize-hidden` | 0.595 |
| `triangle.incoming.output` | 3.145 |
| `extra.triangle-attention-starting.normalize` | 0.695 |
| `extra.triangle-attention-starting.pair-bias` | 0.734 |
| `extra.triangle-attention-starting.project` | 4.266 |
| `extra.triangle-attention-starting.flash` | 29.359 |
| `extra.triangle-attention-starting.output` | 1.367 |
| `extra.triangle-attention-ending.normalize` | 0.755 |
| `extra.triangle-attention-ending.pair-bias` | 0.742 |
| `extra.triangle-attention-ending.project` | 4.266 |
| `extra.triangle-attention-ending.flash` | 29.301 |
| `extra.triangle-attention-ending.output` | 1.379 |
| `extra.pair-transition.normalize-0` | 0.403 |
| `extra.pair-transition.first-0` | 2.884 |
| `extra.pair-transition.second-0` | 3.037 |
| `extra.pair-transition.normalize-49152` | 0.303 |
| `extra.pair-transition.first-49152` | 2.086 |
| `extra.pair-transition.second-49152` | 2.119 |

### Main Evoformer block 0

| Dispatch | ms |
| --- | ---: |
| `msa-row-attention.normalize` | 1.511 |
| `msa-row-attention.pair-normalize` | 0.695 |
| `msa-row-attention.pair-bias` | 1.613 |
| `msa-row-attention.project` | 29.546 |
| `msa-row-attention.flash` | 101.940 |
| `msa-row-attention.output` | 9.026 |
| `msa-column-attention.normalize` | 1.664 |
| `msa-column-attention.project` | 29.763 |
| `msa-column-attention.flash` | 162.056 |
| `msa-column-attention.output` | 8.971 |
| `msa-transition.normalize-0` | 0.253 |
| `msa-transition.first-0` | 5.542 |
| `msa-transition.second-0` | 5.638 |
| `msa-transition.normalize-24576` | 0.244 |
| `msa-transition.first-24576` | 5.796 |
| `msa-transition.second-24576` | 5.581 |
| `msa-transition.normalize-49152` | 0.243 |
| `msa-transition.first-49152` | 5.530 |
| `msa-transition.second-49152` | 5.726 |
| `msa-transition.normalize-73728` | 0.244 |
| `msa-transition.first-73728` | 5.745 |
| `msa-transition.second-73728` | 5.582 |
| `msa-transition.normalize-98304` | 0.248 |
| `msa-transition.first-98304` | 5.617 |
| `msa-transition.second-98304` | 5.591 |
| `msa-transition.normalize-122880` | 0.244 |
| `msa-transition.first-122880` | 5.525 |
| `msa-transition.second-122880` | 5.618 |
| `msa-transition.normalize-147456` | 0.010 |
| `msa-transition.first-147456` | 0.129 |
| `msa-transition.second-147456` | 0.273 |
| `opm.normalize` | 1.449 |
| `opm.project` | 7.143 |

| Offset | Intermediate | Accumulate |
| ---: | ---: | ---: |
| 0 | 7.529 | 66.340 |
| 32 | 7.493 | 66.030 |
| 64 | 7.523 | 66.542 |
| 96 | 7.498 | 66.443 |
| 128 | 7.431 | 66.165 |
| 160 | 7.430 | 66.178 |
| 192 | 7.505 | 66.199 |
| 224 | 7.430 | 66.550 |
| 256 | 7.494 | 66.169 |
| 288 | 7.537 | 66.399 |
| 320 | 7.429 | 66.020 |
| 352 | 7.562 | 66.191 |
| 384 | 7.430 | 66.472 |
| 416 | 7.431 | 66.128 |
| 448 | 7.550 | 66.297 |
| 480 | 6.698 | 57.962 |

| Dispatch | ms |
| --- | ---: |
| `opm.finalize` | 28.746 |
| `outer-product-mean.residual` | 0.538 |
| `triangle.outgoing.normalize-input` | 0.536 |
| `triangle.outgoing.project` | 4.344 |
| `triangle.outgoing.contract` | 35.495 |
| `triangle.outgoing.normalize-hidden` | 0.554 |
| `triangle.outgoing.output` | 3.320 |
| `triangle.incoming.normalize-input` | 0.552 |
| `triangle.incoming.project` | 4.411 |
| `triangle.incoming.contract` | 35.351 |
| `triangle.incoming.normalize-hidden` | 0.530 |
| `triangle.incoming.output` | 3.759 |
| `triangle-attention-starting.normalize` | 1.331 |
| `triangle-attention-starting.pair-bias` | 0.996 |
| `triangle-attention-starting.project` | 4.736 |
| `triangle-attention-starting.flash` | 29.220 |
| `triangle-attention-starting.output` | 1.367 |
| `triangle-attention-ending.normalize` | 0.755 |
| `triangle-attention-ending.pair-bias` | 0.795 |
| `triangle-attention-ending.project` | 4.309 |
| `triangle-attention-ending.flash` | 29.290 |
| `triangle-attention-ending.output` | 1.369 |
| `pair-transition.normalize-0` | 0.403 |
| `pair-transition.first-0` | 2.880 |
| `pair-transition.second-0` | 2.983 |
| `pair-transition.normalize-49152` | 0.303 |
| `pair-transition.first-49152` | 2.085 |
| `pair-transition.second-49152` | 2.115 |

The profile shows that OPM tiled accumulation and extra-MSA row flash dominate
GPU time. This change deliberately leaves those AF2 operations unchanged and
targets the independent host/driver cost of physical buffer creation.

## Reproduction

Apple compact-path browser profile:

```text
http://127.0.0.1:4173/?profile=1&profileRecycle=0&profileExtraBlock=0&profileMainBlock=0
```

GB10 unbounded-path qualification (not run in this work):

```bash
AFWEBGPU_MAX_MS=15000 npm run qualify:hardware
```
