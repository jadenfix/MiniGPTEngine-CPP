# Causal attention demonstration

The GIF and MP4 visualize **actual outputs** of `AttentionMechanism::self_attention`
from source commit `51a7a9f0aa6d3e792ec7b41c2e88d1c565877ea9`.
They are rendered explanatory animations, not screen recordings or model inference.

## Reproduce

On Linux with Python 3 + Pillow, DejaVu fonts, a C++17 `g++` compiler, make and ffmpeg:

```sh
make all test portable-demo
python3 demo/render.py
```

The renderer rebuilds and executes both current tests and `demo/fixture.cpp`, then
uses their numeric output. It refuses to label a changed kernel with the old source
revision. `results.json` records the kernel SHA-256 and exact fixture outputs;
`core-run.txt` records the observed command output. On other operating systems,
change the font paths in the renderer before rendering. Media are 960×540,
16 seconds, 12 frames per second. MP4 uses H.264/yuv420p with fast-start.

## What was actually verified

A three-token fixture repeats values 1, 2 and 3 across each token's 64 channels.
The third token is changed to 1000. All 128 earlier output channels remain exactly
unchanged; the final output remains finite. The emitted first channel outputs are:

| Token | Before | After |
|---|---:|---:|
| 1 | 1 | 1 |
| 2 | 1.99651873 | 1.99651873 |
| 3 | 2.99979353 | 1000 |

The existing `make all test portable-demo` reference/regression lane passed on a
fresh isolated source snapshot using GCC 13.3.0. It covers the analytic two-token
reference, constants, future isolation, independent heads, invalid shapes and
nonfinite inputs. The renamed included demo `main` produces an existing compiler
warning about implicit return in the test translation unit; the tests exited 0.
No source behavior was changed by this media PR.

## Limits

This is an educational eight-head kernel with identity Q/K/V projections and
synthetic data. It does not prove trained model generation, model quality,
performance, compression, GPU acceleration or production integration. Complete
model inference remains blocked as described in the root README. Historical
canned-response programs were not used in these media.
