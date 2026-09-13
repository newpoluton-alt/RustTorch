# Third-party notices

RustTorch original code is available under MIT OR Apache-2.0. Dependencies and
behavioral references retain their own terms.

## PyTorch

RustTorch uses PyTorch/LibTorch through `tch` and adapts selected high-level
behavior from PyTorch v2.13.0, commit `cf30153`.

The following license text is bundled with PyTorch 2.13.0:

```text
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions by Tri Dao:
Copyright (c) 2024 Tri Dao.
All rights reserved.

All contributions by Arm:
Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

Relevant behavioral source areas include `torch/nn/modules`,
`torch/nn/functional.py`, `torch/nn/init.py`, and `torch/optim`. The data
foundation adapts selected map/iterable dataset, sampling, batching,
drop-last, and collation behavior from
`torch/utils/data/dataset.py`, `torch/utils/data/sampler.py`,
`torch/utils/data/distributed.py`, `torch/utils/data/dataloader.py`,
`torch/utils/data/_utils/collate.py`, `torch/utils/data/_utils/fetch.py`,
`torch/utils/data/_utils/worker.py`,
`torch/utils/data/datapipes/datapipe.py`, and
`torch/utils/data/datapipes/_decorator.py` at the version and commit above.
RustTorch's fallible iterators, threads, checkpoint protocols, and
sampler-local ChaCha12 RNG are Rust-specific designs; they do not claim Python
pickling, DataPipe runtime decoration, or PyTorch RNG ordering.

The core model extension follows the same pinned v2.13.0 reference:
`torch/nn/modules/conv.py` (convolution shapes, groups and initialization),
`torch/nn/modules/normalization.py` (layer normalization and affine defaults),
`torch/nn/modules/sparse.py` (embedding initialization, padding and gradients),
`torch/optim/adamw.py`, `torch/optim/rmsprop.py` (optimizer defaults and updates),
and `torch/nn/utils/clip_grad.py` (gradient clipping). Rust builders and error
handling are original implementations over existing fallible LibTorch calls;
no upstream source is vendored. CPU reference fixtures are generated by the
locked Python runtime in `tests/python_reference/generate.py`.

The core model/training extension also follows pinned behavior from
`torch/nn/modules/batchnorm.py`, `instancenorm.py`, `pooling.py`, `activation.py`,
`rnn.py`, `transformer.py`, `torch/nn/functional.py`,
`torch/optim/adam.py`, `sgd.py`, `adagrad.py`, `adadelta.py`, `adamax.py`,
`lr_scheduler.py`, and `torch/amp/{autocast_mode,grad_scaler}.py`.
RustTorch implements typed configuration, masking, state validation and optimizer
updates using existing safe tensor operations; it delegates numerical layer
kernels to LibTorch. No upstream source or model weights are vendored. The pinned
Python fixtures independently generate values, gradients, initialization and
multi-step training references. Existing `serde`/`serde_json` dependencies provide
versioned state encoding; `float_roundtrip` preserves configuration on JSON restore.

The MPS affine correction is an original composition of matrix multiplication
and addition. Its behavioral reference is the pinned
`aten/src/ATen/native/mps/operations/Linear.mm` and the upstream
[biased-linear report](https://github.com/pytorch/pytorch/issues/188438).
RustTorch's CI reproduces the missing bias on Apple M1 virtual hardware; no
upstream implementation code is copied by this correction.

The tensor/differentiation extension follows the same full commit
`cf30153c4c131c8164ee7798e5022d810682e2cb`: `torch/autograd/functional.py`,
`torch/autograd/__init__.py`, `torch/distributions/{normal,bernoulli,categorical,utils}.py`,
`torch/_tensor.py`, and ATen operators described by
`aten/src/ATen/native/native_functions.yaml`. Rust wrappers and recipes are
original compositions over safe `tch` calls. Numerical reference fixtures in
`tests/python_reference/{tensor_workflows,differentiation}.py` call the pinned
public APIs; they do not copy implementation code.

The committed census derives object identities, signatures and source locations
from the pinned documentation's resolved Sphinx domain and canonical schemas
from pinned torchgen. Its manifest records the exact source/runtime identity.
No upstream documentation prose, native libraries or source checkout is vendored.

## Runtime acquisition and external components

With RustTorch's default download feature, `torch-sys` downloads official
PyTorch/LibTorch artifacts into Cargo build storage. None of the
nine RustTorch `.crate` archives redistributes LibTorch, FFmpeg or a downloaded
runtime. NVIDIA drivers and
CUDA toolkits remain system components outside these packages; setup never
installs or modifies them.

## Rust dependency inventory

Generated on 2026-09-13 from the complete locked, all-feature direct and
transitive package set reported by
`cargo metadata --locked --all-features --format-version 1`. All nine RustTorch
workspace packages are excluded. The resulting external inventory has 293
rows; duplicate crate versions are preserved.

| Crate | Version | Declared license |
|---|---:|---|
| `adler2` | `2.0.1` | `0BSD OR MIT OR Apache-2.0` |
| `aes` | `0.8.4` | `MIT OR Apache-2.0` |
| `ahash` | `0.8.12` | `MIT OR Apache-2.0` |
| `aho-corasick` | `1.1.5` | `Unlicense OR MIT` |
| `android_system_properties` | `0.1.6` | `MIT OR Apache-2.0` |
| `anyhow` | `1.0.104` | `MIT OR Apache-2.0` |
| `arrow-array` | `59.3.0` | `Apache-2.0 AND MIT` |
| `arrow-buffer` | `59.3.0` | `Apache-2.0` |
| `arrow-data` | `59.3.0` | `Apache-2.0` |
| `arrow-ipc` | `59.3.0` | `Apache-2.0` |
| `arrow-schema` | `59.3.0` | `Apache-2.0` |
| `arrow-select` | `59.3.0` | `Apache-2.0` |
| `audio-codec-algorithms` | `0.8.1` | `0BSD OR Apache-2.0` |
| `audioadapter` | `5.0.0` | `MIT OR Apache-2.0` |
| `audioadapter-buffers` | `5.1.0` | `MIT OR Apache-2.0` |
| `audioadapter-sample` | `5.2.0` | `MIT OR Apache-2.0` |
| `autocfg` | `1.5.1` | `Apache-2.0 OR MIT` |
| `base64` | `0.13.1` | `MIT/Apache-2.0` |
| `base64` | `0.22.1` | `MIT OR Apache-2.0` |
| `base64` | `0.23.1` | `MIT OR Apache-2.0` |
| `base64ct` | `1.8.3` | `Apache-2.0 OR MIT` |
| `bindgen` | `0.71.1` | `BSD-3-Clause` |
| `bit-set` | `0.8.0` | `Apache-2.0 OR MIT` |
| `bit-vec` | `0.8.0` | `Apache-2.0 OR MIT` |
| `bitflags` | `2.13.1` | `MIT OR Apache-2.0` |
| `block-buffer` | `0.10.4` | `MIT OR Apache-2.0` |
| `bon` | `3.10.1` | `MIT OR Apache-2.0` |
| `bon-macros` | `3.10.1` | `MIT OR Apache-2.0` |
| `bumpalo` | `3.20.3` | `MIT OR Apache-2.0` |
| `bytemuck` | `1.25.2` | `Zlib OR Apache-2.0 OR MIT` |
| `byteorder` | `1.5.0` | `Unlicense OR MIT` |
| `byteorder-lite` | `0.1.0` | `Unlicense OR MIT` |
| `bytes` | `1.12.1` | `MIT` |
| `bzip2` | `0.4.4` | `MIT/Apache-2.0` |
| `bzip2-sys` | `0.1.13+1.0.8` | `MIT/Apache-2.0` |
| `camino` | `1.2.5` | `MIT OR Apache-2.0` |
| `castaway` | `0.2.4` | `MIT` |
| `cc` | `1.4.4` | `MIT OR Apache-2.0` |
| `cexpr` | `0.6.0` | `Apache-2.0/MIT` |
| `cfg-if` | `1.0.4` | `MIT OR Apache-2.0` |
| `chrono` | `0.4.45` | `MIT OR Apache-2.0` |
| `cipher` | `0.4.4` | `MIT OR Apache-2.0` |
| `clang-sys` | `1.9.1` | `Apache-2.0` |
| `compact_str` | `0.9.1` | `MIT` |
| `const-random` | `0.1.18` | `MIT OR Apache-2.0` |
| `const-random-macro` | `0.1.16` | `MIT OR Apache-2.0` |
| `constant_time_eq` | `0.1.5` | `CC0-1.0` |
| `core-foundation-sys` | `0.8.7` | `MIT OR Apache-2.0` |
| `cpufeatures` | `0.2.17` | `MIT OR Apache-2.0` |
| `crc32fast` | `1.5.1` | `MIT OR Apache-2.0` |
| `crossbeam-channel` | `0.5.16` | `MIT OR Apache-2.0` |
| `crossbeam-deque` | `0.8.8` | `MIT OR Apache-2.0` |
| `crossbeam-epoch` | `0.9.21` | `MIT OR Apache-2.0` |
| `crossbeam-utils` | `0.8.22` | `MIT OR Apache-2.0` |
| `crunchy` | `0.2.4` | `MIT` |
| `crypto-common` | `0.1.7` | `MIT OR Apache-2.0` |
| `csv` | `1.4.0` | `Unlicense/MIT` |
| `csv-core` | `0.1.13` | `Unlicense/MIT` |
| `daachorse` | `3.0.3` | `MIT OR Apache-2.0` |
| `darling` | `0.20.11` | `MIT` |
| `darling` | `0.24.1` | `MIT` |
| `darling_core` | `0.20.11` | `MIT` |
| `darling_core` | `0.24.1` | `MIT` |
| `darling_macro` | `0.20.11` | `MIT` |
| `darling_macro` | `0.24.1` | `MIT` |
| `dary_heap` | `0.3.9` | `MIT OR Apache-2.0` |
| `deranged` | `0.5.8` | `MIT OR Apache-2.0` |
| `derive_builder` | `0.20.2` | `MIT OR Apache-2.0` |
| `derive_builder_core` | `0.20.2` | `MIT OR Apache-2.0` |
| `derive_builder_macro` | `0.20.2` | `MIT OR Apache-2.0` |
| `digest` | `0.10.7` | `MIT OR Apache-2.0` |
| `displaydoc` | `0.2.7` | `MIT OR Apache-2.0` |
| `either` | `1.18.0` | `MIT OR Apache-2.0` |
| `equivalent` | `1.0.2` | `Apache-2.0 OR MIT` |
| `errno` | `0.3.14` | `MIT OR Apache-2.0` |
| `esaxx-rs` | `0.1.10` | `Apache-2.0` |
| `extended` | `0.1.0` | `MIT` |
| `fancy-regex` | `0.17.0` | `MIT` |
| `fastrand` | `2.5.0` | `Apache-2.0 OR MIT` |
| `fdeflate` | `0.3.7` | `MIT OR Apache-2.0` |
| `find-msvc-tools` | `0.1.11` | `MIT OR Apache-2.0` |
| `flatbuffers` | `25.12.19` | `Apache-2.0` |
| `flate2` | `1.1.10` | `MIT OR Apache-2.0` |
| `fnv` | `1.0.7` | `Apache-2.0 / MIT` |
| `form_urlencoded` | `1.2.2` | `MIT OR Apache-2.0` |
| `futures-core` | `0.3.34` | `MIT OR Apache-2.0` |
| `futures-task` | `0.3.34` | `MIT OR Apache-2.0` |
| `futures-util` | `0.3.34` | `MIT OR Apache-2.0` |
| `generic-array` | `0.14.7` | `MIT` |
| `getrandom` | `0.2.17` | `MIT OR Apache-2.0` |
| `getrandom` | `0.3.4` | `MIT OR Apache-2.0` |
| `getrandom` | `0.4.3` | `MIT OR Apache-2.0` |
| `glob` | `0.3.4` | `MIT OR Apache-2.0` |
| `half` | `2.7.1` | `MIT OR Apache-2.0` |
| `hashbrown` | `0.17.1` | `MIT OR Apache-2.0` |
| `hmac` | `0.12.1` | `MIT OR Apache-2.0` |
| `iana-time-zone` | `0.1.65` | `MIT OR Apache-2.0` |
| `iana-time-zone-haiku` | `0.1.2` | `MIT OR Apache-2.0` |
| `icu_collections` | `2.3.0` | `Unicode-3.0` |
| `icu_locale_core` | `2.3.0` | `Unicode-3.0` |
| `icu_normalizer` | `2.3.0` | `Unicode-3.0` |
| `icu_normalizer_data` | `2.3.0` | `Unicode-3.0` |
| `icu_properties` | `2.3.0` | `Unicode-3.0` |
| `icu_properties_data` | `2.3.0` | `Unicode-3.0` |
| `icu_provider` | `2.3.1` | `Unicode-3.0` |
| `ident_case` | `1.0.1` | `MIT/Apache-2.0` |
| `idna` | `1.1.0` | `MIT OR Apache-2.0` |
| `idna_adapter` | `1.2.2` | `Apache-2.0 OR MIT` |
| `image` | `0.25.10` | `MIT OR Apache-2.0` |
| `indexmap` | `2.14.1` | `Apache-2.0 OR MIT` |
| `inout` | `0.1.4` | `MIT OR Apache-2.0` |
| `itertools` | `0.13.0` | `MIT OR Apache-2.0` |
| `itertools` | `0.14.0` | `MIT OR Apache-2.0` |
| `itoa` | `1.0.18` | `MIT OR Apache-2.0` |
| `jobserver` | `0.1.35` | `MIT OR Apache-2.0` |
| `js-sys` | `0.3.105` | `MIT OR Apache-2.0` |
| `lazy_static` | `1.5.0` | `MIT OR Apache-2.0` |
| `libc` | `0.2.189` | `MIT OR Apache-2.0` |
| `libloading` | `0.8.9` | `ISC` |
| `libm` | `0.2.16` | `MIT` |
| `linux-raw-sys` | `0.12.1` | `Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT` |
| `litemap` | `0.8.3` | `Unicode-3.0` |
| `log` | `0.4.34` | `MIT OR Apache-2.0` |
| `macro_rules_attribute` | `0.2.3` | `Apache-2.0 OR MIT OR Zlib` |
| `macro_rules_attribute-proc_macro` | `0.2.3` | `Apache-2.0 OR MIT OR Zlib` |
| `matrixmultiply` | `0.3.11` | `MIT/Apache-2.0` |
| `memchr` | `2.8.3` | `Unlicense OR MIT` |
| `minimal-lexical` | `0.2.1` | `MIT/Apache-2.0` |
| `miniz_oxide` | `0.8.9` | `MIT OR Zlib OR Apache-2.0` |
| `miniz_oxide` | `0.9.1` | `MIT OR Zlib OR Apache-2.0` |
| `monostate` | `0.1.18` | `MIT OR Apache-2.0` |
| `monostate-impl` | `0.1.18` | `MIT OR Apache-2.0` |
| `moxcms` | `0.8.1` | `BSD-3-Clause OR Apache-2.0` |
| `ndarray` | `0.16.1` | `MIT OR Apache-2.0` |
| `nom` | `7.1.3` | `MIT` |
| `num-bigint` | `0.5.1` | `MIT OR Apache-2.0` |
| `num-complex` | `0.4.6` | `MIT OR Apache-2.0` |
| `num-conv` | `0.2.2` | `MIT OR Apache-2.0` |
| `num-integer` | `0.1.47` | `MIT OR Apache-2.0` |
| `num-traits` | `0.2.19` | `MIT OR Apache-2.0` |
| `once_cell` | `1.21.4` | `MIT OR Apache-2.0` |
| `parquet` | `59.3.0` | `Apache-2.0` |
| `password-hash` | `0.4.2` | `MIT OR Apache-2.0` |
| `paste` | `1.0.15` | `MIT OR Apache-2.0` |
| `pastey` | `0.2.3` | `MIT OR Apache-2.0` |
| `pbkdf2` | `0.11.0` | `MIT OR Apache-2.0` |
| `percent-encoding` | `2.3.2` | `MIT OR Apache-2.0` |
| `pin-project-lite` | `0.2.17` | `Apache-2.0 OR MIT` |
| `pkg-config` | `0.3.34` | `MIT OR Apache-2.0` |
| `png` | `0.18.1` | `MIT OR Apache-2.0` |
| `portable-atomic` | `1.15.0` | `Apache-2.0 OR MIT` |
| `portable-atomic-util` | `0.2.7` | `Apache-2.0 OR MIT` |
| `potential_utf` | `0.1.6` | `Unicode-3.0` |
| `powerfmt` | `0.2.0` | `MIT OR Apache-2.0` |
| `ppv-lite86` | `0.2.21` | `MIT OR Apache-2.0` |
| `prettyplease` | `0.2.37` | `MIT OR Apache-2.0` |
| `prettyplease` | `0.3.0` | `MIT OR Apache-2.0` |
| `proc-macro2` | `1.0.107` | `MIT OR Apache-2.0` |
| `prost` | `0.14.4` | `Apache-2.0` |
| `prost-derive` | `0.14.4` | `Apache-2.0` |
| `pxfm` | `0.1.30` | `BSD-3-Clause OR Apache-2.0` |
| `quote` | `1.0.47` | `MIT OR Apache-2.0` |
| `r-efi` | `5.3.0` | `MIT OR Apache-2.0 OR LGPL-2.1-or-later` |
| `r-efi` | `6.0.0` | `MIT OR Apache-2.0 OR LGPL-2.1-or-later` |
| `rand` | `0.8.8` | `MIT OR Apache-2.0` |
| `rand` | `0.9.5` | `MIT OR Apache-2.0` |
| `rand_chacha` | `0.3.1` | `MIT OR Apache-2.0` |
| `rand_chacha` | `0.9.0` | `MIT OR Apache-2.0` |
| `rand_core` | `0.6.4` | `MIT OR Apache-2.0` |
| `rand_core` | `0.9.5` | `MIT OR Apache-2.0` |
| `rawpointer` | `0.2.1` | `MIT/Apache-2.0` |
| `rayon` | `1.12.0` | `MIT OR Apache-2.0` |
| `rayon-cond` | `0.4.0` | `Apache-2.0/MIT` |
| `rayon-core` | `1.13.0` | `MIT OR Apache-2.0` |
| `regex` | `1.13.1` | `MIT OR Apache-2.0` |
| `regex-automata` | `0.4.18` | `MIT OR Apache-2.0` |
| `regex-lite` | `0.1.9` | `MIT OR Apache-2.0` |
| `regex-syntax` | `0.8.11` | `MIT OR Apache-2.0` |
| `ring` | `0.17.14` | `Apache-2.0 AND ISC` |
| `rsmpeg` | `0.18.0+ffmpeg.8.0` | `MIT` |
| `rubato` | `5.0.0` | `MIT OR Apache-2.0` |
| `rustc-hash` | `2.1.3` | `Apache-2.0 OR MIT` |
| `rustc_version` | `0.4.1` | `MIT OR Apache-2.0` |
| `rustix` | `1.1.4` | `Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT` |
| `rustls` | `0.23.43` | `Apache-2.0 OR ISC OR MIT` |
| `rustls-pki-types` | `1.15.1` | `MIT OR Apache-2.0` |
| `rustls-webpki` | `0.103.15` | `ISC` |
| `rustversion` | `1.0.23` | `MIT OR Apache-2.0` |
| `rusty_ffmpeg` | `0.16.7+ffmpeg.8` | `MIT` |
| `ryu` | `1.0.23` | `Apache-2.0 OR BSL-1.0` |
| `safetensors` | `0.3.3` | `Apache-2.0` |
| `semver` | `1.0.28` | `MIT OR Apache-2.0` |
| `seq-macro` | `0.3.6` | `MIT OR Apache-2.0` |
| `serde` | `1.0.229` | `MIT OR Apache-2.0` |
| `serde_core` | `1.0.229` | `MIT OR Apache-2.0` |
| `serde_derive` | `1.0.229` | `MIT OR Apache-2.0` |
| `serde_json` | `1.0.151` | `MIT OR Apache-2.0` |
| `sha1` | `0.10.7` | `MIT OR Apache-2.0` |
| `sha2` | `0.10.9` | `MIT OR Apache-2.0` |
| `shlex` | `1.3.0` | `MIT OR Apache-2.0` |
| `shlex` | `2.0.1` | `MIT OR Apache-2.0` |
| `simd-adler32` | `0.3.10` | `MIT` |
| `slab` | `0.4.12` | `MIT` |
| `smallvec` | `1.15.2` | `MIT OR Apache-2.0` |
| `spm_precompiled` | `0.1.4` | `Apache-2.0` |
| `stable_deref_trait` | `1.2.1` | `MIT OR Apache-2.0` |
| `static_assertions` | `1.1.0` | `MIT OR Apache-2.0` |
| `strsim` | `0.11.1` | `MIT` |
| `subtle` | `2.6.1` | `BSD-3-Clause` |
| `symphonia` | `0.6.1` | `MPL-2.0` |
| `symphonia-bundle-flac` | `0.6.1` | `MPL-2.0` |
| `symphonia-codec-pcm` | `0.6.1` | `MPL-2.0` |
| `symphonia-common` | `0.6.1` | `MPL-2.0` |
| `symphonia-core` | `0.6.1` | `MPL-2.0` |
| `symphonia-format-riff` | `0.6.1` | `MPL-2.0` |
| `symphonia-metadata` | `0.6.1` | `MPL-2.0` |
| `syn` | `2.0.119` | `MIT OR Apache-2.0` |
| `syn` | `3.0.4` | `MIT OR Apache-2.0` |
| `synstructure` | `0.13.2` | `MIT` |
| `tch` | `0.26.0` | `MIT/Apache-2.0` |
| `tempfile` | `3.27.0` | `MIT OR Apache-2.0` |
| `thiserror` | `1.0.69` | `MIT OR Apache-2.0` |
| `thiserror` | `2.0.20` | `MIT OR Apache-2.0` |
| `thiserror-impl` | `1.0.69` | `MIT OR Apache-2.0` |
| `thiserror-impl` | `2.0.20` | `MIT OR Apache-2.0` |
| `time` | `0.3.55` | `MIT OR Apache-2.0` |
| `time-core` | `0.1.9` | `MIT OR Apache-2.0` |
| `tiny-keccak` | `2.0.2` | `CC0-1.0` |
| `tinystr` | `0.8.4` | `Unicode-3.0` |
| `tokenizers` | `0.23.2` | `Apache-2.0` |
| `toml_datetime` | `0.6.11` | `MIT OR Apache-2.0` |
| `toml_edit` | `0.22.27` | `MIT OR Apache-2.0` |
| `toml_write` | `0.1.2` | `MIT OR Apache-2.0` |
| `torch-sys` | `0.26.0` | `MIT/Apache-2.0` |
| `twox-hash` | `2.1.4` | `MIT` |
| `typenum` | `1.20.1` | `MIT OR Apache-2.0` |
| `unicode-ident` | `1.0.24` | `(MIT OR Apache-2.0) AND Unicode-3.0` |
| `unicode-normalization-alignments` | `0.1.12` | `MIT/Apache-2.0` |
| `unicode-segmentation` | `1.13.3` | `MIT OR Apache-2.0` |
| `unicode_categories` | `0.1.1` | `MIT OR Apache-2.0` |
| `untrusted` | `0.9.0` | `ISC` |
| `ureq` | `2.12.1` | `MIT OR Apache-2.0` |
| `url` | `2.5.8` | `MIT OR Apache-2.0` |
| `utf8_iter` | `1.0.4` | `Apache-2.0 OR MIT` |
| `vcpkg` | `0.2.15` | `MIT/Apache-2.0` |
| `version_check` | `0.9.5` | `MIT/Apache-2.0` |
| `visibility` | `0.1.1` | `Zlib OR MIT OR Apache-2.0` |
| `wasi` | `0.11.1+wasi-snapshot-preview1` | `Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT` |
| `wasip2` | `1.0.4+wasi-0.2.12` | `Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT` |
| `wasm-bindgen` | `0.2.128` | `MIT OR Apache-2.0` |
| `wasm-bindgen-macro` | `0.2.128` | `MIT OR Apache-2.0` |
| `wasm-bindgen-macro-support` | `0.2.128` | `MIT OR Apache-2.0` |
| `wasm-bindgen-shared` | `0.2.128` | `MIT OR Apache-2.0` |
| `webpki-roots` | `0.26.11` | `CDLA-Permissive-2.0` |
| `webpki-roots` | `1.0.9` | `CDLA-Permissive-2.0` |
| `windowfunctions` | `0.1.1` | `MIT` |
| `windows-core` | `0.62.2` | `MIT OR Apache-2.0` |
| `windows-implement` | `0.60.2` | `MIT OR Apache-2.0` |
| `windows-interface` | `0.59.3` | `MIT OR Apache-2.0` |
| `windows-link` | `0.2.1` | `MIT OR Apache-2.0` |
| `windows-result` | `0.4.1` | `MIT OR Apache-2.0` |
| `windows-strings` | `0.5.1` | `MIT OR Apache-2.0` |
| `windows-sys` | `0.52.0` | `MIT OR Apache-2.0` |
| `windows-targets` | `0.52.6` | `MIT OR Apache-2.0` |
| `windows_aarch64_gnullvm` | `0.52.6` | `MIT OR Apache-2.0` |
| `windows_aarch64_msvc` | `0.52.6` | `MIT OR Apache-2.0` |
| `windows_i686_gnu` | `0.52.6` | `MIT OR Apache-2.0` |
| `windows_i686_gnullvm` | `0.52.6` | `MIT OR Apache-2.0` |
| `windows_i686_msvc` | `0.52.6` | `MIT OR Apache-2.0` |
| `windows_x86_64_gnu` | `0.52.6` | `MIT OR Apache-2.0` |
| `windows_x86_64_gnullvm` | `0.52.6` | `MIT OR Apache-2.0` |
| `windows_x86_64_msvc` | `0.52.6` | `MIT OR Apache-2.0` |
| `winnow` | `0.7.15` | `MIT` |
| `wit-bindgen` | `0.57.1` | `Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT` |
| `writeable` | `0.6.4` | `Unicode-3.0` |
| `yoke` | `0.8.3` | `Unicode-3.0` |
| `yoke-derive` | `0.8.2` | `Unicode-3.0` |
| `zerocopy` | `0.8.56` | `BSD-2-Clause OR Apache-2.0 OR MIT` |
| `zerocopy-derive` | `0.8.56` | `BSD-2-Clause OR Apache-2.0 OR MIT` |
| `zerofrom` | `0.1.8` | `Unicode-3.0` |
| `zerofrom-derive` | `0.1.7` | `Unicode-3.0` |
| `zeroize` | `1.9.0` | `Apache-2.0 OR MIT` |
| `zerotrie` | `0.2.5` | `Unicode-3.0` |
| `zerovec` | `0.11.8` | `Unicode-3.0` |
| `zerovec-derive` | `0.11.6` | `Unicode-3.0` |
| `zip` | `0.6.6` | `MIT` |
| `zlib-rs` | `0.6.7` | `Zlib` |
| `zmij` | `1.0.23` | `MIT` |
| `zstd` | `0.11.2+zstd.1.5.2` | `MIT` |
| `zstd-safe` | `5.0.2+zstd.1.5.2` | `MIT/Apache-2.0` |
| `zstd-sys` | `2.0.16+zstd.1.5.7` | `MIT/Apache-2.0` |
| `zune-core` | `0.5.3` | `MIT OR Apache-2.0 OR Zlib` |
| `zune-jpeg` | `0.5.15` | `MIT OR Apache-2.0 OR Zlib` |

## Documentation inventory tooling

Maintainer-only exact pins: Sphinx 9.1.0 (BSD-2-Clause), MyST-Parser 5.1.0
(MIT), PyYAML 6.0.3 (MIT), and MyST-NB 1.4.0 (BSD-3-Clause). `uv.lock`
records their complete transitive resolution and artifact hashes. These tools
are needed to resolve documentation directives and canonical YAML schemas;
standard-library text matching cannot recover those semantic objects. They are
not Rust/runtime dependencies and are excluded from crate distributions.
Refresh runs in an isolated temporary configuration with notebook execution
disabled; dependency review and locked artifact checks apply to updates.

## Python parity tooling

- PyTorch 2.13.0: BSD-3-Clause; the complete notice is reproduced above.
- SafeTensors 0.8.0: Apache-2.0.
- ONNX 1.22.0: Apache-2.0; its Python checker/reference dependencies include
  ml-dtypes 0.6.0 (Apache-2.0) and protobuf 7.36.1 (BSD-3-Clause).
- NumPy 2.5.2: BSD-3-Clause plus the licenses for bundled components listed
  in its installed distribution.

The authoritative license files included with each resolved crate, source
archive, native library, or Python package distribution control. This notice
summarizes their declared licenses and does not replace those files.

## Distributed, deployment and framework references

The 0.4 behavior is checked against PyTorch 2.13.0 commit
`cf30153c4c131c8164ee7798e5022d810682e2cb`, specifically
`torch/distributed/distributed_c10d.py`, `torch/nn/parallel/distributed.py`,
`torch/distributed/fsdp/fully_sharded_data_parallel.py`,
`torch/distributed/fsdp/_fully_shard/_fully_shard.py`, `torch/_export/serde`,
`torch/export`, `torch/jit`, `torch/testing`, `torch/autograd/gradcheck.py`,
and `torch/utils/benchmark`. The TCP transport, typed portable schema, tests
and application recipes are original Rust implementations over existing tensor
operations. Gloo is used by the Python numerical fixture; no Gloo implementation,
headers or native c10d bridge is copied into RustTorch.

The protobuf field tags/enums in `src/deployment/onnx.rs` follow the ONNX schema
at [ONNX 1.22.0](https://github.com/onnx/onnx/tree/2bb50465112feca9003e1ed654d77f01ff1415ca)
(commit `2bb50465112feca9003e1ed654d77f01ff1415ca`, Apache-2.0). The narrow Rust
message types were transcribed from that schema; no model weights or upstream
implementation prose is included. `prost` supplies protobuf encoding/decoding,
and the official ONNX checker/reference evaluator verifies generated artifacts.
The portable interchange profile is IR 10/opset 18; package version and supported
interchange versions are separate contracts.

User-region profiling writes the documented Chrome Trace Event JSON format
consumed by [Perfetto](https://perfetto.dev/docs/getting-started/other-formats).
This is an independently produced interchange record, with no tracing runtime
or Perfetto source dependency. No kernel-level instrumentation is claimed.

## Domain dependency and native boundary review

The optional domain packages reuse `rusttorch-data` instead of adding another
loader. Their established upstream libraries handle formats whose complete
parsers/codecs are outside the standard library: `image` for PNG/JPEG;
`tokenizers` for local tokenizer files; Symphonia for WAV/FLAC; Rubato with
`audioadapter-buffers` for resampling; `csv` for quoted/multiline records;
and Arrow/Parquet for columnar data. `rsmpeg`/`rusty_ffmpeg` provide the FFmpeg
bridge. Versions and licenses, including Symphonia's MPL-2.0 terms, appear in
the inventory above. These dependencies retain their licenses and are not
relicensed as original RustTorch code. Feature selection excludes downloads,
remote tokenizer loading, unnecessary image/audio codecs and Parquet compression.

All optional dependencies compile with Rust 1.88. Input limits and malformed
fixture tests bound public ingestion; upstream native parsers still own internal
scratch allocations. The explicit FFmpeg audio `unsafe` boundary reads one
owned packed-f32 AVFrame after sample count, byte capacity, non-null pointer and
alignment checks. The frame stays alive until the tensor constructor copies its
contents; borrowed native memory never escapes. Native codec fixtures exercise
both audio and video paths and rejection limits. Dependency review, pinned
locks and package inspection are release gates; no performance or universal
format-security claim follows merely from choosing Rust dependencies.

The CI native profile builds unmodified
[FFmpeg 8.1.2 source](https://ffmpeg.org/releases/ffmpeg-8.1.2.tar.xz), SHA-256
`464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c`.
`scripts/build-ffmpeg.py` records the exact configure invocation and runtime
ABI/license/configuration audit. It enables shared libraries and disables GPL,
nonfree, version-3-only components, external autodetection and network protocols.
Only FFV1/PCM decoding and Matroska/WAV/file input are enabled in this test
profile. The resulting libraries report LGPL-2.1-or-later, following
[FFmpeg's linking guidance](https://ffmpeg.org/legal.html).
Native libraries are used for testing and are never included in RustTorch crate
archives. A user-selected system or vcpkg FFmpeg build may have different codec,
license and redistribution requirements. Local Homebrew GPL-enabled execution
is additional development evidence, never the official LGPL profile evidence.
