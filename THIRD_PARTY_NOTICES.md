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
`torch/nn/functional.py`, `torch/nn/init.py`, and `torch/optim`. Files that
substantially translate logic should carry a concise source-path attribution.

## Rust dependency inventory

Generated on 2026-08-30 from the complete direct and transitive package set
reported by `cargo metadata --offline --format-version 1` for the checked-in
`Cargo.lock`. The RustTorch workspace package itself is excluded.

| Crate | Version | Declared license |
|---|---:|---|
| `adler2` | `2.0.1` | `0BSD OR MIT OR Apache-2.0` |
| `aes` | `0.8.4` | `MIT OR Apache-2.0` |
| `anyhow` | `1.0.104` | `MIT OR Apache-2.0` |
| `autocfg` | `1.5.1` | `Apache-2.0 OR MIT` |
| `base64ct` | `1.8.3` | `Apache-2.0 OR MIT` |
| `block-buffer` | `0.10.4` | `MIT OR Apache-2.0` |
| `byteorder` | `1.5.0` | `Unlicense OR MIT` |
| `bzip2` | `0.4.4` | `MIT/Apache-2.0` |
| `bzip2-sys` | `0.1.13+1.0.8` | `MIT/Apache-2.0` |
| `cc` | `1.4.4` | `MIT OR Apache-2.0` |
| `cfg-if` | `1.0.4` | `MIT OR Apache-2.0` |
| `cipher` | `0.4.4` | `MIT OR Apache-2.0` |
| `constant_time_eq` | `0.1.5` | `CC0-1.0` |
| `cpufeatures` | `0.2.17` | `MIT OR Apache-2.0` |
| `crc32fast` | `1.5.1` | `MIT OR Apache-2.0` |
| `crossbeam-utils` | `0.8.22` | `MIT OR Apache-2.0` |
| `crunchy` | `0.2.4` | `MIT` |
| `crypto-common` | `0.1.7` | `MIT OR Apache-2.0` |
| `deranged` | `0.5.8` | `MIT OR Apache-2.0` |
| `digest` | `0.10.7` | `MIT OR Apache-2.0` |
| `find-msvc-tools` | `0.1.11` | `MIT OR Apache-2.0` |
| `flate2` | `1.1.10` | `MIT OR Apache-2.0` |
| `generic-array` | `0.14.7` | `MIT` |
| `getrandom` | `0.2.17` | `MIT OR Apache-2.0` |
| `getrandom` | `0.4.3` | `MIT OR Apache-2.0` |
| `half` | `2.7.1` | `MIT OR Apache-2.0` |
| `hmac` | `0.12.1` | `MIT OR Apache-2.0` |
| `inout` | `0.1.4` | `MIT OR Apache-2.0` |
| `itoa` | `1.0.18` | `MIT OR Apache-2.0` |
| `jobserver` | `0.1.35` | `MIT OR Apache-2.0` |
| `lazy_static` | `1.5.0` | `MIT OR Apache-2.0` |
| `libc` | `0.2.189` | `MIT OR Apache-2.0` |
| `matrixmultiply` | `0.3.11` | `MIT/Apache-2.0` |
| `memchr` | `2.8.3` | `Unlicense OR MIT` |
| `miniz_oxide` | `0.9.1` | `MIT OR Zlib OR Apache-2.0` |
| `ndarray` | `0.16.1` | `MIT OR Apache-2.0` |
| `num-complex` | `0.4.6` | `MIT OR Apache-2.0` |
| `num-conv` | `0.2.2` | `MIT OR Apache-2.0` |
| `num-integer` | `0.1.47` | `MIT OR Apache-2.0` |
| `num-traits` | `0.2.19` | `MIT OR Apache-2.0` |
| `password-hash` | `0.4.2` | `MIT OR Apache-2.0` |
| `pbkdf2` | `0.11.0` | `MIT OR Apache-2.0` |
| `pkg-config` | `0.3.34` | `MIT OR Apache-2.0` |
| `portable-atomic` | `1.15.0` | `Apache-2.0 OR MIT` |
| `portable-atomic-util` | `0.2.7` | `Apache-2.0 OR MIT` |
| `powerfmt` | `0.2.0` | `MIT OR Apache-2.0` |
| `ppv-lite86` | `0.2.21` | `MIT OR Apache-2.0` |
| `proc-macro2` | `1.0.107` | `MIT OR Apache-2.0` |
| `quote` | `1.0.47` | `MIT OR Apache-2.0` |
| `r-efi` | `6.0.0` | `MIT OR Apache-2.0 OR LGPL-2.1-or-later` |
| `rand` | `0.8.8` | `MIT OR Apache-2.0` |
| `rand_chacha` | `0.3.1` | `MIT OR Apache-2.0` |
| `rand_core` | `0.6.4` | `MIT OR Apache-2.0` |
| `rawpointer` | `0.2.1` | `MIT/Apache-2.0` |
| `safetensors` | `0.3.3` | `Apache-2.0` |
| `serde` | `1.0.229` | `MIT OR Apache-2.0` |
| `serde_core` | `1.0.229` | `MIT OR Apache-2.0` |
| `serde_derive` | `1.0.229` | `MIT OR Apache-2.0` |
| `serde_json` | `1.0.151` | `MIT OR Apache-2.0` |
| `sha1` | `0.10.7` | `MIT OR Apache-2.0` |
| `sha2` | `0.10.9` | `MIT OR Apache-2.0` |
| `shlex` | `2.0.1` | `MIT OR Apache-2.0` |
| `simd-adler32` | `0.3.10` | `MIT` |
| `subtle` | `2.6.1` | `BSD-3-Clause` |
| `syn` | `2.0.119` | `MIT OR Apache-2.0` |
| `syn` | `3.0.4` | `MIT OR Apache-2.0` |
| `tch` | `0.26.0` | `MIT/Apache-2.0` |
| `thiserror` | `1.0.69` | `MIT OR Apache-2.0` |
| `thiserror` | `2.0.20` | `MIT OR Apache-2.0` |
| `thiserror-impl` | `1.0.69` | `MIT OR Apache-2.0` |
| `thiserror-impl` | `2.0.20` | `MIT OR Apache-2.0` |
| `time` | `0.3.55` | `MIT OR Apache-2.0` |
| `time-core` | `0.1.9` | `MIT OR Apache-2.0` |
| `torch-sys` | `0.26.0` | `MIT/Apache-2.0` |
| `typenum` | `1.20.1` | `MIT OR Apache-2.0` |
| `unicode-ident` | `1.0.24` | `(MIT OR Apache-2.0) AND Unicode-3.0` |
| `version_check` | `0.9.5` | `MIT/Apache-2.0` |
| `wasi` | `0.11.1+wasi-snapshot-preview1` | `Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT` |
| `zerocopy` | `0.8.56` | `BSD-2-Clause OR Apache-2.0 OR MIT` |
| `zerocopy-derive` | `0.8.56` | `BSD-2-Clause OR Apache-2.0 OR MIT` |
| `zip` | `0.6.6` | `MIT` |
| `zmij` | `1.0.23` | `MIT` |
| `zstd` | `0.11.2+zstd.1.5.2` | `MIT` |
| `zstd-safe` | `5.0.2+zstd.1.5.2` | `MIT/Apache-2.0` |
| `zstd-sys` | `2.0.16+zstd.1.5.7` | `MIT/Apache-2.0` |

## Python parity tooling

- PyTorch 2.13.0: BSD-3-Clause; the complete notice is reproduced above.
- SafeTensors 0.8.0: Apache-2.0.
- NumPy 2.5.2: BSD-3-Clause plus the licenses for bundled components listed
  in its installed distribution.

The authoritative license files included with each resolved crate, source
archive, native library, or Python package distribution control. This notice
summarizes their declared licenses and does not replace those files.
