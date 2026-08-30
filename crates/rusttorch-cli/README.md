# RustTorch setup CLI

`rusttorch-cli` installs the `rusttorch` bootstrap command. Run it in a Cargo
project after adding the RustTorch library dependency:

```sh
cargo install rusttorch-cli
cargo add rusttorch
rusttorch setup --backend auto
cargo run
```

The accepted setup commands are exactly:

```sh
rusttorch setup --backend auto
rusttorch setup --backend cpu
rusttorch setup --backend cuda-12.6
```

Setup locates the Cargo workspace root, writes project-local managed settings
for CPU or CUDA, and runs `cargo check`. It does not install globally managed
LibTorch files, NVIDIA drivers, or CUDA toolkits. `auto` preserves an active
LibTorch/Python/TORCH selection; otherwise it chooses CUDA 12.6 only on a
Linux or Windows NVIDIA system with a compatible driver, and chooses CPU on
the remaining supported systems. The macOS CPU LibTorch distribution can use
MPS when supported.

See the [RustTorch README](https://github.com/newpoluton-alt/RustTorch) for
configuration ownership, target isolation, offline setup, and dynamic-loader
caveats.
