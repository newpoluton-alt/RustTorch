# Native runtime capabilities

RustTorch exposes scoped reverse-mode recording through `no_grad` and
`with_grad` in both `rusttorch` and `rusttorch_core`. An inference loop can
temporarily record one calculation to measure its input sensitivity. The
surrounding thread resumes its previous recording mode on return, error, or
panic. Keep the low-level `no_grad_guard` on its creating thread and drop
nested guards in reverse order. See the [runnable core examples](../crates/rusttorch-core/src/lib.rs).

## Results on the shipping runtime

The feasibility tests ran on macOS 27.0.1 arm64 with CPython 3.14.7, the
PyTorch 2.13.0 wheel at `cf30153c4c131c8164ee7798e5022d810682e2cb`, Apple
Clang 21.0.0, C++20 and Rust 1.98.0. The Rust crate uses `tch` 0.26.0 at
`4227b89b72059b0651ff83a38637693574e0a2d6`. These results apply to that
build. The native executables under [tests/native-runtime](../tests/native-runtime)
are development probes.

| Boundary | Executed operation and result | Rust integration |
|---|---|---|
| Scoped gradients | A nested `no_grad` unwound, an independent thread continued recording, and training recorded again afterward. `with_grad` inside inference gave the same input gradient as the pinned reference and restored inference mode. | Available through the core crate and facade. [Rust regression](../crates/rusttorch-core/tests/public_api.rs) and [reference case](../tests/python_reference/differentiation.py). |
| Tensor ownership | A transposed CPU tensor survived dropping the original owner. Mutation was visible through its alias. | Use existing `Tensor` views and `shallow_clone`. |
| Device and scalar representation | LibTorch created a meta tensor and read UInt16 values on CPU. A separate executable used only the AOTI C shim to create a UInt16 meta tensor, duplicate the handle, destroy the first handle and read the correct metadata. The XPU identifier parsed and allocation raised a native error on this XPU-less build. | The current `tch::Device` and `Kind` lack the corresponding variants. Extend their checked conversions before exposing these values on ordinary Rust tensors. |
| Native generator | A private CPU generator drew samples, saved its state after advancement, drew twice more, and reproduced the first subsequent draw exactly after restoration. | A pinned C++ or upstream binding is needed for generator ownership and get/set state. RustTorch's existing `TensorRng` uses ChaCha12. |
| Forward AD | One native dual level propagated a tangent through a square on a noncontiguous input. After exception unwinding, opening a new level found no tangent on the retained output. | `tch` wraps make/unpack operations. Public dual-level entry/exit and safe lifecycle management remain open. |
| Custom backward | A C++ autograd function rounded its forward value and returned the incoming seed in backward. A throwing gradient hook propagated an exception. Removing the hook allowed a new differentiation to succeed. | Rust callback registration, panic translation, ownership across engine threads and cancellation still need a reviewed interface. |
| MPS stream | A native stream guard ran and synchronized a real tensor reduction. It restored stream 0 after unwinding. Requesting a new stream also returned ID 0 on this build. | Switching between distinct accelerator streams needs a device and backend with independent streams. |
| Compiled model | AOTInductor compiled an exported Float32 square function for input shape `[3]`. A standalone C++ process opened the resulting package with `AOTIModelPackageLoader` and matched eager output on `[-3, 2, 0]`. | A production loader needs an explicit trusted-code boundary and ownership of the runner for the lifetime of each call. Existing RustTorch graph imports have their own [documented scope](deployment.md). |
| Native process group | Two native executables opened a file store and Gloo group on loopback. Both waited for sum all-reduce and read 3 from rank inputs 1 and 2. | The wheel exports the symbols but omits transitive Gloo headers. Compilation succeeded with the matching upstream Gloo checkout. A second collective failed after peer exit and a fresh pair successfully summed again. Production abort, shutdown and platform packaging remain open. |

## Reproduce the probes

From the repository root, use a new external build directory and the project
environment. The default two CTest cases need the pinned CPU-capable LibTorch,
a C++20 compiler and CMake.

```sh
. scripts/dev-env.sh
probe_build=$(mktemp -d)
torch_cmake=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')
cmake -S tests/native-runtime -B "$probe_build" -DCMAKE_PREFIX_PATH="$torch_cmake"
cmake --build "$probe_build" --parallel 3
ctest --test-dir "$probe_build" --output-on-failure
cargo test -p rusttorch-core --test public_api
cargo test -p rusttorch-core --doc
```

The following commands add the compiled-model, Gloo and MPS cases on the
Apple Silicon host. The model is built from the checked-in square function and
executes native code. Gloo headers must match the submodule recorded by the
same PyTorch commit. The launcher uses Python's standard library to start and
reap the two C++ workers. On a CPU-only host omit `-DSTREAM_DEVICE=mps:0`.

```sh
python tests/native-runtime/export_square.py "$probe_build/square.pt2"
git clone --filter=blob:none --no-checkout https://github.com/pytorch/gloo.git "$probe_build/gloo-src"
git -C "$probe_build/gloo-src" checkout --detach bcd1672ee07538123ea8f4fac76832efc58fb8ef
cmake -S tests/native-runtime -B "$probe_build" \
  -DAOTI_PACKAGE="$probe_build/square.pt2" -DGLOO_INCLUDE_DIR="$probe_build/gloo-src" \
  -DSTREAM_DEVICE=mps:0 -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python"
cmake --build "$probe_build" --parallel 3
ctest --test-dir "$probe_build" --output-on-failure
```

All six CTest cases passed on macOS arm64, including the peer-loss and recovery scenario. The two default tests also passed on Linux arm64 with GCC 12.2.0, CMake 3.25.1 and the pinned 2.13.0 CPU wheel. Windows execution awaits the hosted runner. One clean three-job Ninja build of the
four probe executables took 4.5 seconds on this host after configuration.
Before adding the peer-loss scenario, the executables totaled 1,697,456 bytes; the locally generated model occupied
329,465 bytes. Cargo builds no probe code. These measurements exclude the
installed runtime, model compilation and the external header checkout.

## Binding decisions and remaining checks

Use the existing safe bindings for gradient scopes. Prefer a small upstream
`tch` extension for owned CPU generators and dual-level lifecycle so all
Rust tensor consumers can share the implementation. A direct stable C ABI
adapter is feasible for the tested metadata and handle operations. Use the
runtime getters for device/dtype constants, check each error return and
release each owning handle with its matching destructor. Interchanging
`torch-sys` tensor pointers and opaque AOTI handles by cast has no established
ownership or ABI contract. [Pinned C shim](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/torch/csrc/inductor/aoti_torch/c/shim.h)

For generators, lock state reads and writes and hold an owning reference across
each call. For forward AD, enforce matching level entry/exit, unwind cleanup
and concurrent-call semantics. For callbacks, contain both Rust panics and C++
exceptions at the language boundary before releasing callback state. The
compiled-package loader and process-group APIs use version-pinned C++ headers.
Their production bindings require error, lifetime and teardown tests on Linux,
macOS and Windows. Multi-device and multi-node behavior remains unverified.
The probes introduce no production Rust dependency or FFI. The external
PyTorch and Gloo sources and build products stay outside the repository.
[Pinned generated bindings](https://github.com/LaurentMazare/tch-rs/tree/4227b89b72059b0651ff83a38637693574e0a2d6)
and [Gloo source and license](https://github.com/pytorch/gloo/tree/bcd1672ee07538123ea8f4fac76832efc58fb8ef).

## Production binding packets for review

Each packet can be reviewed independently. The contracts below describe the
first user-visible operation and the test that must pass before exposing it.

| Packet | First contract and implementation route | Acceptance before documenting support |
|---|---|---|
| W2 owned CPU generator | Map `python:torch.Generator.get_state` and `set_state` onto an owned LibTorch CPU generator through a checked `tch` extension. Keep snapshots separate from the existing RustTorch `TensorRngState` in `src/reproducibility.rs`. | Advance the generator, restore it and reproduce the next two draws. Reject malformed state without advancing it. Verify state ownership after the original handle drops and serialize draws against concurrent state changes. |
| W4 forward AD | Add native dual-level entry and exit to the existing make/unpack operations for `python:torch.autograd.forward_ad.dual_level`, `make_dual` and `unpack_dual`. Scope the level lifetime to the call in `src/autograd.rs`. | For a noncontiguous primal and tangent, compare both value and tangent with the reference. Exercise nested entry, mismatch errors, concurrent callers, unwinding and retained outputs after exit. A Rust backward callback is a separate packet requiring a reviewed panic and lifetime boundary. |
| W10 compiled CPU packages | Provide a trusted, version-checked loader for `python:torch._inductor.aoti_load_package` in `src/deployment/`. The importer must distinguish native code packages from the existing portable PT2 interpretation. | Load the locally compiled square program in a fresh process, compare outputs on multiple inputs, check wrong dtype/shape errors, and release every handle after constructor/run failure. Qualify the loader on each platform it advertises. |
| W11 native CPU collectives | Map `python:torch.distributed.FileStore` and the CPU sum subset of `python:torch.distributed.all_reduce` onto the proven c10d/Gloo route under `src/distributed/`. Pin the matching native header distribution and make waits bounded. | Launch two Rust processes, check the sum on both ranks, interrupt a participant and join both processes within the deadline. Recreate a group and complete another sum. Exercise explicit abort and outstanding work during teardown before promising cancellation. |

The executable probes above provide reference outputs for these packets.
Production interfaces that add FFI or change package dependencies require the
issue and architecture review described in [CONTRIBUTING.md](../CONTRIBUTING.md).
