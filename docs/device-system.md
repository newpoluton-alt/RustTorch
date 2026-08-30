# Device system

`DeviceSpec` separates a caller's request from the resolved `tch::Device`:

```rust,ignore
pub enum DeviceSpec { Auto, Cpu, Cuda(usize), Mps }
```

`Auto` tries usable CUDA device 0, then usable MPS, then CPU. The resolved
device is observable through the model. There is no generic `Gpu` variant
because CUDA and MPS have different capabilities.

Explicit requests default to error. Requesting unavailable CUDA/MPS or an
invalid CUDA index must not fall back to CPU. Forward execution checks model
and input devices and does not silently move inputs.

Capability reporting covers CPU, CUDA availability and device count, cuDNN,
and MPS. Detection uses safe backend APIs where the pinned `tch` exposes them.
MPS otherwise uses one cached, tiny, fallible tensor operation on
`tch::Device::Mps`; failure becomes “unavailable,” not a panic. Detection does
not invoke Python at application runtime.

Moving a model moves every tensor registered in its `VarStore` while preserving
names and train/eval state. The 0.1 module set registers parameters and has no
dedicated persistent-buffer API. Weight files are device-neutral and are loaded
onto the target model device. Rebuild optimizers after movement unless
optimizer-state movement is explicitly verified.

Default training dtype is float32. Mixed precision is not a cross-backend
compatibility promise.
