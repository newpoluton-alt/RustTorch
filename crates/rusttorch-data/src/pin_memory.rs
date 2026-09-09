use std::{collections::BTreeMap, marker::PhantomData};

use rusttorch_core::{Device, Result, Tensor};

use crate::Bytes;

/// Builder state for loading without recursive batch pinning.
#[derive(Clone, Copy, Debug, Default)]
pub struct PinDisabled;

/// Builder state for enabled recursive batch pinning.
#[derive(Clone, Copy, Debug, Default)]
pub struct PinEnabled<M>(pub(crate) PhantomData<M>);

/// Automatic pinning mode selecting CUDA device zero when available.
#[derive(Clone, Copy, Debug, Default)]
pub struct Auto;

/// Explicit pinning mode validated against an available CUDA device.
#[derive(Clone, Copy, Debug, Default)]
pub struct Explicit;

/// Effective recursive pinning behavior for a built loader.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PinMemoryStatus {
    /// Pinning was not requested.
    Disabled,
    /// Automatic pinning was requested but no CUDA accelerator is available.
    DisabledNoAccelerator,
    /// Batches are pinned for this available CUDA device.
    Enabled(Device),
}

/// Recursively moves a value into host memory pinned for `device`.
///
/// Tensor storage delegates to LibTorch. Supported containers preserve their
/// shape while consuming their values, and scalar/string/byte leaves are
/// returned unchanged without cloning.
pub trait PinMemory: Sized {
    /// Pins this value for the requested accelerator.
    fn pin_memory(self, device: Device) -> Result<Self>;
}

impl PinMemory for Tensor {
    fn pin_memory(self, device: Device) -> Result<Self> {
        self.f_pin_memory(device).map_err(Into::into)
    }
}

macro_rules! impl_identity_pin {
    ($($type:ty),+ $(,)?) => {
        $(
            impl PinMemory for $type {
                fn pin_memory(self, _device: Device) -> Result<Self> {
                    Ok(self)
                }
            }
        )+
    };
}

impl_identity_pin!(u8, i8, i16, i32, i64, f32, f64, bool, String, Bytes);

impl<T> PinMemory for Vec<T>
where
    T: PinMemory,
{
    fn pin_memory(self, device: Device) -> Result<Self> {
        self.into_iter()
            .map(|value| value.pin_memory(device))
            .collect()
    }
}

impl<T> PinMemory for Option<T>
where
    T: PinMemory,
{
    fn pin_memory(self, device: Device) -> Result<Self> {
        self.map(|value| value.pin_memory(device)).transpose()
    }
}

impl<K, V> PinMemory for BTreeMap<K, V>
where
    K: Ord,
    V: PinMemory,
{
    fn pin_memory(self, device: Device) -> Result<Self> {
        self.into_iter()
            .map(|(key, value)| value.pin_memory(device).map(|value| (key, value)))
            .collect()
    }
}

macro_rules! impl_tuple_pin {
    ($(($type:ident, $value:ident)),+ $(,)?) => {
        impl<$($type),+> PinMemory for ($($type,)+)
        where
            $($type: PinMemory),+
        {
            fn pin_memory(self, device: Device) -> Result<Self> {
                let ($($value,)+) = self;
                Ok(($($value.pin_memory(device)?,)+))
            }
        }
    };
}

impl_tuple_pin!((A, a), (B, b));
impl_tuple_pin!((A, a), (B, b), (C, c));
impl_tuple_pin!((A, a), (B, b), (C, c), (D, d));
impl_tuple_pin!((A, a), (B, b), (C, c), (D, d), (E, e));
impl_tuple_pin!((A, a), (B, b), (C, c), (D, d), (E, e), (F, f));
impl_tuple_pin!((A, a), (B, b), (C, c), (D, d), (E, e), (F, f), (G, g));
impl_tuple_pin!(
    (A, a),
    (B, b),
    (C, c),
    (D, d),
    (E, e),
    (F, f),
    (G, g),
    (H, h)
);
