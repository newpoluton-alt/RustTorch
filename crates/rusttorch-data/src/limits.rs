//! Shared finite limits for decoding and parsing untrusted local data.

use std::io::{self, Read};

use rusttorch_core::{Result, RustTorchError};
use serde::{Deserialize, Serialize};

/// Resource ceilings applied before allocation whenever metadata permits.
///
/// These bound logical input/output payloads, not native decoder RSS. Loader
/// queue bytes remain a separate [`crate::MemoryFootprint`] budget.
///
/// ```
/// use rusttorch_data::ResourceLimits;
/// let limits = ResourceLimits::default();
/// limits.check("tokens", 128, limits.max_tokens)?;
/// assert!(limits.check("tokens", usize::MAX, limits.max_tokens).is_err());
/// # Ok::<(), rusttorch_core::RustTorchError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum encoded bytes per local file or bounded input (64 MiB).
    pub max_encoded_bytes: usize,
    /// Maximum decoded sample or explicitly bounded batch payload (256 MiB).
    pub max_decoded_bytes: usize,
    /// Maximum elements in one tensor (64 million).
    pub max_tensor_elements: usize,
    /// Maximum image width or height (16,384).
    pub max_image_dimension: usize,
    /// Maximum records in a dataset or explicitly bounded batch (1 million).
    pub max_records: usize,
    /// Maximum fields per tabular record (4,096).
    pub max_columns: usize,
    /// Maximum UTF-8 bytes per field or text sample (1 MiB).
    pub max_string_bytes: usize,
    /// Maximum tokens per sequence or padded batch (1 million).
    pub max_tokens: usize,
    /// Maximum Arrow record-batch or Parquet row-group payload (256 MiB).
    pub max_columnar_bytes: usize,
    /// Maximum media duration in whole seconds (one hour).
    pub max_media_seconds: u64,
    /// Maximum streams per media container (32).
    pub max_streams: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_encoded_bytes: 64 * 1024 * 1024,
            max_decoded_bytes: 256 * 1024 * 1024,
            max_tensor_elements: 64 * 1024 * 1024,
            max_image_dimension: 16_384,
            max_records: 1_000_000,
            max_columns: 4_096,
            max_string_bytes: 1024 * 1024,
            max_tokens: 1_000_000,
            max_columnar_bytes: 256 * 1024 * 1024,
            max_media_seconds: 3_600,
            max_streams: 32,
        }
    }
}

impl ResourceLimits {
    /// Explicitly disables logical ceilings; integer overflow is still rejected.
    pub const fn unlimited() -> Self {
        Self {
            max_encoded_bytes: usize::MAX,
            max_decoded_bytes: usize::MAX,
            max_tensor_elements: usize::MAX,
            max_image_dimension: usize::MAX,
            max_records: usize::MAX,
            max_columns: usize::MAX,
            max_string_bytes: usize::MAX,
            max_tokens: usize::MAX,
            max_columnar_bytes: usize::MAX,
            max_media_seconds: u64::MAX,
            max_streams: usize::MAX,
        }
    }

    /// Rejects a value above its named ceiling.
    pub fn check(&self, field: &'static str, actual: usize, maximum: usize) -> Result<()> {
        if actual > maximum {
            return Err(RustTorchError::InvalidConfiguration {
                field,
                reason: format!("{actual} exceeds resource limit {maximum}"),
            });
        }
        Ok(())
    }

    /// Checks multiplication and the resulting tensor-element ceiling.
    pub fn tensor_elements(&self, dimensions: &[usize]) -> Result<usize> {
        for &dimension in dimensions {
            self.check("tensor dimension", dimension, i64::MAX as usize)?;
        }
        let elements = dimensions
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| RustTorchError::InvalidConfiguration {
                field: "tensor elements",
                reason: "shape product overflows usize".into(),
            })?;
        self.check(
            "tensor elements",
            elements,
            self.max_tensor_elements.min(i64::MAX as usize),
        )?;
        Ok(elements)
    }

    /// Reads at most the encoded ceiling plus one sentinel byte, then rejects
    /// oversized input. The limit is enforced even if file metadata changes.
    pub fn read_encoded(&self, reader: impl Read) -> io::Result<Vec<u8>> {
        let mut bytes = Vec::new();
        reader
            .take((self.max_encoded_bytes as u64).saturating_add(1))
            .read_to_end(&mut bytes)?;
        if bytes.len() > self.max_encoded_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "encoded resource limit exceeded",
            ));
        }
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_oversize_inputs_and_shape_overflow() {
        let limits = ResourceLimits {
            max_encoded_bytes: 2,
            max_tensor_elements: 8,
            ..Default::default()
        };
        assert_eq!(limits.read_encoded(&b"ab"[..]).unwrap(), b"ab");
        assert!(limits.read_encoded(&b"abc"[..]).is_err());
        assert!(limits.tensor_elements(&[3, 3]).is_err());
        assert!(
            ResourceLimits::unlimited()
                .tensor_elements(&[usize::MAX, 2])
                .is_err()
        );
    }
}
