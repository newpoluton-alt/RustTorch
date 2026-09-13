use super::{Result, Row, TabularError};
use arrow_array::{
    Array, Float32Array, Float64Array, Int32Array, Int64Array, LargeStringArray, RecordBatch,
    StringArray,
};
use arrow_schema::DataType;
use rusttorch_data::ResourceLimits;
use std::{fs::File, path::Path};
/// Lazy row iterator over Arrow record batches. Supports Utf8/LargeUtf8,
/// Int32/Int64 and Float32/Float64 columns; nested/dictionary types fail explicitly.
pub struct ArrowRows<I> {
    batches: I,
    batch: Option<RecordBatch>,
    row: usize,
    total: usize,
    limits: ResourceLimits,
    done: bool,
}
impl<I> ArrowRows<I> {
    /// Adapts a batch reader and enforces per-batch bytes, fields and row limits.
    pub fn new(batches: I, limits: ResourceLimits) -> Self {
        Self {
            batches,
            batch: None,
            row: 0,
            total: 0,
            limits,
            done: false,
        }
    }
}
impl<I> Iterator for ArrowRows<I>
where
    I: Iterator<Item = std::result::Result<RecordBatch, arrow_schema::ArrowError>>,
{
    type Item = Result<Row>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let result = (|| {
            loop {
                if self.batch.as_ref().is_none_or(|b| self.row >= b.num_rows()) {
                    let Some(batch) = self.batches.next() else {
                        return Ok(None);
                    };
                    let batch = batch?;
                    self.limits.check(
                        "Arrow columns",
                        batch.num_columns(),
                        self.limits.max_columns,
                    )?;
                    self.limits.check(
                        "Arrow batch bytes",
                        batch.get_array_memory_size(),
                        self.limits.max_columnar_bytes,
                    )?;
                    if batch.num_rows() > self.limits.max_records.saturating_sub(self.total) {
                        return Err(TabularError::Invalid("Arrow row limit exceeded".into()));
                    }
                    let schema = batch.schema();
                    let mut names = std::collections::BTreeSet::new();
                    for f in schema.fields() {
                        self.limits.check(
                            "Arrow column name",
                            f.name().len(),
                            self.limits.max_string_bytes,
                        )?;
                        if f.name().is_empty() || !names.insert(f.name()) {
                            return Err(TabularError::Invalid(
                                "Arrow column names must be nonempty and unique".into(),
                            ));
                        }
                        if !matches!(
                            f.data_type(),
                            DataType::Utf8
                                | DataType::LargeUtf8
                                | DataType::Int32
                                | DataType::Int64
                                | DataType::Float32
                                | DataType::Float64
                        ) {
                            return Err(TabularError::Invalid(format!(
                                "unsupported Arrow type {}",
                                f.data_type()
                            )));
                        }
                    }
                    self.row = 0;
                    self.batch = Some(batch);
                    continue;
                }
                let batch = self.batch.as_ref().expect("batch assigned above");
                let fields = batch
                    .schema()
                    .fields()
                    .iter()
                    .zip(batch.columns())
                    .map(|(field, a)| {
                        let value = if a.is_null(self.row) {
                            None
                        } else {
                            Some(match a.data_type() {
                                DataType::Utf8 => a
                                    .as_any()
                                    .downcast_ref::<StringArray>()
                                    .expect("Arrow Utf8 type invariant")
                                    .value(self.row)
                                    .to_owned(),
                                DataType::LargeUtf8 => a
                                    .as_any()
                                    .downcast_ref::<LargeStringArray>()
                                    .expect("Arrow LargeUtf8 invariant")
                                    .value(self.row)
                                    .to_owned(),
                                DataType::Int32 => a
                                    .as_any()
                                    .downcast_ref::<Int32Array>()
                                    .expect("Arrow Int32 invariant")
                                    .value(self.row)
                                    .to_string(),
                                DataType::Int64 => a
                                    .as_any()
                                    .downcast_ref::<Int64Array>()
                                    .expect("Arrow Int64 invariant")
                                    .value(self.row)
                                    .to_string(),
                                DataType::Float32 => a
                                    .as_any()
                                    .downcast_ref::<Float32Array>()
                                    .expect("Arrow Float32 invariant")
                                    .value(self.row)
                                    .to_string(),
                                DataType::Float64 => a
                                    .as_any()
                                    .downcast_ref::<Float64Array>()
                                    .expect("Arrow Float64 invariant")
                                    .value(self.row)
                                    .to_string(),
                                _ => unreachable!("validated Arrow datatype"),
                            })
                        };
                        (field.name().clone(), value)
                    })
                    .collect();
                self.row += 1;
                self.total += 1;
                let row = Row { fields };
                row.validate(self.limits)?;
                return Ok(Some(row));
            }
        })();
        match result {
            Ok(Some(row)) => Some(Ok(row)),
            Ok(None) => {
                self.done = true;
                None
            }
            Err(e) => {
                self.done = true;
                Some(Err(e))
            }
        }
    }
}
fn bounded_file(path: &Path, limits: ResourceLimits) -> Result<File> {
    let file = File::open(path)?;
    let n = usize::try_from(file.metadata()?.len()).unwrap_or(usize::MAX);
    limits.check("columnar encoded bytes", n, limits.max_encoded_bytes)?;
    Ok(file)
}
/// Opens an Arrow IPC file (file framing, not stream framing). The file size and
/// each decoded record batch are checked; internal decoder allocation is upstream-owned.
pub fn open_arrow(
    path: impl AsRef<Path>,
    limits: ResourceLimits,
) -> Result<ArrowRows<arrow_ipc::reader::FileReader<File>>> {
    let reader =
        arrow_ipc::reader::FileReader::try_new(bounded_file(path.as_ref(), limits)?, None)?;
    Ok(ArrowRows::new(reader, limits))
}
/// Opens local Parquet and rejects oversized declared uncompressed row groups
/// before decoding. Uses batches of at most 1,024 rows. Compression codecs are
/// not enabled by default; unsupported encodings return a Parquet error.
#[cfg(feature = "parquet")]
pub fn open_parquet(
    path: impl AsRef<Path>,
    limits: ResourceLimits,
) -> Result<ArrowRows<parquet::arrow::arrow_reader::ParquetRecordBatchReader>> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let builder = ParquetRecordBatchReaderBuilder::try_new(bounded_file(path.as_ref(), limits)?)?;
    let mut total = 0usize;
    for group in builder.metadata().row_groups() {
        let bytes = usize::try_from(group.total_byte_size())
            .map_err(|_| TabularError::Invalid("invalid Parquet row-group byte count".into()))?;
        limits.check("Parquet row-group bytes", bytes, limits.max_columnar_bytes)?;
        let rows = usize::try_from(group.num_rows())
            .map_err(|_| TabularError::Invalid("invalid Parquet row count".into()))?;
        total = total
            .checked_add(rows)
            .ok_or_else(|| TabularError::Invalid("Parquet row count overflow".into()))?;
        limits.check("Parquet rows", total, limits.max_records)?;
    }
    let reader = builder
        .with_batch_size(1024.min(limits.max_records.max(1)))
        .build()?;
    Ok(ArrowRows::new(reader, limits))
}
