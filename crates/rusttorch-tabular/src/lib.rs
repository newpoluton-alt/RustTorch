//! Stream local records, fit preprocessing on training data, and reuse that
//! immutable state for validation and inference.
//!
//! ```
//! use rusttorch_tabular::{CsvRecords, Column, FittedPreprocessor, MissingValue, Schema};
//! use rusttorch_data::ResourceLimits;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let limits = ResourceLimits::default();
//! let schema = Schema::new(vec![Column::Numeric { name: "age".into(), missing: MissingValue::Reject }])?;
//! let training = CsvRecords::new("age\n20\n40\n".as_bytes(), limits)?;
//! let fitted = FittedPreprocessor::fit(schema, training, limits)?;
//! let state = fitted.to_json()?;
//! let validation = CsvRecords::new("age\n30\n".as_bytes(), limits)?;
//! let restored = FittedPreprocessor::from_json(&state, limits)?;
//! let sample = restored.transform(&validation.into_iter().next().unwrap()?)?;
//! assert_eq!(Vec::<f32>::try_from(&sample.numeric)?, [0.]);
//! # Ok(()) }
//! ```
#![deny(missing_docs)]
#![doc = include_str!("../README.md")]
use rusttorch_core::{Device, Tensor};
use rusttorch_data::{
    CollateError, DefaultCollate, DefaultConvert, MemoryFootprint, PinMemory, ResourceLimits,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{BufRead, BufReader, Read},
};
/// Parsing, schema, fitted-state or tensor failure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TabularError {
    /// CSV syntax/type error with record position retained.
    #[error(transparent)]
    Csv(#[from] csv::Error),
    /// JSON syntax error with line and column retained.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Local reader failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Invalid schema, cell or fitted state.
    #[error("{0}")]
    Invalid(String),
    /// Finite input/output limit exceeded.
    #[error(transparent)]
    Limit(#[from] rusttorch_core::RustTorchError),
    /// LibTorch failure.
    #[error(transparent)]
    Tensor(#[from] tch::TchError),
    /// Arrow reader or schema error.
    #[cfg(feature = "arrow")]
    #[error(transparent)]
    Arrow(#[from] arrow_schema::ArrowError),
    /// Parquet reader error.
    #[cfg(feature = "parquet")]
    #[error(transparent)]
    Parquet(#[from] parquet::errors::ParquetError),
}
/// Result returned by tabular operations.
pub type Result<T> = std::result::Result<T, TabularError>;
/// One named row. Empty/missing cells are `None`; numbers remain decimal strings
/// until the selected schema parses them. Nested JSON values are rejected.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Row {
    /// Column names and optional UTF-8 cell values.
    pub fields: BTreeMap<String, Option<String>>,
}
impl Row {
    fn validate(&self, limits: ResourceLimits) -> Result<()> {
        limits.check("row columns", self.fields.len(), limits.max_columns)?;
        for (name, value) in &self.fields {
            limits.check("column name", name.len(), limits.max_string_bytes)?;
            if let Some(v) = value {
                limits.check("field bytes", v.len(), limits.max_string_bytes)?;
            }
        }
        Ok(())
    }
}
impl MemoryFootprint for Row {
    fn resident_bytes(&self) -> usize {
        self.fields.resident_bytes()
    }
}
impl DefaultConvert for Row {
    type Output = Self;
    fn default_convert(self) -> std::result::Result<Self, CollateError> {
        Ok(self)
    }
}
/// Streaming CSV reader with mandatory unique headers and finite total input.
/// Quoted separators/newlines are handled by the established `csv` parser.
pub struct CsvRecords<R: Read> {
    reader: csv::Reader<std::io::Take<R>>,
    headers: Vec<String>,
    limits: ResourceLimits,
    count: usize,
    done: bool,
}
impl<R: Read> CsvRecords<R> {
    /// Creates a reader; a zero-byte/duplicate-header input is rejected.
    pub fn new(reader: R, limits: ResourceLimits) -> Result<Self> {
        let mut reader = csv::ReaderBuilder::new()
            .from_reader(reader.take((limits.max_encoded_bytes as u64).saturating_add(1)));
        let headers = reader
            .headers()?
            .iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        limits.check("CSV columns", headers.len(), limits.max_columns)?;
        let unique = headers.iter().collect::<BTreeSet<_>>();
        if headers.is_empty()
            || unique.len() != headers.len()
            || headers.iter().any(String::is_empty)
        {
            return Err(TabularError::Invalid(
                "CSV requires nonempty unique headers".into(),
            ));
        }
        for h in &headers {
            limits.check("CSV header bytes", h.len(), limits.max_string_bytes)?;
        }
        Ok(Self {
            reader,
            headers,
            limits,
            count: 0,
            done: false,
        })
    }
}
impl<R: Read> Iterator for CsvRecords<R> {
    type Item = Result<Row>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let result = (|| {
            let mut record = csv::StringRecord::new();
            let has = self.reader.read_record(&mut record)?;
            self.limits.check(
                "CSV encoded bytes",
                usize::try_from(self.reader.position().byte()).unwrap_or(usize::MAX),
                self.limits.max_encoded_bytes,
            )?;
            if !has {
                return Ok(None);
            }
            self.count = self
                .count
                .checked_add(1)
                .ok_or_else(|| TabularError::Invalid("record counter overflow".into()))?;
            self.limits
                .check("CSV records", self.count, self.limits.max_records)?;
            let row = Row {
                fields: self
                    .headers
                    .iter()
                    .cloned()
                    .zip(record.iter().map(|s| {
                        if s.is_empty() {
                            None
                        } else {
                            Some(s.to_owned())
                        }
                    }))
                    .collect(),
            };
            row.validate(self.limits)?;
            Ok(Some(row))
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
/// Bounded JSON Lines reader. Each line is one flat object; blank lines fail.
pub struct JsonLines<R: BufRead> {
    reader: R,
    limits: ResourceLimits,
    line: usize,
    encoded: usize,
    done: bool,
}
impl<R: BufRead> JsonLines<R> {
    /// Creates a lazy reader without consuming input.
    pub fn new(reader: R, limits: ResourceLimits) -> Self {
        Self {
            reader,
            limits,
            line: 0,
            encoded: 0,
            done: false,
        }
    }
}
impl<R: BufRead> Iterator for JsonLines<R> {
    type Item = Result<Row>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let result = (|| {
            let mut bytes = Vec::new();
            let remaining = self.limits.max_encoded_bytes.saturating_sub(self.encoded);
            let ceiling = remaining
                .min(self.limits.max_string_bytes)
                .saturating_add(1);
            let n = (&mut self.reader)
                .take(ceiling as u64)
                .read_until(b'\n', &mut bytes)?;
            if n == 0 {
                return Ok(None);
            }
            self.encoded = self
                .encoded
                .checked_add(n)
                .ok_or_else(|| TabularError::Invalid("encoded byte count overflow".into()))?;
            self.limits.check(
                "JSONL encoded bytes",
                self.encoded,
                self.limits.max_encoded_bytes,
            )?;
            self.limits
                .check("JSONL line bytes", n, self.limits.max_string_bytes)?;
            self.line += 1;
            self.limits
                .check("JSONL records", self.line, self.limits.max_records)?;
            let object: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&bytes)?;
            let fields=object.into_iter().map(|(key,v)|{let value=match v{serde_json::Value::Null=>None,serde_json::Value::String(v)=>Some(v),serde_json::Value::Number(v)=>Some(v.to_string()),_=>return Err(TabularError::Invalid(format!("JSONL line {} column {key}: expected scalar string, number or null",self.line)))};Ok((key,value))}).collect::<Result<_>>()?;
            let row = Row { fields };
            row.validate(self.limits)?;
            Ok(Some(row))
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
/// Missing numeric value behavior.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum MissingValue {
    /// Fail on missing numeric cells.
    Reject,
    /// Substitute the training mean, yielding normalized zero.
    Mean,
}
/// One model input column; numeric and categorical output orders follow schema order.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Column {
    /// Numeric feature standardized with training population variance.
    Numeric {
        /// Exact input column name.
        name: String,
        /// Missing-value policy.
        missing: MissingValue,
    },
    /// Categorical feature with sorted training vocabulary and Int64 output.
    Categorical {
        /// Exact input column name.
        name: String,
        /// Reserve ID zero for missing/unknown values; otherwise reject.
        unknown_zero: bool,
    },
}
impl Column {
    fn name(&self) -> &str {
        match self {
            Self::Numeric { name, .. } | Self::Categorical { name, .. } => name,
        }
    }
}
/// Nonempty schema with unique column names. Extra input fields are ignored.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Schema {
    columns: Vec<Column>,
}
impl<'de> Deserialize<'de> for Schema {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Fields {
            columns: Vec<Column>,
        }
        let fields = Fields::deserialize(deserializer)?;
        Self::new(fields.columns).map_err(serde::de::Error::custom)
    }
}
impl Schema {
    /// Validates unique nonempty names.
    pub fn new(columns: Vec<Column>) -> Result<Self> {
        let names = columns.iter().map(Column::name).collect::<BTreeSet<_>>();
        if columns.is_empty() || names.len() != columns.len() || names.contains("") {
            return Err(TabularError::Invalid(
                "schema requires unique nonempty column names".into(),
            ));
        }
        Ok(Self { columns })
    }
    /// Columns in model input order.
    pub fn columns(&self) -> &[Column] {
        &self.columns
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
enum FittedColumn {
    Numeric { mean: f64, scale: f64 },
    Categorical { values: Vec<String> },
}
/// Immutable fitted training statistics and categorical vocabularies.
#[derive(Clone, Debug, Serialize)]
pub struct FittedPreprocessor {
    version: u32,
    schema: Schema,
    columns: Vec<FittedColumn>,
    #[serde(skip)]
    limits: ResourceLimits,
}
#[derive(Deserialize)]
struct SerializedFit {
    version: u32,
    schema: Schema,
    columns: Vec<FittedColumn>,
}
impl FittedPreprocessor {
    /// Fits once over training rows. Numeric variance uses a stable online update;
    /// validation/test transformation never modifies these statistics.
    pub fn fit(
        schema: Schema,
        rows: impl IntoIterator<Item = Result<Row>>,
        limits: ResourceLimits,
    ) -> Result<Self> {
        limits.check("schema columns", schema.columns.len(), limits.max_columns)?;
        for column in &schema.columns {
            limits.check("column name", column.name().len(), limits.max_string_bytes)?;
        }
        let mut stats = vec![(0usize, 0f64, 0f64); schema.columns.len()];
        let mut vocab = vec![BTreeSet::new(); schema.columns.len()];
        let mut count = 0;
        let mut vocabulary_bytes = 0usize;
        for row in rows {
            let row = row?;
            row.validate(limits)?;
            count += 1;
            limits.check("training records", count, limits.max_records)?;
            for (i, c) in schema.columns.iter().enumerate() {
                let value = row.fields.get(c.name()).and_then(Option::as_deref);
                match c {
                    Column::Numeric { missing, .. } => {
                        let Some(value) = value else {
                            if *missing == MissingValue::Reject {
                                return Err(TabularError::Invalid(format!(
                                    "missing numeric column {}",
                                    c.name()
                                )));
                            }
                            continue;
                        };
                        let x = parse_number(value, c.name())?;
                        let (n, mean, m2) = &mut stats[i];
                        *n += 1;
                        let d = x - *mean;
                        *mean += d / (*n as f64);
                        *m2 += d * (x - *mean);
                        if !mean.is_finite() || !m2.is_finite() {
                            return Err(TabularError::Invalid(
                                "training statistics overflow".into(),
                            ));
                        }
                    }
                    Column::Categorical { unknown_zero, .. } => {
                        if let Some(v) = value {
                            if !vocab[i].contains(v) {
                                vocabulary_bytes =
                                    vocabulary_bytes.checked_add(v.len()).ok_or_else(|| {
                                        TabularError::Invalid("vocabulary size overflow".into())
                                    })?;
                                limits.check(
                                    "fitted vocabulary bytes",
                                    vocabulary_bytes,
                                    limits.max_decoded_bytes,
                                )?;
                                limits.check(
                                    "vocabulary size",
                                    vocab[i].len().saturating_add(1),
                                    limits.max_records,
                                )?;
                                vocab[i].insert(v.to_owned());
                            }
                        } else if !unknown_zero {
                            return Err(TabularError::Invalid(format!(
                                "missing category {}",
                                c.name()
                            )));
                        }
                    }
                }
            }
        }
        if count == 0 {
            return Err(TabularError::Invalid(
                "cannot fit empty training input".into(),
            ));
        }
        let columns = schema
            .columns
            .iter()
            .enumerate()
            .map(|(i, c)| match c {
                Column::Numeric { .. } => {
                    let (n, mean, m2) = stats[i];
                    if n == 0 {
                        return Err(TabularError::Invalid(format!(
                            "no numeric training values for {}",
                            c.name()
                        )));
                    }
                    let scale = (m2 / (n as f64)).max(0.).sqrt();
                    Ok(FittedColumn::Numeric {
                        mean,
                        scale: if scale == 0. { 1. } else { scale },
                    })
                }
                Column::Categorical { .. } => Ok(FittedColumn::Categorical {
                    values: std::mem::take(&mut vocab[i]).into_iter().collect(),
                }),
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            version: 1,
            schema,
            columns,
            limits,
        })
    }
    /// Transforms one row using stored statistics; does not fit or mutate state.
    pub fn transform(&self, row: &Row) -> Result<TabularSample> {
        row.validate(self.limits)?;
        let mut numeric = Vec::new();
        let mut categorical = Vec::new();
        for (c, f) in self.schema.columns.iter().zip(&self.columns) {
            let v = row.fields.get(c.name()).and_then(Option::as_deref);
            match (c, f) {
                (Column::Numeric { missing, .. }, FittedColumn::Numeric { mean, scale }) => {
                    let x = match v {
                        Some(v) => parse_number(v, c.name())?,
                        None if *missing == MissingValue::Mean => *mean,
                        None => {
                            return Err(TabularError::Invalid(format!(
                                "missing numeric column {}",
                                c.name()
                            )));
                        }
                    };
                    let normalized = ((x - mean) / scale) as f32;
                    if !normalized.is_finite() {
                        return Err(TabularError::Invalid(
                            "normalized feature is not finite Float".into(),
                        ));
                    }
                    numeric.push(normalized);
                }
                (
                    Column::Categorical { unknown_zero, .. },
                    FittedColumn::Categorical { values },
                ) => {
                    let id = v
                        .and_then(|v| values.binary_search_by(|s| s.as_str().cmp(v)).ok())
                        .map(|i| i as i64 + 1);
                    categorical.push(match id {
                        Some(id) => id,
                        None if *unknown_zero => 0,
                        None => {
                            return Err(TabularError::Invalid(format!(
                                "unknown or missing category {}",
                                c.name()
                            )));
                        }
                    });
                }
                _ => {
                    return Err(TabularError::Invalid(
                        "fitted column type differs from schema".into(),
                    ));
                }
            }
        }
        self.limits
            .tensor_elements(&[numeric.len().saturating_add(categorical.len())])?;
        Ok(TabularSample {
            numeric: Tensor::f_from_slice(&numeric)?,
            categorical: Tensor::f_from_slice(&categorical)?,
        })
    }
    /// Serializes a versioned immutable fitted state.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }
    /// Restores state after validating sizes, finite statistics, column kinds,
    /// sorted unique vocabularies and supported schema version.
    pub fn from_json(json: &str, limits: ResourceLimits) -> Result<Self> {
        limits.check("fitted state bytes", json.len(), limits.max_encoded_bytes)?;
        let state: SerializedFit = serde_json::from_str(json)?;
        let schema = Schema::new(state.schema.columns)?;
        limits.check("fitted columns", schema.columns.len(), limits.max_columns)?;
        if state.version != 1 || state.columns.len() != schema.columns.len() {
            return Err(TabularError::Invalid(
                "unsupported fitted state version or column count".into(),
            ));
        }
        let mut vocabulary_bytes = 0usize;
        for (c, f) in schema.columns.iter().zip(&state.columns) {
            limits.check("column name", c.name().len(), limits.max_string_bytes)?;
            match (c, f) {
                (Column::Numeric { .. }, FittedColumn::Numeric { mean, scale })
                    if mean.is_finite() && scale.is_finite() && *scale > 0. => {}
                (Column::Categorical { .. }, FittedColumn::Categorical { values }) => {
                    limits.check("vocabulary size", values.len(), limits.max_records)?;
                    if values.windows(2).any(|w| w[0] >= w[1]) {
                        return Err(TabularError::Invalid(
                            "vocabulary must be sorted and unique".into(),
                        ));
                    }
                    for v in values {
                        vocabulary_bytes =
                            vocabulary_bytes.checked_add(v.len()).ok_or_else(|| {
                                TabularError::Invalid("vocabulary size overflow".into())
                            })?;
                        limits.check(
                            "fitted vocabulary bytes",
                            vocabulary_bytes,
                            limits.max_decoded_bytes,
                        )?;
                        limits.check("category bytes", v.len(), limits.max_string_bytes)?;
                    }
                }
                _ => {
                    return Err(TabularError::Invalid(
                        "invalid fitted statistics or column kind".into(),
                    ));
                }
            }
        }
        Ok(Self {
            version: 1,
            schema,
            columns: state.columns,
            limits,
        })
    }
    /// Lazily converts a record iterator into model inputs suitable for
    /// `rusttorch_data::batches` or a sharded `StreamDataLoader` source.
    pub fn transform_rows<'a, I>(
        &'a self,
        rows: I,
    ) -> impl Iterator<Item = Result<TabularSample>> + 'a
    where
        I: IntoIterator<Item = Result<Row>> + 'a,
        I::IntoIter: 'a,
    {
        rows.into_iter().map(|row| self.transform(&row?))
    }
}
fn parse_number(value: &str, name: &str) -> Result<f64> {
    let x = value
        .parse::<f64>()
        .map_err(|_| TabularError::Invalid(format!("column {name}: invalid number {value:?}")))?;
    if !x.is_finite() {
        return Err(TabularError::Invalid(format!(
            "column {name}: non-finite number"
        )));
    }
    Ok(x)
}
/// One model-ready row, with separate floating-point and categorical inputs.
#[derive(Debug)]
pub struct TabularSample {
    /// Float standardized features, shape `[numeric_columns]`.
    pub numeric: Tensor,
    /// Int64 category IDs, shape `[categorical_columns]`.
    pub categorical: Tensor,
}
/// Tabular model batch with one leading sample dimension.
#[derive(Debug)]
pub struct TabularBatch {
    /// Float `[batch,numeric_columns]` features.
    pub numeric: Tensor,
    /// Int64 `[batch,categorical_columns]` IDs.
    pub categorical: Tensor,
}
impl DefaultCollate for TabularSample {
    type Batch = TabularBatch;
    fn default_collate(samples: Vec<Self>) -> std::result::Result<TabularBatch, CollateError> {
        let (n, c): (Vec<_>, Vec<_>) = samples
            .into_iter()
            .map(|s| (s.numeric, s.categorical))
            .unzip();
        Ok(TabularBatch {
            numeric: Tensor::default_collate(n)?,
            categorical: Tensor::default_collate(c)?,
        })
    }
}
impl DefaultConvert for TabularSample {
    type Output = Self;
    fn default_convert(self) -> std::result::Result<Self, CollateError> {
        Ok(self)
    }
}
impl MemoryFootprint for TabularSample {
    fn resident_bytes(&self) -> usize {
        self.numeric
            .resident_bytes()
            .saturating_add(self.categorical.resident_bytes())
    }
}
impl MemoryFootprint for TabularBatch {
    fn resident_bytes(&self) -> usize {
        self.numeric
            .resident_bytes()
            .saturating_add(self.categorical.resident_bytes())
    }
}
impl PinMemory for TabularSample {
    fn pin_memory(mut self, d: Device) -> rusttorch_core::Result<Self> {
        self.numeric = self.numeric.pin_memory(d)?;
        self.categorical = self.categorical.pin_memory(d)?;
        Ok(self)
    }
}
impl PinMemory for TabularBatch {
    fn pin_memory(mut self, d: Device) -> rusttorch_core::Result<Self> {
        self.numeric = self.numeric.pin_memory(d)?;
        self.categorical = self.categorical.pin_memory(d)?;
        Ok(self)
    }
}

/// Reads a local JSON Lines file through a buffered bounded iterator.
pub fn open_jsonl(
    path: impl AsRef<std::path::Path>,
    limits: ResourceLimits,
) -> Result<JsonLines<BufReader<std::fs::File>>> {
    Ok(JsonLines::new(
        BufReader::new(std::fs::File::open(path)?),
        limits,
    ))
}
/// Reads a local CSV file through a buffered bounded iterator.
pub fn open_csv(
    path: impl AsRef<std::path::Path>,
    limits: ResourceLimits,
) -> Result<CsvRecords<BufReader<std::fs::File>>> {
    CsvRecords::new(BufReader::new(std::fs::File::open(path)?), limits)
}

#[cfg(feature = "arrow")]
mod columnar;
#[cfg(feature = "parquet")]
pub use columnar::open_parquet;
#[cfg(feature = "arrow")]
pub use columnar::{ArrowRows, open_arrow};

/// Finite immutable rows transformed lazily for ordinary worker DataLoader use.
/// For datasets larger than memory, use CSV/JSONL/Arrow streaming iterators.
pub struct TabularDataset {
    rows: Vec<Row>,
    preprocessing: FittedPreprocessor,
}
impl TabularDataset {
    /// Validates row counts and field limits using the fitted state's policy.
    pub fn new(rows: Vec<Row>, preprocessing: FittedPreprocessor) -> Result<Self> {
        preprocessing.limits.check(
            "tabular records",
            rows.len(),
            preprocessing.limits.max_records,
        )?;
        let mut bytes = 0usize;
        for row in &rows {
            row.validate(preprocessing.limits)?;
            bytes = bytes.saturating_add(row.resident_bytes());
        }
        preprocessing.limits.check(
            "retained tabular bytes",
            bytes,
            preprocessing.limits.max_decoded_bytes,
        )?;
        Ok(Self {
            rows,
            preprocessing,
        })
    }
}
impl rusttorch_data::Dataset for TabularDataset {
    fn get_batch_with_context(
        &self,
        indices: &[usize],
        context: &rusttorch_data::WorkerContext,
    ) -> Result<Vec<Self::Sample>> {
        indices
            .iter()
            .map(|&index| {
                context.check().map_err(|_| {
                    TabularError::Invalid("sample loading cancelled or timed out".into())
                })?;
                self.get(index)
            })
            .collect()
    }
    type Sample = TabularSample;
    type Error = TabularError;
    fn len(&self) -> usize {
        self.rows.len()
    }
    fn get(&self, index: usize) -> Result<TabularSample> {
        self.preprocessing.transform(
            self.rows
                .get(index)
                .ok_or_else(|| TabularError::Invalid("tabular index out of range".into()))?,
        )
    }
}
