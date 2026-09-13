# RustTorch Tabular

Turn local CSV, JSON Lines, Arrow and Parquet data into model-ready numeric and
categorical tensors. Fit preprocessing once on training data, save it, and use
the same immutable statistics on validation, test and inference rows.

## Installation

Enable `rusttorch = { version = "0.4", features = ["tabular"] }` for CSV/JSONL,
or add `columnar` for Arrow/Parquet. Direct package users select `arrow` or
`parquet` separately. CSV users do not compile the columnar stack. All tensor
packages share the existing LibTorch runtime.

## Features

The `tabular` facade feature enables CSV/JSONL and fitted preprocessing.
`columnar` also enables Arrow/Parquet; direct consumers can select `arrow` or
`parquet`. The direct package has no default features.
`download-libtorch` obtains the tensor runtime; `doc-only` builds documentation
without native linking and cannot execute tensor operations.

## Native runtime

Tensor operations require **LibTorch 2.13.0**. Enable `download-libtorch`, or set
`LIBTORCH` to an extracted distribution and add its library directory to `PATH`
on Windows or the platform's shared-library search path. All RustTorch packages
in one application must share the same runtime. Documentation can be checked
with `cargo doc -p rusttorch-tabular --no-default-features --features doc-only`.

## Example: Fit training data and batch validation rows

```rust
use rusttorch_data::{DataLoader, ResourceLimits};
use rusttorch_tabular::{Column, CsvRecords, FittedPreprocessor, MissingValue,
    Schema, TabularDataset};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let limits = ResourceLimits::default();
    let schema = Schema::new(vec![
        Column::Numeric { name: "age".into(), missing: MissingValue::Mean },
        Column::Categorical { name: "city".into(), unknown_zero: true },
    ])?;
    let training = CsvRecords::new("age,city\n20,Berlin\n40,Paris\n".as_bytes(), limits)?;
    let fit = FittedPreprocessor::fit(schema, training, limits)?;
    let saved = fit.to_json()?; // Store beside the model checkpoint.

    let fit = FittedPreprocessor::from_json(&saved, limits)?;
    let rows = CsvRecords::new("age,city\n30,Berlin\n,Unknown\n".as_bytes(), limits)?
        .collect::<Result<Vec<_>, _>>()?;
    let dataset = TabularDataset::new(rows, fit)?;
    let mut loader = DataLoader::builder(dataset).batch_size(2).workers(2).build()?;
    let batch = loader.iter().next().unwrap()?;
    assert_eq!(Vec::<Vec<f32>>::try_from(&batch.numeric)?, [[0.], [0.]]);
    assert_eq!(Vec::<Vec<i64>>::try_from(&batch.categorical)?, [[1], [0]]);
    Ok(())
}
```

Numeric features use the training mean and population standard deviation. A
constant feature uses scale one. `MissingValue::Mean` produces normalized zero;
`Reject` fails. Numeric values and resulting Float features must be finite.
Categorical vocabularies are sorted; known IDs start at one. ID zero represents
missing/unknown values only when explicitly enabled. Numeric and categorical
output order follows the schema, in separate `Float` and `Int64` tensors.
Extra input fields are ignored. No statistics are recomputed during iteration.

## Stream records without retaining the full dataset

```no_run
use rusttorch_data::{batches, ResourceLimits};
use rusttorch_tabular::{FittedPreprocessor, open_jsonl};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let limits = ResourceLimits::default();
    let fit = FittedPreprocessor::from_json(&std::fs::read_to_string("preprocessing.json")?, limits)?;
    let rows = open_jsonl("events.jsonl", limits)?;
    for samples in batches(fit.transform_rows(rows), 128, false)? {
        let samples = samples?; // Owned TabularSample values, loaded lazily.
        println!("{} rows", samples.len());
    }
    Ok(())
}
```

`open_csv`/`CsvRecords` delegate quoting and multiline fields to the CSV parser.
Headers must be unique and nonempty. `open_jsonl`/`JsonLines` accept one flat
object per line with string, number or null values. Blank lines, arrays and
nested values return contextual errors. Empty CSV fields and JSON null are
missing values. Reader errors are yielded once, then iteration terminates.

## Read Arrow and Parquet

With the direct package's `parquet` feature enabled:

```no_run
# #[cfg(feature="parquet")]
# fn run() -> Result<(), Box<dyn std::error::Error>> {
use rusttorch_data::ResourceLimits;
use rusttorch_tabular::{open_arrow, open_parquet};

let limits = ResourceLimits::default();
for row in open_arrow("features.arrow", limits)? {
    println!("{:?}", row?.fields);
}
for row in open_parquet("features.parquet", limits)? {
    println!("{:?}", row?.fields);
}
# Ok(()) }
```

The Arrow reader uses IPC **file** framing. Both readers support Utf8/LargeUtf8,
Int32/Int64 and Float32/Float64 with nulls. Other types fail explicitly. Parquet
reads batches of at most 1,024 rows and validates declared uncompressed row-group
sizes before decoding. Compression codecs are not enabled in the initial
profile; unsupported compression returns the upstream error.

## Limits and evidence

`ResourceLimits` covers total encoded input, columns, fields/JSONL lines, record
counts, retained rows, tensor elements, Arrow batch bytes and Parquet row-group
bytes. Metadata is checked before decoding where available; upstream readers
own their internal scratch allocations. Fitted-state JSON validates version,
finite scales, field kinds and sorted unique vocabulary entries on restore.

`tests/pipelines.rs` checks CSV/JSONL equivalence, quoted newlines, immutable
training statistics, worker tensor batches, corrupt state and input limits.
Optional tests write/read actual Arrow IPC and Parquet files and compare their
rows to CSV. Existing `rusttorch::data` APIs remain unchanged.

| Input or operation | Executable evidence |
|---|---|
| CSV quoting/newlines and JSON scalar/null cells | `quoted_csv_newlines_and_typed_json_cells_are_preserved` |
| Train-only fit, immutable restore and worker batches | `training_fit_is_immutable_and_worker_batches_match_jsonl` |
| Arrow IPC files and batch byte ceilings | `arrow_ipc_rows_equal_csv_and_enforce_batch_budget` |
| Uncompressed Parquet and row-group metadata ceilings | `parquet_rows_equal_csv_and_metadata_limits_precede_decode` |
| Malformed records, states and aggregate fitted vocabulary limits | `invalid_states_records_and_limits_fail_once` |
