use rusttorch_data::{DataLoader, ResourceLimits};
use rusttorch_tabular::*;
fn schema() -> Schema {
    Schema::new(vec![
        Column::Numeric {
            name: "age".into(),
            missing: MissingValue::Mean,
        },
        Column::Categorical {
            name: "color".into(),
            unknown_zero: true,
        },
    ])
    .unwrap()
}
#[test]
fn training_fit_is_immutable_and_worker_batches_match_jsonl() {
    let limits = ResourceLimits::default();
    let rows = CsvRecords::new(&b"age,color\n20,red\n40,blue\n"[..], limits)
        .unwrap()
        .collect::<Result<Vec<_>>>()
        .unwrap();
    let fit = FittedPreprocessor::fit(schema(), rows.clone().into_iter().map(Ok), limits).unwrap();
    let saved = fit.to_json().unwrap();
    let restored = FittedPreprocessor::from_json(&saved, limits).unwrap();
    let data = TabularDataset::new(rows, restored).unwrap();
    let mut loader = DataLoader::builder(data)
        .batch_size(2)
        .workers(2)
        .build()
        .unwrap();
    let batch = loader.iter().next().unwrap().unwrap();
    assert_eq!(
        Vec::<Vec<f32>>::try_from(&batch.numeric).unwrap(),
        [[-1.], [1.]]
    );
    assert_eq!(
        Vec::<Vec<i64>>::try_from(&batch.categorical).unwrap(),
        [[2], [1]]
    );
    let val = JsonLines::new(
        &b"{\"age\":30,\"color\":\"green\"}\n{\"age\":null,\"color\":null}\n"[..],
        limits,
    )
    .collect::<Result<Vec<_>>>()
    .unwrap();
    for row in val {
        let sample = fit.transform(&row).unwrap();
        assert_eq!(Vec::<f32>::try_from(&sample.numeric).unwrap(), [0.]);
        assert_eq!(Vec::<i64>::try_from(&sample.categorical).unwrap(), [0]);
    }
    assert_eq!(fit.to_json().unwrap(), saved);
}
#[test]
fn quoted_csv_newlines_and_typed_json_cells_are_preserved() {
    let rows = CsvRecords::new(
        &b"name,n\n\"a,b\",1\n\"two\nlines\",2\n"[..],
        Default::default(),
    )
    .unwrap()
    .collect::<Result<Vec<_>>>()
    .unwrap();
    assert_eq!(rows[0].fields["name"], Some("a,b".into()));
    assert_eq!(rows[1].fields["name"], Some("two\nlines".into()));
    assert!(
        JsonLines::new(&b"{\"a\":[]}\n"[..], Default::default())
            .next()
            .unwrap()
            .is_err()
    );
}
#[test]
fn invalid_states_records_and_limits_fail_once() {
    let limits = ResourceLimits {
        max_string_bytes: 3,
        ..Default::default()
    };
    let mut rows = CsvRecords::new(&b"a\nlong-value\nnext\n"[..], limits).unwrap();
    assert!(rows.next().unwrap().is_err());
    assert!(rows.next().is_none());
    assert!(CsvRecords::new(&b"a,a\n1,2\n"[..], Default::default()).is_err());
    assert!(FittedPreprocessor::fit(schema(), std::iter::empty(), Default::default()).is_err());
    let fit = FittedPreprocessor::fit(
        schema(),
        CsvRecords::new(&b"age,color\n20,red\n40,blue\n"[..], Default::default()).unwrap(),
        Default::default(),
    )
    .unwrap();
    assert!(
        FittedPreprocessor::from_json(
            &fit.to_json()
                .unwrap()
                .replace("\"version\":1", "\"version\":2"),
            Default::default()
        )
        .is_err()
    );
    assert!(
        FittedPreprocessor::fit(
            schema(),
            CsvRecords::new(&b"age,color\n20,red\n40,blue\n"[..], Default::default()).unwrap(),
            ResourceLimits {
                max_decoded_bytes: 6,
                ..Default::default()
            },
        )
        .is_err()
    );
    let mut json = JsonLines::new(
        &b"{\"a\":\"long\"}\n{}\n"[..],
        ResourceLimits {
            max_string_bytes: 5,
            ..Default::default()
        },
    );
    assert!(json.next().unwrap().is_err());
    assert!(json.next().is_none());
}

#[test]
fn deserialized_schemas_preserve_constructor_invariants_and_wire_shape() {
    let valid = schema();
    let json = serde_json::to_value(&valid).unwrap();
    assert!(json.as_object().unwrap().contains_key("columns"));
    assert_eq!(
        serde_json::from_value::<Schema>(json.clone()).unwrap(),
        valid
    );
    let column = json["columns"][0].clone();
    for columns in [
        serde_json::json!([]),
        serde_json::json!([column, column]),
        serde_json::json!([{"Numeric":{"name":"","missing":"Reject"}}]),
    ] {
        assert!(serde_json::from_value::<Schema>(serde_json::json!({"columns": columns})).is_err());
    }
}
#[cfg(feature = "arrow")]
mod columnar {
    use super::*;
    use arrow_array::{Float64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use std::{
        path::PathBuf,
        sync::{
            Arc,
            atomic::{AtomicU64, Ordering},
        },
    };
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            let p = std::env::temp_dir().join(format!(
                "rusttorch-columnar-{}-{}",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn batch() -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                Field::new("age", DataType::Float64, true),
                Field::new("color", DataType::Utf8, true),
            ])),
            vec![
                Arc::new(Float64Array::from(vec![Some(20.), Some(40.)])),
                Arc::new(StringArray::from(vec![Some("red"), Some("blue")])),
            ],
        )
        .unwrap()
    }
    #[test]
    fn arrow_ipc_rows_equal_csv_and_enforce_batch_budget() {
        let t = Temp::new();
        let p = t.0.join("data.arrow");
        let batch = batch();
        let mut w = arrow_ipc::writer::FileWriter::try_new(
            std::fs::File::create(&p).unwrap(),
            &batch.schema(),
        )
        .unwrap();
        w.write(&batch).unwrap();
        w.finish().unwrap();
        drop(w);
        let rows = open_arrow(&p, Default::default())
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let csv = CsvRecords::new(&b"age,color\n20,red\n40,blue\n"[..], Default::default())
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(rows, csv);
        let mut r = open_arrow(
            &p,
            ResourceLimits {
                max_columnar_bytes: 1,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(r.next().unwrap().is_err());
        assert!(r.next().is_none());
    }
    #[test]
    fn empty_arrow_batches_still_validate_column_names() {
        for name in ["", "oversized"] {
            let batch = RecordBatch::new_empty(Arc::new(ArrowSchema::new(vec![Field::new(
                name,
                DataType::Utf8,
                true,
            )])));
            let mut rows = ArrowRows::new(
                [Ok(batch)].into_iter(),
                ResourceLimits {
                    max_string_bytes: 4,
                    ..Default::default()
                },
            );
            assert!(rows.next().unwrap().is_err());
            assert!(rows.next().is_none());
        }
        let valid = RecordBatch::new_empty(Arc::new(ArrowSchema::new(vec![Field::new(
            "name",
            DataType::Utf8,
            true,
        )])));
        assert!(
            ArrowRows::new([Ok(valid)].into_iter(), Default::default())
                .next()
                .is_none()
        );
    }
    #[cfg(feature = "parquet")]
    #[test]
    fn parquet_rows_equal_csv_and_metadata_limits_precede_decode() {
        let t = Temp::new();
        let p = t.0.join("data.parquet");
        let b = batch();
        let mut w = parquet::arrow::ArrowWriter::try_new(
            std::fs::File::create(&p).unwrap(),
            b.schema(),
            None,
        )
        .unwrap();
        w.write(&b).unwrap();
        w.close().unwrap();
        let rows = open_parquet(&p, Default::default())
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let csv = CsvRecords::new(&b"age,color\n20,red\n40,blue\n"[..], Default::default())
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(rows, csv);
        assert!(
            open_parquet(
                &p,
                ResourceLimits {
                    max_columnar_bytes: 1,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
}
