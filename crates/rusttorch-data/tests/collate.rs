use std::{collections::BTreeMap, convert::Infallible};

use rusttorch_core::{Kind, RustTorchError, Tensor};
use rusttorch_data::{
    Bytes, Collate, CollateError, DefaultCollator, DefaultConverter, FnCollate, VecCollate,
};

fn tensor_values_i64(tensor: &Tensor) -> Vec<i64> {
    Vec::<i64>::try_from(tensor).expect("the test tensor is one-dimensional int64")
}

#[test]
fn tensors_stack_on_a_new_leading_dimension() {
    let mut collator = DefaultCollator;
    let first = Tensor::from_slice(&[1_i64, 2]);
    let second = Tensor::from_slice(&[3_i64, 4]);
    let source_pointers = [first.data_ptr(), second.data_ptr()];
    let batch = collator
        .collate(vec![first, second])
        .expect("equal dense tensors stack");

    assert_eq!(batch.size(), [2, 2]);
    assert_eq!(batch.kind(), Kind::Int64);
    assert!(!source_pointers.contains(&batch.data_ptr()));
    assert_eq!(batch.int64_value(&[0, 0]), 1);
    assert_eq!(batch.int64_value(&[1, 1]), 4);
}

#[test]
fn numeric_scalars_use_their_exact_libtorch_kinds() {
    macro_rules! assert_kind {
        ($values:expr, $kind:expr) => {{
            let mut collator = DefaultCollator;
            let tensor = collator.collate($values).expect("numeric values convert");
            assert_eq!(tensor.size(), [2]);
            assert_eq!(tensor.kind(), $kind);
        }};
    }

    assert_kind!(vec![1_u8, 2], Kind::Uint8);
    assert_kind!(vec![1_i8, 2], Kind::Int8);
    assert_kind!(vec![1_i16, 2], Kind::Int16);
    assert_kind!(vec![1_i32, 2], Kind::Int);
    assert_kind!(vec![1_i64, 2], Kind::Int64);
    assert_kind!(vec![1_f32, 2.0], Kind::Float);
    assert_kind!(vec![1_f64, 2.0], Kind::Double);
    assert_kind!(vec![true, false], Kind::Bool);
}

#[test]
fn strings_and_bytes_remain_records() {
    let mut collator = DefaultCollator;
    let strings = collator
        .collate(vec!["one".to_owned(), "two".to_owned()])
        .expect("strings collate");
    assert_eq!(strings, ["one", "two"]);

    let bytes = collator
        .collate(vec![Bytes(vec![1, 2]), Bytes(vec![3])])
        .expect("byte records collate");
    assert_eq!(bytes, [Bytes(vec![1, 2]), Bytes(vec![3])]);
}

#[test]
fn tuples_collate_each_field_through_arity_eight() {
    let mut collator = DefaultCollator;
    let pair = collator
        .collate(vec![(1_i64, "one".to_owned()), (2, "two".to_owned())])
        .expect("pairs collate");
    assert_eq!(tensor_values_i64(&pair.0), [1, 2]);
    assert_eq!(pair.1, ["one", "two"]);

    let eight = collator
        .collate(vec![
            (1_i64, 2_i64, 3_i64, 4_i64, 5_i64, 6_i64, 7_i64, 8_i64),
            (
                11_i64, 12_i64, 13_i64, 14_i64, 15_i64, 16_i64, 17_i64, 18_i64,
            ),
        ])
        .expect("eight-tuples collate");
    assert_eq!(tensor_values_i64(&eight.0), [1, 11]);
    assert_eq!(tensor_values_i64(&eight.1), [2, 12]);
    assert_eq!(tensor_values_i64(&eight.2), [3, 13]);
    assert_eq!(tensor_values_i64(&eight.3), [4, 14]);
    assert_eq!(tensor_values_i64(&eight.4), [5, 15]);
    assert_eq!(tensor_values_i64(&eight.5), [6, 16]);
    assert_eq!(tensor_values_i64(&eight.6), [7, 17]);
    assert_eq!(tensor_values_i64(&eight.7), [8, 18]);
}

#[test]
fn equal_vectors_transpose_before_recursive_collation() {
    let mut collator = DefaultCollator;
    let columns = collator
        .collate(vec![vec![1_i64, 2], vec![3, 4]])
        .expect("equal vectors collate");

    assert_eq!(columns.len(), 2);
    assert_eq!(tensor_values_i64(&columns[0]), [1, 3]);
    assert_eq!(tensor_values_i64(&columns[1]), [2, 4]);

    let empty_columns: Vec<Tensor> = collator
        .collate(vec![Vec::<i64>::new(), Vec::new()])
        .expect("equal empty vectors collate to an empty sequence");
    assert!(empty_columns.is_empty());
}

#[test]
fn mismatched_vector_lengths_are_typed_errors() {
    let mut collator = DefaultCollator;
    let error = collator
        .collate(vec![vec![1_i64, 2], vec![3]])
        .expect_err("mismatched vectors must fail");

    assert!(matches!(
        error,
        CollateError::SequenceLengthMismatch {
            sample: 1,
            expected: 2,
            actual: 1
        }
    ));
}

#[test]
fn options_require_uniform_presence_and_recurse() {
    let mut collator = DefaultCollator;
    let values = collator
        .collate(vec![Some(1_i64), Some(2)])
        .expect("present values collate");
    assert_eq!(
        tensor_values_i64(values.as_ref().expect("all values are present")),
        [1, 2]
    );

    let absent: Option<Tensor> = collator
        .collate(vec![None::<i64>, None])
        .expect("absent values collate");
    assert!(absent.is_none());

    let error = collator
        .collate(vec![Some(1_i64), None])
        .expect_err("mixed presence must fail");
    assert!(matches!(
        error,
        CollateError::OptionPresenceMismatch {
            sample: 1,
            expected_some: true,
            actual_some: false
        }
    ));
}

#[test]
fn maps_require_identical_keys_and_recurse_over_values() {
    let mut collator = DefaultCollator;
    let first = BTreeMap::from([("left", 1_i64), ("right", 2)]);
    let second = BTreeMap::from([("left", 3_i64), ("right", 4)]);
    let batch = collator
        .collate(vec![first, second])
        .expect("matching maps collate");
    assert_eq!(tensor_values_i64(&batch["left"]), [1, 3]);
    assert_eq!(tensor_values_i64(&batch["right"]), [2, 4]);

    let error = collator
        .collate(vec![
            BTreeMap::from([("left", 1_i64)]),
            BTreeMap::from([("right", 2_i64)]),
        ])
        .expect_err("different keys must fail");
    assert!(matches!(error, CollateError::MapKeysMismatch { sample: 1 }));
}

#[test]
fn empty_batches_and_incompatible_tensors_return_errors() {
    let mut collator = DefaultCollator;
    let empty = collator
        .collate(Vec::<i64>::new())
        .expect_err("an empty top-level batch must fail");
    assert!(matches!(empty, CollateError::EmptyBatch));

    let incompatible = collator
        .collate(vec![
            Tensor::from_slice(&[1_i64]),
            Tensor::from_slice(&[2_i64, 3]),
        ])
        .expect_err("incompatible tensors must fail without panicking");
    assert!(matches!(
        incompatible,
        CollateError::LibTorch {
            source: RustTorchError::Backend(_)
        }
    ));
}

#[test]
fn default_converter_preserves_leaves_and_recurses_without_transposing() {
    let mut converter = DefaultConverter;

    let tensor = converter
        .convert(Tensor::from_slice(&[1_i64, 2]))
        .expect("tensor converts");
    assert_eq!(tensor_values_i64(&tensor), [1, 2]);
    assert_eq!(converter.convert(7_i64).expect("integer converts"), 7);
    assert_eq!(
        converter
            .convert("value".to_owned())
            .expect("string converts"),
        "value"
    );
    assert_eq!(
        converter.convert(Bytes(vec![1, 2])).expect("bytes convert"),
        Bytes(vec![1, 2])
    );
    assert_eq!(
        converter
            .convert(Some(vec![1_i64, 2, 3]))
            .expect("option vector converts"),
        Some(vec![1, 2, 3])
    );
    assert_eq!(
        converter
            .convert((vec![1_i64, 2], "row".to_owned()))
            .expect("tuple converts"),
        (vec![1, 2], "row".to_owned())
    );
    assert_eq!(
        converter
            .convert(BTreeMap::from([("values", vec![1_i64, 2])]))
            .expect("map converts"),
        BTreeMap::from([("values", vec![1, 2])])
    );
}

#[test]
fn default_converter_collate_requires_exactly_one_sample() {
    let mut converter = DefaultConverter;
    assert_eq!(
        converter
            .collate(vec![vec![1_i64, 2]])
            .expect("one sample converts"),
        vec![1, 2]
    );
    assert!(matches!(
        converter.collate(Vec::<Vec<i64>>::new()),
        Err(CollateError::ConversionBatchSize { actual: 0 })
    ));
    assert!(matches!(
        converter.collate(vec![vec![1_i64], vec![2]]),
        Err(CollateError::ConversionBatchSize { actual: 2 })
    ));
}

#[test]
fn custom_and_vector_collators_are_explicit_escape_hatches() -> Result<(), Infallible> {
    let mut collate = FnCollate::new(|samples: Vec<Vec<i64>>| {
        Ok::<_, Infallible>(samples.into_iter().flatten().collect::<Vec<_>>())
    });
    assert_eq!(collate.collate(vec![vec![1, 2], vec![3]])?, vec![1, 2, 3]);

    struct NotClone(i64);
    let mut vec_collate = VecCollate;
    let batch = vec_collate.collate(vec![NotClone(4), NotClone(5)])?;
    assert_eq!(
        batch.into_iter().map(|value| value.0).collect::<Vec<_>>(),
        [4, 5]
    );
    Ok(())
}
