use std::{cell::RefCell, panic::catch_unwind, rc::Rc};

use rusttorch_core::RustTorchError;
use rusttorch_data::{
    BatchSource, FnBatchSource, FnSampler, RandomSampler, Sampler, SequentialSampler,
    SubsetRandomSampler, WeightedRandomSampler,
};

fn assert_invalid_configuration<T>(result: rusttorch_core::Result<T>, field: &'static str) {
    assert!(matches!(
        result,
        Err(RustTorchError::InvalidConfiguration {
            field: actual,
            ..
        }) if actual == field
    ));
}

fn assert_caught_invalid_configuration<T>(
    caught: std::thread::Result<rusttorch_core::Result<T>>,
    field: &'static str,
) {
    let result = caught.unwrap_or_else(|_| panic!("constructor panicked for `{field}`"));
    assert_invalid_configuration(result, field);
}

#[test]
fn random_sampler_preserves_the_empty_length_error_contract() {
    assert!(matches!(
        RandomSampler::new(0, 7),
        Err(RustTorchError::InvalidConfiguration {
            field: "length",
            reason,
        }) if reason == "must be greater than zero"
    ));
}

#[test]
fn random_sampler_with_replacement_is_seeded_and_bounded() {
    let first = RandomSampler::with_replacement(4, 12, 7)
        .expect("positive length and sample count must be valid")
        .collect::<Vec<_>>();
    let second = RandomSampler::with_replacement(4, 12, 7)
        .expect("positive length and sample count must be valid")
        .collect::<Vec<_>>();
    let different_seed = RandomSampler::with_replacement(4, 12, 8)
        .expect("positive length and sample count must be valid")
        .collect::<Vec<_>>();

    assert_eq!(first.len(), 12);
    assert!(first.iter().all(|&index| index < 4));
    assert_eq!(first, second);
    assert_ne!(first, different_seed);
}

#[test]
fn random_sampler_without_replacement_repeats_complete_permutations() {
    let indices = RandomSampler::without_replacement(3, 8, 7)
        .expect("positive length and sample count must be valid")
        .collect::<Vec<_>>();

    assert_eq!(indices.len(), 8);
    for permutation in [&indices[..3], &indices[3..6]] {
        let mut sorted = permutation.to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, [0, 1, 2]);
    }
    assert_ne!(&indices[..3], &indices[3..6]);
    assert!(indices[6..].iter().all(|&index| index < 3));
}

#[test]
fn random_sampler_constructors_validate_length_and_sample_count() {
    assert_invalid_configuration(RandomSampler::with_replacement(0, 1, 7), "length");
    assert_invalid_configuration(RandomSampler::with_replacement(1, 0, 7), "num_samples");
    assert_invalid_configuration(RandomSampler::without_replacement(0, 1, 7), "length");
    assert_invalid_configuration(RandomSampler::without_replacement(1, 0, 7), "num_samples");
}

#[test]
fn random_sampler_rejects_unrepresentable_storage_without_panicking() {
    for caught in [
        catch_unwind(|| RandomSampler::new(usize::MAX, 7)),
        catch_unwind(|| RandomSampler::without_replacement(usize::MAX, 1, 7)),
    ] {
        assert_caught_invalid_configuration(caught, "length");
    }
    for caught in [
        catch_unwind(|| RandomSampler::with_replacement(1, usize::MAX, 7)),
        catch_unwind(|| RandomSampler::without_replacement(1, usize::MAX, 7)),
    ] {
        assert_caught_invalid_configuration(caught, "num_samples");
    }
}

#[test]
fn subset_random_sampler_yields_one_seeded_permutation() -> rusttorch_core::Result<()> {
    let first = SubsetRandomSampler::new(vec![9, 4, 7], 3)?.collect::<Vec<_>>();
    let second = SubsetRandomSampler::new(vec![9, 4, 7], 3)?.collect::<Vec<_>>();
    let different_seed = SubsetRandomSampler::new(vec![9, 4, 7], 4)?.collect::<Vec<_>>();

    let mut sorted = first.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, vec![4, 7, 9]);
    assert_eq!(first, second);
    assert_ne!(first, different_seed);
    Ok(())
}

#[test]
fn weighted_random_sampler_validates_inputs() {
    assert_invalid_configuration(WeightedRandomSampler::new(vec![], 1, true, 7), "weights");
    assert_invalid_configuration(
        WeightedRandomSampler::new(vec![1.0], 0, true, 7),
        "num_samples",
    );
    assert_invalid_configuration(
        WeightedRandomSampler::new(vec![1.0, -1.0], 1, true, 7),
        "weights",
    );
    assert_invalid_configuration(
        WeightedRandomSampler::new(vec![1.0, f64::INFINITY], 1, true, 7),
        "weights",
    );
    assert_invalid_configuration(
        WeightedRandomSampler::new(vec![0.0, 0.0], 1, true, 7),
        "weights",
    );
    assert_invalid_configuration(
        WeightedRandomSampler::new(vec![1.0, 1.0], 3, false, 7),
        "num_samples",
    );
}

#[test]
fn weighted_random_sampler_with_replacement_is_deterministic_and_has_cardinality() {
    let first = WeightedRandomSampler::new(vec![1.0, 2.0, 4.0], 16, true, 7)
        .expect("valid weights must sample")
        .collect::<Vec<_>>();
    let second = WeightedRandomSampler::new(vec![1.0, 2.0, 4.0], 16, true, 7)
        .expect("valid weights must sample")
        .collect::<Vec<_>>();
    let only_nonzero = WeightedRandomSampler::new(vec![0.0, 0.0, 1.0], 4, true, 7)
        .expect("one positive weight must sample")
        .collect::<Vec<_>>();

    assert_eq!(first.len(), 16);
    assert_eq!(first, second);
    assert!(first.iter().all(|&index| index < 3));
    assert_eq!(only_nonzero, vec![2, 2, 2, 2]);
}

#[test]
fn weighted_random_sampler_rejects_unrepresentable_storage_without_panicking() {
    assert_caught_invalid_configuration(
        catch_unwind(|| WeightedRandomSampler::new(vec![1.0], usize::MAX, true, 7)),
        "num_samples",
    );
}

#[test]
fn weighted_random_sampler_without_replacement_is_unique_and_fills_zero_weights() {
    let indices = WeightedRandomSampler::new(vec![1.0, 0.0, 0.0], 3, false, 7)
        .expect("zero weights may fill after positive weights")
        .collect::<Vec<_>>();

    assert_eq!(indices[0], 0);
    let mut sorted = indices;
    sorted.sort_unstable();
    assert_eq!(sorted, vec![0, 1, 2]);
}

#[test]
fn weighted_random_sampler_keeps_equal_subnormal_weights_random() {
    let weight = f64::from_bits(1);
    let mut selected = (0..16)
        .map(|seed| {
            WeightedRandomSampler::new(vec![weight, weight], 1, false, seed)
                .expect("positive finite subnormal weights must be valid")
                .next()
                .expect("one sample was requested")
        })
        .collect::<Vec<_>>();
    selected.sort_unstable();
    selected.dedup();

    assert_eq!(selected, vec![0, 1]);
}

#[test]
fn concrete_samplers_create_fresh_epoch_aware_iterators() {
    let mut sequential = SequentialSampler::new(4);
    assert_eq!(sequential.epoch(), 0);
    assert_eq!(Sampler::exact_len(&sequential), Some(4));
    assert_eq!(sequential.iter().collect::<Vec<_>>(), vec![0, 1, 2, 3]);
    assert_eq!(sequential.iter().collect::<Vec<_>>(), vec![0, 1, 2, 3]);
    sequential.set_epoch(5);
    assert_eq!(sequential.epoch(), 5);

    let mut random = RandomSampler::new(8, 42).expect("positive length must be valid");
    assert_eq!(random.epoch(), 0);
    let epoch_zero = random.iter().collect::<Vec<_>>();
    assert_eq!(epoch_zero, random.iter().collect::<Vec<_>>());
    random.set_epoch(1);
    assert_eq!(random.epoch(), 1);
    let epoch_one = random.iter().collect::<Vec<_>>();
    assert_ne!(epoch_zero, epoch_one);
    assert_eq!(epoch_one, random.iter().collect::<Vec<_>>());
}

#[test]
fn subset_and_weighted_samplers_create_fresh_epoch_iterators() -> rusttorch_core::Result<()> {
    let mut subset = SubsetRandomSampler::new((0..8).collect(), 42)?;
    let subset_epoch_zero = subset.iter().collect::<Vec<_>>();
    assert_eq!(subset_epoch_zero, subset.iter().collect::<Vec<_>>());
    subset.set_epoch(1);
    assert_eq!(subset.epoch(), 1);
    assert_ne!(subset_epoch_zero, subset.iter().collect::<Vec<_>>());

    let mut weighted = WeightedRandomSampler::new(vec![1.0, 2.0, 3.0, 4.0], 3, false, 42)?;
    let weighted_epoch_zero = weighted.iter().collect::<Vec<_>>();
    assert_eq!(weighted_epoch_zero, weighted.iter().collect::<Vec<_>>());
    weighted.set_epoch(1);
    assert_eq!(weighted.epoch(), 1);
    assert_ne!(weighted_epoch_zero, weighted.iter().collect::<Vec<_>>());
    Ok(())
}

#[test]
fn fn_sampler_factories_are_fresh_and_length_is_optional() {
    let sized_calls = Rc::new(RefCell::new(Vec::new()));
    let sized_observer = Rc::clone(&sized_calls);
    let mut sized = FnSampler::new(Some(3), move |epoch| {
        sized_observer.borrow_mut().push(epoch);
        vec![epoch as usize, 10, 20].into_iter()
    });

    assert_eq!(sized.epoch(), 0);
    assert_eq!(sized.exact_len(), Some(3));
    assert_eq!(sized.iter().collect::<Vec<_>>(), vec![0, 10, 20]);
    sized.set_epoch(2);
    assert_eq!(sized.epoch(), 2);
    assert_eq!(sized.iter().collect::<Vec<_>>(), vec![2, 10, 20]);
    assert_eq!(&*sized_calls.borrow(), &[0, 2]);

    let unsized_calls = Rc::new(RefCell::new(Vec::new()));
    let unsized_observer = Rc::clone(&unsized_calls);
    let mut unsized_sampler = FnSampler::new(None, move |epoch| {
        unsized_observer.borrow_mut().push(epoch);
        0..(epoch as usize + 2)
    });

    assert_eq!(unsized_sampler.epoch(), 0);
    assert_eq!(unsized_sampler.exact_len(), None);
    assert_eq!(unsized_sampler.iter().collect::<Vec<_>>(), vec![0, 1]);
    unsized_sampler.set_epoch(1);
    assert_eq!(unsized_sampler.epoch(), 1);
    assert_eq!(unsized_sampler.iter().collect::<Vec<_>>(), vec![0, 1, 2]);
    assert_eq!(&*unsized_calls.borrow(), &[0, 1]);
}

#[test]
fn fn_batch_source_factories_are_fresh_and_length_is_optional() {
    let sized_calls = Rc::new(RefCell::new(Vec::new()));
    let sized_observer = Rc::clone(&sized_calls);
    let mut sized = FnBatchSource::new(Some(2), move |epoch| {
        sized_observer.borrow_mut().push(epoch);
        vec![vec![epoch as usize], vec![8, 9]].into_iter()
    });

    assert_eq!(sized.epoch(), 0);
    assert_eq!(sized.exact_len(), Some(2));
    assert_eq!(sized.iter().collect::<Vec<_>>(), vec![vec![0], vec![8, 9]]);
    sized.set_epoch(3);
    assert_eq!(sized.epoch(), 3);
    assert_eq!(sized.iter().collect::<Vec<_>>(), vec![vec![3], vec![8, 9]]);
    assert_eq!(&*sized_calls.borrow(), &[0, 3]);

    let unsized_calls = Rc::new(RefCell::new(Vec::new()));
    let unsized_observer = Rc::clone(&unsized_calls);
    let mut unsized_source = FnBatchSource::new(None, move |epoch| {
        unsized_observer.borrow_mut().push(epoch);
        vec![vec![epoch as usize, 5]].into_iter()
    });

    assert_eq!(unsized_source.epoch(), 0);
    assert_eq!(unsized_source.exact_len(), None);
    assert_eq!(unsized_source.iter().collect::<Vec<_>>(), vec![vec![0, 5]]);
    unsized_source.set_epoch(4);
    assert_eq!(unsized_source.epoch(), 4);
    assert_eq!(unsized_source.iter().collect::<Vec<_>>(), vec![vec![4, 5]]);
    assert_eq!(&*unsized_calls.borrow(), &[0, 4]);
}
