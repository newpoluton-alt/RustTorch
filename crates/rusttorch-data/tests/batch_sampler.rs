use std::{cell::RefCell, panic::catch_unwind, rc::Rc};

use rusttorch_core::RustTorchError;
use rusttorch_data::{BatchSampler, BatchSource, FnBatchSource, FnSampler};

#[test]
fn batch_sampler_keeps_or_drops_the_exact_tail() -> rusttorch_core::Result<()> {
    let kept = BatchSampler::new(0..5, 2, false)?;
    assert_eq!(kept.exact_len(), Some(3));
    assert_eq!(
        kept.collect::<Vec<_>>(),
        vec![vec![0, 1], vec![2, 3], vec![4]]
    );

    let dropped = BatchSampler::new(0..5, 2, true)?;
    assert_eq!(dropped.exact_len(), Some(2));
    assert_eq!(dropped.collect::<Vec<_>>(), vec![vec![0, 1], vec![2, 3]]);
    Ok(())
}

#[test]
fn batch_sampler_accepts_a_one_shot_custom_iterator() -> rusttorch_core::Result<()> {
    let indices = vec![8, 3, 5, 1].into_iter().filter(|index| index % 2 == 1);

    assert_eq!(
        BatchSampler::new(indices, 2, false)?.collect::<Vec<_>>(),
        vec![vec![3, 5], vec![1]]
    );
    Ok(())
}

#[test]
fn batch_sampler_validates_public_storage_dimensions_without_panicking() {
    assert!(matches!(
        BatchSampler::new(0..1, 0, false),
        Err(RustTorchError::InvalidConfiguration {
            field: "batch_size",
            ..
        })
    ));

    let result = catch_unwind(|| BatchSampler::new(0..1, usize::MAX, false));
    assert!(result.is_ok(), "constructor must not panic");
    assert!(matches!(
        result.expect("constructor did not panic"),
        Err(RustTorchError::InvalidConfiguration {
            field: "batch_size",
            ..
        })
    ));
}

#[test]
fn batch_sampler_recreates_sized_sampler_iterations_and_forwards_epoch()
-> rusttorch_core::Result<()> {
    let calls = Rc::new(RefCell::new(Vec::new()));
    let observer = Rc::clone(&calls);
    let sampler = FnSampler::new(Some(5), move |epoch| {
        observer.borrow_mut().push(epoch);
        epoch as usize..epoch as usize + 5
    });
    let mut batches = BatchSampler::new(sampler, 2, false)?;

    assert_eq!(BatchSource::exact_len(&batches), Some(3));
    assert_eq!(BatchSource::epoch(&batches), 0);
    assert_eq!(
        BatchSource::iter(&batches).collect::<Vec<_>>(),
        vec![vec![0, 1], vec![2, 3], vec![4]]
    );
    assert_eq!(BatchSource::iter(&batches).count(), 3);

    BatchSource::set_epoch(&mut batches, 7);
    assert_eq!(BatchSource::epoch(&batches), 7);
    assert_eq!(
        BatchSource::iter(&batches).collect::<Vec<_>>(),
        vec![vec![7, 8], vec![9, 10], vec![11]]
    );
    assert_eq!(&*calls.borrow(), &[0, 0, 7]);
    Ok(())
}

#[test]
fn batch_sampler_preserves_an_unsized_sampler_source() -> rusttorch_core::Result<()> {
    let sampler = FnSampler::new(None, |epoch| 0..epoch as usize + 3);
    let mut batches = BatchSampler::new(sampler, 2, true)?;

    assert_eq!(BatchSource::exact_len(&batches), None);
    assert_eq!(
        BatchSource::iter(&batches).collect::<Vec<_>>(),
        vec![vec![0, 1]]
    );
    BatchSource::set_epoch(&mut batches, 2);
    assert_eq!(
        BatchSource::iter(&batches).collect::<Vec<_>>(),
        vec![vec![0, 1], vec![2, 3]]
    );
    Ok(())
}

#[test]
fn fn_batch_source_recreates_nonuniform_batches_for_each_epoch() {
    let sized_calls = Rc::new(RefCell::new(Vec::new()));
    let sized_observer = Rc::clone(&sized_calls);
    let mut sized = FnBatchSource::new(Some(2), move |epoch| {
        sized_observer.borrow_mut().push(epoch);
        vec![vec![epoch as usize], vec![4, 7, 9]].into_iter()
    });

    assert_eq!(sized.exact_len(), Some(2));
    assert_eq!(
        sized.iter().collect::<Vec<_>>(),
        vec![vec![0], vec![4, 7, 9]]
    );
    sized.set_epoch(3);
    assert_eq!(
        sized.iter().collect::<Vec<_>>(),
        vec![vec![3], vec![4, 7, 9]]
    );
    assert_eq!(&*sized_calls.borrow(), &[0, 3]);

    let mut unsized_source = FnBatchSource::new(None, |epoch| {
        vec![vec![1, 2], vec![epoch as usize], vec![8, 9, 10]].into_iter()
    });
    assert_eq!(unsized_source.exact_len(), None);
    assert_eq!(unsized_source.iter().count(), 3);
    unsized_source.set_epoch(5);
    assert_eq!(
        unsized_source.iter().collect::<Vec<_>>(),
        vec![vec![1, 2], vec![5], vec![8, 9, 10]]
    );
}
