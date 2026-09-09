use std::panic::catch_unwind;

use rusttorch_core::RustTorchError;
use rusttorch_data::{DistributedSampler, Sampler};

#[test]
fn distributed_sampler_cyclically_pads_equal_rank_lengths() -> rusttorch_core::Result<()> {
    let ranks = (0..3)
        .map(|rank| {
            DistributedSampler::new(10, 3, rank, false, 42, false).map(Iterator::collect::<Vec<_>>)
        })
        .collect::<rusttorch_core::Result<Vec<_>>>()?;

    assert_eq!(
        ranks,
        vec![vec![0, 3, 6, 9], vec![1, 4, 7, 0], vec![2, 5, 8, 1]]
    );
    assert!(ranks.iter().all(|indices| indices.len() == 4));
    Ok(())
}

#[test]
fn distributed_sampler_truncates_to_disjoint_rank_strides() -> rusttorch_core::Result<()> {
    let ranks = (0..3)
        .map(|rank| {
            DistributedSampler::new(10, 3, rank, false, 42, true).map(Iterator::collect::<Vec<_>>)
        })
        .collect::<rusttorch_core::Result<Vec<_>>>()?;

    assert_eq!(ranks, vec![vec![0, 3, 6], vec![1, 4, 7], vec![2, 5, 8]]);
    let mut all = ranks.into_iter().flatten().collect::<Vec<_>>();
    all.sort_unstable();
    assert_eq!(all, (0..9).collect::<Vec<_>>());
    Ok(())
}

#[test]
fn distributed_sampler_validates_replica_and_rank_configuration() {
    assert!(matches!(
        DistributedSampler::new(10, 0, 0, false, 42, false),
        Err(RustTorchError::InvalidConfiguration {
            field: "replicas",
            ..
        })
    ));
    assert!(matches!(
        DistributedSampler::new(10, 3, 3, false, 42, false),
        Err(RustTorchError::InvalidConfiguration { field: "rank", .. })
    ));
}

#[test]
fn distributed_sampler_rejects_unrepresentable_padding_and_storage_without_panicking() {
    for result in [
        catch_unwind(|| DistributedSampler::new(usize::MAX, 2, 0, false, 42, false)),
        catch_unwind(|| DistributedSampler::new(1, usize::MAX, 0, false, 42, false)),
    ] {
        assert!(result.is_ok(), "constructor must not panic");
        assert!(matches!(
            result.expect("constructor did not panic"),
            Err(RustTorchError::InvalidConfiguration {
                field: "length",
                ..
            })
        ));
    }

    let unshuffled =
        catch_unwind(|| DistributedSampler::new(usize::MAX - 1, usize::MAX, 0, false, 0, true));
    let sampler = unshuffled
        .expect("drop-last construction must not panic")
        .expect("the discarded unshuffled range needs no allocation");
    assert!(sampler.is_empty());

    let shuffled =
        catch_unwind(|| DistributedSampler::new(usize::MAX - 1, usize::MAX, 0, true, 0, true));
    assert!(matches!(
        shuffled.expect("shuffled drop-last construction must not panic"),
        Err(RustTorchError::InvalidConfiguration {
            field: "length",
            ..
        })
    ));
}

#[test]
fn distributed_sampler_is_deterministic_per_epoch_and_creates_fresh_iterations()
-> rusttorch_core::Result<()> {
    let mut sampler = DistributedSampler::new(20, 2, 1, true, 91, false)?;
    let peer = DistributedSampler::new(20, 2, 1, true, 91, false)?;
    let epoch_zero = Sampler::iter(&sampler).collect::<Vec<_>>();

    assert_eq!(sampler.epoch(), 0);
    assert_eq!(sampler.len(), 10);
    assert_eq!(Sampler::exact_len(&sampler), Some(10));
    assert_eq!(epoch_zero, Sampler::iter(&sampler).collect::<Vec<_>>());
    assert_eq!(epoch_zero, Sampler::iter(&peer).collect::<Vec<_>>());

    sampler.set_epoch(1);
    let epoch_one = Sampler::iter(&sampler).collect::<Vec<_>>();
    assert_eq!(sampler.epoch(), 1);
    assert_ne!(epoch_zero, epoch_one);
    assert_eq!(epoch_one, Sampler::iter(&sampler).collect::<Vec<_>>());
    Ok(())
}

#[test]
fn distributed_sampler_tracks_direct_iteration_position_and_resets_on_epoch()
-> rusttorch_core::Result<()> {
    let mut sampler = DistributedSampler::new(7, 2, 0, false, 12, false)?;

    assert_eq!(sampler.position(), 0);
    assert_eq!(sampler.next(), Some(0));
    assert_eq!(sampler.position(), 1);
    assert_eq!(sampler.by_ref().count(), 3);
    assert_eq!(sampler.position(), sampler.len());

    sampler.set_epoch(4);
    assert_eq!(sampler.position(), 0);
    assert_eq!(sampler.epoch(), 4);
    assert_eq!(sampler.by_ref().count(), sampler.len());
    Ok(())
}

#[test]
fn distributed_sampler_accepts_empty_datasets_for_every_valid_rank() -> rusttorch_core::Result<()> {
    for drop_last in [false, true] {
        for rank in 0..3 {
            let sampler = DistributedSampler::new(0, 3, rank, true, 42, drop_last)?;
            assert_eq!(sampler.len(), 0);
            assert_eq!(sampler.position(), 0);
            assert_eq!(Sampler::iter(&sampler).count(), 0);
        }
    }
    Ok(())
}

#[test]
fn distributed_sampler_does_not_change_libtorch_global_rng() -> rusttorch_core::Result<()> {
    tch::manual_seed(1_234);
    let expected = tch::Tensor::randn([8], (tch::Kind::Float, tch::Device::Cpu));

    tch::manual_seed(1_234);
    let _indices = DistributedSampler::new(64, 4, 2, true, 42, false)?.collect::<Vec<_>>();
    let actual = tch::Tensor::randn([8], (tch::Kind::Float, tch::Device::Cpu));

    assert_eq!(
        Vec::<f32>::try_from(&actual).expect("random tensor must convert"),
        Vec::<f32>::try_from(&expected).expect("random tensor must convert")
    );
    Ok(())
}
