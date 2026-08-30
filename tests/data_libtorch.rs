use rusttorch::{Device, Kind, Tensor, data::RandomSampler, manual_seed};

#[test]
fn random_sampler_does_not_change_libtorch_global_rng() {
    manual_seed(1_234);
    let expected = Tensor::randn([8], (Kind::Float, Device::Cpu));

    manual_seed(1_234);
    let _indices = RandomSampler::new(64, 42)
        .expect("positive length must be valid")
        .collect::<Vec<_>>();
    let actual = Tensor::randn([8], (Kind::Float, Device::Cpu));

    assert_eq!(
        Vec::<f32>::try_from(&actual).expect("random tensor must convert"),
        Vec::<f32>::try_from(&expected).expect("random tensor must convert")
    );
}
