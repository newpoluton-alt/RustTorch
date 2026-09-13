use rusttorch::{
    Device, Kind, Tensor,
    distributed::{DistributedDataParallel, GroupOptions, ProcessGroup},
    nn::{LinearConfig, VarStore},
    optim::Adam,
};
use std::{
    env,
    net::{SocketAddr, TcpListener},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rank: usize = env::var("RUSTTORCH_RANK").unwrap_or("0".into()).parse()?;
    let world: usize = env::var("RUSTTORCH_WORLD").unwrap_or("1".into()).parse()?;
    let address: SocketAddr = env::var("RUSTTORCH_ADDRESS")
        .unwrap_or("127.0.0.1:29500".into())
        .parse()?;
    let options = GroupOptions::new(env::var("RUSTTORCH_RUN").unwrap_or("local-fit".into()));
    let mut group = if rank == 0 {
        ProcessGroup::accept(TcpListener::bind(address)?, world, options)?
    } else {
        ProcessGroup::connect(address, rank, world, options)?
    };

    let mut store = VarStore::new(Device::Cpu);
    store.set_kind(Kind::Double);
    let model = LinearConfig::new(2, 1).build(&store.root())?;
    let replica = DistributedDataParallel::new(&mut group, &store)?;
    let mut optimizer = Adam::builder().learning_rate(0.03).build(&store)?;
    // Each rank supplies a different local batch with the same number of samples.
    let x = Tensor::from_slice(&[rank as f64, 1., rank as f64 + 1., 1.]).reshape([2, 2]);
    let y = x.narrow(1, 0, 1) * 2. + 0.5;
    for _ in 0..50 {
        replica.sync_buffers(&mut group)?;
        let prediction = model.forward(&x)?;
        let loss = prediction.f_sub(&y)?.f_square()?.f_mean(Kind::Double)?;
        replica.backward_step(&mut group, &mut optimizer, &loss)?;
    }
    group.barrier()?;
    if rank == 0 {
        println!("predictions: {:?}", model.forward(&x)?);
    }
    Ok(())
}
