# Train on several CPU processes

Use a `ProcessGroup` when several Rust programs need to exchange tensors. Choose
`DistributedDataParallel` for an existing model built with `VarStore` and normal
layers. Choose `ShardedTrainer` for a functional model whose persistent parameter
and optimizer storage should be divided among workers.

These APIs run without Python. They currently communicate dense CPU `Float`,
`Double` and `Int64` tensors over TCP. Training parameters use `Float` or `Double`.
Accelerator transports and mixed precision are not supported by this backend.

| Task | API | Result |
| --- | --- | --- |
| Combine per-worker metrics | `ProcessGroup::all_reduce` | A detached sum, mean, minimum, maximum or product on every worker |
| Initialize identical replicas | `DistributedDataParallel::new` | Rank-zero weights and buffers copied to every model |
| Train an ordinary model | `sync_buffers`, `backward_step` | Local forward computation with averaged gradients |
| Accumulate microbatches | `sync_gradients` | Synchronize once before the optimizer update |
| Divide persistent training state | `ShardedTrainer::train_step` | Local parameter, gradient and optimizer-moment shards |
| Resume with the same workers | `checkpoint`, `restore` | Exact named weights, moments, settings and step counters |
| Resume with a different worker count | `consolidate_checkpoint`, `restore_full` | Parameter and optimizer shards repartitioned together |

## Launch a replicated training job

Save this program as an example or binary in a project depending on RustTorch.
With no environment variables it runs with one worker. To use two workers, start
it twice with the same address, world size and run identifier, and distinct ranks:

```text
RUSTTORCH_RANK=0 RUSTTORCH_WORLD=2 RUSTTORCH_RUN=fit-42 cargo run --example distributed_training
RUSTTORCH_RANK=1 RUSTTORCH_WORLD=2 RUSTTORCH_RUN=fit-42 cargo run --example distributed_training
```

Set the environment variables using your shell's syntax; on PowerShell use
`$env:RUSTTORCH_RANK = "0"`, for example. `RUSTTORCH_ADDRESS` defaults to
`127.0.0.1:29500`. Rank zero listens at that address; the other workers connect to
it. Use an address reachable by every worker when running on several machines.

```no_run
use std::{env, net::{SocketAddr, TcpListener}};
use rusttorch::{Device, Kind, Tensor, nn::{LinearConfig, VarStore}, optim::Adam,
    distributed::{DistributedDataParallel, GroupOptions, ProcessGroup}};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rank: usize = env::var("RUSTTORCH_RANK").unwrap_or("0".into()).parse()?;
    let world: usize = env::var("RUSTTORCH_WORLD").unwrap_or("1".into()).parse()?;
    let address: SocketAddr = env::var("RUSTTORCH_ADDRESS")
        .unwrap_or("127.0.0.1:29500".into()).parse()?;
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
    if rank == 0 { println!("predictions: {:?}", model.forward(&x)?); }
    Ok(())
}
```

Register every model parameter and persistent buffer before constructing the
replica. Construction verifies names, dimensions and dtypes, then broadcasts
rank-zero values. `sync_buffers` repeats the buffer broadcast before a forward
pass. This is useful for running statistics; it does not combine local batch
normalization statistics into synchronized batch normalization.

For map datasets, use `DistributedSampler` with this group's `world_size()` and
`rank()`. Give all workers the same shuffle seed and epoch. The sampler pads or
truncates the index sequence according to its configuration so workers can make
the same number of updates. See the [DataLoader guide](../crates/rusttorch-data/README.md)
for dataset construction and epoch handling.

## Accumulate several microbatches

Clear gradients once, scale each microbatch's loss, and synchronize after all
contributing backward passes. This avoids communicating after every microbatch.
All ranks must synchronize at the same update boundary.

```no_run
use rusttorch::{Tensor, Result, optim::Optimizer,
    distributed::{DistributedDataParallel, ProcessGroup}};

fn accumulated_step(group: &mut ProcessGroup, replica: &DistributedDataParallel,
                    optimizer: &mut Optimizer, losses: &[Tensor]) -> Result<()> {
    optimizer.try_zero_grad()?;
    for loss in losses {
        loss.f_div_scalar(losses.len() as f64)?.f_backward()?;
    }
    replica.sync_gradients(group)?;
    optimizer.try_step()?;
    Ok(())
}
```

Averaging gradients from equal-size local mean losses gives the global-batch
mean gradient. For unequal sample counts, compute local **sum** losses and scale
each by `world_size / global_sample_count` before backward. Every rank must still
participate in the synchronization. If a parameter is unused everywhere it is
skipped; a parameter used by only some ranks is rejected instead of silently
using inconsistent optimizer updates.

## Train with parameter and optimizer shards

A `ShardedTrainer` accepts named tensors and an optimizer builder. Its closure
receives full named parameters for one forward/backward computation. The trainer
then averages and partitions gradients and updates local shards. The closure
must use every supplied parameter and return one tracked scalar loss.

```
use std::net::TcpListener;
use rusttorch::{Kind, Tensor, optim::Adam,
    distributed::{GroupOptions, ProcessGroup, ShardedTrainer}};

// Replace the one-worker group with the launch code above to use several ranks.
let listener = TcpListener::bind("127.0.0.1:0").unwrap();
let mut group = ProcessGroup::accept(listener, 1, GroupOptions::new("functional-fit"))?;
let mut trainer = ShardedTrainer::new(&mut group, vec![
    ("weight".into(), Tensor::from_slice(&[0_f64, 0.])),
    ("bias".into(), Tensor::from(0_f64)),
], |store| Adam::builder().learning_rate(0.05).build(store))?;
let features = Tensor::from_slice(&[1_f64, 0., 0., 1., 1., 1.]).reshape([3, 2]);
let targets = Tensor::from_slice(&[2_f64, -1., 1.]);
for _ in 0..20 {
    trainer.train_step(&mut group, |parameters| {
        let predictions = features.f_matmul(&parameters["weight"])?
            .f_add(&parameters["bias"])?;
        Ok(predictions.f_sub(&targets)?.f_square()?.f_mean(Kind::Double)?)
    })?;
}
let weights = trainer.full_parameters(&mut group)?;
assert_eq!(weights["weight"].size(), [2]);
# Ok::<(), rusttorch::RustTorchError>(())
```

Between steps, each rank stores a padded flat slice of each parameter and the
matching moment slices. `local_parameter_elements()` reports this persistent
parameter count, including padding. During a step, full parameters and local
activations are temporarily present. This implementation gathers one parameter
group for the entire forward/backward computation. It does not install module
hooks or overlap per-layer gathering with computation. Avoid retaining the
closure's full tensors or external copies of the initial weights if you need the
persistent storage reduction.

Any of RustTorch's seven optimizer families can be supplied to the builder. The
current functional trainer registers every parameter in group zero. Use
`optimizer_mut()` to update a learning rate or apply a scheduler consistently on
all ranks. Training verifies that optimizer settings and step counters agree
before the next update.

## Save, resume and change the worker count

A shard checkpoint includes local weights and optimizer moments, named parameter
metadata, the optimizer family and settings, and per-parameter step counters.
`checkpoint` is collective: every worker validates the complete checkpoint set
before receiving its own record. Records include a session/capture identifier,
so a shard from another job or capture is rejected even at the same optimizer
step. Save all rank records from the same call. Use a
new checkpoint directory and publish a completion manifest only after every
worker reports its write succeeded. A partially written set is not a checkpoint.

```no_run
use rusttorch::{distributed::{ProcessGroup, ShardedTrainer, ShardedCheckpoint}, optim::Adam};

fn save_and_restore(group: &mut ProcessGroup, trainer: &mut ShardedTrainer)
    -> Result<ShardedTrainer, Box<dyn std::error::Error>> {
    let checkpoint = trainer.checkpoint(group)?;
    let encoded = serde_json::to_vec(&checkpoint)?;
    // Store encoded bytes under this rank's file in a new checkpoint directory.
    // The round trip below also demonstrates decoding a stored record.
    let decoded: ShardedCheckpoint = serde_json::from_slice(&encoded)?;
    let restored = ShardedTrainer::restore(group, &decoded,
        |store| Adam::builder().build(store))?;
    Ok(restored)
}
```

Choose the same optimizer family when restoring; saved settings replace the
builder's initial rates. A new process-group session is expected after a restart.
A shard record's original rank and world size must match the receiving worker.

To resume on a different number of workers, obtain a consolidated checkpoint
before shutting down the original job. Save one full record, start the new group,
load that same record on every worker, then call `restore_full`:

```no_run
use rusttorch::{distributed::{ProcessGroup, ShardedTrainer, ConsolidatedCheckpoint}, optim::Adam};

fn restore_on_new_workers(group: &mut ProcessGroup, bytes: &[u8])
    -> Result<ShardedTrainer, Box<dyn std::error::Error>> {
    let checkpoint: ConsolidatedCheckpoint = serde_json::from_slice(bytes)?;
    Ok(ShardedTrainer::restore_full(group, &checkpoint,
        |store| Adam::builder().build(store))?)
}
// In the original job:
// let full = trainer.consolidate_checkpoint(&mut group)?;
// let bytes = serde_json::to_vec(&full)?;
```

Parameters and moment tensors are repartitioned together; retaining only model
weights would lose the optimizer's training history. Consolidation allocates
full weights and optimizer state on every participant, so checkpoint exchange
also needs to fit the configured message/working-set limit. For offline
consolidation, `ShardedCheckpoint::consolidate(&records)` accepts rank records in
any order and rejects missing, duplicate or inconsistent records.

These records contain no executable code or pickle. They do not include dataset
cursors, DataLoader checkpoints, application buffers, scheduler state, random
number generator state or an incomplete accumulation window. Save those with the
training checkpoint when exact end-to-end input replay is needed. Changing world
size also changes floating-point reduction order; numerical agreement is tested,
but bitwise identical training across different worker counts is not promised.

## Exchange tensors and handle failures

Collectives return independent tensors detached from autograd. `all_gather` and
`gather` return tensors in source-rank order. `scatter` takes one tensor per rank
on its root; other workers pass an empty slice. `reduce_scatter` takes one input
per destination on every worker and returns the reduction for the calling rank.
`all_to_all` sends entry `r` to rank `r`. Inputs to these tensor collectives must
share shapes and dtypes.

`send(tensor, destination, tag)` and `recv(source, tag)` involve only their two
workers, and can connect two nonzero ranks. Both calls block; the matching receive
must run while its sender is waiting. Opposing sends without receives time out.
Do not mix conflicting collective and point-to-point order across ranks.

Every communication operation has a finite deadline. Native tensor kernels and
serialization run synchronously and cannot be preempted by a socket deadline.
Mismatched operation order, roots, tensor
specifications, rank sessions, or point-to-point tags return an error. A transport
or collective error permanently fails the group and closes local connections;
other ranks fail on disconnect or their own deadlines. `abort()` lets a worker
end a failed job explicitly. Recreate the group and restore the last completed
checkpoint. A failure during final optimizer updates can leave different ranks
at different steps; do not attempt to continue that session.

Communication uses a full mesh of TCP connections for peer-to-peer calls and a
rank-zero coordinator for collectives. The default 64 MiB limit applies to each
encoded message and estimated collective response working set; split large
inputs or deliberately raise `GroupOptions::max_message_bytes`. TCP is
unencrypted and has no peer authentication. Run mutually trusted workers on a
controlled network. Session names prevent accidental run mixing; they are not
credentials. This backend makes no multi-node throughput claim.
