use std::{
    collections::VecDeque,
    mem::size_of,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::worker::{WorkerHooks, WorkerRunContext, WorkerTask};
use crate::{CancellationToken, WorkerCheckpoint};
use crossbeam_channel::{Receiver, Sender, bounded, select};
use rusttorch_core::{Result, RustTorchError};

pub(crate) fn error(reason: &str) -> RustTorchError {
    RustTorchError::InvalidConfiguration {
        field: "worker_checkpoint",
        reason: reason.to_owned(),
    }
}

enum Command {
    Apply,
    Rollback { generation: u64, boundary: u64 },
}

pub(crate) struct Lane<S> {
    id: usize,
    pending: Option<S>,
    snapshots: VecDeque<(u64, u64, S)>,
    capacity: usize,
    committed: Arc<AtomicU64>,
    commands: Receiver<Command>,
    initialized: Sender<(usize, Result<()>)>,
    snapshots_out: Sender<(usize, u64, Result<S>)>,
}

pub(crate) struct Barrier<S> {
    commands: Vec<Sender<Command>>,
    pub(crate) initialized: Receiver<(usize, Result<()>)>,
    snapshots: Receiver<(usize, u64, Result<S>)>,
    pub(crate) committed: Arc<AtomicU64>,
}

impl<S: Clone + Send + 'static> Barrier<S> {
    pub(crate) fn new(
        workers: usize,
        prefetch: usize,
        boundary: u64,
        states: Option<Vec<S>>,
    ) -> Result<(Self, Vec<Option<Lane<S>>>)> {
        let capacity = prefetch
            .checked_add(1)
            .ok_or_else(|| error("snapshot capacity overflow"))?;
        let (initialized_tx, initialized) = bounded(workers);
        let (snapshots_tx, snapshots) = bounded(workers);
        let committed = Arc::new(AtomicU64::new(boundary));
        let mut commands = Vec::with_capacity(workers);
        let mut lanes = Vec::with_capacity(workers);
        for id in 0..workers {
            let (tx, rx) = bounded(1);
            commands.push(tx);
            lanes.push(Some(Lane {
                id,
                pending: states.as_ref().map(|states| states[id].clone()),
                snapshots: VecDeque::with_capacity(capacity),
                capacity,
                committed: Arc::clone(&committed),
                commands: rx,
                initialized: initialized_tx.clone(),
                snapshots_out: snapshots_tx.clone(),
            }));
        }
        Ok((
            Self {
                commands,
                initialized,
                snapshots,
                committed,
            },
            lanes,
        ))
    }

    pub(crate) fn apply(&self) -> Result<()> {
        for command in &self.commands {
            command
                .send(Command::Apply)
                .map_err(|_| error("worker closed before apply"))?;
        }
        Ok(())
    }

    pub(crate) fn request(&self, generation: u64, boundary: u64) -> Result<()> {
        for command in &self.commands {
            command
                .send(Command::Rollback {
                    generation,
                    boundary,
                })
                .map_err(|_| error("worker closed before rollback"))?;
        }
        Ok(())
    }

    pub(crate) fn collect(&self, generation: u64) -> Result<Vec<S>> {
        let mut states: Vec<Option<S>> = (0..self.commands.len()).map(|_| None).collect();
        // Quiescence acknowledged by every worker: all replies must already exist.
        for _ in 0..states.len() {
            let (id, received_generation, state) = self
                .snapshots
                .try_recv()
                .map_err(|_| error("missing rollback acknowledgment"))?;
            if id >= states.len() || states[id].is_some() || received_generation != generation {
                return Err(error(
                    "stale, duplicate, or wrong-lane rollback acknowledgment",
                ));
            }
            states[id] = Some(state?);
        }
        if self.snapshots.try_recv().is_ok() {
            return Err(error("duplicate rollback acknowledgment"));
        }
        Ok(states.into_iter().map(Option::unwrap).collect())
    }
}

impl<T> WorkerHooks<T> for Lane<T::State>
where
    T: WorkerCheckpoint,
    T::State: Send + 'static,
{
    const CANCEL_ON_ERROR: bool = false;
    fn initialized(&mut self, transform: &mut T, shutdown: &CancellationToken) -> bool {
        let valid = self
            .pending
            .as_ref()
            .map_or(Ok(()), |state| transform.validate_snapshot(state));
        if self.initialized.send((self.id, valid)).is_err() {
            return false;
        }
        select! {
            recv(self.commands) -> command => match command {
                Ok(Command::Apply) => {
                    if let Some(state) = self.pending.take() { transform.restore_validated(&state); }
                    true
                },
                _ => false,
            },
            recv(shutdown.signal()) -> _ => false,
        }
    }

    fn before_task(&mut self, transform: &T, task: &WorkerTask) {
        let boundary = self.committed.load(Ordering::Acquire);
        while self
            .snapshots
            .front()
            .is_some_and(|(_, sequence, _)| *sequence < boundary)
        {
            self.snapshots.pop_front();
        }
        assert!(
            self.snapshots.len() < self.capacity,
            "checkpoint snapshot ring exhausted"
        );
        self.snapshots
            .push_back((task.generation, task.batch_sequence, transform.snapshot()));
    }

    fn quiesced(
        &mut self,
        transform: &mut T,
        run: &WorkerRunContext,
        shutdown: &CancellationToken,
    ) {
        select! {
            recv(self.commands) -> command => {
                let Ok(Command::Rollback { generation, boundary }) = command else { return; };
                let state = if generation != run.generation {
                    Err(error("stale rollback request"))
                } else {
                    let state = self.snapshots.iter().find(|(g, sequence, _)| *g == generation && *sequence >= boundary)
                        .map(|(_, _, state)| state.clone()).unwrap_or_else(|| transform.snapshot());
                    transform.validate_snapshot(&state).map(|()| {
                        transform.restore_validated(&state);
                        state
                    })
                };
                self.snapshots.clear();
                let _ = self.snapshots_out.send((self.id, generation, state));
            },
            recv(shutdown.signal()) -> _ => {},
        }
    }
}

pub(crate) fn extra_capacity<S, E>(workers: usize, prefetch: usize) -> Result<usize> {
    use crate::worker::channel_slot_size;
    let ring = prefetch
        .checked_add(1)
        .and_then(|n| n.checked_mul(size_of::<(u64, u64, S)>()))
        .ok_or_else(|| error("snapshot ring allocation overflow"))?;
    // Typed rings, control/reply slots, lane handles and committed occurrence ledger.
    let per_lane = ring
        .checked_add(size_of::<Lane<S>>())
        .and_then(|n| n.checked_add(size_of::<Sender<Command>>()))
        .and_then(|n| n.checked_add(channel_slot_size::<Command>()))
        .and_then(|n| n.checked_add(channel_slot_size::<(usize, Result<()>)>()))
        .and_then(|n| n.checked_add(channel_slot_size::<(usize, u64, Result<S>)>()))
        .and_then(|n| n.checked_add(prefetch.checked_mul(size_of::<(u64, u64)>() + 64)?))
        .and_then(|n| n.checked_add(prefetch.checked_mul(size_of::<u64>() + 64)?))
        .and_then(|n| n.checked_add(prefetch.checked_mul(size_of::<(u64, E)>() + 64)?))
        .and_then(|n| n.checked_add(size_of::<S>().checked_mul(8)?))
        .and_then(|n| n.checked_add(6144))
        .ok_or_else(|| error("checkpoint bookkeeping allocation overflow"))?;
    workers
        .checked_mul(per_lane)
        .and_then(|n| n.checked_add(size_of::<Barrier<S>>() + 64))
        .ok_or_else(|| error("checkpoint allocation overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rollback_acknowledgments_reject_missing_stale_and_duplicate_lanes() {
        let (barrier, lanes) = Barrier::<()>::new(2, 1, 0, None).unwrap();
        assert!(barrier.collect(4).is_err());
        let sender = &lanes[0].as_ref().unwrap().snapshots_out;
        sender.send((0, 3, Ok(()))).unwrap();
        assert!(barrier.collect(4).is_err());
        sender.send((0, 4, Ok(()))).unwrap();
        sender.send((0, 4, Ok(()))).unwrap();
        assert!(barrier.collect(4).is_err());
        sender.send((2, 4, Ok(()))).unwrap();
        assert!(barrier.collect(4).is_err());
    }

    #[test]
    fn typed_snapshot_and_error_sizes_participate_in_capacity_limit() {
        let small = extra_capacity::<(), ()>(8, 8).unwrap();
        let large_state = extra_capacity::<[u8; 1024 * 1024], ()>(8, 8).unwrap();
        let large_error = extra_capacity::<(), [u8; 1024 * 1024]>(8, 8).unwrap();
        assert!(small < 64 * 1024 * 1024);
        assert!(large_state > 64 * 1024 * 1024);
        assert!(large_error > 64 * 1024 * 1024);
        assert!(extra_capacity::<(), ()>(usize::MAX, usize::MAX).is_err());
    }
}
