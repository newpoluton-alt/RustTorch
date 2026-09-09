use std::{
    collections::BTreeMap,
    mem::size_of,
    sync::{Arc, Condvar, Mutex, MutexGuard},
};

use rusttorch_core::Tensor;

use crate::Bytes;

/// Builder state for item-bounded loading without byte accounting.
#[derive(Clone, Copy, Debug, Default)]
pub struct MemoryDisabled;

/// Builder state for post-transform byte-bounded prefetch.
#[derive(Clone, Copy, Debug, Default)]
pub struct MemoryEnabled;

/// Conservative logical payload bytes retained by one prefetched value.
///
/// The estimate describes the final post-transform payload held by loader
/// result queues. It is not a process-RSS measurement: allocator metadata and
/// tensor backing-storage aliasing are intentionally outside this contract.
/// All built-in arithmetic saturates at [`usize::MAX`].
pub trait MemoryFootprint {
    /// Returns the conservative logical payload size in bytes.
    fn resident_bytes(&self) -> usize;
}

impl MemoryFootprint for Tensor {
    fn resident_bytes(&self) -> usize {
        let Some(bytes) = self.numel().checked_mul(self.kind().elt_size_in_bytes()) else {
            return usize::MAX;
        };
        bytes
    }
}

macro_rules! impl_scalar_footprint {
    ($($type:ty),+ $(,)?) => {
        $(
            impl MemoryFootprint for $type {
                fn resident_bytes(&self) -> usize {
                    size_of::<Self>()
                }
            }
        )+
    };
}

impl_scalar_footprint!(u8, i8, i16, i32, i64, f32, f64, bool);

impl MemoryFootprint for String {
    fn resident_bytes(&self) -> usize {
        self.capacity()
    }
}

impl MemoryFootprint for Bytes {
    fn resident_bytes(&self) -> usize {
        self.0.capacity()
    }
}

impl<T> MemoryFootprint for Vec<T>
where
    T: MemoryFootprint,
{
    fn resident_bytes(&self) -> usize {
        saturating_sum(self.iter().map(MemoryFootprint::resident_bytes))
    }
}

impl<T> MemoryFootprint for Option<T>
where
    T: MemoryFootprint,
{
    fn resident_bytes(&self) -> usize {
        self.as_ref().map_or(0, MemoryFootprint::resident_bytes)
    }
}

impl<K, V> MemoryFootprint for BTreeMap<K, V>
where
    K: MemoryFootprint,
    V: MemoryFootprint,
{
    fn resident_bytes(&self) -> usize {
        saturating_sum(
            self.iter()
                .flat_map(|(key, value)| [key.resident_bytes(), value.resident_bytes()]),
        )
    }
}

macro_rules! impl_tuple_footprint {
    ($(($type:ident, $index:tt)),+ $(,)?) => {
        impl<$($type),+> MemoryFootprint for ($($type,)+)
        where
            $($type: MemoryFootprint),+
        {
            fn resident_bytes(&self) -> usize {
                saturating_sum([$((self.$index).resident_bytes()),+])
            }
        }
    };
}

impl_tuple_footprint!((A, 0), (B, 1));
impl_tuple_footprint!((A, 0), (B, 1), (C, 2));
impl_tuple_footprint!((A, 0), (B, 1), (C, 2), (D, 3));
impl_tuple_footprint!((A, 0), (B, 1), (C, 2), (D, 3), (E, 4));
impl_tuple_footprint!((A, 0), (B, 1), (C, 2), (D, 3), (E, 4), (F, 5));
impl_tuple_footprint!((A, 0), (B, 1), (C, 2), (D, 3), (E, 4), (F, 5), (G, 6));
impl_tuple_footprint!(
    (A, 0),
    (B, 1),
    (C, 2),
    (D, 3),
    (E, 4),
    (F, 5),
    (G, 6),
    (H, 7)
);

fn saturating_sum(values: impl IntoIterator<Item = usize>) -> usize {
    let mut total = 0usize;
    for value in values {
        let Some(next) = total.checked_add(value) else {
            return usize::MAX;
        };
        total = next;
    }
    total
}

#[derive(Debug)]
pub(crate) enum BudgetError {
    Oversize { limit: usize, actual: usize },
    Cancelled,
    SequenceAlreadyAdmitted,
}

#[derive(Default)]
struct BudgetState {
    used: usize,
    next_ordered: u64,
    waiters: usize,
    cancelled: bool,
}

pub(crate) struct ByteBudget {
    limit: usize,
    state: Mutex<BudgetState>,
    changed: Condvar,
}

impl ByteBudget {
    pub(crate) fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self {
            limit,
            state: Mutex::new(BudgetState::default()),
            changed: Condvar::new(),
        })
    }

    pub(crate) fn acquire(self: &Arc<Self>, bytes: usize) -> Result<BytePermit, BudgetError> {
        self.acquire_inner(None, bytes)
    }

    pub(crate) fn acquire_ordered(
        self: &Arc<Self>,
        sequence: u64,
        bytes: usize,
    ) -> Result<BytePermit, BudgetError> {
        self.acquire_inner(Some(sequence), bytes)
    }

    fn acquire_inner(
        self: &Arc<Self>,
        sequence: Option<u64>,
        bytes: usize,
    ) -> Result<BytePermit, BudgetError> {
        let mut state = lock_recover(&self.state);
        loop {
            if state.cancelled {
                return Err(BudgetError::Cancelled);
            }
            if bytes > self.limit {
                return Err(BudgetError::Oversize {
                    limit: self.limit,
                    actual: bytes,
                });
            }
            if sequence.is_some_and(|sequence| sequence < state.next_ordered) {
                return Err(BudgetError::SequenceAlreadyAdmitted);
            }
            let sequence_ready = sequence.is_none_or(|sequence| sequence == state.next_ordered);
            if sequence_ready && bytes <= self.limit - state.used {
                state.used = state.used.saturating_add(bytes);
                if sequence.is_some() {
                    state.next_ordered = state
                        .next_ordered
                        .checked_add(1)
                        .ok_or(BudgetError::SequenceAlreadyAdmitted)?;
                    self.changed.notify_all();
                }
                return Ok(BytePermit {
                    budget: Arc::clone(self),
                    bytes,
                });
            }
            state.waiters = state.waiters.saturating_add(1);
            self.changed.notify_all();
            state = match self.changed.wait(state) {
                Ok(state) => state,
                Err(poisoned) => poisoned.into_inner(),
            };
            state.waiters = state.waiters.saturating_sub(1);
        }
    }

    pub(crate) fn cancel(&self) {
        let mut state = lock_recover(&self.state);
        state.cancelled = true;
        self.changed.notify_all();
    }
}

fn lock_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

pub(crate) struct BytePermit {
    budget: Arc<ByteBudget>,
    bytes: usize,
}

pub(crate) trait MemoryPolicy {
    type Permit: Send + 'static;

    fn permit(acquired: Option<BytePermit>) -> Self::Permit;
}

impl MemoryPolicy for MemoryDisabled {
    type Permit = ();

    fn permit(acquired: Option<BytePermit>) -> Self::Permit {
        debug_assert!(acquired.is_none());
    }
}

impl MemoryPolicy for MemoryEnabled {
    type Permit = BytePermit;

    fn permit(acquired: Option<BytePermit>) -> Self::Permit {
        acquired.expect("enabled byte accounting acquires a permit before publication")
    }
}

impl Drop for BytePermit {
    fn drop(&mut self) {
        let mut state = lock_recover(&self.budget.state);
        state.used = state.used.saturating_sub(self.bytes);
        self.budget.changed.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::thread;

    use super::*;

    #[test]
    fn permit_release_and_cancellation_wake_waiters() {
        let budget = ByteBudget::new(4);
        let first = budget.acquire(4).unwrap();
        let waiting = Arc::clone(&budget);
        let (sender, receiver) = mpsc::sync_channel(0);
        let handle = thread::spawn(move || {
            sender.send(waiting.acquire(1).is_ok()).unwrap();
        });
        drop(first);
        assert!(receiver.recv().unwrap());
        handle.join().unwrap();

        let held = budget.acquire(4).unwrap();
        let waiting = Arc::clone(&budget);
        let (sender, receiver) = mpsc::sync_channel(0);
        let handle = thread::spawn(move || {
            sender.send(waiting.acquire(1).is_err()).unwrap();
        });
        budget.cancel();
        assert!(receiver.recv().unwrap());
        drop(held);
        handle.join().unwrap();
    }

    #[test]
    fn exact_limit_never_overgrants_and_release_wakes_the_blocked_acquire() {
        let budget = ByteBudget::new(5);
        let first = budget.acquire(3).unwrap();
        let second = budget.acquire(2).unwrap();
        let waiting = Arc::clone(&budget);
        let (admitted_sender, admitted_receiver) = mpsc::sync_channel(0);
        let handle = thread::spawn(move || {
            let permit = waiting.acquire(1).unwrap();
            admitted_sender.send(()).unwrap();
            drop(permit);
        });

        let mut state = lock_recover(&budget.state);
        while state.waiters == 0 {
            state = match budget.changed.wait(state) {
                Ok(state) => state,
                Err(poisoned) => poisoned.into_inner(),
            };
        }
        assert_eq!(state.used, 5);
        drop(state);
        assert!(matches!(
            admitted_receiver.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
        drop(first);
        admitted_receiver.recv().unwrap();
        drop(second);
        handle.join().unwrap();
    }
}
