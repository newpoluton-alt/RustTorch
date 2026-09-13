use super::{decode, invalid};
use crate::{Device, Kind, Result, Tensor, no_grad};
use serde::{Deserialize, Serialize};
use std::{
    io::{Read, Write},
    net::{Shutdown, SocketAddr, TcpListener, TcpStream},
    thread,
    time::{Duration, Instant},
};

const VERSION: u32 = 1;
const MAX_RANKS: usize = 1024;

/// Rendezvous identity, operation deadline and encoded-message memory limit.
///
/// All ranks must use the same session and message limit. Choose a fresh session
/// for each run; the name detects accidental cross-run connections, not hostile
/// peers. For example, `GroupOptions::new("image-training-42")` uses a 30-second
/// deadline and a 64 MiB limit **including JSON encoding** for each message.
///
/// ```
/// use std::time::Duration;
/// use rusttorch::distributed::GroupOptions;
/// let mut options = GroupOptions::new("experiment-42");
/// options.timeout = Duration::from_secs(60);
/// options.max_message_bytes = 128 * 1024 * 1024;
/// ```
#[derive(Debug, Clone)]
pub struct GroupOptions {
    /// Nonempty run identifier, at most 256 UTF-8 bytes.
    pub session: String,
    /// Positive timeout for the entire rendezvous or individual operation.
    pub timeout: Duration,
    /// Maximum encoded message and estimated collective response working set.
    /// Split large collective inputs or explicitly increase this limit.
    pub max_message_bytes: usize,
}
impl GroupOptions {
    /// Creates options for a mutually trusted worker session.
    pub fn new(session: impl Into<String>) -> Self {
        Self {
            session: session.into(),
            timeout: Duration::from_secs(30),
            max_message_bytes: 64 * 1024 * 1024,
        }
    }
    fn validate(&self, world: usize, rank: usize) -> Result<()> {
        if !(1..=MAX_RANKS).contains(&world) || rank >= world {
            return Err(invalid(
                "world size must be in 1..=1024 and rank must be smaller",
            ));
        }
        if self.session.is_empty()
            || self.session.len() > 256
            || self.timeout.is_zero()
            || Instant::now().checked_add(self.timeout).is_none()
            || self.max_message_bytes < 1024
            || self.max_message_bytes > 1024 * 1024 * 1024
        {
            return Err(invalid(
                "invalid session, timeout, or message limit (1024..=1 GiB)",
            ));
        }
        Ok(())
    }
}

/// Reduction applied in deterministic rank order on the coordinator.
///
/// `Mean` requires floating tensors; other operations also accept `Int64`.
/// Integer overflow follows the native tensor kernel. Reduction results are
/// detached from autograd; synchronize parameter gradients explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReduceOp {
    /// Elementwise sum.
    Sum,
    /// Elementwise sum divided by world size.
    Mean,
    /// Elementwise minimum.
    Min,
    /// Elementwise maximum.
    Max,
    /// Elementwise product.
    Product,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum Dtype {
    Float,
    Double,
    Int64,
}
impl Dtype {
    pub(super) fn kind(self) -> Kind {
        match self {
            Self::Float => Kind::Float,
            Self::Double => Kind::Double,
            Self::Int64 => Kind::Int64,
        }
    }
    fn from_kind(kind: Kind) -> Result<Self> {
        match kind {
            Kind::Float => Ok(Self::Float),
            Kind::Double => Ok(Self::Double),
            Kind::Int64 => Ok(Self::Int64),
            _ => Err(invalid(
                "CPU transport accepts only Float, Double, and Int64",
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Payload {
    pub(super) shape: Vec<i64>,
    pub(super) dtype: Dtype,
    pub(super) bytes: Vec<u8>,
}
impl Payload {
    pub(super) fn capture(tensor: &Tensor, limit: usize) -> Result<Self> {
        if !tensor.defined()
            || tensor.device() != Device::Cpu
            || tensor.is_sparse()
            || tensor.is_mkldnn()
        {
            return Err(invalid("transport requires defined dense CPU tensors"));
        }
        let dtype = Dtype::from_kind(tensor.kind())?;
        let length = tensor
            .numel()
            .checked_mul(dtype.kind().elt_size_in_bytes())
            .filter(|n| *n <= limit)
            .ok_or_else(|| invalid("tensor exceeds message limit"))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| invalid("cannot allocate tensor payload"))?;
        bytes.resize(length, 0);
        tensor
            .f_contiguous()?
            .f_copy_data_u8(&mut bytes, tensor.numel())?;
        if cfg!(target_endian = "big") {
            swap(&mut bytes, dtype.kind().elt_size_in_bytes());
        }
        let value = Self {
            shape: tensor.size(),
            dtype,
            bytes,
        };
        value.validate(limit)?;
        Ok(value)
    }
    pub(super) fn validate(&self, limit: usize) -> Result<()> {
        if self.shape.len() > 64 || self.bytes.len() > limit {
            return Err(invalid("invalid tensor rank or byte count"));
        }
        let n = self
            .shape
            .iter()
            .try_fold(1usize, |n, &d| {
                usize::try_from(d).ok().and_then(|d| n.checked_mul(d))
            })
            .ok_or_else(|| invalid("tensor dimensions overflow or are negative"))?;
        if n.checked_mul(self.dtype.kind().elt_size_in_bytes()) != Some(self.bytes.len()) {
            return Err(invalid("tensor byte count differs from shape and dtype"));
        }
        Ok(())
    }
    pub(super) fn tensor(&self) -> Result<Tensor> {
        self.validate(1024 * 1024 * 1024)?;
        if cfg!(target_endian = "big") {
            let mut bytes = self.bytes.clone();
            swap(&mut bytes, self.dtype.kind().elt_size_in_bytes());
            Ok(Tensor::f_from_data_size(
                &bytes,
                &self.shape,
                self.dtype.kind(),
            )?)
        } else {
            Ok(Tensor::f_from_data_size(
                &self.bytes,
                &self.shape,
                self.dtype.kind(),
            )?)
        }
    }
    fn same_spec(&self, other: &Self) -> bool {
        self.shape == other.shape && self.dtype == other.dtype
    }
}
fn swap(bytes: &mut [u8], width: usize) {
    for item in bytes.chunks_exact_mut(width) {
        item.reverse();
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum Operation {
    Barrier,
    Bytes,
    Broadcast(usize),
    AllReduce(ReduceOp),
    Reduce(usize, ReduceOp),
    AllGather,
    Gather(usize),
    Scatter(usize),
    ReduceScatter(ReduceOp),
    AllToAll,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    sequence: u64,
    operation: Operation,
    error: Option<String>,
    tensors: Vec<Payload>,
    bytes: Vec<u8>,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Response {
    sequence: u64,
    error: Option<String>,
    tensors: Vec<Payload>,
    bytes: Vec<Vec<u8>>,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Hello {
    version: u32,
    rank: usize,
    world: usize,
    session: String,
    limit: usize,
    address: SocketAddr,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum Frame {
    Hello(Hello),
    Addresses(Vec<SocketAddr>),
    Request(Request),
    Response(Response),
    Point {
        sequence: u64,
        tag: u64,
        tensor: Payload,
    },
    Ack {
        sequence: u64,
        tag: u64,
    },
}

/// A blocking CPU TCP group with one ordered stream per pair of ranks.
///
/// Rank zero calls [`Self::accept`] with a bound listener; other ranks call
/// [`Self::connect`] with its address. Every collective must be called by every
/// rank in the same order. Point-to-point calls involve only their two ranks.
/// Do not interleave conflicting collective and point-to-point orders.
///
/// Each call owns a finite deadline. Any transport/protocol/collective failure
/// closes this rank's connections and makes subsequent calls fail immediately.
/// Other participants observe disconnects or their own deadlines. This backend
/// has O(world-size²) sockets and gathers collective data at rank zero; it is a
/// correctness-oriented CPU implementation, with no scaling-performance claim.
/// Deadlines bound communication waits; native tensor kernels and serialization
/// are synchronous and cannot be preempted by the socket deadline.
///
/// ```
/// use rusttorch::{Tensor, distributed::{GroupOptions, ProcessGroup, ReduceOp}};
/// let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
/// let mut group = ProcessGroup::accept(listener, 1, GroupOptions::new("collectives"))?;
/// let value = Tensor::from_slice(&[1_f64, 2.]);
/// let copies = group.all_gather(&value)?;
/// assert_eq!(copies.len(), group.world_size());
/// let destination = group.scatter(&copies, 0)?;
/// let reduced = group.reduce_scatter(&copies, ReduceOp::Sum)?;
/// let exchanged = group.all_to_all(&copies)?;
/// assert_eq!(destination.size(), reduced.size());
/// assert_eq!(exchanged.len(), 1);
/// group.barrier()?;
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
#[derive(Debug)]
pub struct ProcessGroup {
    rank: usize,
    world: usize,
    options: GroupOptions,
    peers: Vec<Option<TcpStream>>,
    sequence: u64,
    sends: Vec<u64>,
    receives: Vec<u64>,
    failed: bool,
}
impl ProcessGroup {
    /// Starts rank zero using a listener already bound by the application.
    ///
    /// Binding `127.0.0.1:0` and sharing `listener.local_addr()` avoids port
    /// reservation races in tests. Other ranks must join before the deadline.
    pub fn accept(listener: TcpListener, world: usize, options: GroupOptions) -> Result<Self> {
        options.validate(world, 0)?;
        let deadline = deadline(options.timeout)?;
        listener.set_nonblocking(true).map_err(io)?;
        let mut peers: Vec<Option<TcpStream>> = (0..world).map(|_| None).collect();
        let address = listener.local_addr().map_err(io)?;
        let mut addresses = vec![address; world];
        for _ in 1..world {
            let mut stream = accept(&listener, deadline)?;
            let hello = hello(read(&mut stream, deadline, options.max_message_bytes)?)?;
            check_hello(&hello, world, &options)?;
            if hello.rank == 0 || peers[hello.rank].is_some() {
                return Err(invalid("duplicate or invalid rendezvous rank"));
            }
            addresses[hello.rank] = hello.address;
            peers[hello.rank] = Some(stream);
        }
        for stream in peers.iter_mut().flatten() {
            write(
                stream,
                &Frame::Addresses(addresses.clone()),
                deadline,
                options.max_message_bytes,
            )?;
        }
        let mut group = Self {
            rank: 0,
            world,
            options,
            peers,
            sequence: 0,
            sends: vec![0; world],
            receives: vec![0; world],
            failed: false,
        };
        let timeout = group.options.timeout;
        group.options.timeout = remaining(deadline)?;
        group.barrier()?;
        group.options.timeout = timeout;
        Ok(group)
    }

    /// Joins rank zero's address as a nonzero rank.
    ///
    /// For two workers, rank zero uses `accept(listener, 2, options)` and rank
    /// one uses `connect(address, 1, 2, options)`. All workers must be able to
    /// connect to each other's advertised local IP addresses.
    pub fn connect(
        address: SocketAddr,
        rank: usize,
        world: usize,
        options: GroupOptions,
    ) -> Result<Self> {
        options.validate(world, rank)?;
        if rank == 0 {
            return Err(invalid("rank zero must use ProcessGroup::accept"));
        }
        let deadline = deadline(options.timeout)?;
        let mut coordinator = connect(address, deadline)?;
        let listener = TcpListener::bind(SocketAddr::new(
            coordinator.local_addr().map_err(io)?.ip(),
            0,
        ))
        .map_err(io)?;
        listener.set_nonblocking(true).map_err(io)?;
        let local = listener.local_addr().map_err(io)?;
        let greeting = |rank| {
            Frame::Hello(Hello {
                version: VERSION,
                rank,
                world,
                session: options.session.clone(),
                limit: options.max_message_bytes,
                address: local,
            })
        };
        write(
            &mut coordinator,
            &greeting(rank),
            deadline,
            options.max_message_bytes,
        )?;
        let addresses = match read(&mut coordinator, deadline, options.max_message_bytes)? {
            Frame::Addresses(values) if values.len() == world => values,
            _ => return Err(invalid("invalid rendezvous address list")),
        };
        let mut peers: Vec<Option<TcpStream>> = (0..world).map(|_| None).collect();
        peers[0] = Some(coordinator);
        for other in (rank + 1)..world {
            let mut stream = connect(addresses[other], deadline)?;
            write(
                &mut stream,
                &greeting(rank),
                deadline,
                options.max_message_bytes,
            )?;
            peers[other] = Some(stream);
        }
        for _ in 1..rank {
            let mut stream = accept(&listener, deadline)?;
            let hello = hello(read(&mut stream, deadline, options.max_message_bytes)?)?;
            check_hello(&hello, world, &options)?;
            if hello.rank == 0 || hello.rank >= rank || peers[hello.rank].is_some() {
                return Err(invalid("invalid peer mesh rank"));
            }
            peers[hello.rank] = Some(stream);
        }
        let mut group = Self {
            rank,
            world,
            options,
            peers,
            sequence: 0,
            sends: vec![0; world],
            receives: vec![0; world],
            failed: false,
        };
        let timeout = group.options.timeout;
        group.options.timeout = remaining(deadline)?;
        group.barrier()?;
        group.options.timeout = timeout;
        Ok(group)
    }
    /// This worker's zero-based rank.
    pub fn rank(&self) -> usize {
        self.rank
    }
    /// Number of ranks in the group.
    pub fn world_size(&self) -> usize {
        self.world
    }
    /// Rendezvous identity selected by the application.
    pub fn session(&self) -> &str {
        &self.options.session
    }
    /// Changes this rank's communication budget for subsequent operations.
    ///
    /// For example, use a long rendezvous budget, then
    /// `group.set_timeout(std::time::Duration::from_secs(5))?` for training.
    /// The timeout must be positive and fit the monotonic clock. It need not
    /// match other ranks; changing it does not recover a failed group.
    pub fn set_timeout(&mut self, timeout: Duration) -> Result<()> {
        self.ready()?;
        if timeout.is_zero() || Instant::now().checked_add(timeout).is_none() {
            return Err(invalid(
                "communication timeout must be positive and fit the clock",
            ));
        }
        self.options.timeout = timeout;
        Ok(())
    }
    /// Whether a previous operation permanently failed this group.
    pub fn is_failed(&self) -> bool {
        self.failed
    }
    /// Encoded-message limit, also used by checkpoint exchange.
    pub fn max_message_bytes(&self) -> usize {
        self.options.max_message_bytes
    }
    /// Closes all connections. Other ranks' blocked operations fail or time out.
    pub fn abort(&mut self) {
        self.failed = true;
        for stream in &mut self.peers {
            if let Some(stream) = stream.take() {
                let _ = stream.shutdown(Shutdown::Both);
            }
        }
    }
    fn ready(&self) -> Result<Instant> {
        if self.failed {
            Err(invalid("process group failed; create a new session"))
        } else {
            deadline(self.options.timeout)
        }
    }
    fn finish<T>(&mut self, result: Result<T>) -> Result<T> {
        if result.is_err() {
            self.abort();
        }
        result
    }
    /// Waits until every rank reaches this collective.
    pub fn barrier(&mut self) -> Result<()> {
        self.call(Operation::Barrier, Ok(vec![]), vec![])
            .map(|_| ())
    }
    /// Gathers bounded, non-executable byte payloads in rank order.
    ///
    /// Useful for agreeing on model metadata or exchanging JSON checkpoint
    /// records. Bytes are not interpreted as Python objects or pickle.
    pub fn all_gather_bytes(&mut self, bytes: &[u8]) -> Result<Vec<Vec<u8>>> {
        let valid = if bytes.len() > self.options.max_message_bytes {
            Err(invalid("byte payload exceeds message limit"))
        } else {
            Ok(vec![])
        };
        self.call(
            Operation::Bytes,
            valid,
            if bytes.len() <= self.options.max_message_bytes {
                bytes.to_vec()
            } else {
                vec![]
            },
        )
        .map(|v| v.bytes)
    }
    /// Broadcasts a tensor from `root`; every rank supplies the same tensor spec.
    pub fn broadcast(&mut self, tensor: &Tensor, root: usize) -> Result<Tensor> {
        self.one(Operation::Broadcast(root), &[tensor])
    }
    /// Reduces a tensor and returns an independent result on every rank.
    pub fn all_reduce(&mut self, tensor: &Tensor, op: ReduceOp) -> Result<Tensor> {
        self.one(Operation::AllReduce(op), &[tensor])
    }
    /// Reduces to `root`; other ranks receive `None`.
    pub fn reduce(&mut self, tensor: &Tensor, op: ReduceOp, root: usize) -> Result<Option<Tensor>> {
        Ok(self
            .tensors(Operation::Reduce(root, op), &[tensor])?
            .into_iter()
            .next())
    }
    /// Collects equal-spec tensors in rank order on every rank.
    pub fn all_gather(&mut self, tensor: &Tensor) -> Result<Vec<Tensor>> {
        self.tensors(Operation::AllGather, &[tensor])
    }
    /// Collects equal-spec tensors on `root`; other ranks receive an empty vector.
    pub fn gather(&mut self, tensor: &Tensor, root: usize) -> Result<Vec<Tensor>> {
        self.tensors(Operation::Gather(root), &[tensor])
    }
    /// Sends one equal-spec tensor per rank from `root`; other ranks pass `&[]`.
    pub fn scatter(&mut self, tensors: &[Tensor], root: usize) -> Result<Tensor> {
        self.one(
            Operation::Scatter(root),
            &tensors.iter().collect::<Vec<_>>(),
        )
    }
    /// Reduces each rank-indexed input and returns this rank's reduced tensor.
    ///
    /// Every rank supplies `world_size()` equal-spec tensors. `Mean` is useful
    /// for partitioning a full model gradient among optimizer owners.
    pub fn reduce_scatter(&mut self, tensors: &[Tensor], op: ReduceOp) -> Result<Tensor> {
        self.one(
            Operation::ReduceScatter(op),
            &tensors.iter().collect::<Vec<_>>(),
        )
    }
    /// Sends entry `r` to rank `r`; returns one tensor from each source rank.
    pub fn all_to_all(&mut self, tensors: &[Tensor]) -> Result<Vec<Tensor>> {
        self.tensors(Operation::AllToAll, &tensors.iter().collect::<Vec<_>>())
    }
    fn one(&mut self, op: Operation, tensors: &[&Tensor]) -> Result<Tensor> {
        self.tensors(op, tensors)?
            .into_iter()
            .next()
            .ok_or_else(|| invalid("collective returned no tensor"))
    }
    fn tensors(&mut self, op: Operation, tensors: &[&Tensor]) -> Result<Vec<Tensor>> {
        let data = tensors
            .iter()
            .map(|t| Payload::capture(t, self.options.max_message_bytes))
            .collect();
        let response = self.call(op, data, vec![])?;
        let result = response.tensors.iter().map(Payload::tensor).collect();
        self.finish(result)
    }
    pub(super) fn checkpoint_stamp(&self) -> String {
        format!("{}:{}", self.options.session, self.sequence)
    }

    pub(super) fn agree(&mut self, status: Result<()>) -> Result<()> {
        self.call(Operation::Barrier, status.map(|_| vec![]), vec![])
            .map(|_| ())
    }
    pub(super) fn identical(&mut self, bytes: &[u8]) -> Result<()> {
        let values = self.all_gather_bytes(bytes)?;
        let result = if values.iter().all(|v| v == bytes) {
            Ok(())
        } else {
            Err(invalid("rank metadata or checkpoint identity differs"))
        };
        self.finish(result)
    }
    fn call(
        &mut self,
        operation: Operation,
        tensors: Result<Vec<Payload>>,
        bytes: Vec<u8>,
    ) -> Result<Response> {
        let result = (|| {
            let deadline = self.ready()?;
            let (tensors, error) = match tensors {
                Ok(v) => (v, None),
                Err(e) => (vec![], Some(e.to_string())),
            };
            let request = Request {
                sequence: self.sequence,
                operation,
                tensors,
                bytes,
                error,
            };
            let response = if self.rank == 0 {
                let mut requests = vec![request];
                for stream in self.peers.iter_mut().skip(1).flatten() {
                    match read(stream, deadline, self.options.max_message_bytes)? {
                        Frame::Request(r) => requests.push(r),
                        _ => return Err(invalid("collective received a different operation type")),
                    }
                }
                let computation = evaluate(&requests, self.world, self.options.max_message_bytes);
                let responses = match computation {
                    Ok(v) => v,
                    Err(e) => (0..self.world)
                        .map(|_| Response {
                            sequence: self.sequence,
                            error: Some(e.to_string()),
                            tensors: vec![],
                            bytes: vec![],
                        })
                        .collect(),
                };
                let mut responses = responses.into_iter();
                let own = responses.next().expect("nonempty group");
                for (stream, response) in self.peers.iter_mut().skip(1).flatten().zip(responses) {
                    write(
                        stream,
                        &Frame::Response(response),
                        deadline,
                        self.options.max_message_bytes,
                    )?;
                }
                own
            } else {
                let stream = self.peers[0].as_mut().expect("connected coordinator");
                write(
                    stream,
                    &Frame::Request(request),
                    deadline,
                    self.options.max_message_bytes,
                )?;
                match read(stream, deadline, self.options.max_message_bytes)? {
                    Frame::Response(r) => r,
                    _ => return Err(invalid("invalid collective response")),
                }
            };
            if response.sequence != self.sequence {
                return Err(invalid("collective response sequence mismatch"));
            }
            if let Some(error) = &response.error {
                return Err(invalid(error.clone()));
            }
            for tensor in &response.tensors {
                tensor.validate(self.options.max_message_bytes)?;
            }
            self.sequence = self
                .sequence
                .checked_add(1)
                .ok_or_else(|| invalid("collective sequence exhausted"))?;
            Ok(response)
        })();
        self.finish(result)
    }
    /// Sends a tensor to one other rank and waits for its matching receive.
    ///
    /// Tags and per-direction sequence numbers detect mismatched calls. Other
    /// ranks need not participate; a receive with the same source/tag must run
    /// concurrently. Calling blocking send on both peers first will time out.
    ///
    /// ```no_run
    /// # fn example(group: &mut rusttorch::distributed::ProcessGroup) -> rusttorch::Result<()> {
    /// use rusttorch::Tensor;
    /// if group.rank() == 0 {
    ///     group.send(&Tensor::from_slice(&[0.2_f32, 0.8]), 1, 42)?;
    /// } else if group.rank() == 1 {
    ///     let scores = group.recv(0, 42)?;
    ///     assert_eq!(scores.size(), [2]);
    /// }
    /// group.barrier()?;
    /// # Ok(()) }
    /// ```
    pub fn send(&mut self, tensor: &Tensor, destination: usize, tag: u64) -> Result<()> {
        let result = (|| {
            let deadline = self.ready()?;
            if destination >= self.world || destination == self.rank {
                return Err(invalid("invalid point-to-point destination"));
            }
            let tensor = Payload::capture(tensor, self.options.max_message_bytes)?;
            let sequence = self.sends[destination];
            let stream = self.peers[destination].as_mut().expect("connected peer");
            write(
                stream,
                &Frame::Point {
                    sequence,
                    tag,
                    tensor,
                },
                deadline,
                self.options.max_message_bytes,
            )?;
            match read(stream, deadline, self.options.max_message_bytes)? {
                Frame::Ack {
                    sequence: s,
                    tag: t,
                } if s == sequence && t == tag => (),
                _ => return Err(invalid("point-to-point acknowledgement mismatch")),
            }
            self.sends[destination] = sequence
                .checked_add(1)
                .ok_or_else(|| invalid("point-to-point sequence exhausted"))?;
            Ok(())
        })();
        self.finish(result)
    }
    /// Receives one detached tensor from `source` with the expected tag.
    pub fn recv(&mut self, source: usize, tag: u64) -> Result<Tensor> {
        let result = (|| {
            let deadline = self.ready()?;
            if source >= self.world || source == self.rank {
                return Err(invalid("invalid point-to-point source"));
            }
            let sequence = self.receives[source];
            let stream = self.peers[source].as_mut().expect("connected peer");
            let tensor = match read(stream, deadline, self.options.max_message_bytes)? {
                Frame::Point {
                    sequence: s,
                    tag: t,
                    tensor,
                } if s == sequence && t == tag => tensor.tensor()?,
                _ => return Err(invalid("point-to-point source sequence or tag mismatch")),
            };
            write(
                stream,
                &Frame::Ack { sequence, tag },
                deadline,
                self.options.max_message_bytes,
            )?;
            self.receives[source] = sequence
                .checked_add(1)
                .ok_or_else(|| invalid("point-to-point sequence exhausted"))?;
            Ok(tensor)
        })();
        self.finish(result)
    }
}
impl Drop for ProcessGroup {
    fn drop(&mut self) {
        self.abort();
    }
}

fn evaluate(requests: &[Request], world: usize, limit: usize) -> Result<Vec<Response>> {
    let first = &requests[0];
    // Bound copies before constructing rank-indexed response vectors. The
    // coordinator deliberately favors predictable memory over large collectives.
    let working_set = requests
        .iter()
        .try_fold(0usize, |total, request| {
            request.tensors.iter().try_fold(
                total.checked_add(request.bytes.len())?,
                |total, tensor| {
                    total
                        .checked_add(tensor.bytes.len())?
                        .checked_add(tensor.shape.len().checked_mul(8)?)?
                        .checked_add(64)
                },
            )
        })
        .and_then(|bytes| bytes.checked_mul(world));
    if working_set.is_none_or(|bytes| bytes > limit) {
        return Err(invalid(
            "collective response working set exceeds message limit; split the inputs",
        ));
    }
    for request in requests {
        if let Some(e) = &request.error {
            return Err(invalid(e.clone()));
        }
        if request.sequence != first.sequence || request.operation != first.operation {
            return Err(invalid(
                "collective sequence, operation, root or reduction mismatch",
            ));
        }
        if first.operation != Operation::Bytes && !request.bytes.is_empty() {
            return Err(invalid("unexpected bytes in tensor collective"));
        }
        for tensor in &request.tensors {
            tensor.validate(limit)?;
        }
    }
    let empty = || Response {
        sequence: first.sequence,
        error: None,
        tensors: vec![],
        bytes: vec![],
    };
    let mut result: Vec<_> = (0..world).map(|_| empty()).collect();
    let root = match first.operation {
        Operation::Broadcast(r)
        | Operation::Reduce(r, _)
        | Operation::Gather(r)
        | Operation::Scatter(r) => Some(r),
        _ => None,
    };
    if root.is_some_and(|r| r >= world) {
        return Err(invalid("collective root is outside the group"));
    }
    let counts = |expected: usize| -> Result<()> {
        if requests.iter().any(|r| r.tensors.len() != expected) {
            Err(invalid("collective tensor count mismatch"))
        } else {
            Ok(())
        }
    };
    match first.operation {
        Operation::Barrier => counts(0)?,
        Operation::Bytes => {
            counts(0)?;
            let bytes: Vec<_> = requests.iter().map(|r| r.bytes.clone()).collect();
            for out in &mut result {
                out.bytes = bytes.clone();
            }
        }
        Operation::Scatter(root) => {
            if requests[root].tensors.len() != world
                || requests
                    .iter()
                    .enumerate()
                    .any(|(r, v)| r != root && !v.tensors.is_empty())
            {
                return Err(invalid("scatter requires world-size inputs only at root"));
            }
            same(&requests[root].tensors)?;
            for (out, tensor) in result.iter_mut().zip(&requests[root].tensors) {
                out.tensors.push(tensor.clone());
            }
        }
        Operation::ReduceScatter(_) | Operation::AllToAll => {
            counts(world)?;
            let all: Vec<_> = requests
                .iter()
                .flat_map(|r| r.tensors.iter().cloned())
                .collect();
            same(&all)?;
            for (rank, out) in result.iter_mut().enumerate() {
                let inputs: Vec<_> = requests.iter().map(|r| r.tensors[rank].clone()).collect();
                out.tensors = match first.operation {
                    Operation::ReduceScatter(op) => vec![reduce(&inputs, op, limit)?],
                    _ => inputs,
                };
            }
        }
        _ => {
            counts(1)?;
            let inputs: Vec<_> = requests.iter().map(|r| r.tensors[0].clone()).collect();
            same(&inputs)?;
            match first.operation {
                Operation::Broadcast(root) => {
                    for out in &mut result {
                        out.tensors.push(inputs[root].clone());
                    }
                }
                Operation::AllReduce(op) => {
                    let tensor = reduce(&inputs, op, limit)?;
                    for out in &mut result {
                        out.tensors.push(tensor.clone());
                    }
                }
                Operation::Reduce(root, op) => {
                    result[root].tensors.push(reduce(&inputs, op, limit)?)
                }
                Operation::AllGather => {
                    for out in &mut result {
                        out.tensors = inputs.clone();
                    }
                }
                Operation::Gather(root) => result[root].tensors = inputs,
                _ => unreachable!(),
            }
        }
    }
    Ok(result)
}
fn same(tensors: &[Payload]) -> Result<()> {
    if tensors.windows(2).any(|v| !v[0].same_spec(&v[1])) {
        Err(invalid("collective tensor shape or dtype mismatch"))
    } else {
        Ok(())
    }
}
fn reduce(inputs: &[Payload], op: ReduceOp, limit: usize) -> Result<Payload> {
    if op == ReduceOp::Mean && inputs[0].dtype == Dtype::Int64 {
        return Err(invalid("mean reduction requires floating tensors"));
    }
    no_grad(|| {
        let mut result = inputs[0].tensor()?;
        for input in &inputs[1..] {
            let value = input.tensor()?;
            result = match op {
                ReduceOp::Sum | ReduceOp::Mean => result.f_add(&value)?,
                ReduceOp::Min => result.f_minimum(&value)?,
                ReduceOp::Max => result.f_maximum(&value)?,
                ReduceOp::Product => result.f_mul(&value)?,
            };
        }
        if op == ReduceOp::Mean {
            result = result.f_div_scalar(inputs.len() as f64)?;
        }
        Payload::capture(&result, limit)
    })
}
fn hello(frame: Frame) -> Result<Hello> {
    if let Frame::Hello(v) = frame {
        Ok(v)
    } else {
        Err(invalid("expected rendezvous greeting"))
    }
}
fn check_hello(hello: &Hello, world: usize, options: &GroupOptions) -> Result<()> {
    if hello.version != VERSION
        || hello.world != world
        || hello.rank >= world
        || hello.session != options.session
        || hello.limit != options.max_message_bytes
    {
        Err(invalid(
            "rendezvous version, rank, world, session or message limit mismatch",
        ))
    } else {
        Ok(())
    }
}
fn io(error: std::io::Error) -> crate::RustTorchError {
    if matches!(
        error.kind(),
        std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
    ) {
        invalid("TCP communication timed out")
    } else {
        invalid(format!("TCP communication failed: {error}"))
    }
}
fn deadline(timeout: Duration) -> Result<Instant> {
    Instant::now()
        .checked_add(timeout)
        .ok_or_else(|| invalid("timeout exceeds clock range"))
}
fn remaining(deadline: Instant) -> Result<Duration> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|d| !d.is_zero())
        .ok_or_else(|| invalid("TCP operation timed out"))
}
fn accept(listener: &TcpListener, deadline: Instant) -> Result<TcpStream> {
    loop {
        match listener.accept() {
            Ok((stream, _)) => {
                stream.set_nonblocking(false).map_err(io)?;
                stream.set_nodelay(true).map_err(io)?;
                return Ok(stream);
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(remaining(deadline)?.min(Duration::from_millis(2)))
            }
            Err(e) => return Err(io(e)),
        }
    }
}
fn connect(address: SocketAddr, deadline: Instant) -> Result<TcpStream> {
    loop {
        match TcpStream::connect_timeout(
            &address,
            remaining(deadline)?.min(Duration::from_millis(100)),
        ) {
            Ok(stream) => {
                stream.set_nonblocking(false).map_err(io)?;
                stream.set_nodelay(true).map_err(io)?;
                return Ok(stream);
            }
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::ConnectionRefused
                        | std::io::ErrorKind::TimedOut
                        | std::io::ErrorKind::WouldBlock
                ) =>
            {
                thread::sleep(remaining(deadline)?.min(Duration::from_millis(5)))
            }
            Err(e) => return Err(io(e)),
        }
    }
}
fn read_exact(stream: &mut TcpStream, mut bytes: &mut [u8], deadline: Instant) -> Result<()> {
    while !bytes.is_empty() {
        stream
            .set_read_timeout(Some(remaining(deadline)?))
            .map_err(io)?;
        match stream.read(bytes) {
            Ok(0) => return Err(invalid("peer disconnected")),
            Ok(n) => bytes = &mut bytes[n..],
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => (),
            Err(e) => return Err(io(e)),
        }
    }
    Ok(())
}
fn write_all(stream: &mut TcpStream, mut bytes: &[u8], deadline: Instant) -> Result<()> {
    while !bytes.is_empty() {
        stream
            .set_write_timeout(Some(remaining(deadline)?))
            .map_err(io)?;
        match stream.write(bytes) {
            Ok(0) => return Err(invalid("peer stopped accepting bytes")),
            Ok(n) => bytes = &bytes[n..],
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => (),
            Err(e) => return Err(io(e)),
        }
    }
    Ok(())
}
fn read(stream: &mut TcpStream, deadline: Instant, limit: usize) -> Result<Frame> {
    let mut header = [0; 8];
    read_exact(stream, &mut header, deadline)?;
    let length = usize::try_from(u64::from_be_bytes(header))
        .ok()
        .filter(|n| *n <= limit)
        .ok_or_else(|| invalid("peer message exceeds limit"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| invalid("cannot allocate peer message"))?;
    bytes.resize(length, 0);
    read_exact(stream, &mut bytes, deadline)?;
    decode(&bytes)
}
struct LimitedBytes {
    bytes: Vec<u8>,
    limit: usize,
}
impl Write for LimitedBytes {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self
            .bytes
            .len()
            .checked_add(bytes.len())
            .is_none_or(|n| n > self.limit)
        {
            return Err(std::io::Error::other(
                "encoded message exceeds configured limit",
            ));
        }
        self.bytes
            .try_reserve(bytes.len())
            .map_err(|_| std::io::Error::other("cannot allocate encoded message"))?;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
fn write(stream: &mut TcpStream, frame: &Frame, deadline: Instant, limit: usize) -> Result<()> {
    let mut output = LimitedBytes {
        bytes: Vec::new(),
        limit,
    };
    serde_json::to_writer(&mut output, frame)
        .map_err(|e| invalid(format!("cannot encode bounded peer message: {e}")))?;
    write_all(stream, &(output.bytes.len() as u64).to_be_bytes(), deadline)?;
    write_all(stream, &output.bytes, deadline)
}
