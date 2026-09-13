//! Measure application regions and benchmark completed tensor work.
//!
//! [`Profiler`] records bounded CPU wall-clock regions and exports Chrome trace
//! JSON for Perfetto. It does not intercept operators, allocations or GPU kernels.
//! Use [`benchmark`] with an explicit synchronization closure for device work.
//!
//! ```
//! use rusttorch::{profiling::{Profiler, benchmark, BenchmarkOptions}, Tensor};
//! let profile = Profiler::new(100)?;
//! let x = Tensor::from_slice(&[1_f32, 2., 3.]);
//! let y = profile.record("square", || Ok(x.f_square()?))?;
//! assert_eq!(y.size(), [3]);
//! let trace = profile.chrome_trace()?;
//! assert!(trace.contains("square"));
//! let result = benchmark(BenchmarkOptions { warmup: 2, samples: 5 },
//!     || Ok(x.f_square()?), || Ok(()))?;
//! assert_eq!(result.seconds.len(), 5);
//! # Ok::<(), rusttorch::RustTorchError>(())
//! ```

use crate::{Result, RustTorchError};
use serde::Serialize;
use std::{collections::HashMap, sync::Mutex, thread::ThreadId, time::Instant};

#[derive(Debug, Serialize)]
struct Event {
    name: String,
    cat: &'static str,
    ph: &'static str,
    pid: u32,
    tid: u64,
    ts: f64,
    dur: f64,
    args: EventArgs,
    #[serde(skip)]
    finished: bool,
}

#[derive(Debug, Serialize)]
struct EventArgs {
    outcome: &'static str,
}

#[derive(Debug, Default)]
struct Events {
    values: Vec<Event>,
    threads: HashMap<ThreadId, u64>,
}

/// A bounded, thread-safe recorder for explicitly named application regions.
///
/// One recorder can be shared by reference across scoped threads. A region
/// measures elapsed host time, including scheduling, locks and any explicit
/// device synchronization performed inside it. Nested spans on a thread must
/// finish in reverse order to remain properly nested in trace viewers.
#[derive(Debug)]
pub struct Profiler {
    origin: Instant,
    capacity: usize,
    events: Mutex<Events>,
}

impl Profiler {
    /// Creates a recorder retaining at most `capacity` events, in `1..=1_000_000`.
    ///
    /// A full recorder rejects a new span before executing [`Self::record`]'s
    /// closure. Export and [`Self::clear`] between windows in a long-running job.
    pub fn new(capacity: usize) -> Result<Self> {
        if !(1..=1_000_000).contains(&capacity) {
            return Err(invalid("event capacity must be in 1..=1_000_000"));
        }
        Ok(Self {
            origin: Instant::now(),
            capacity,
            events: Mutex::new(Events::default()),
        })
    }

    /// Starts a region that finishes when its guard is dropped.
    ///
    /// Labels must contain 1–1,024 bytes. Bind the returned guard to a named
    /// variable and drop it at the intended boundary; `let _ = ...` ends it
    /// immediately. A guard must finish on the thread where it started.
    pub fn span(&self, label: impl Into<String>) -> Result<Span<'_>> {
        let name = label.into();
        if name.is_empty() || name.len() > 1024 {
            return Err(invalid("span label must contain 1..=1024 bytes"));
        }
        let mut events = self
            .events
            .lock()
            .map_err(|_| invalid("profiler state poisoned"))?;
        if events.values.len() == self.capacity {
            return Err(invalid(
                "profile event capacity reached; export and clear the completed window",
            ));
        }
        let next = events.threads.len() as u64 + 1;
        let thread = std::thread::current().id();
        let tid = *events.threads.entry(thread).or_insert(next);
        events
            .values
            .try_reserve(1)
            .map_err(|_| invalid("cannot reserve trace event"))?;
        let index = events.values.len();
        let start = Instant::now();
        events.values.push(Event {
            name,
            cat: "rusttorch.application",
            ph: "X",
            pid: std::process::id(),
            tid,
            ts: start.duration_since(self.origin).as_secs_f64() * 1_000_000.,
            dur: 0.,
            args: EventArgs {
                outcome: "complete",
            },
            finished: false,
        });
        Ok(Span {
            profiler: self,
            index,
            start,
            failed: false,
            thread_bound: std::marker::PhantomData,
        })
    }

    /// Measures a fallible closure, recording whether it returned an error.
    ///
    /// Panics propagate normally while the guard records `"panic"`. Use explicit
    /// synchronization inside the closure when measuring completed device work.
    pub fn record<T>(
        &self,
        label: impl Into<String>,
        work: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let mut span = self.span(label)?;
        let result = work();
        span.failed = result.is_err();
        result
    }

    /// Serializes completed events in Chrome trace JSON, with microsecond timestamps.
    ///
    /// Wait for active spans to finish before exporting. Save the returned string
    /// to a `.json` file and open it in a trace viewer such as Perfetto. Names are
    /// serialized as JSON data; quotes and control characters are escaped.
    pub fn chrome_trace(&self) -> Result<String> {
        let events = self
            .events
            .lock()
            .map_err(|_| invalid("profiler state poisoned"))?;
        if events.values.iter().any(|event| !event.finished) {
            return Err(invalid("finish active spans before exporting"));
        }
        #[derive(Serialize)]
        struct Trace<'a> {
            #[serde(rename = "traceEvents")]
            events: &'a [Event],
            #[serde(rename = "displayTimeUnit")]
            unit: &'static str,
        }
        serde_json::to_string(&Trace {
            events: &events.values,
            unit: "ms",
        })
        .map_err(|error| invalid(&format!("cannot serialize trace: {error}")))
    }

    /// Discards a completed window and its thread metadata without resetting the clock.
    pub fn clear(&self) -> Result<()> {
        let mut events = self
            .events
            .lock()
            .map_err(|_| invalid("profiler state poisoned"))?;
        if events.values.iter().any(|event| !event.finished) {
            return Err(invalid("finish active spans before clearing"));
        }
        events.values.clear();
        events.threads.clear();
        Ok(())
    }
}

/// Ends an application region on drop; created by [`Profiler::span`].
///
/// This guard intentionally cannot move to another thread. It holds no lock
/// while application code runs, allowing nested regions and parallel workers.
#[derive(Debug)]
pub struct Span<'a> {
    profiler: &'a Profiler,
    index: usize,
    start: Instant,
    failed: bool,
    thread_bound: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl Drop for Span<'_> {
    fn drop(&mut self) {
        if let Ok(mut events) = self.profiler.events.lock() {
            let event = &mut events.values[self.index];
            event.dur = self.start.elapsed().as_secs_f64() * 1_000_000.;
            event.args.outcome = if std::thread::panicking() {
                "panic"
            } else if self.failed {
                "error"
            } else {
                "complete"
            };
            event.finished = true;
        }
    }
}

/// Number of untimed warmup calls and retained benchmark samples.
///
/// Both counts are bounded at 1,000,000; `samples` must be positive. Each sample
/// times one invocation, so amortize tiny operations in your closure yourself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BenchmarkOptions {
    /// Warmup invocations before samples are collected; default 5.
    pub warmup: usize,
    /// Number of measured invocations; default 20.
    pub samples: usize,
}

impl Default for BenchmarkOptions {
    fn default() -> Self {
        Self {
            warmup: 5,
            samples: 20,
        }
    }
}

/// Raw seconds and descriptive statistics for one benchmark configuration.
///
/// Compare equivalent work using the same build, device, runtime, thread count
/// and input. A timing distribution alone makes no cross-library speed claim.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Measurement {
    /// One elapsed time per measured call, in execution order.
    pub seconds: Vec<f64>,
    /// Arithmetic mean in seconds.
    pub mean: f64,
    /// Median in seconds, averaging the middle pair for an even sample count.
    pub median: f64,
    /// Population standard deviation in seconds.
    pub standard_deviation: f64,
    /// Minimum elapsed seconds.
    pub minimum: f64,
    /// Maximum elapsed seconds.
    pub maximum: f64,
}

/// Measures warm, explicitly synchronized work without timing output destruction.
///
/// `synchronize` runs before and after every warmup and measured call. Use
/// `|| Ok(())` only for synchronous CPU work or deliberate host-dispatch timing.
/// For CUDA, call `tch::Cuda::synchronize(device)`; for another asynchronous
/// backend supply its real synchronization primitive or copy a required result
/// to CPU inside the work closure. This API cannot infer an asynchronous job's
/// completion. Work/synchronization errors stop collection immediately.
pub fn benchmark<T>(
    options: BenchmarkOptions,
    mut work: impl FnMut() -> Result<T>,
    mut synchronize: impl FnMut() -> Result<()>,
) -> Result<Measurement> {
    if options.samples == 0 || options.samples > 1_000_000 || options.warmup > 1_000_000 {
        return Err(invalid(
            "benchmark requires 1..=1_000_000 samples and at most 1_000_000 warmup calls",
        ));
    }
    for _ in 0..options.warmup {
        synchronize()?;
        let output = work()?;
        synchronize()?;
        std::hint::black_box(output);
    }
    let mut seconds = Vec::new();
    seconds
        .try_reserve_exact(options.samples)
        .map_err(|_| invalid("cannot reserve benchmark samples"))?;
    for _ in 0..options.samples {
        synchronize()?;
        let start = Instant::now();
        let output = work()?;
        synchronize()?;
        seconds.push(start.elapsed().as_secs_f64());
        std::hint::black_box(output);
    }
    let mean = seconds.iter().sum::<f64>() / seconds.len() as f64;
    let standard_deviation =
        (seconds.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / seconds.len() as f64).sqrt();
    let mut sorted = seconds.clone();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    let median = if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.
    } else {
        sorted[middle]
    };
    Ok(Measurement {
        minimum: sorted[0],
        maximum: sorted[sorted.len() - 1],
        mean,
        median,
        standard_deviation,
        seconds,
    })
}

fn invalid(reason: &str) -> RustTorchError {
    RustTorchError::InvalidConfiguration {
        field: "profiling",
        reason: reason.into(),
    }
}
