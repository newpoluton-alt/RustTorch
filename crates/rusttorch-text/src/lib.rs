//! Tokenize local text and produce padded model inputs with explicit policies.
//!
//! ```
//! use rusttorch_text::{EncodedSequence, Padding, TextCollator};
//! use rusttorch_data::Collate;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let sequences = vec![EncodedSequence::from_ids(vec![2, 3]), EncodedSequence::from_ids(vec![4])];
//! let batch = TextCollator::new(Padding::Longest { pad_id: 0 }, Default::default())?.collate(sequences)?;
//! assert_eq!(batch.input_ids.size(), [2, 2]);
//! assert_eq!(Vec::<Vec<i64>>::try_from(&batch.attention_mask)?, [[1,1],[1,0]]);
//! # Ok(()) }
//! ```
#![deny(missing_docs)]
#![doc = include_str!("../README.md")]
use rusttorch_core::{Device, RustTorchError, Tensor};
use rusttorch_data::{
    BatchSource, BatchSourceCheckpoint, Collate, CollateError, Dataset, DefaultConvert,
    DistributedConfiguration, DistributedSampler, DistributedSamplerState, MemoryFootprint,
    PinMemory, ResourceLimits, Sampler, SamplerCheckpoint,
};
use serde::{Deserialize, Serialize};
use std::{fs::File, path::Path};
/// Advanced Hugging Face models, normalizers and post-processors.
pub use tokenizers;
/// Text processing failure with preserved native source chains.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TextError {
    /// Tokenizer validation/encoding failure.
    #[error("tokenizer: {0}")]
    Tokenizer(#[source] tokenizers::Error),
    /// Invalid input or policy.
    #[error("{0}")]
    Invalid(String),
    /// Local tokenizer file failure.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Resource/configuration failure.
    #[error(transparent)]
    Limit(#[from] RustTorchError),
    /// Tensor allocation/conversion failure.
    #[error(transparent)]
    Tensor(#[from] tch::TchError),
}
/// Result returned by text operations.
pub type Result<T> = std::result::Result<T, TextError>;
/// One unpadded sequence, including post-processor special tokens.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EncodedSequence {
    /// Vocabulary IDs.
    pub ids: Vec<u32>,
    /// Segment/type identifiers aligned to IDs.
    pub type_ids: Vec<u32>,
    /// One for special tokens, zero for ordinary tokens.
    pub special_tokens_mask: Vec<u32>,
}
impl EncodedSequence {
    /// Constructs a single-segment sequence with no marked special tokens.
    pub fn from_ids(ids: Vec<u32>) -> Self {
        let n = ids.len();
        Self {
            ids,
            type_ids: vec![0; n],
            special_tokens_mask: vec![0; n],
        }
    }
    fn validate(&self, limits: ResourceLimits) -> Result<()> {
        limits.check("tokens", self.ids.len(), limits.max_tokens)?;
        if self.ids.len() != self.type_ids.len()
            || self.ids.len() != self.special_tokens_mask.len()
            || self.special_tokens_mask.iter().any(|&v| v > 1)
        {
            return Err(TextError::Invalid(
                "sequence fields must have equal length and binary special-token mask".into(),
            ));
        }
        Ok(())
    }
}
impl MemoryFootprint for EncodedSequence {
    fn resident_bytes(&self) -> usize {
        self.ids
            .capacity()
            .saturating_add(self.type_ids.capacity())
            .saturating_add(self.special_tokens_mask.capacity())
            .saturating_mul(4)
    }
}
impl DefaultConvert for EncodedSequence {
    type Output = Self;
    fn default_convert(self) -> std::result::Result<Self, CollateError> {
        Ok(self)
    }
}
/// Explicit right-truncation policy. Truncation occurs after special-token
/// insertion and can remove trailing special tokens; use tokenizer-native
/// postprocessing when a model requires preserving a closing separator.
#[derive(Clone, Copy, Debug)]
pub enum Truncation {
    /// Reject sequences longer than this limit.
    Reject(usize),
    /// Keep at most the first specified number of tokens.
    Right(usize),
}
/// Adapter over a locally configured Hugging Face tokenizer.
#[derive(Clone)]
pub struct HfTokenizer {
    inner: tokenizers::Tokenizer,
    limits: ResourceLimits,
    truncation: Truncation,
    add_special_tokens: bool,
}
impl HfTokenizer {
    /// Creates an adapter, disabling native padding/truncation so the specified
    /// RustTorch policies are the only length transformations.
    pub fn new(
        mut inner: tokenizers::Tokenizer,
        truncation: Truncation,
        add_special_tokens: bool,
        limits: ResourceLimits,
    ) -> Result<Self> {
        let n = match truncation {
            Truncation::Reject(n) | Truncation::Right(n) => n,
        };
        if n == 0 {
            return Err(TextError::Invalid("token limit must be positive".into()));
        }
        limits.check("token policy", n, limits.max_tokens)?;
        inner.with_padding(None);
        inner.with_truncation(None).map_err(TextError::Tokenizer)?;
        Ok(Self {
            inner,
            limits,
            truncation,
            add_special_tokens,
        })
    }
    /// Loads bounded local tokenizer JSON. No model or vocabulary is downloaded.
    pub fn from_file(
        path: impl AsRef<Path>,
        truncation: Truncation,
        add_special_tokens: bool,
        limits: ResourceLimits,
    ) -> Result<Self> {
        let bytes = limits.read_encoded(File::open(path)?)?;
        Self::new(
            tokenizers::Tokenizer::from_bytes(bytes).map_err(TextError::Tokenizer)?,
            truncation,
            add_special_tokens,
            limits,
        )
    }
    /// Read-only access to vocabulary and advanced tokenizer configuration.
    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }
    /// Number of vocabulary entries, including added tokens.
    pub fn vocabulary_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
    /// Encodes one bounded UTF-8 sample with deterministic length policy.
    pub fn encode(&self, text: &str) -> Result<EncodedSequence> {
        self.limits
            .check("text bytes", text.len(), self.limits.max_string_bytes)?;
        let e = self
            .inner
            .encode(text, self.add_special_tokens)
            .map_err(TextError::Tokenizer)?;
        self.limits
            .check("encoded tokens", e.len(), self.limits.max_tokens)?;
        let n = match self.truncation {
            Truncation::Reject(n) => {
                if e.len() > n {
                    return Err(TextError::Invalid(
                        "sequence exceeds truncation policy".into(),
                    ));
                }
                e.len()
            }
            Truncation::Right(n) => e.len().min(n),
        };
        Ok(EncodedSequence {
            ids: e.get_ids()[..n].to_vec(),
            type_ids: e.get_type_ids()[..n].to_vec(),
            special_tokens_mask: e.get_special_tokens_mask()[..n].to_vec(),
        })
    }
}
/// Local strings encoded lazily during map-style loading.
pub struct TextDataset {
    texts: Vec<String>,
    tokenizer: HfTokenizer,
}
impl TextDataset {
    /// Validates total record count and per-string byte limits before iteration.
    pub fn new(texts: Vec<String>, tokenizer: HfTokenizer) -> Result<Self> {
        tokenizer
            .limits
            .check("text records", texts.len(), tokenizer.limits.max_records)?;
        for t in &texts {
            tokenizer
                .limits
                .check("text bytes", t.len(), tokenizer.limits.max_string_bytes)?;
        }
        Ok(Self { texts, tokenizer })
    }
    /// Measures encoded lengths using exactly the same policy as [`Dataset::get`].
    pub fn lengths(&self) -> Result<Vec<usize>> {
        self.texts
            .iter()
            .map(|t| self.tokenizer.encode(t).map(|s| s.ids.len()))
            .collect()
    }
}
impl Dataset for TextDataset {
    fn get_batch_with_context(
        &self,
        indices: &[usize],
        context: &rusttorch_data::WorkerContext,
    ) -> Result<Vec<Self::Sample>> {
        indices
            .iter()
            .map(|&index| {
                context.check().map_err(|_| {
                    TextError::Invalid("sample loading cancelled or timed out".into())
                })?;
                self.get(index)
            })
            .collect()
    }

    type Sample = EncodedSequence;
    type Error = TextError;
    fn len(&self) -> usize {
        self.texts.len()
    }
    fn get(&self, index: usize) -> Result<EncodedSequence> {
        self.tokenizer.encode(
            self.texts
                .get(index)
                .ok_or_else(|| TextError::Invalid("text index out of range".into()))?,
        )
    }
}
/// Right-padding policy; long sequences are rejected, never silently truncated.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Padding {
    /// Pad to the longest sample in the batch; all-empty batches have width zero.
    Longest {
        /// Vocabulary padding ID.
        pad_id: u32,
    },
    /// Pad every batch to exactly this width.
    Fixed {
        /// Vocabulary padding ID.
        pad_id: u32,
        /// Required token width.
        length: usize,
    },
}
/// Model-ready integer tensors and original unpadded lengths.
#[derive(Debug)]
pub struct TextBatch {
    /// Padded `[batch,length]` Int64 vocabulary IDs.
    pub input_ids: Tensor,
    /// Int64 mask: one for valid tokens, zero for padding.
    pub attention_mask: Tensor,
    /// Padded Int64 segment IDs.
    pub token_type_ids: Tensor,
    /// Int64 mask for original special tokens; padded positions are zero.
    pub special_tokens_mask: Tensor,
    /// Original token lengths.
    pub lengths: Vec<usize>,
}
impl MemoryFootprint for TextBatch {
    fn resident_bytes(&self) -> usize {
        self.input_ids
            .resident_bytes()
            .saturating_add(self.attention_mask.resident_bytes())
            .saturating_add(self.token_type_ids.resident_bytes())
            .saturating_add(self.special_tokens_mask.resident_bytes())
            .saturating_add(
                self.lengths
                    .capacity()
                    .saturating_mul(std::mem::size_of::<usize>()),
            )
    }
}
impl PinMemory for TextBatch {
    fn pin_memory(mut self, d: Device) -> rusttorch_core::Result<Self> {
        self.input_ids = self.input_ids.pin_memory(d)?;
        self.attention_mask = self.attention_mask.pin_memory(d)?;
        self.token_type_ids = self.token_type_ids.pin_memory(d)?;
        self.special_tokens_mask = self.special_tokens_mask.pin_memory(d)?;
        Ok(self)
    }
}
/// Stateless deterministic collator for encoded sequences.
#[derive(Clone, Copy, Debug)]
pub struct TextCollator {
    padding: Padding,
    limits: ResourceLimits,
}
impl TextCollator {
    /// Validates the fixed padding width against the token ceiling.
    pub fn new(padding: Padding, limits: ResourceLimits) -> Result<Self> {
        if let Padding::Fixed { length, .. } = padding {
            limits.check("padding width", length, limits.max_tokens)?;
        }
        Ok(Self { padding, limits })
    }
}
impl Collate<EncodedSequence> for TextCollator {
    type Batch = TextBatch;
    type Error = TextError;
    fn collate(&mut self, samples: Vec<EncodedSequence>) -> Result<TextBatch> {
        if samples.is_empty() {
            return Err(TextError::Invalid(
                "cannot collate an empty text batch".into(),
            ));
        }
        self.limits
            .check("text batch records", samples.len(), self.limits.max_records)?;
        for s in &samples {
            s.validate(self.limits)?;
        }
        let longest = samples.iter().map(|s| s.ids.len()).max().unwrap_or(0);
        let (pad, width) = match self.padding {
            Padding::Longest { pad_id } => (pad_id, longest),
            Padding::Fixed { pad_id, length } => (pad_id, length),
        };
        if longest > width {
            return Err(TextError::Invalid("sample exceeds padding width".into()));
        }
        let n = self.limits.tensor_elements(&[samples.len(), width])?;
        self.limits
            .check("padded tokens", n, self.limits.max_tokens)?;
        self.limits.check(
            "text batch bytes",
            n.checked_mul(32)
                .and_then(|bytes| {
                    samples
                        .len()
                        .checked_mul(std::mem::size_of::<usize>())
                        .and_then(|lengths| bytes.checked_add(lengths))
                })
                .ok_or_else(|| TextError::Invalid("text batch byte overflow".into()))?,
            self.limits.max_decoded_bytes,
        )?;
        let mut ids = vec![pad as i64; n];
        let mut masks = vec![0i64; n];
        let mut types = vec![0i64; n];
        let mut specials = vec![0i64; n];
        let lengths = samples.iter().map(|s| s.ids.len()).collect();
        for (row, s) in samples.iter().enumerate() {
            for i in 0..s.ids.len() {
                let j = row * width + i;
                ids[j] = s.ids[i] as i64;
                masks[j] = 1;
                types[j] = s.type_ids[i] as i64;
                specials[j] = s.special_tokens_mask[i] as i64;
            }
        }
        let shape = [samples.len() as i64, width as i64];
        Ok(TextBatch {
            input_ids: Tensor::f_from_slice(&ids)?.f_reshape(shape)?,
            attention_mask: Tensor::f_from_slice(&masks)?.f_reshape(shape)?,
            token_type_ids: Tensor::f_from_slice(&types)?.f_reshape(shape)?,
            special_tokens_mask: Tensor::f_from_slice(&specials)?.f_reshape(shape)?,
            lengths,
        })
    }
}
impl rusttorch_data::Checkpointable for TextCollator {
    type State = (Padding, ResourceLimits);
    fn save_state(&self) -> Self::State {
        (self.padding, self.limits)
    }
    fn validate_state(&self, s: &Self::State) -> rusttorch_core::Result<()> {
        if *s != self.save_state() {
            return Err(invalid("text collator policy differs"));
        }
        Ok(())
    }
    fn load_validated(&mut self, _: &Self::State) {}
}

/// Token-budget and distributed-rank policy. Padded token cost is
/// `batch_size * longest_sequence`, and every sequence must fit by itself.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenBudgetConfig {
    /// Maximum padded tokens in a batch.
    pub max_tokens: usize,
    /// Maximum samples, including empty sequences, in a batch.
    pub max_samples: usize,
    /// Total distributed ranks.
    pub replicas: usize,
    /// This rank, starting at zero.
    pub rank: usize,
    /// Shuffle with the supplied seed and epoch.
    pub shuffle: bool,
    /// Base permutation seed.
    pub seed: u64,
    /// Drop the incomplete distributed tail instead of padding rank indices.
    pub drop_rank_tail: bool,
}
impl Default for TokenBudgetConfig {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            max_samples: 64,
            replicas: 1,
            rank: 0,
            shuffle: false,
            seed: 0,
            drop_rank_tail: false,
        }
    }
}
/// Checkpointable variable-size batches over measured sequence lengths.
/// Distributed ranks receive equal sample counts, but can receive different
/// batch counts; synchronized training must account for uneven batch counts.
pub struct TokenBudgetSampler {
    lengths: Vec<usize>,
    config: TokenBudgetConfig,
    sampler: DistributedSampler,
}
/// Versioned token-budget checkpoint containing immutable sequence lengths.
#[derive(Clone, Serialize, Deserialize)]
pub struct TokenBudgetState {
    version: u32,
    lengths: Vec<usize>,
    config: TokenBudgetConfig,
    sampler: DistributedSamplerState,
    next_batch: u64,
    next_sample: u64,
}
fn invalid(reason: &str) -> RustTorchError {
    RustTorchError::InvalidConfiguration {
        field: "token budget",
        reason: reason.into(),
    }
}
impl TokenBudgetSampler {
    /// Validates positive budgets and every measured length before batching.
    pub fn new(
        lengths: Vec<usize>,
        config: TokenBudgetConfig,
        limits: ResourceLimits,
    ) -> rusttorch_core::Result<Self> {
        limits.check("token lengths", lengths.len(), limits.max_records)?;
        limits.check("token budget", config.max_tokens, limits.max_tokens)?;
        if config.max_tokens == 0
            || config.max_samples == 0
            || lengths.iter().any(|&n| n > config.max_tokens)
        {
            return Err(invalid(
                "positive budgets required and every sequence must fit",
            ));
        }
        let sampler = DistributedSampler::new(
            lengths.len(),
            config.replicas,
            config.rank,
            config.shuffle,
            config.seed,
            config.drop_rank_tail,
        )?;
        Ok(Self {
            lengths,
            config,
            sampler,
        })
    }
    fn group(&self, indices: impl Iterator<Item = usize>) -> Vec<Vec<usize>> {
        let mut batches = Vec::new();
        let mut batch = Vec::new();
        let mut longest = 0;
        for index in indices {
            let next = longest.max(self.lengths[index]);
            if !batch.is_empty()
                && (batch.len() == self.config.max_samples
                    || next
                        .checked_mul(batch.len() + 1)
                        .is_none_or(|n| n > self.config.max_tokens))
            {
                batches.push(std::mem::take(&mut batch));
                longest = 0;
            }
            longest = longest.max(self.lengths[index]);
            batch.push(index);
        }
        if !batch.is_empty() {
            batches.push(batch);
        }
        batches
    }
    fn boundary(&self, batch: u64, sample: u64, epoch: u64) -> rusttorch_core::Result<()> {
        let mut sampler = DistributedSampler::new(
            self.lengths.len(),
            self.config.replicas,
            self.config.rank,
            self.config.shuffle,
            self.config.seed,
            self.config.drop_rank_tail,
        )?;
        sampler.set_epoch(epoch);
        let batches = self.group(sampler.iter());
        let b = usize::try_from(batch).map_err(|_| invalid("batch position overflow"))?;
        if b > batches.len() || batches[..b].iter().map(Vec::len).sum::<usize>() as u64 != sample {
            return Err(invalid("checkpoint is not a token-budget batch boundary"));
        }
        Ok(())
    }
}
impl BatchSource for TokenBudgetSampler {
    type Iter = std::vec::IntoIter<Vec<usize>>;
    fn iter(&self) -> Self::Iter {
        self.group(self.sampler.iter()).into_iter()
    }
    fn exact_len(&self) -> Option<usize> {
        Some(self.group(self.sampler.iter()).len())
    }
    fn epoch(&self) -> u64 {
        self.sampler.epoch()
    }
    fn set_epoch(&mut self, epoch: u64) {
        self.sampler.set_epoch(epoch);
    }
}
impl BatchSourceCheckpoint for TokenBudgetSampler {
    type State = TokenBudgetState;
    fn kind(&self) -> String {
        "token_budget/distributed/v1".into()
    }
    fn checkpoint_kind_from_state(_: &Self::State) -> String {
        "token_budget/distributed/v1".into()
    }
    fn checkpoint_state(&self, b: u64, s: u64) -> rusttorch_core::Result<Self::State> {
        self.boundary(b, s, self.sampler.epoch())?;
        Ok(TokenBudgetState {
            version: 1,
            lengths: self.lengths.clone(),
            config: self.config.clone(),
            sampler: self.sampler.checkpoint_state(s)?,
            next_batch: b,
            next_sample: s,
        })
    }
    fn validate_checkpoint_state(
        &self,
        state: &Self::State,
        epoch: u64,
        b: u64,
        s: u64,
    ) -> rusttorch_core::Result<()> {
        if state.version != 1
            || state.lengths != self.lengths
            || state.config != self.config
            || state.next_batch != b
            || state.next_sample != s
        {
            return Err(invalid("checkpoint token-budget configuration differs"));
        }
        self.sampler
            .validate_checkpoint_state(&state.sampler, epoch, s)?;
        self.boundary(b, s, epoch)
    }
    fn restore_checkpoint_iter_validated(&mut self, state: &Self::State) -> Self::Iter {
        let indices = self
            .sampler
            .restore_checkpoint_iter_validated(&state.sampler);
        self.group(indices).into_iter()
    }
    fn distributed_configuration(&self) -> Option<DistributedConfiguration> {
        self.sampler.distributed_configuration()
    }
    fn checkpoint_distributed_configuration_from_state(
        state: &Self::State,
    ) -> rusttorch_core::Result<Option<DistributedConfiguration>> {
        DistributedSampler::checkpoint_distributed_configuration_from_state(&state.sampler)
    }
}
