use rusttorch_data::{BatchSource, BatchSourceCheckpoint, Collate, DataLoader, ResourceLimits};
use rusttorch_text::*;
fn tokenizer() -> HfTokenizer {
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(
            [
                ("[UNK]".into(), 0),
                ("hello".into(), 1),
                ("world".into(), 2),
            ]
            .into_iter()
            .collect(),
        )
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut t = tokenizers::Tokenizer::new(model);
    t.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
    ));
    HfTokenizer::new(t, Truncation::Reject(8), false, Default::default()).unwrap()
}
#[test]
fn tokenizer_to_worker_loader_produces_ids_masks_and_budget_batches() {
    let dataset = TextDataset::new(
        vec!["hello world".into(), "hello".into(), "unknown".into()],
        tokenizer(),
    )
    .unwrap();
    let lengths = dataset.lengths().unwrap();
    assert_eq!(lengths, [2, 1, 1]);
    let sampler = TokenBudgetSampler::new(
        lengths,
        TokenBudgetConfig {
            max_tokens: 4,
            max_samples: 2,
            ..Default::default()
        },
        Default::default(),
    )
    .unwrap();
    let mut loader = DataLoader::builder(dataset)
        .batch_sampler(sampler)
        .collate(TextCollator::new(Padding::Longest { pad_id: 0 }, Default::default()).unwrap())
        .workers(2)
        .build()
        .unwrap();
    let first = loader.iter().next().unwrap().unwrap();
    assert_eq!(
        Vec::<Vec<i64>>::try_from(&first.input_ids).unwrap(),
        [[1, 2], [1, 0]]
    );
    assert_eq!(
        Vec::<Vec<i64>>::try_from(&first.attention_mask).unwrap(),
        [[1, 1], [1, 0]]
    );
}
#[test]
fn padding_policies_and_limits_reject_invalid_sequences() {
    let mut collate = TextCollator::new(
        Padding::Fixed {
            pad_id: 9,
            length: 2,
        },
        Default::default(),
    )
    .unwrap();
    assert!(
        collate
            .collate(vec![EncodedSequence::from_ids(vec![1, 2, 3])])
            .is_err()
    );
    let mut collate = TextCollator::new(
        Padding::Longest { pad_id: 0 },
        ResourceLimits {
            max_tokens: 3,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(
        collate
            .collate(vec![
                EncodedSequence::from_ids(vec![1, 2]),
                EncodedSequence::from_ids(vec![3])
            ])
            .is_err()
    );
    let mut broken = EncodedSequence::from_ids(vec![1]);
    broken.type_ids.clear();
    assert!(collate.collate(vec![broken]).is_err());
    assert!(collate.collate(Vec::new()).is_err());
    let b = collate
        .collate(vec![EncodedSequence::from_ids(vec![])])
        .unwrap();
    assert_eq!(b.input_ids.size(), [1, 0]);
}
#[test]
fn token_budget_resume_replays_only_remaining_complete_batches() {
    let lengths = vec![2, 3, 1, 4, 2, 1];
    let cfg = TokenBudgetConfig {
        max_tokens: 6,
        max_samples: 3,
        shuffle: true,
        seed: 4,
        ..Default::default()
    };
    let mut sampler =
        TokenBudgetSampler::new(lengths.clone(), cfg.clone(), Default::default()).unwrap();
    sampler.set_epoch(3);
    let batches = sampler.iter().collect::<Vec<_>>();
    for b in &batches {
        assert!(b.len() * b.iter().map(|&i| lengths[i]).max().unwrap() <= 6);
    }
    let state = sampler
        .checkpoint_state(1, batches[0].len() as u64)
        .unwrap();
    let serialized = serde_json::to_string(&state).unwrap();
    let state: TokenBudgetState = serde_json::from_str(&serialized).unwrap();
    sampler
        .validate_checkpoint_state(&state, 3, 1, batches[0].len() as u64)
        .unwrap();
    assert_eq!(
        sampler
            .restore_checkpoint_iter_validated(&state)
            .collect::<Vec<_>>(),
        batches[1..]
    );
    let mut fresh =
        TokenBudgetSampler::new(lengths.clone(), cfg.clone(), Default::default()).unwrap();
    fresh
        .validate_checkpoint_state(&state, 3, 1, batches[0].len() as u64)
        .unwrap();
    assert_eq!(
        fresh
            .restore_checkpoint_iter_validated(&state)
            .collect::<Vec<_>>(),
        batches[1..]
    );
    assert!(
        sampler
            .validate_checkpoint_state(&state, 3, 1, 999)
            .is_err()
    );
    let other = TokenBudgetSampler::new(vec![1; 6], cfg, Default::default()).unwrap();
    assert!(
        other
            .validate_checkpoint_state(&state, 3, 1, batches[0].len() as u64)
            .is_err()
    );
}
#[test]
fn distributed_token_batches_preserve_disjoint_rank_indices() {
    let lengths = vec![1; 8];
    let mut all = Vec::new();
    for rank in 0..2 {
        let config = TokenBudgetConfig {
            max_tokens: 2,
            max_samples: 2,
            replicas: 2,
            rank,
            ..Default::default()
        };
        let sampler = TokenBudgetSampler::new(lengths.clone(), config, Default::default()).unwrap();
        all.extend(sampler.iter().flatten());
    }
    all.sort();
    assert_eq!(all, (0..8).collect::<Vec<_>>());
}

#[test]
fn empty_sequences_still_obey_batch_record_and_length_byte_limits() {
    for limits in [
        ResourceLimits {
            max_records: 1,
            ..Default::default()
        },
        ResourceLimits {
            max_decoded_bytes: std::mem::size_of::<usize>(),
            ..Default::default()
        },
    ] {
        let mut collator = TextCollator::new(Padding::Longest { pad_id: 0 }, limits).unwrap();
        assert!(
            collator
                .collate(vec![EncodedSequence::from_ids(vec![]); 2])
                .is_err()
        );
    }
}
