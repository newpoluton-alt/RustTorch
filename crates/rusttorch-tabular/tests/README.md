# Tabular fixture provenance

`pipelines.rs` defines original short CSV and JSONL records as source literals. Feature-gated tests create temporary Arrow IPC and Parquet files using the pinned upstream Rust writers, then read them through the public RustTorch adapters. The temporary files are removed after each test. This exercises real file framing, nulls, fitted statistics, category IDs and resource ceilings without shipping external dataset records. All synthetic records and generation code are contributed under MIT OR Apache-2.0.
