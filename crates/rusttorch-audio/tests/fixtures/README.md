# Audio fixture provenance

All fixtures are original synthetic test data contributed under MIT OR Apache-2.0. No external dataset records, personal data, pretrained weights or vendored native binaries are included.

- `mono.wav`: Python standard-library `wave` and `struct` output: mono, 8,000 Hz, signed little-endian PCM16; samples `[0,8192,-8192,16384]` repeated four times. Their normalized values are exactly `[0,0.25,-0.25,0.5]`.
- `mono.flac`: losslessly encoded from that WAV using FFmpeg 8.1.1 (`ffmpeg -i mono.wav -c:a flac mono.flac`). Tests compare both decoded waveforms exactly.
- `numerical.json`: generated from original deterministic values by `../generate_reference.py`. The script verifies the exact version and commit in `compat/pytorch_reference.toml` before computing CPU Double STFT, HTK Mel, orthonormal log-Mel MFCCs and input gradients. It rounds flattened outputs to ten decimal places and writes sorted JSON.

To verify the numerical fixture, configure the project environment and run `python crates/rusttorch-audio/tests/generate_reference.py --check` from the repository root. To regenerate it after an intentional reference change, omit `--check` and review the result with the pinned reference update.

## Checked-in SHA-256 values

| File | SHA-256 |
|---|---|
| `mono.flac` | `8997803558e96f69a3387f9d85d95b061f098e7db74a5190ab52446f60687261` |
| `mono.wav` | `0d907a6ea13be3df7ed0eda7015853a6d964ed40d74b8bc1d0fa2c5c5221a67a` |
| `numerical.json` | `58380e5720ddeb1bde917479553c44ccb5779f29cd2e896b2289450b0717f4b2` |
