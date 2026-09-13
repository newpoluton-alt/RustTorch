# Codec fixture provenance

All fixtures are original synthetic test data contributed under MIT OR Apache-2.0. No external dataset records, personal data, pretrained weights or vendored native binaries are included.

- `mono.wav`: the same original 16-frame mono 8,000 Hz PCM16 waveform used by `rusttorch-audio`; values `[0,8192,-8192,16384]` repeated four times.
- `tiny.mkv`: original FFmpeg 8.1.1 synthetic output, created with:

```sh
ffmpeg -f lavfi -i color=c=red:s=8x8:r=4:d=0.75 \
  -f lavfi -i sine=frequency=440:sample_rate=8000:duration=0.75 \
  -c:v ffv1 -pix_fmt bgr0 -c:a pcm_s16le -shortest tiny.mkv
```

The container contains three 8×8 red FFV1 frames at 4 fps and 6,000 PCM audio frames. Container identifiers may differ when regenerated; tests assert decoded contents, timestamps and seek behavior. Encoder binaries are not redistributed. The decoder tests also pass with the checksummed FFmpeg 8.1.2 LGPL-only shared-library profile built by `scripts/build-ffmpeg.py`.

## Checked-in SHA-256 values

| File | SHA-256 |
|---|---|
| `mono.wav` | `0d907a6ea13be3df7ed0eda7015853a6d964ed40d74b8bc1d0fa2c5c5221a67a` |
| `tiny.mkv` | `3b54e891e5752373db815c3bc4fdcbf0e9775636ffbd341618686bdf365ef67c` |
