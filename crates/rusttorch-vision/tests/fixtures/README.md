# Vision fixture provenance

All fixtures are original synthetic test data contributed under MIT OR Apache-2.0. No external dataset records, personal data, pretrained weights or vendored native binaries are included.

- `rgb.png`: a 2×2 RGB8 PNG generated using Python's `struct` and `zlib`, with rows red/green and blue/white, no filtering, and standard IHDR/IDAT/IEND chunks.
- `rgb.jpg`: JPEG encoded from `rgb.png` with FFmpeg 8.1.1 (`ffmpeg -i rgb.png rgb.jpg`). JPEG decoding is tested with shape/range expectations, not exact lossy bytes.
- `images.idx`: big-endian IDX image header `(2051, 2, 2, 2)` followed by bytes `[0,64,128,255,255,128,64,0]`.
- `labels.idx`: big-endian IDX label header `(2049, 2)` followed by labels `[3,8]`.
- `cifar10.cifar`: one CIFAR-10 binary-format record: label 2 followed by 1,024 red bytes of 10, green bytes of 20 and blue bytes of 30.
- `cifar100.cifar`: one CIFAR-100 binary-format record: coarse label 1, fine label 17 and the same RGB planes. The `.cifar` suffix avoids the repository's model-artifact `.bin` exclusion; the format is unchanged.
- `coco.json`: one original COCO-style image record referring to `rgb.png`, one category-4 box and one visible keypoint. No upstream COCO annotation is copied.

The IDX and CIFAR headers exercise their public file layouts. These tiny records are not samples from MNIST, CIFAR or COCO training datasets.

## Checked-in SHA-256 values

| File | SHA-256 |
|---|---|
| `cifar10.cifar` | `67f70182d8a73a80098137a13c5d4f70ac965a426d0970cb5b5e66b30935e6af` |
| `cifar100.cifar` | `77046eba433af8b6e47bbea27abcdc308af05b2479de9b46de3a9d1177cd5c44` |
| `coco.json` | `8af2e5ad69944ed08945af7a5f3b0310f4cb2eb422b21e5037d8f99f0cada305` |
| `images.idx` | `77910d91376df113a1ee707eeea7392599c7e7a9f422569499e40f874283d673` |
| `labels.idx` | `9367b1f1935f1f124248bd3ba34da0805bf19249aa97ffc062709c026fdcb80a` |
| `rgb.jpg` | `101bda429469eeaf7fd3eaf358dff64d85c497e3551e248408860adf75a1c470` |
| `rgb.png` | `e6d66889131220f931fddfb05730d647a0992456c63ae0a8154b4ae32ff219ef` |
