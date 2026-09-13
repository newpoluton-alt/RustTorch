#!/usr/bin/env python3
"""Generate CPU numerical evidence for the explicitly scoped audio conventions.

Reference: PyTorch torch.stft/torch.matmul/torch.log and autograd at the exact
version and commit in compat/pytorch_reference.toml. Original deterministic
synthetic data; no external datasets, model weights or audio are downloaded.
"""
import argparse
import json
from pathlib import Path
import tomllib
import torch

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).with_name("fixtures") / "numerical.json"


def generate():
    reference = tomllib.loads((ROOT / "compat/pytorch_reference.toml").read_text())
    if torch.__version__.split("+")[0] != reference["pytorch_version"]:
        raise SystemExit("PyTorch version differs from the pinned reference")
    if torch.version.git_version != reference["pytorch_commit"]:
        raise SystemExit("PyTorch commit differs from the pinned reference")
    torch.set_num_threads(1)
    x = torch.tensor([[0., .25, -.125, .5, .75, -.5, .125, .25,
                       -.25, .375, .125, -.625, .75, .5, -.25, .125]],
                     dtype=torch.float64, requires_grad=True)
    n_fft, hop, rate, n_mels, n_mfcc = 8, 4, 8000, 3, 2
    power = torch.stft(x, n_fft, hop, n_fft, torch.hann_window(n_fft, dtype=x.dtype),
                       center=False, normalized=False, onesided=True,
                       return_complex=True).abs().pow(2)
    max_mel = 2595. * torch.log10(torch.tensor(1. + rate / 1400., dtype=x.dtype))
    mel_points = torch.linspace(0., max_mel, n_mels + 2, dtype=x.dtype)
    hz = 700. * (torch.pow(10., mel_points / 2595.) - 1.)
    frequencies = torch.arange(n_fft // 2 + 1, dtype=x.dtype) * rate / n_fft
    filters = torch.stack([torch.minimum((frequencies-hz[m])/(hz[m+1]-hz[m]),
                                         (hz[m+2]-frequencies)/(hz[m+2]-hz[m+1])).clamp_min(0)
                           for m in range(n_mels)])
    mel = filters @ power
    k = torch.arange(n_mfcc, dtype=x.dtype).unsqueeze(1)
    n = torch.arange(n_mels, dtype=x.dtype).unsqueeze(0)
    dct = torch.cos(torch.pi * k * (n + .5) / n_mels) * (2./n_mels)**.5
    dct[0] /= 2.**.5
    mfcc = dct @ mel.clamp_min(1e-10).log()
    mfcc.sum().backward()

    def values(t):
        return [round(v, 10) for v in t.detach().reshape(-1).tolist()]
    data = {"pytorch_version": reference["pytorch_version"],
            "pytorch_commit": reference["pytorch_commit"], "input": values(x),
            "power": values(power), "mel": values(mel), "mfcc": values(mfcc),
            "gradient": values(x.grad), "filter": values(filters)}
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    contents = generate()
    if args.check:
        if OUTPUT.read_text() != contents:
            raise SystemExit("audio numerical fixture is stale")
        print("audio numerical fixture matches pinned CPU reference")
    else:
        OUTPUT.write_text(contents)
