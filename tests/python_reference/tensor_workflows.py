"""Original CPU fixtures using pinned PyTorch 2.13.0 public operations.

Source API scope: torch/indexing, torch.linalg, torch.fft, torch.special,
torch.sparse and quantization at v2.13.0 (cf30153). No upstream code is copied.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from common import write_json


def generate(directory: Path) -> None:
    if torch.__version__.split("+", 1)[0] != "2.13.0":
        raise RuntimeError("tensor workflow parity requires PyTorch 2.13.0")
    out = {}
    x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([2, 0, 2])
    out["select"] = x.index_select(0, ids)
    out["gather"] = x.gather(1, torch.tensor([[1], [0], [1]])).squeeze(1)
    out["scatter"] = torch.zeros_like(x).index_add(0, ids, x)
    out["masked"] = torch.where(x > 3., torch.zeros_like(x), x)
    out["standardize"] = (x - x.mean(0, keepdim=True)) / x.std(0, correction=0, keepdim=True).clamp_min(1e-8)
    out["flatten"] = x.reshape(1, 3, 2).transpose(1, 2).reshape(1, 6).clone()
    out["select"].square().sum().backward()
    out["select_grad"] = x.grad
    a = torch.tensor([[4., 1.], [1., 3.]], dtype=torch.float64)
    out["solve"] = torch.linalg.solve(a, torch.tensor([[1.], [2.]], dtype=torch.float64))
    out["det"] = torch.linalg.det(a)
    out["svd_values"] = torch.linalg.svd(a).S
    out["eigenvalues"] = torch.linalg.eigh(a).eigenvalues
    out["norm"] = torch.linalg.norm(a, ord=2, dim=(0, 1))
    design = torch.tensor([[1., 0.], [1., 1.], [1., 2.]], dtype=torch.float64)
    fit = torch.linalg.lstsq(design, torch.tensor([[1.], [2.5], [5.5]], dtype=torch.float64), driver="gelsd")
    out.update(lstsq=fit.solution, lstsq_residuals=fit.residuals, lstsq_rank=fit.rank, lstsq_singular_values=fit.singular_values)
    signal = torch.tensor([1., 2., -1., 0., 3.], dtype=torch.float64, requires_grad=True)
    spectrum = torch.fft.rfft(signal)
    out["rfft"] = torch.view_as_real(spectrum)
    filtered = torch.fft.irfft(spectrum * torch.tensor([1., 0.5, 0.], dtype=torch.float64), n=5)
    filtered.square().sum().backward()
    out["filtered"] = filtered
    out["filter_grad"] = signal.grad
    fft = torch.fft.fft(torch.complex(signal, signal * 0.5), norm="ortho")
    out["fft"] = torch.view_as_real(fft)
    out["ifft"] = torch.view_as_real(torch.fft.ifft(fft, norm="ortho"))
    z = torch.tensor([-3., 0., 2.], dtype=torch.float64)
    out["normal_cdf"] = torch.special.ndtr(z)
    out["log_normal_cdf"] = torch.special.log_ndtr(z)
    out["log_gamma"] = torch.special.gammaln(torch.tensor([0.5, 1., 3.], dtype=torch.float64))
    coords = torch.tensor([[0, 0, 1], [0, 0, 1]])
    values = torch.tensor([1., 2., 4.], dtype=torch.float64)
    sparse = torch.sparse_coo_tensor(coords, values, (2, 2), check_invariants=True).coalesce()
    differentiable_source = torch.tensor([[3., 0.], [0., 4.]], dtype=torch.float64, requires_grad=True)
    result = torch.sparse.mm(differentiable_source.to_sparse(), a)
    result.square().sum().backward()
    out["sparse_mm"] = result
    out["sparse_grad"] = differentiable_source.grad
    out["csr_mm"] = torch.sparse.mm(sparse.to_sparse_csr(), a)
    out["sparse_dense"] = sparse.to_dense()
    q = torch.quantize_per_tensor(torch.tensor([-100., -0.25, 0., 0.25, 100.]), 0.1, 0, torch.qint8)
    out["quantized"] = q.int_repr()
    out["dequantized"] = q.dequantize()
    write_json(directory / "tensor_workflows.json", {name: value.detach().double().reshape(-1).tolist() for name, value in out.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    generate(parser.parse_args().output_dir)
