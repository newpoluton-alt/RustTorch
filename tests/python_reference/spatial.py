"""Emit deterministic CPU references for RustTorch's spatial layers.

Behavior reference: PyTorch v2.13.0, cf30153c4c131c8164ee7798e5022d810682e2cb,
torch/nn/modules/{batchnorm,instancenorm,normalization,conv,pooling,activation}.py.
No upstream source is copied; this script exercises the installed public API.
"""

import json
import math

import torch
from torch import nn


def values(tensor):
    return tensor.detach().flatten().tolist()


def input_tensor(shape):
    return (torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape) / 7 - 1).requires_grad_()


def gradients(output):
    factors = torch.arange(output.numel(), dtype=torch.float64).reshape(output.shape) / 10 + 0.2
    (output * factors).sum().backward()


def normalization(dim, instance, cumulative):
    shape = [2, 3] + [2] * dim
    layer_type = getattr(nn, f"{'Instance' if instance else 'Batch'}Norm{dim}d")
    layer = layer_type(3, affine=True, track_running_stats=True,
                       momentum=None if cumulative else 0.1, dtype=torch.float64)
    x = input_tensor(shape)
    first = layer(x)
    gradients(first)
    second = layer(x.detach() * 2 + 1)
    layer.eval()
    return {
        "first": values(first), "second": values(second), "eval": values(layer(x.detach())),
        "input_grad": values(x.grad), "weight_grad": values(layer.weight.grad),
        "bias_grad": values(layer.bias.grad),
        "running_mean": values(layer.running_mean), "running_var": values(layer.running_var),
        "count": layer.num_batches_tracked.item(),
    }


def transposed(dim):
    torch.manual_seed(700 + dim)
    layer = getattr(nn, f"ConvTranspose{dim}d")(
        2, 4, [2] * dim, stride=[2] * dim, padding=[1] * dim,
        output_padding=[1] * dim, dilation=[2] * dim, groups=2, dtype=torch.float64)
    initial_weight = values(layer.weight)
    initial_bias = values(layer.bias)
    with torch.no_grad():
        layer.weight.copy_(torch.arange(layer.weight.numel(), dtype=torch.float64).reshape(layer.weight.shape) / 11 - 0.2)
        layer.bias.copy_(torch.tensor([-0.2, -0.1, 0, 0.1], dtype=torch.float64))
    x = input_tensor([2, 2] + [3] * dim)
    output = layer(x)
    gradients(output)
    return {"output": values(output), "shape": list(output.shape),
            "initial_weight": initial_weight, "initial_bias": initial_bias,
            "input_grad": values(x.grad), "weight_grad": values(layer.weight.grad),
            "bias_grad": values(layer.bias.grad),
            "requested": values(layer(x.detach(), output_size=[5] * dim))}


def pooling(dim, kind):
    x = input_tensor([2, 2] + [5] * dim)
    if kind == "max":
        layer = getattr(nn, f"MaxPool{dim}d")([3] * dim, stride=[2] * dim, padding=[1] * dim,
                                             dilation=[2] * dim, ceil_mode=True, return_indices=True)
    elif kind == "avg":
        kwargs = {} if dim == 1 else {"divisor_override": 4}
        layer = getattr(nn, f"AvgPool{dim}d")([3] * dim, stride=[2] * dim, padding=[1] * dim,
                                             ceil_mode=True, count_include_pad=False, **kwargs)
    elif kind == "adaptive_max":
        layer = getattr(nn, f"AdaptiveMaxPool{dim}d")([2] * dim, return_indices=True)
    else:
        layer = getattr(nn, f"AdaptiveAvgPool{dim}d")([2] * dim)
    output = layer(x)
    indices = None
    if isinstance(output, tuple):
        output, indices = output
    gradients(output)
    result = {"output": values(output), "shape": list(output.shape), "input_grad": values(x.grad)}
    if indices is not None:
        result["indices"] = values(indices)
    return result


def group_norm():
    layer = nn.GroupNorm(3, 6, eps=1e-4, bias=False, dtype=torch.float64)
    x = input_tensor([2, 6, 2, 3])
    output = layer(x)
    gradients(output)
    return {"output": values(output), "input_grad": values(x.grad), "weight_grad": values(layer.weight.grad)}


def activation(layer):
    x = torch.tensor([-1000., -2., -0.1, 0., 0.2, 2., 1000.], dtype=torch.float64, requires_grad=True)
    output = layer(x)
    gradients(output)
    return {"output": values(output), "input_grad": values(x.grad)}


def main():
    assert torch.__version__.split("+", 1)[0] == "2.13.0", torch.__version__
    assert torch.version.git_version == "cf30153c4c131c8164ee7798e5022d810682e2cb"
    torch.set_num_threads(1)
    results = {"version": torch.__version__, "commit": torch.version.git_version}
    for dim in (1, 2, 3):
        for instance in (False, True):
            for cumulative in (False, True):
                results[f"{'instance' if instance else 'batch'}{dim}_{cumulative}"] = normalization(dim, instance, cumulative)
        results[f"transpose{dim}"] = transposed(dim)
        for kind in ("max", "avg", "adaptive_max", "adaptive_avg"):
            results[f"{kind}{dim}"] = pooling(dim, kind)
    results["group"] = group_norm()
    for name, layer in {
        "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "silu": nn.SiLU(),
        "softmax": nn.Softmax(-1), "log_softmax": nn.LogSoftmax(-1),
        "leaky_relu": nn.LeakyReLU(0.2), "elu": nn.ELU(1.2),
    }.items():
        results[name] = activation(layer)
    print(json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()
