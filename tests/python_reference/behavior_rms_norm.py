"""Paired token-normalization case for the pinned RMSNorm implementation."""

import torch


def test_trainable_noncontiguous_token_features(device):
    norm = torch.nn.RMSNorm([2, 3], dtype=torch.float32, device=device)
    with torch.no_grad():
        norm.weight.copy_(torch.arange(1, 7, dtype=torch.float32, device=device).reshape(2, 3) / 4)
    x = torch.arange(1, 13, dtype=torch.float32, device=device).reshape(2, 3, 2).transpose(1, 2)
    x.requires_grad_()
    assert not x.is_contiguous()
    output = norm(x)
    torch.testing.assert_close(output, torch.nn.functional.rms_norm(x, [2, 3], norm.weight))
    output.sum().backward()
    for gradient in (x.grad, norm.weight.grad):
        assert gradient.device == output.device
        assert gradient.dtype == output.dtype and gradient.layout == output.layout
    norm.reset_parameters()

    def values(tensor):
        return tensor.detach().flatten().cpu().tolist()

    return {
        "device": output.device.type, "dtype": str(output.dtype).removeprefix("torch."),
        "layout": str(output.layout).removeprefix("torch."),
        "metadata": {"shape": list(output.shape), "input_stride": list(x.stride()),
                     "output_stride": list(output.stride()), "input_gradient_stride": list(x.grad.stride()),
                     "weight_shape": list(norm.weight.shape)},
        "values": {"output": values(output), "input_gradient": values(x.grad),
                   "weight_gradient": values(norm.weight.grad), "reset_weight": values(norm.weight)},
    }
