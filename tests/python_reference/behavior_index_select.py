"""Reference for indexed selection at the pinned PyTorch commit."""

import torch


def test_repeated_rows_backward(device):
    x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32, device=device, requires_grad=True)
    rows = torch.tensor([2, 0, 2], dtype=torch.int64, device=device)
    selected = torch.index_select(x, 0, rows)
    selected.square().sum().backward()
    grad = x.grad
    assert grad.device == selected.device and grad.dtype == selected.dtype
    assert grad.layout == selected.layout
    return {
        "device": selected.device.type,
        "dtype": str(selected.dtype).removeprefix("torch."),
        "layout": str(selected.layout).removeprefix("torch."),
        "metadata": {
            "shape": list(selected.shape),
            "stride": list(selected.stride()),
            "gradient_shape": list(grad.shape),
            "gradient_stride": list(grad.stride()),
        },
        "values": {
            "selected": selected.detach().flatten().cpu().tolist(),
            "gradient": grad.flatten().cpu().tolist(),
        },
    }


def test_rejects_float_indices(device):
    x = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device)
    rows = torch.tensor([1.0], dtype=torch.float32, device=device)
    try:
        torch.index_select(x, 0, rows)
    except RuntimeError as error:
        if "dtype" not in str(error) or "index" not in str(error):
            raise
    else:
        raise AssertionError("index_select accepted float indices")
    return {
        "device": x.device.type, "dtype": str(x.dtype).removeprefix("torch."),
        "layout": str(x.layout).removeprefix("torch."),
        "metadata": {"error": "index_dtype", "shape": list(x.shape)},
        "values": {"input_after_error": x.cpu().tolist()},
    }
