"""Pinned PyTorch 2.13 loss/gradient/scaler fixtures for RustTorch training APIs."""
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
from common import tensor_data, write_json


def main(directory: Path) -> None:
    results = {}
    values = [-1.5, 0.25, 2.0, 4.0]
    targets = [0.5, -0.25, 1.0, 2.0]
    for name, function in {
        "mse_none": lambda x, t: F.mse_loss(x, t, reduction="none"),
        "l1_mean": lambda x, t: F.l1_loss(x, t),
        "smooth_l1": lambda x, t: F.smooth_l1_loss(x, t, beta=0.75, reduction="sum"),
        "huber": lambda x, t: F.huber_loss(x, t, delta=1.5, reduction="sum"),
    }.items():
        x = torch.tensor(values, requires_grad=True)
        loss = function(x, torch.tensor(targets))
        loss.sum().backward()
        results[name] = {"loss": tensor_data(loss), "grad": tensor_data(x.grad)}
    for name, probabilities in (("bce", True), ("bce_logits", False)):
        x = torch.tensor([0.2, 0.7, 0.9] if probabilities else [-1.0, 0.5, 2.0], requires_grad=True)
        target = torch.tensor([0., 1., 0.])
        weight = torch.tensor([1., 2., 0.5])
        if probabilities:
            loss = F.binary_cross_entropy(x, target, weight, reduction="sum")
        else:
            loss = F.binary_cross_entropy_with_logits(x, target, weight, pos_weight=torch.tensor([2.]), reduction="sum")
        loss.backward()
        results[name] = {"loss": tensor_data(loss), "grad": tensor_data(x.grad)}
    for name in ("ce", "ce_probabilities", "nll"):
        x = torch.tensor([[1., -1., 2.], [0.5, 1.5, -0.5], [-2., 1., 0.]], requires_grad=True)
        classes = torch.tensor([2, -1, 0])
        weights = torch.tensor([0.5, 2., 1.])
        if name == "ce":
            loss = F.cross_entropy(x, classes, weights, ignore_index=-1, label_smoothing=0.15)
        elif name == "ce_probabilities":
            target = torch.tensor([[0.1, 0.2, 0.7], [0., 0.5, 0.5], [1., 0., 0.]])
            loss = F.cross_entropy(x, target, weights, label_smoothing=0.1, reduction="sum")
        else:
            loss = F.nll_loss(F.log_softmax(x, -1), classes, weights, ignore_index=-1)
        loss.backward()
        results[name] = {"loss": tensor_data(loss), "grad": tensor_data(x.grad)}

    scales = {}
    for name, options in (("scaler", dict(init_scale=8., growth_interval=2)),
                          ("scaler_custom", dict(init_scale=3.3, growth_interval=1,
                                                 growth_factor=1.1, backoff_factor=0.7))):
        p = torch.nn.Parameter(torch.tensor([2.]))
        optimizer = torch.optim.SGD([p], lr=0.1)
        scaler = torch.amp.GradScaler("cpu", **options)
        steps = []
        for nonfinite in (False, False, True, False):
            optimizer.zero_grad()
            loss = p.square().sum() if not nonfinite else (p * float("inf")).sum()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            steps.append({"parameter": p.item(), "scale": scaler.get_scale(), "tracker": scaler.state_dict()["_growth_tracker"]})
        scales[name] = steps
    write_json(directory / "training.json", {"losses": results, **scales})


if __name__ == "__main__":
    main(Path(sys.argv[1]))
