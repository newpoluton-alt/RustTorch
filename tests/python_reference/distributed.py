"""Original CPU/Gloo fixtures for PyTorch 2.13.0 at cf30153.

Exercises real spawned processes, native collectives, DDP and one-group FSDP.
No Python process participates in RustTorch runtime execution.
"""
from datetime import timedelta
from pathlib import Path
import multiprocessing as mp
import sys
import tempfile
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.fsdp import FullyShardedDataParallel

from common import write_json

assert torch.__version__.split("+")[0] == "2.13.0"


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor([0.2], dtype=torch.float64))
        self.layer = torch.nn.Linear(3, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.layer.weight.copy_(torch.tensor([[0.4, -0.7, 0.1]], dtype=torch.float64))

    def forward(self, x):
        return self.layer(x).squeeze(-1) + self.bias


def batch(rank, world, step):
    count = 6 // world
    data, target = [], []
    for index in range(rank * count, (rank + 1) * count):
        x = index * 0.2 + step * 0.01
        data.append([x, 1 - x, x * x])
        target.append(0.6 * x - 0.4)
    return torch.tensor(data, dtype=torch.float64), torch.tensor(target, dtype=torch.float64)


def parameters(model):
    return {name: tensor.detach().flatten().tolist() for name, tensor in model.named_parameters()}


def worker(rank, world, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=Path(rendezvous).as_uri(),
                            rank=rank, world_size=world, timeout=timedelta(seconds=20))
    try:
        result = {}
        if world == 3:
            value = torch.tensor([rank + 1., rank + 2.], dtype=torch.float64)
            for name, op in (("sum", dist.ReduceOp.SUM), ("min", dist.ReduceOp.MIN),
                             ("max", dist.ReduceOp.MAX), ("product", dist.ReduceOp.PRODUCT)):
                reduced = value.clone()
                dist.all_reduce(reduced, op)
                result[name] = reduced.tolist()
            result["mean"] = (torch.tensor(result["sum"], dtype=torch.float64) / world).tolist()
            broadcast = value.clone()
            dist.broadcast(broadcast, 2)
            result["broadcast"] = broadcast.tolist()
            gathered = [torch.empty_like(value) for _ in range(world)]
            dist.all_gather(gathered, value)
            result["all_gather"] = [t.tolist() for t in gathered]
            reduced = torch.empty(1, dtype=torch.float64)
            inputs = torch.tensor([rank + 1. + r for r in range(world)], dtype=torch.float64)
            dist.reduce_scatter_single(reduced, inputs)
            result["reduce_scatter"] = reduced.tolist()
            inputs = torch.tensor([10. * rank + r for r in range(world)], dtype=torch.float64)
            exchanged = torch.empty_like(inputs)
            dist.all_to_all_single(exchanged, inputs)
            result["all_to_all"] = exchanged.tolist()
        else:
            model = Model()
            replica = DistributedDataParallel(model)
            optimizer = torch.optim.Adam(replica.parameters(), lr=0.03, amsgrad=True)
            for step in range(4):
                x, target = batch(rank, world, step)
                optimizer.zero_grad()
                (replica(x) - target).square().mean().backward()
                optimizer.step()
            result["ddp"] = parameters(model)
            model = Model()
            sharded = FullyShardedDataParallel(model, device_id=torch.device("cpu"))
            optimizer = torch.optim.Adam(sharded.parameters(), lr=0.03, amsgrad=True)
            for step in range(4):
                x, target = batch(rank, world, step)
                optimizer.zero_grad()
                (sharded(x) - target).square().mean().backward()
                optimizer.step()
            with FullyShardedDataParallel.summon_full_params(sharded):
                result["fsdp"] = parameters(model)
        write_json(Path(output) / f"rank-{rank}.json", result)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def generate(world, directory):
    import json
    directory.mkdir(parents=True)
    context = mp.get_context("spawn")
    processes = [context.Process(target=worker, args=(rank, world, str(directory / "rendezvous"), str(directory))) for rank in range(world)]
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + 60
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
            if process.is_alive() or process.exitcode != 0:
                raise RuntimeError(f"Gloo reference worker failed or timed out: {process.exitcode}")
        return [json.loads((directory / f"rank-{rank}.json").read_text()) for rank in range(world)]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(5)
                if process.is_alive():
                    process.kill()
                    process.join()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="rusttorch-gloo-reference-") as temporary:
        root = Path(temporary)
        result = {"collectives": generate(3, root / "collectives"),
                  "training": generate(2, root / "training")}
        write_json(Path(sys.argv[1]) / "distributed.json", result)
