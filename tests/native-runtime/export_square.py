"""Compile a CPU package for the native AOTInductor loader probe."""

import sys

import torch


class Square(torch.nn.Module):
    def forward(self, value):
        return value.square()


if __name__ == "__main__":
    (path,) = sys.argv[1:]
    if torch.version.git_version != "cf30153c4c131c8164ee7798e5022d810682e2cb":
        raise RuntimeError("export requires the pinned PyTorch 2.13.0 build")
    exported = torch.export.export(Square(), (torch.ones(3, dtype=torch.float32),))
    torch._inductor.aoti_compile_and_package(exported, package_path=path)
