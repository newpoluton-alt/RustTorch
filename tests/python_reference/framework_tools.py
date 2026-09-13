"""Pinned numerical reference for tensor comparisons and derivative checks."""
import json
import torch

assert torch.__version__.split('+')[0] == '2.13.0'

actual = torch.tensor([[1., float('nan')], [float('inf'), -float('inf')]], dtype=torch.float64)
expected = torch.tensor([[1.1, float('nan')], [float('inf'), float('inf')]], dtype=torch.float64)
reports = []
for equal_nan in [False, True]:
    mask = ~torch.isclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=equal_nan)
    reports.append({'mismatches': int(mask.sum()), 'first': mask.nonzero()[0].tolist()})

class IncorrectDerivative(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.square()

    @staticmethod
    def backward(ctx, gradient):
        return gradient

x = torch.tensor([[.2, -.4], [.1, .3]], dtype=torch.float64).T.requires_grad_(True)
settings = dict(eps=1e-6, atol=1e-5, rtol=1e-3, raise_exception=False)
good = torch.autograd.gradcheck(torch.tanh, (x,), **settings)
bad = torch.autograd.gradcheck(IncorrectDerivative.apply, (x,), **settings)
print(json.dumps({'comparisons': reports, 'gradcheck_good': good, 'gradcheck_bad': bad}))
