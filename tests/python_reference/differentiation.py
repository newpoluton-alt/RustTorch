"""Original CPU reference fixtures for pinned 2.13.0 functional AD/distributions."""
from pathlib import Path
import sys
import torch
from common import tensor_data, write_json

assert torch.__version__.split('+')[0] == '2.13.0'
result = {}
x = torch.tensor([0.2, -0.7, 1.1], dtype=torch.float64, requires_grad=True)
v = torch.tensor([0.3, 0.4, -0.2], dtype=torch.float64)
f = lambda x: torch.stack([x.sin().sum(), (x.square() * x.exp()).sum()])
result['jacobian'] = tensor_data(torch.autograd.functional.jacobian(f, x))
result['jvp'] = tensor_data(torch.autograd.functional.jvp(f, x, v)[1])
result['vjp'] = tensor_data(torch.autograd.functional.vjp(f, x, torch.tensor([0.5, -0.8], dtype=x.dtype))[1])
result['hessian'] = tensor_data(torch.autograd.functional.hessian(lambda x: (x.sin() * x.roll(1)).sum(), x))
first = torch.autograd.grad(x.pow(4).sum(), x, create_graph=True)[0]
second = torch.autograd.grad(first.sum(), x, create_graph=True)[0]
result['third'] = tensor_data(torch.autograd.grad(second.sum(), x)[0])
mean = torch.tensor([0.1, -0.3], dtype=torch.float64, requires_grad=True)
scale = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
normal = torch.distributions.Normal(mean, scale)
torch.manual_seed(847)
draws = normal.rsample((3,))
result['normal_draws'] = tensor_data(draws)
result['normal_draw_grads'] = [tensor_data(g) for g in torch.autograd.grad(draws.square().sum(), (mean, scale))]
result['normal_scores'] = tensor_data(normal.log_prob(torch.tensor([[0.4], [-1.2]], dtype=mean.dtype)))
result['normal_entropy'] = tensor_data(normal.entropy())
for family in ('bernoulli', 'categorical'):
    for parameter in ('probs', 'logits'):
        values = [[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]] if parameter == 'probs' else [[-1.2, 0.3, 2.1], [0.1, 0.7, -0.2]]
        p = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        cls = torch.distributions.Bernoulli if family == 'bernoulli' else torch.distributions.Categorical
        distribution = cls(**{parameter:p})
        observations = torch.tensor([1.,0.,1.], dtype=p.dtype) if family == 'bernoulli' else torch.tensor([[0],[2]])
        scores = distribution.log_prob(observations)
        entropy = distribution.entropy()
        torch.manual_seed(847)
        samples = distribution.sample((3,))
        result[family+'_'+parameter] = {'scores':tensor_data(scores), 'entropy':tensor_data(entropy), 'samples':tensor_data(samples), 'grad':tensor_data(torch.autograd.grad((scores+entropy).sum(), p)[0])}
write_json(Path(sys.argv[1]) / 'differentiation.json', result)
