"""Pinned-runtime numerical reference for RustTorch optimizer and scheduler state.

Run with the project's locked Python environment. Only generated tiny tensors
are used; output is JSON, with no pickle or external data.
"""
import json
import torch


def optimizer_case(name):
    p = torch.tensor([0.4, -0.7, 1.2], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([-0.2, 0.5], dtype=torch.float64, requires_grad=True)
    groups = [dict(params=[p], lr=0.03, weight_decay=0.02),
              dict(params=[q], lr=0.012, weight_decay=0.0)]
    options = {
        'adam': (torch.optim.Adam, dict(betas=(0.8, 0.91), eps=1e-6, amsgrad=True)),
        'adamw': (torch.optim.AdamW, dict(betas=(0.8, 0.91), eps=1e-6, amsgrad=True)),
        'sgd': (torch.optim.SGD, dict(momentum=0.8, nesterov=True)),
        'rmsprop': (torch.optim.RMSprop, dict(alpha=0.8, eps=1e-5, momentum=0.7, centered=True)),
        'adagrad': (torch.optim.Adagrad, dict(lr_decay=0.03, initial_accumulator_value=0.2, eps=1e-6)),
        'adadelta': (torch.optim.Adadelta, dict(rho=0.8, eps=1e-5)),
        'adamax': (torch.optim.Adamax, dict(betas=(0.8, 0.91), eps=1e-6)),
    }
    cls, kwargs = options[name]
    optimizer = cls(groups, foreach=False, **kwargs)
    trajectory = []
    for step in range(8):
        if step == 3:
            optimizer.param_groups[1]['lr'] = 0.007
        if step == 6:
            for group in optimizer.param_groups:
                group['lr'] = 0.009
        optimizer.zero_grad(set_to_none=True)
        if step != 5:
            p.grad = torch.tensor([(step + 1) * 0.1, -0.2, (step % 3 - 1) * 0.15], dtype=torch.float64)
        if step not in (1, 4):
            q.grad = torch.tensor([0.3, -0.1 * (step + 1)], dtype=torch.float64)
        optimizer.step()
        trajectory.append(p.detach().tolist() + q.detach().tolist())
    states = {}
    for name, parameter in [('p', p), ('q', q)]:
        states[name] = {key: value.detach().flatten().tolist()
                       for key, value in optimizer.state[parameter].items() if key != 'step'}
    return dict(trajectory=trajectory, states=states)


def schedule_case(name):
    p = torch.tensor([1.0], requires_grad=True)
    q = torch.tensor([1.0], requires_grad=True)
    optimizer = torch.optim.SGD([dict(params=[p], lr=0.1), dict(params=[q], lr=0.04)])
    cases = {
        'step': lambda: torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.5),
        'exponential': lambda: torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8),
        'multi': lambda: torch.optim.lr_scheduler.MultiStepLR(optimizer, [0, 2, 2, 5], gamma=0.5),
        'cosine': lambda: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3, eta_min=0.005),
        'plateau': lambda: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1,
            threshold=0.01, threshold_mode='abs', cooldown=1, min_lr=[0.02, 0.005], eps=1e-10),
    }
    scheduler = cases[name]()
    rates = [[group['lr'] for group in optimizer.param_groups]]
    for metric in [1., 1., 1., 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]:
        optimizer.step()
        scheduler.step(metric) if name == 'plateau' else scheduler.step()
        rates.append([group['lr'] for group in optimizer.param_groups])
    return rates


if __name__ == '__main__':
    assert torch.__version__.split('+')[0] == '2.13.0', torch.__version__
    print(json.dumps(dict(
        optimizers={name: optimizer_case(name) for name in ['adam', 'adamw', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax']},
        schedulers={name: schedule_case(name) for name in ['step', 'exponential', 'multi', 'cosine', 'plateau']},
    )))
