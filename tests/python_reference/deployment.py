"""Pinned deployment round trips: torch.export, ONNX checker/reference and JIT.

Run by tests/deployment.rs; generated artifacts stay in its temporary directory.
No downloaded models or arbitrary pickle input is used.
"""
from pathlib import Path
import json
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
import torch
from safetensors.torch import save_file


def inputs():
    return torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(3, 2)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor([[1., 2., 3.], [-1., 2., -3.]]))
            self.head.bias.copy_(torch.tensor([.5, 1.]))
        self.scale = torch.tensor([2., 3.])

    def forward(self, x):
        return torch.relu(self.head(x)) * self.scale


def generate(directory):
    assert torch.__version__.split('+')[0] == '2.13.0'
    assert onnx.__version__ == '1.22.0'
    model = Classifier()
    exported = torch.export.export(
        model, (inputs(),),
        dynamic_shapes={'x': {0: torch.export.Dim('batch', min=1, max=8)}},
    )
    torch.export.save(exported, directory / 'reference.pt2')
    x = inputs().requires_grad_(True)
    output = model(x)
    output.sum().backward()
    save_file({'output': output.detach(), 'input_gradient': x.grad,
               'weight_gradient': model.head.weight.grad}, directory / 'expected.safetensors')

    graph = helper.make_graph(
        [helper.make_node('Gemm', ['x', 'head.weight', 'head.bias'], ['linear'], transB=1),
         helper.make_node('Relu', ['linear'], ['relu']),
         helper.make_node('Mul', ['relu', 'scale'], ['score'])],
        'classifier',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, ['batch', 3])],
        [helper.make_tensor_value_info('score', TensorProto.FLOAT, ['batch', 2])],
        [numpy_helper.from_array(t.detach().numpy(), name)
         for name, t in [('head.weight', model.head.weight), ('head.bias', model.head.bias),
                         ('scale', model.scale)]],
    )
    graph_model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)], ir_version=10)
    onnx.checker.check_model(graph_model, full_check=True)
    onnx.save(graph_model, directory / 'reference.onnx')
    torch.jit.trace(model.eval(), inputs()).save(str(directory / 'reference.pt'))
    (directory / 'reference.pt.guards.json').write_text(json.dumps([1,[{
        'name':'x','dtype':'Float','dimensions':[
            {'Symbol':{'name':'batch','min':1,'max':8}},{'Known':3}]}]]))
    conditional = {}
    for predicate in [False,True]:
        x = torch.tensor([2.,3.],requires_grad=True)
        y = torch.cond(torch.tensor(predicate),lambda x:x*x,lambda x:x+x,(x,))
        y.sum().backward()
        conditional[f'output_{int(predicate)}'] = y.detach()
        conditional[f'gradient_{int(predicate)}'] = x.grad
    save_file(conditional,directory / 'conditional.safetensors')


def leaves(value):
    if isinstance(value, dict):
        return [v for x in value.values() for v in leaves(x)]
    if isinstance(value, (tuple, list)):
        return [v for x in value for v in leaves(x)]
    return [value]


def verify(directory):
    for path in sorted(directory.glob('*.rust.pt2')) + [directory / 'rust.pt2']:
        model = torch.export.load(path).module()
        for batch in [2, 3, 8]:
            x = torch.arange(batch * 3, dtype=torch.float32).reshape(batch, 3)
            actual = leaves(model(x))[0]
            torch.testing.assert_close(actual, Classifier()(x))
        x = inputs().requires_grad_(True)
        leaves(model(x))[0].sum().backward()
        baseline = inputs().requires_grad_(True)
        Classifier()(baseline).sum().backward()
        torch.testing.assert_close(x.grad, baseline.grad)
    for path in sorted(directory.glob('*.rust.onnx')) + [directory / 'rust.onnx']:
        model = onnx.load(path, load_external_data=False)
        onnx.checker.check_model(model, full_check=True)
        runner = ReferenceEvaluator(model)
        for batch in [2, 3, 8]:
            x = torch.arange(batch * 3, dtype=torch.float32).reshape(batch, 3)
            actual = runner.run(None, {'x': x.numpy()})[0]
            np.testing.assert_allclose(actual, Classifier()(x).detach().numpy(), rtol=1e-5, atol=1e-6)
    native = torch.jit.load(str(directory / 'rust.pt'))
    for batch in [2,3,8]:
        x = torch.arange(batch * 3, dtype=torch.float32).reshape(batch,3)
        torch.testing.assert_close(native(x),Classifier()(x))
    print('PT2 values/gradients, ONNX checker/reference and actual TorchScript round trips passed')


if __name__ == '__main__':
    mode, directory = sys.argv[1:]
    {'generate': generate, 'verify': verify}[mode](Path(directory))
