"""Pinned PyTorch 2.13.0 (cf30153) sequence-module numerical reference.

Uses public torch.nn APIs from modules/rnn.py, activation.py, transformer.py.
This original fixture generator does not copy upstream implementation code.
Run through scripts/run-python-parity.sh; generated tensors remain in target/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import tensor_data, state_data, write_json


def assign(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for index, (name, parameter) in enumerate(sorted(model.named_parameters())):
            values = torch.arange(parameter.numel()).reshape_as(parameter).float()
            if "norm" in name and name.endswith("weight"):
                values = 0.7 + values * 0.03
            else:
                values = values * 0.013 + (index % 5 - 2) * 0.04
            parameter.copy_(values)


def values(shape: tuple[int, ...], offset: float = -0.3) -> torch.Tensor:
    count = 1
    for size in shape:
        count *= size
    return (torch.arange(count).float().reshape(shape) / 17 + offset).requires_grad_()


def finish(model, output, inputs, extra=()):
    loss = output.square().mean()
    for item in extra:
        loss = loss + item.square().mean()
    loss.backward()
    return {
        "output": tensor_data(output),
        "extra": [tensor_data(item) for item in extra],
        "state": state_data(model),
        "parameter_grads": {name: tensor_data(p.grad) for name, p in model.named_parameters()},
        "input_grads": [tensor_data(item.grad) for item in inputs],
    }


def recurrent_examples():
    result = {}
    for name, family, kwargs, shape, explicit, training in [
        ("rnn_tanh", "rnn", {}, (3, 2, 2), False, False),
        ("rnn_relu", "rnn", dict(nonlinearity="relu", num_layers=2, bidirectional=True, batch_first=True, dropout=1.), (2, 3, 2), True, True),
        ("rnn_dropout", "rnn", dict(num_layers=2, dropout=0.4), (3, 2, 2), False, True),
        ("lstm", "lstm", dict(num_layers=2, bidirectional=True, batch_first=True), (2, 3, 2), True, False),
        ("lstm_projection", "lstm", dict(proj_size=2, bias=False, batch_first=True), (3, 2), True, False),
        ("gru", "gru", dict(bias=False, batch_first=True), (3, 2), True, True),
        ("gru_bidirectional", "gru", dict(num_layers=2, bidirectional=True, dropout=0.3), (3, 2, 2), False, True),
    ]:
        torch.manual_seed(127)
        model = {"rnn": torch.nn.RNN, "lstm": torch.nn.LSTM, "gru": torch.nn.GRU}[family](2, 3, **kwargs)
        initial = state_data(model)
        assign(model)
        model.train(training)
        value = values(shape)
        states = []
        if explicit:
            layers = kwargs.get("num_layers", 1) * (2 if kwargs.get("bidirectional", False) else 1)
            state_shape = (layers,) if len(shape) == 2 else (layers, 2)
            states.append(values((*state_shape, kwargs.get("proj_size", 3)), 0.1))
            if family == "lstm":
                states.append(values((*state_shape, 3), -0.1))
        state = tuple(states) if family == "lstm" and explicit else states[0] if explicit else None
        torch.manual_seed(541)
        output, final = model(value, state)
        extra = final if family == "lstm" else (final,)
        result[name] = {
            "family": family,
            "kwargs": kwargs,
            "training": training,
            "input": tensor_data(value),
            "states": [tensor_data(s) for s in states],
            "initial": initial,
            **finish(model, output, [value, *states], extra),
        }
    return result


def attention_examples():
    result = {}
    for name, batch_first, unbatched, bias, dropout, training, causal, per_head in [
        ("attention", False, False, True, 0., False, False, False),
        ("attention_causal", True, False, True, 0., True, True, True),
        ("attention_unbatched", True, True, False, 0., False, False, True),
        ("attention_dropout", True, False, True, 0.4, True, False, False),
        ("attention_eval", True, False, True, 0.9, False, False, False),
    ]:
        torch.manual_seed(127)
        model = torch.nn.MultiheadAttention(4, 2, dropout=dropout, bias=bias, batch_first=batch_first)
        initial = state_data(model)
        assign(model)
        model.train(training)
        shape = (3, 4) if unbatched else (2, 3, 4) if batch_first else (3, 2, 4)
        source_shape = (4, 4) if unbatched else (2, 4, 4) if batch_first else (4, 2, 4)
        query, key, value = values(shape), values(source_shape, 0.2), values(source_shape, -0.1)
        padding = torch.tensor([False, False, False, True]) if unbatched else torch.tensor([[False, False, False, True], [False, False, True, False]])
        attention = torch.ones(3, 4, dtype=torch.bool).triu(1) if causal else torch.zeros(3, 4)
        if not causal:
            attention[:, 1] = -0.3
            padding = torch.zeros_like(padding, dtype=torch.float32).masked_fill(padding, float("-inf"))
        if unbatched:
            attention = attention.unsqueeze(0).repeat(2, 1, 1)
        torch.manual_seed(541)
        output, weights = model(query, key, value, key_padding_mask=padding, attn_mask=attention, average_attn_weights=not per_head)
        # Infinity is encoded as a string; Rust reconstructs it explicitly.
        padding_json = [["-inf" if v == float("-inf") else v for v in row] for row in padding.tolist()] if padding.ndim == 2 else ["-inf" if v == float("-inf") else v for v in padding.tolist()]
        result[name] = {
            "batch_first": batch_first, "bias": bias, "dropout": dropout,
            "training": training, "causal": causal, "per_head": per_head,
            "query": tensor_data(query), "key": tensor_data(key), "value": tensor_data(value),
            "padding": padding_json, "padding_bool": padding.dtype == torch.bool,
            "attention": tensor_data(attention), "attention_bool": attention.dtype == torch.bool,
            "initial": initial,
            "weights": tensor_data(weights),
            **finish(model, output, [query, key, value], (weights,)),
        }
    return result


def transformer_examples():
    result = {}
    for name, family, kwargs, unbatched, training in [
        ("encoder_layer", "encoder_layer", {}, False, False),
        ("encoder_pre_norm", "encoder_layer", dict(norm_first=True, activation="gelu", bias=False, batch_first=True), True, False),
        ("decoder_layer", "decoder_layer", dict(batch_first=True), False, False),
        ("decoder_pre_norm", "decoder_layer", dict(norm_first=True, activation="gelu", batch_first=True, bias=False), True, True),
        ("encoder_stack", "encoder", dict(batch_first=True), False, False),
        ("encoder_dropout", "encoder_layer", dict(batch_first=True, dropout=0.3), False, True),
        ("decoder_stack", "decoder", dict(norm_first=True), False, False),
        ("decoder_dropout", "decoder_layer", dict(norm_first=True, dropout=0.3), False, True),
        ("transformer", "transformer", dict(batch_first=True), False, False),
    ]:
        torch.manual_seed(127)
        base = {"d_model": 4, "nhead": 2, "dim_feedforward": 7, "dropout": 0., **kwargs}
        if family == "transformer":
            model = torch.nn.Transformer(**base, num_encoder_layers=2, num_decoder_layers=2)
        elif family in ("encoder", "decoder"):
            layer_type = torch.nn.TransformerEncoderLayer if family == "encoder" else torch.nn.TransformerDecoderLayer
            stack_type = torch.nn.TransformerEncoder if family == "encoder" else torch.nn.TransformerDecoder
            model = stack_type(layer_type(**base), 2, norm=torch.nn.LayerNorm(4))
        else:
            model = (torch.nn.TransformerEncoderLayer if family == "encoder_layer" else torch.nn.TransformerDecoderLayer)(**base)
        initial = state_data(model)
        assign(model)
        model.train(training)
        batch_first = kwargs.get("batch_first", False)
        shape = (3, 4) if unbatched else (2, 3, 4) if batch_first else (3, 2, 4)
        memory_shape = (4, 4) if unbatched else (2, 4, 4) if batch_first else (4, 2, 4)
        value = values(shape)
        memory = values(memory_shape, 0.2)
        causal = torch.ones(3, 3, dtype=torch.bool).triu(1)
        source_padding = torch.tensor([False, False, False, True]) if unbatched else torch.tensor([[False, False, False, True], [False, True, False, False]])
        target_padding = torch.tensor([False, False, True]) if unbatched else torch.tensor([[False, False, True], [False, True, False]])
        torch.manual_seed(541)
        if family == "transformer":
            output = model(memory, value, tgt_mask=causal, src_key_padding_mask=source_padding, tgt_key_padding_mask=target_padding, memory_key_padding_mask=source_padding)
            inputs = [value, memory]
        elif family in ("encoder", "encoder_layer"):
            output = model(value, src_mask=causal, src_key_padding_mask=target_padding) if family == "encoder_layer" else model(value, mask=causal, src_key_padding_mask=target_padding)
            inputs = [value]
        else:
            output = model(value, memory, tgt_mask=causal, tgt_key_padding_mask=target_padding, memory_key_padding_mask=source_padding)
            inputs = [value, memory]
        result[name] = {
            "family": family, "kwargs": kwargs, "training": training,
            "initial": initial if family != "transformer" else None,
            "input": tensor_data(value), "memory": tensor_data(memory),
            "target_padding": tensor_data(target_padding), "source_padding": tensor_data(source_padding),
            **finish(model, output, inputs),
        }
    return result


def generate(output_dir: Path) -> None:
    if torch.__version__.split("+", 1)[0] != "2.13.0":
        raise RuntimeError("sequence parity requires the locked PyTorch 2.13.0 environment")
    write_json(output_dir / "sequence.json", {
        "recurrent": recurrent_examples(),
        "attention": attention_examples(),
        "transformer": transformer_examples(),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    generate(parser.parse_args().output_dir)
