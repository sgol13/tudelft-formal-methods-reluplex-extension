import torch
import torch.nn as nn


def conv1d_to_mlp(layer: torch.nn.Conv1d, input_size: int) -> torch.nn.Linear:
    assert isinstance(layer, nn.Conv1d), f"Conv1d layer expected, got {type(layer)}."
    assert layer.groups == 1, "Grouped convolutions are not supported."
    assert layer.dilation[0] == 1, "Dilation is not supported."

    C_in = layer.in_channels
    C_out = layer.out_channels
    K = layer.kernel_size[0]
    stride = layer.stride[0]
    padding = layer.padding[0]

    L_in = input_size
    L_out = (L_in + 2 * padding - K) // stride + 1

    in_features = C_in * L_in
    out_features = C_out * L_out

    device = layer.weight.device
    dtype = layer.weight.dtype

    W_dense = torch.zeros((out_features, in_features), device=device, dtype=dtype)
    b_dense = torch.zeros((out_features,), device=device, dtype=dtype)

    weight = layer.weight.detach().clone().view(C_out, C_in, K)
    bias = layer.bias.detach().clone() if layer.bias is not None else torch.zeros(C_out, device=device, dtype=dtype)

    for co in range(C_out):
        for pos in range(L_out):
            out_index = co * L_out + pos
            b_dense[out_index] = bias[co]
            for ci in range(C_in):
                for k in range(K):
                    in_pos = pos * stride + k - padding
                    if 0 <= in_pos < L_in:
                        in_index = ci * L_in + in_pos
                        W_dense[out_index, in_index] = weight[co, ci, k]

    fc = nn.Linear(in_features, out_features, bias=True)
    fc.weight.data.copy_(W_dense)
    fc.bias.data.copy_(b_dense)

    fc.to(device=device, dtype=dtype)
    return fc


def sequential_to_mlp(model: nn.Sequential, input_size: int) -> nn.Sequential:
    layers = []

    for layer in model:
        print(input_size)
        if isinstance(layer, nn.Conv1d):
            linear = conv1d_to_mlp(layer, input_size)

            x = torch.zeros(1, input_size * layer.in_channels)
            output = linear(x)
            input_size = output.shape[-1] // layer.out_channels

            layers.append(linear)

        elif isinstance(layer, nn.Linear):
            layers.append(layer)
            input_size = layer.out_features

        elif isinstance(layer, nn.Flatten):
            continue

        elif isinstance(layer, nn.ReLU):
            layers.append(nn.ReLU())

        else:
            raise NotImplementedError(f"Layer type {type(layer)} is not supported.")

    return nn.Sequential(*layers)
