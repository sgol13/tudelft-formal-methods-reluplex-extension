import torch
import torch.nn as nn


def conv1d_to_mlp(layer: torch.nn.Conv1d, input_size: int) -> torch.nn.Linear:
    assert isinstance(layer, torch.nn.Conv1d), f"Conv1d layer expected, got {type(layer)}."
    assert layer.in_channels == 1, "Only single input channel is supported."
    assert layer.groups == 1, "Grouped convolutions are not supported."

    # conv params
    C_in = layer.in_channels
    C_out = layer.out_channels
    K = layer.kernel_size[0]
    stride = layer.stride[0]
    padding = layer.padding[0]
    dilation = layer.dilation[0]

    L_in = input_size
    L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) // stride + 1

    in_features = C_in * L_in
    out_features = C_out * L_out

    device = layer.weight.device
    dtype = layer.weight.dtype

    # prepare dense weight and bias on the same device/dtype as the conv layer
    W_dense = torch.zeros((out_features, in_features), device=device, dtype=dtype)
    b_dense = torch.zeros((out_features,), device=device, dtype=dtype)

    # conv weight shape: (C_out, C_in, K)
    weight = layer.weight.detach().clone().view(C_out, C_in, K)
    bias = layer.bias.detach().clone() if layer.bias is not None else torch.zeros(C_out, device=device, dtype=dtype)

    # IMPORTANT: iterate channels outer, positions inner to match conv flattening:
    # conv output memory order (1, C_out, L_out) flattened -> for co in range(C_out): for pos in range(L_out)
    for co in range(C_out):
        for pos in range(L_out):
            out_index = co * L_out + pos
            b_dense[out_index] = bias[co]
            for ci in range(C_in):
                for k in range(K):
                    in_pos = pos * stride + k * dilation - padding
                    if 0 <= in_pos < L_in:
                        in_index = ci * L_in + in_pos
                        W_dense[out_index, in_index] = weight[co, ci, k]

    # build linear and copy parameters
    fc = nn.Linear(in_features, out_features, bias=True)
    fc.weight.data.copy_(W_dense)
    fc.bias.data.copy_(b_dense)

    # put Linear on same device/dtype as conv layer
    fc.to(device=device, dtype=dtype)
    return fc