import torch
import torch.nn as nn


def conv1d_to_mlp(conv1d: torch.nn.Conv1d, input_size: int, device: torch.device) -> torch.nn.Linear:
    assert isinstance(conv1d, nn.Conv1d), f"Conv1d layer expected, got {type(conv1d)}."
    assert conv1d.groups == 1, "Grouped convolutions are not supported."
    assert conv1d.dilation[0] == 1, "Dilation is not supported."

    C_in = conv1d.in_channels
    C_out = conv1d.out_channels
    K = conv1d.kernel_size[0]
    stride = conv1d.stride[0]
    padding = conv1d.padding[0]

    L_in = input_size
    L_out = (L_in + 2 * padding - K) // stride + 1

    in_features = C_in * L_in
    out_features = C_out * L_out

    dtype = conv1d.weight.dtype

    W_dense = torch.zeros((out_features, in_features), device=device, dtype=dtype)
    b_dense = torch.zeros((out_features,), device=device, dtype=dtype)

    weight = conv1d.weight.detach().clone().view(C_out, C_in, K)
    bias = conv1d.bias.detach().clone() if conv1d.bias is not None else torch.zeros(C_out, device=device, dtype=dtype)

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

    linear = nn.Linear(in_features, out_features, bias=True)
    linear.weight.data.copy_(W_dense)
    linear.bias.data.copy_(b_dense)

    linear.to(device=device, dtype=dtype)
    return linear


def avg_pool1d_to_mlp(avg_pool1d: nn.AvgPool1d, input_size: int, in_channels: int, device: torch.device) -> nn.Linear:
    kernel_size = avg_pool1d.kernel_size if isinstance(avg_pool1d.kernel_size, int) else avg_pool1d.kernel_size[0]
    stride = avg_pool1d.stride if isinstance(avg_pool1d.stride, int) else avg_pool1d.stride[0]
    padding = avg_pool1d.padding if isinstance(avg_pool1d.padding, int) else avg_pool1d.padding[0]

    L_out = (input_size + 2 * padding - kernel_size) // stride + 1

    in_features = input_size * in_channels
    out_features = L_out * in_channels
    dtype = torch.get_default_dtype()

    W_dense = torch.zeros((out_features, in_features), dtype=dtype, device=device)

    for c in range(in_channels):
        for out_pos in range(L_out):
            for k in range(kernel_size):
                in_pos = out_pos * stride + k - padding
                if 0 <= in_pos < input_size:
                    in_index = c * input_size + in_pos
                    out_index = c * L_out + out_pos
                    W_dense[out_index, in_index] = 1.0 / kernel_size

    linear = nn.Linear(in_features, out_features, bias=True, device=device, dtype=dtype)
    linear.weight.data.copy_(W_dense)
    linear.bias.data.zero_()

    return linear


def sequential_to_mlp(model: nn.Sequential, input_size: int, device: torch.device) -> nn.Sequential:
    layers = []

    x = torch.zeros(1, input_size, device=device)
    channels = 1

    for layer in model:
        if isinstance(layer, nn.Conv1d):
            input_size = x.shape[-1] // channels
            linear = conv1d_to_mlp(layer, input_size, device=device)

            x = linear(x)
            channels = layer.out_channels

            layers.append(linear)

        elif isinstance(layer, nn.Linear):
            layers.append(layer)

        elif isinstance(layer, nn.AvgPool1d):
            input_size = x.shape[-1] // channels
            linear = avg_pool1d_to_mlp(layer, input_size, channels, device=device)

            x = linear(x)

            layers.append(linear)

        elif isinstance(layer, nn.Flatten):
            channels = 1
            continue

        elif isinstance(layer, nn.ReLU):
            layers.append(nn.ReLU())

        else:
            raise NotImplementedError(f"Layer type {type(layer)} is not supported.")

    return nn.Sequential(*layers)
