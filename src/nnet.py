from torch import nn
from datetime import datetime
import torch
import numpy as np


def save_as_nnet(
    sequential: nn.Sequential, path: str, X: torch.Tensor, y: torch.Tensor
):
    linear_layers: list[nn.Linear] = []
    for layer in sequential:
        if isinstance(layer, nn.Linear):
            linear_layers.append(layer)
        elif isinstance(layer, nn.ReLU):
            continue
        else:
            raise ValueError(f"Unsupported layer type: {type(layer)}")

    nnet_rows: list[str] = []

    print(f"Loaded X: {X.shape}, y: {y.shape}")

    # .nnet format source: https://github.com/sisl/NNet
    # 1: Header text. This can be any number of lines so long as they begin with "//"
    nnet_rows.append(f'// {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    # 2: Four values: Number of layers, number of inputs, number of outputs, and maximum layer size
    nnet_rows.append(
        f"{len(linear_layers)},"
        f"{linear_layers[0].in_features},"
        f"{linear_layers[-1].out_features},"
        f"{max(layer.out_features for layer in linear_layers)}"
    )

    # 3: A sequence of values describing the network layer sizes.
    # Begin with the input size, then the size of the first layer, second layer, and so on until the output layer size.
    layer_sizes = [linear_layers[0].in_features] + [
        layer.out_features for layer in linear_layers
    ]

    nnet_rows.append(",".join(map(str, layer_sizes)))

    # 4: A flag that is no longer used, can be ignored
    nnet_rows.append("0")

    X = X.to(device="cpu")

    mins = torch.min(X, dim=0).values
    maxs = torch.max(X, dim=0).values

    assert mins.shape[0] == 40
    assert maxs.shape[0] == 40

    y = y.type(torch.float32)

    means = torch.cat([X.mean(dim=0), torch.Tensor([y.mean()])], dim=0).cpu().numpy()
    ranges = (
        torch.cat([maxs - mins, torch.Tensor([y.max() - y.min()])], dim=0).cpu().numpy()
    )

    mins = mins.numpy()
    maxs = maxs.numpy()

    assert means.shape[0] == 41 and ranges.shape[0] == 41

    # 5: Minimum values of inputs (used to keep inputs within expected range)
    mins = ",".join(mins.astype(str))
    nnet_rows.append(mins)

    # 6: Maximum values of inputs (used to keep inputs within expected range)
    maxes = ",".join(maxs.astype(str))
    nnet_rows.append(maxes)

    # 7: Mean values of inputs and one value for all outputs (used for normalization)
    means = ",".join(means.astype(str))
    nnet_rows.append(means)

    # 8: Range values of inputs and one value for all outputs (used for normalization)
    ranges = ",".join(ranges.astype(str))
    nnet_rows.append(ranges)

    # 9+: Begin defining the weight matrix for the first layer, followed by the bias vector.
    # The weights and biases for the second layer follow after,
    # until the weights and biases for the output layer are defined.

    for layer in linear_layers:
        # weights (row-wise)
        weights = layer.weight.detach().cpu().numpy()
        for row in weights:
            nnet_rows.append(",".join(map(str, row)))

        biases = layer.bias.detach().cpu().numpy()
        for bias in biases:
            nnet_rows.append(str(bias))

    with open(path, "w") as f:
        f.write("\n".join(nnet_rows))
