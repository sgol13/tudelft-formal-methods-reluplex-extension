from mlp_conversions import conv1d_to_mlp, sequential_to_mlp

import pytest
import torch
import torch.nn as nn


class TestMlpConversions:
    @pytest.mark.parametrize(
        "conv1d",
        [
            # one input, one output channels
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=0),

            # one input, multiple output channels
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=0),

            # multiple input channels
            nn.Conv1d(in_channels=3, out_channels=2, kernel_size=4, stride=1, padding=0),

            # padding and stride
            nn.Conv1d(in_channels=3, out_channels=2, kernel_size=4, stride=3, padding=2),

            nn.Conv1d(in_channels=3, out_channels=5, kernel_size=5, stride=5, padding=0),
        ]
    )
    def test_conv1d_to_mlp(self, conv1d: nn.Conv1d):
        # given
        input_size = 20
        x = torch.randn(1, conv1d.in_channels, input_size)

        # when
        linear = conv1d_to_mlp(conv1d, input_size)

        # then
        conv1d_output = conv1d(x).view(-1)
        linear_output = linear(x.view(1, -1))

        # Check if outputs are close
        assert torch.allclose(conv1d_output, linear_output, atol=1e-4)

    @pytest.mark.parametrize(
        "sequential",
        [
            # simple sequential model
            nn.Sequential(
                nn.Conv1d(1, 5, 5, stride=5, padding=0),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(20, 10)
            ),

            # two conv layers (stride == kernel_size)
            nn.Sequential(
                nn.Conv1d(1, 3, 5, stride=5, padding=0),
                nn.ReLU(),
                nn.Conv1d(3, 5, 2, stride=5, padding=0),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(5, 10)
            ),

            # two conv layers (stride < kernel_size)
            nn.Sequential(
                nn.Conv1d(1, 3, 5, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv1d(3, 5, 2, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(35, 10)
            ),

            # no linear layer
            nn.Sequential(
                nn.Conv1d(1, 3, 5, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv1d(3, 5, 2, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )
        ]
    )
    def test_sequential_to_mlp(self, sequential):
        # given
        input_size = 20
        x = torch.randn(1, input_size)

        # when
        mlp = sequential_to_mlp(sequential, input_size)

        # then
        sequential_output = sequential(x.view(-1, 1, x.shape[-1]))
        mlp_output = mlp(x)

        # Check if outputs are close
        assert torch.allclose(sequential_output, mlp_output, atol=1e-4)
