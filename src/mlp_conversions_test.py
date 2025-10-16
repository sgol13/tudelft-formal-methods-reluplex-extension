from mlp_conversions import conv1d_to_mlp, sequential_to_mlp, avg_pool1d_to_mlp

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
        ]
    )
    def test_conv1d_to_mlp(self, conv1d: nn.Conv1d):
        # given
        input_size = 20
        x = torch.randn(1, conv1d.in_channels, input_size)

        # when
        linear = conv1d_to_mlp(conv1d, input_size, device=torch.device('cpu'))

        # then
        conv1d_output = conv1d(x).view(-1)
        linear_output = linear(x.view(1, -1))

        assert torch.allclose(conv1d_output, linear_output, atol=1e-4)

    @pytest.mark.parametrize(
        ("avg_pool1d", "in_channels"),
        [
            # simple example
            (nn.AvgPool1d(kernel_size=2, stride=1, padding=0), 1),

            # larger kernel and stride
            (nn.AvgPool1d(kernel_size=4, stride=2, padding=0), 1),

            # padding
            (nn.AvgPool1d(kernel_size=4, stride=2, padding=1), 1),

            # more channels
            (nn.AvgPool1d(kernel_size=4, stride=2, padding=1), 3),
        ]
    )
    def test_avg_pool1d_to_mlp(self, avg_pool1d: nn.AvgPool1d, in_channels: int):
        # given
        input_size = 20
        x = torch.randn(1, in_channels, input_size)

        # when
        mlp = avg_pool1d_to_mlp(avg_pool1d, input_size, in_channels, device=torch.device('cpu'))

        # then
        max_pool_output = avg_pool1d(x).view(-1)
        mlp_output = mlp(x.view(1, -1))

        assert torch.allclose(max_pool_output, mlp_output, atol=1e-4)

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
            ),

            # average pooling
            nn.Sequential(
                nn.Conv1d(1, 5, 5, stride=5, padding=0),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=2, stride=2, padding=0),
                nn.Flatten(),
                nn.Linear(10, 10)
            ),

            # double average pooling
            nn.Sequential(
                nn.Conv1d(1, 3, 5, stride=2, padding=0),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=2, stride=2, padding=0),
                nn.Conv1d(3, 5, 2, stride=1, padding=0),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=2, stride=1, padding=1),
                nn.Flatten(),
                nn.Linear(20, 10)
            ),
        ]
    )
    def test_sequential_to_mlp(self, sequential):
        # given
        input_size = 20
        x = torch.randn(1, input_size)

        # when
        mlp = sequential_to_mlp(sequential, input_size, device=torch.device('cpu'))

        # then
        sequential_output = sequential(x.view(-1, 1, x.shape[-1]))
        # mlp_output = mlp(x)
        #
        # assert torch.allclose(sequential_output, mlp_output, atol=1e-4)
