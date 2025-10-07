from mlp_conversions import conv1d_to_mlp
import torch


def test_conv1d_to_mlp():
    # Generate random Conv1d layer
    conv1d = torch.nn.Conv1d(in_channels=1, out_channels=3, kernel_size=5, stride=2, padding=1)
    input_size = 20

    # Generate random input tensor
    x = torch.randn(1, 1, input_size)  # Batch size = 1, Channels = 1, Length = 20

    # Convert Conv1d to Linear
    linear = conv1d_to_mlp(conv1d, input_size)

    # Compute outputs
    conv1d_output = conv1d(x)
    linear_output = linear(x.view(1, -1))  # Flatten input for Linear layer

    # Check if outputs are close
    assert torch.allclose(conv1d_output.view(-1), linear_output, atol=1e-4), "Outputs do not match!"
