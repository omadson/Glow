from torch import nn
from nflows import transforms

class Conv2dSameSize(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size):
        same_padding = kernel_size // 2  # Padding that would keep the spatial dims the same
        super().__init__(in_channels, out_channels, kernel_size, padding=same_padding)
        
class ConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.net = nn.Sequential(
            Conv2dSameSize(in_channels, hidden_channels, kernel_size=3),
            nn.ReLU(),
            Conv2dSameSize(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            Conv2dSameSize(hidden_channels, out_channels, kernel_size=3),
        )

    def forward(self, inputs, context=None):
        return self.net.forward(inputs)

class ReshapeTransform(transforms.Transform):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, inputs, context=None):
        if tuple(inputs.shape[1:]) != self.input_shape:
            raise RuntimeError('Unexpected inputs shape ({}, but expecting {})'
                               .format(tuple(inputs.shape[1:]), self.input_shape))
        return inputs.reshape(-1, *self.output_shape), torch.zeros(inputs.shape[0], device=device)

    def inverse(self, inputs, context=None):
        if tuple(inputs.shape[1:]) != self.output_shape:
            raise RuntimeError('Unexpected inputs shape ({}, but expecting {})'
                               .format(tuple(inputs.shape[1:]), self.output_shape))
        return inputs.reshape(-1, *self.input_shape), torch.zeros(inputs.shape[0], device=device)