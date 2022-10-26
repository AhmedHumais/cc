
import torch.nn as nn

class Conv5x5(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv5x5, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(2)
        else:
            self.pad = nn.ZeroPad2d(2)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 5)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out