import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    """(Conv => BN => Pool => ReLU )"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.MaxPool1d(2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class FC(nn.Module):
    """Flatten => FC => out(h)"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1408, 704),
            nn.Linear(704, 352),
            nn.Linear(352, 2)
        )

    def forward(self, x):
        return self.fc(x)

