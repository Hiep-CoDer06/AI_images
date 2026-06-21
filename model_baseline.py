import torch
import torch.nn as nn
import math


def ratio2filtersize(ratio: float, channel: str = "AWGN") -> int:
    """Convert compression ratio to channel use (filter size)."""
    if channel.lower() == "awgn":
        n_filters = int(math.floor(1024 * ratio))
    else:
        n_filters = int(math.floor(512 * ratio))
    n_filters = max(8, n_filters)
    return n_filters


class _Encoder(nn.Module):
    """CNN-based encoder for Deep JSCC."""

    def __init__(self, in_channels: int = 3, n_filters: int = 1024):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        H = 32
        W = 32
        conv_out_size = 512 * H * W

        self.fc = nn.Linear(conv_out_size, n_filters)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class _Decoder(nn.Module):
    """CNN-based decoder for Deep JSCC."""

    def __init__(self, in_features: int = 1024, out_channels: int = 3, n_filters: int = 1024):
        super().__init__()
        H = 32
        W = 32
        conv_out_size = 512 * H * W

        self.fc = nn.Linear(in_features, conv_out_size)
        self.act = nn.ReLU(inplace=True)

        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc(x))
        x = x.view(x.size(0), 512, 32, 32)
        x = self.act(self.bn1(self.deconv1(x)))
        x = self.act(self.bn2(self.deconv2(x)))
        x = self.act(self.bn3(self.deconv3(x)))
        x = self.deconv4(x)
        return x


class DeepJSCC(nn.Module):
    """Baseline Deep JSCC model (without FIS controller)."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        ratio: float = 0.1667,
        channel: str = "AWGN",
    ):
        super().__init__()
        n_filters = ratio2filtersize(ratio, channel)
        self.encoder = _Encoder(in_channels=in_channels, n_filters=n_filters)
        self.decoder = _Decoder(in_features=n_filters, out_channels=out_channels, n_filters=n_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
