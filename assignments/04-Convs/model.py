import torch
import torch.nn as nn


class Model(torch.nn.Module):
    """
    The Model is here
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """
        Neural Network Architecture
        """

        super(Model, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(8 * 8 * 64, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The Forward pass
        """

        x = self.conv_layers(x).view(-1, 8 * 8 * 64)
        x = self.fc_layers(x)

        return x
