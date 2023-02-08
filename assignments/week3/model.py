import torch
from typing import Callable
import torch.nn as nn


class MLP(nn.Module):
    """
    MLP class implementation is here
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.hidden_count = hidden_count
        self.activation = torch.nn.functional.relu
        self.initializer = initializer

        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.bn_1 = torch.nn.BatchNorm1d(hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.num_classes)

        # self.initializer(self.fc1.weight)
        # self.initializer(self.fc2.weight)

        # self.parameters = nn.ModuleList([self.fc1, self.fc2, self.bn_1]).parameters()

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        x = self.activation(self.bn_1(self.fc1(x)))
        x = self.fc2(x)
        return x
