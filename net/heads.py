""" Projection and Prediction Heads for Self-supervised Learning """

# Copyright (c) 2021. Lightly AG and its affiliates.
# All Rights Reserved

from typing import List, Optional, Tuple
import math
import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Base class for all projection and prediction heads.

    Args:
        blocks:
            List of tuples, each denoting one block of the projection head MLP.
            Each tuple reads (in_features, out_features, batch_norm_layer,
            non_linearity_layer).

    Examples:
        >>> # the following projection head has two blocks
        >>> # the first block uses batch norm an a ReLU non-linearity
        >>> # the second block is a simple linear layer
        >>> projection_head = ProjectionHead([
        >>>     (256, 256, nn.BatchNorm1d(256), nn.ReLU()),
        >>>     (256, 128, None, None)
        >>> ])

    """

    def __init__(
        self, 
        blocks: List[Tuple[int, int, Optional[nn.Module], Optional[nn.Module]]]
    ):
        super(ProjectionHead, self).__init__()

        layers = []
        for input_dim, output_dim, batch_norm, non_linearity in blocks:
            use_bias = not bool(batch_norm)
            layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
            if batch_norm:
                layers.append(batch_norm)
            if non_linearity:
                layers.append(non_linearity)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """Computes one forward pass through the projection head.

        Args:
            x:
                Input of shape bsz x num_ftrs.

        """
        return self.layers(x)


class DINOProjectionHead(ProjectionHead):
    """Projection head used in DINO.

    "The projection head consists of a 3-layer multi-layer perceptron (MLP) 
    with hidden dimension 2048 followed by l2 normalization and a weight
    normalized fully connected layer with K dimensions, which is similar to the
    design from SwAV [1]." [0]

    - [0]: DINO, 2021, https://arxiv.org/abs/2104.14294
    - [1]: SwAV, 2020, https://arxiv.org/abs/2006.09882

    Attributes:
        input_dim:
            The input dimension of the head.
        hidden_dim:
            The hidden dimension.
        bottleneck_dim:
            Dimension of the bottleneck in the last layer of the head.
        output_dim:
            The output dimension of the head.
        batch_norm:
            Whether to use batch norm or not. Should be set to False when using
            a vision transformer backbone.
        freeze_last_layer:
            Number of epochs during which we keep the output layer fixed. 
            Typically doing so during the first epoch helps training. Try 
            increasing this value if the loss does not decrease.
        norm_last_layer:
            Whether or not to weight normalize the last layer of the DINO head.
            Not normalizing leads to better performance but can make the 
            training unstable.
    
    """
    def __init__(
        self, 
        input_dim: int = 2048, 
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        output_dim: int = 65536,
        batch_norm: bool = False,
        freeze_last_layer: int = -1,
        norm_last_layer: bool = True,
    ):
        bn = nn.BatchNorm1d(hidden_dim) if batch_norm else None

        super().__init__([
            (input_dim, hidden_dim, bn, nn.GELU()),
            (hidden_dim, hidden_dim, bn, nn.GELU()),
            (hidden_dim, bottleneck_dim, None, None),
        ])
        self.apply(self._init_weights)
        self.freeze_last_layer = freeze_last_layer
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, output_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        # Option to normalize last layer. 
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
        
    def cancel_last_layer_gradients(self, current_epoch: int):
        """Cancel last layer gradients to stabilize the training.
        
        """
        if current_epoch >= self.freeze_last_layer:
            return
        for param in self.last_layer.parameters():
            param.grad = None

    def _init_weights(self, module):
        """Initializes layers with a truncated normal distribution.
        
        """
        if isinstance(module, nn.Linear):
            _no_grad_trunc_normal(
                module.weight, 
                mean=0, 
                std=0.2, 
                a=-2, 
                b=2,
            )
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes one forward pass through the head.
        
        """
        x = self.layers(x)
        # l2 normalization
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class SimCLRProjectionHead(ProjectionHead):
    """Projection head used for SimCLR.

    "We use a MLP with one hidden layer to obtain zi = g(h) = W_2 * σ(W_1 * h)
    where σ is a ReLU non-linearity." [0]

    [0] SimCLR, 2020, https://arxiv.org/abs/2002.05709

    """
    def __init__(self,
                 input_dim: int = 2048,
                 hidden_dim: int = 2048,
                 output_dim: int = 128):
        super(SimCLRProjectionHead, self).__init__([
            (input_dim, hidden_dim, None, nn.ReLU()),
            (hidden_dim, output_dim, None, None),
        ])


def _no_grad_trunc_normal(
    tensor: torch.Tensor,
    mean: float,
    std: float,
    a: float,
    b: float,
) -> torch.Tensor:
    """Initializes the input tensor with a truncated normal distribution.

    This method is based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf

    Args:
        tensor:
            The tensor to initialize.
        mean:
            Mean of the distribution.
        std:
            Standard deviation of the distribution.
        a:
            Minimum value of the distribution, values below will be clamped.
        b:
            Maximum value of the distribution, values above will be clamped.

    """
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor