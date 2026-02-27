"""Loss functions"""
from typing import Sequence, Optional, Tuple, Callable, Union

import torch
import numpy as np

from torch import nn
from torch.nn import functional as F

from my_elektronn3.modules.lovasz_losses import lovasz_softmax


class MCCLoss(torch.nn.Module):
    """Matthews Correlation Coefficient Loss (From Herrera et al. 2019)
    Claims to work better for datasets with class imbalance

    Works for n-dimensional data. 
    Input:
    ``output`` - tensor of model output. 
    ``targets`` - list of tensors, first entry is target. 

    Assuming that the ``output`` tensor to be compared to the ``target`` has the shape (N, C, D, H, W), 
    the ``target`` can either have the same shape (N, C, D, H, W) (one-hot encoded) or (N, D, H, W) 
    (with dense class indices, as in ``torch.nn.CrossEntropyLoss``). If the latter shape is detected, 
    the ``target`` is automatically internally converted to a one-hot tensor for loss calculation.

    Args:
        apply_softmax: If ``True``, a softmax operation is applied to the
            ``output`` tensor before loss calculation. This is necessary if
            your model does not already apply softmax as the last layer.
            If ``False``, ``output`` is assumed to already contain softmax
            probabilities.
    """
    def __init__(
            self,
            apply_softmax: bool = True,
    ):
        super().__init__()
        if apply_softmax:
            self.softmax = torch.nn.Softmax(dim=1)
        else:
            self.softmax = lambda x: x  # Identity (no softmax)

    def forward(self, output, targets):
        # Apply softmax to output
        probs = self.softmax(output)

        # print(output.shape)
        # print(targets[0].shape)
        # print(probs.shape)
        
        torch_TP = probs[1] * targets[0]
        torch_TN = (1-probs[1]) * (1-targets[0])
        torch_FP = probs[0] * (1-targets[0])
        torch_FN = (1-probs[0]) * targets[0]

        # sum all values
        TP = torch_TP.sum() + 1e-5
        TN = torch_TN.sum() + 1e-5
        FP = torch_FP.sum() + 1e-5
        FN = torch_FN.sum() + 1e-5

        # print(f'TP: {TP}')
        # print(f'TN: {TN}')
        # print(f'FP: {FP}')
        # print(f'FN: {FN}')

        # MCC
        MCC_n = ((TP*TN) - (FP*FN))
        MCC_d = ((TP+FP) * (FN+TN) * (FP+TN) * (TP+FN)).sqrt()
        MCC = MCC_n / MCC_d
        
        # # rescale
        # MCC = (MCC+1)/2

        # Final Loss
        loss = 1-MCC
        
        return loss