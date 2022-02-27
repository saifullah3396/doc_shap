"""
Defines the ouputs of the different model/sub-models as dataclasses.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SimpleLossOutput:
    """
    Data class for simple loss output

    Args:
        loss: Classification loss.
    """

    loss: Optional[torch.FloatTensor] = None


@dataclass
class ClassificationModelOutput:
    """
    Data class for outputs of any classifier

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Classifier output logits.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
