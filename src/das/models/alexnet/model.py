""" PyTorch lightning module for the visual backbone of the AlexNetv2 model. """


import torch
from das.utils.basic_utils import create_logger
from torch import nn

from ..model import TorchHubBaseModelForImageClassification

logger = create_logger(__name__)


class AlexNetForImageClassification(TorchHubBaseModelForImageClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def load_model(cls, model_name, num_labels, use_timm=False, pretrained=True):
        model = super(AlexNetForImageClassification, cls).load_model(
            model_name, num_labels, use_timm=use_timm, pretrained=pretrained
        )
        return cls.update_classifier_for_labels(model, num_labels=num_labels)

    @classmethod
    def update_classifier_for_labels(cls, model, num_labels):
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_labels)
        return model

    def setup_for_features(self):
        super().setup_for_features()
        self.model.classifier[6] = nn.Identity()


SUPPORTED_TASKS = {
    "image_classification": AlexNetForImageClassification,
}
