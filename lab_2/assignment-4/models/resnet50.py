import torch
from torch import nn
from transformers import AutoModelForImageClassification

class PretrainedResNet50(nn.Module):
    def __init__(self, num_classes: int = 21, pretrained_model_name: str = "microsoft/resnet-50"):
        super().__init__()
        self.model = AutoModelForImageClassification.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.model(images)
        if hasattr(outputs, "logits"):
            return outputs.logits
        return outputs
