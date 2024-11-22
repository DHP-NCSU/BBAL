from typing import Optional
from torchvision.models import resnet18
import torch.nn as nn
from .base_model import BaseModel

class ResNet18(BaseModel):
    """
    Encapsulate the pretrained ResNet18 model
    
    Parameters
    ----------
    n_classes : int, default(256)
        the new number of classes
    device: Optional[str] 'cuda' or 'cpu', default(None)
            if None: cuda will be used if it is available
    """

    def __init__(self, n_classes: int = 256, device: Optional[str] = None):
        super().__init__(n_classes, device)
        self.model = resnet18(pretrained=True, progress=True)
        self._freeze_all_layers()
        self._change_last_layer()

    def _change_last_layer(self) -> None:
        """
        Change last layer to accept n_classes instead of 1000 classes
        """
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.n_classes)