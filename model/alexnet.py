from typing import Optional
from torchvision.models import alexnet
import torch.nn as nn
from .base_model import BaseModel

class AlexNet(BaseModel):
    """
    Encapsulate the pretrained AlexNet model
    
    Parameters
    ----------
    n_classes : int, default(256)
        the new number of classes
    device: Optional[str] 'cuda' or 'cpu', default(None)
            if None: cuda will be used if it is available
    """

    def __init__(self, n_classes: int = 256, device: Optional[str] = None):
        super().__init__(n_classes, device)
        self.model = alexnet(pretrained=True, progress=True)
        self._freeze_all_layers()
        self._change_last_layer()

    def _change_last_layer(self) -> None:
        """
        Change last layer to accept n_classes instead of 1000 classes
        """
        self.model.classifier[6] = nn.Linear(4096, self.n_classes)

    def _add_softmax_layer(self) -> None:
        """
        Add softmax layer to alexnet model (optional)
        """
        self.model = nn.Sequential(self.model, nn.LogSoftmax(dim=1))