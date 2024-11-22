from typing import Optional
from .alexnet import AlexNet
from .resnet import ResNet18

def create_model(model_name: str, n_classes: int, device: Optional[str] = None):
    """
    Factory function to create the appropriate model
    
    Parameters
    ----------
    model_name : str
        Name of the model to create ('alexnet' or 'resnet18')
    n_classes : int
        Number of classes for the model
    device : Optional[str]
        Device to use for the model
        
    Returns
    -------
    BaseModel
        The instantiated model
    """
    model_map = {
        'alexnet': AlexNet,
        'resnet18': ResNet18
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(model_map.keys())}")
    
    return model_map[model_name](n_classes=n_classes, device=device)