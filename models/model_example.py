import torch.nn import nn
from models.model_registry import MODEL_EXAMPLE


@MODEL_EXAMPLE.register('MyModel')
class MyModel(nn.Module):
    def __init__():
        # Create the net
        pass

    def forward(img):
        # Pass the image through the net
        pass
