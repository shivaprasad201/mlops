import numpy as np
from PIL import Image
import torch

from src.api.app import preprocess_pil_to_tensor
from src.model import SimpleCNN

def test_preprocess_pil_to_tensor():
    img = Image.fromarray(np.zeros((300, 300, 3), dtype=np.uint8))
    x = preprocess_pil_to_tensor(img)
    assert isinstance(x, torch.Tensor)
    assert x.shape == (1, 3, 224, 224)

def test_model_forward_shape():
    model = SimpleCNN(num_classes=2)
    x = torch.zeros((1, 3, 224, 224))
    y = model(x)
    assert y.shape == (1, 2)
