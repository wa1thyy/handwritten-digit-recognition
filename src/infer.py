import io
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from .model import SmallCNN

def load_model(ckpt_path="checkpoints/best.pt", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, device

_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

def preprocess(img: Image.Image):
    return _transform(img).unsqueeze(0)  # (1,1,28,28)

@torch.no_grad()
def predict(model, device, img: Image.Image):
    x = preprocess(img).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
    pred = int(probs.argmax())
    return pred, probs.tolist()