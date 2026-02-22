from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.model import SimpleCNN
from src.utils.metrics import classification_metrics, confusion

def evaluate(data_dir: str = "data/processed", model_path: str = "models/model.pt", device: str = "cpu"):
    data_dir = Path(data_dir)
    test_dir = data_dir / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Expected {test_dir}. Run preprocess first.")

    tfm = transforms.Compose([transforms.ToTensor()])
    ds = datasets.ImageFolder(str(test_dir), transform=tfm)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    model = SimpleCNN(num_classes=2).to(device)
    sd = torch.load(model_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            y_true.extend(yb.numpy().tolist())
            y_pred.extend(preds)

    m = classification_metrics(y_true, y_pred)
    cm = confusion(y_true, y_pred)
    return m, cm

if __name__ == "__main__":
    metrics, cm = evaluate()
    print("Test metrics:", metrics)
    print("Confusion matrix:\n", cm)
