import os
import warnings
# Suppress macOS LibreSSL / urllib3 incompatibility noise
warnings.filterwarnings("ignore", category=Warning, module="urllib3")
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import mlflow
import mlflow.pytorch

from src.model import SimpleCNN
from src.utils.metrics import classification_metrics, confusion

# Resolve paths relative to the mlops/ project root (parent of src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = str(_PROJECT_ROOT / "data/processed")
DEFAULT_MODEL_PATH = str(_PROJECT_ROOT / "models/model.pt")

def _best_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def train(
    data_dir: str = DEFAULT_DATA_DIR,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = _best_device()
):
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Processed data not found. Expected {train_dir} and {val_dir}. "
            "Run preprocess first."
        )

    # ImageNet mean/std — standard for CNNs trained on 224x224 RGB
    _mean = [0.485, 0.456, 0.406]
    _std  = [0.229, 0.224, 0.225]

    tfm_train = transforms.Compose([
        # Geometry augmentation
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        # Colour augmentation
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=_mean, std=_std),
    ])
    tfm_eval = transforms.Compose([
        transforms.Resize((224, 224)),   # guard: already 224 from preprocess
        transforms.ToTensor(),
        transforms.Normalize(mean=_mean, std=_std),
    ])

    train_ds = datasets.ImageFolder(str(train_dir), transform=tfm_train)
    val_ds = datasets.ImageFolder(str(val_dir), transform=tfm_eval)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = SimpleCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Metadata (params, metrics, tags) → mlops/mlflow.db  (SQLite, absolute)
    # Artifacts (model.pt, plots)      → mlops/mlruns/    (file store, absolute)
    _db_path       = _PROJECT_ROOT / "mlflow.db"
    _artifact_path = _PROJECT_ROOT / "mlruns"
    mlflow.set_tracking_uri(f"sqlite:///{_db_path}")
    _client = mlflow.tracking.MlflowClient()
    if _client.get_experiment_by_name("cats-vs-dogs-mlops") is None:
        _client.create_experiment("cats-vs-dogs-mlops",
                                  artifact_location=str(_artifact_path))
    mlflow.set_experiment("cats-vs-dogs-mlops")
    with mlflow.start_run():
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "device": device,
            "optimizer": "Adam",
            "scheduler": "CosineAnnealingLR",
            "augmentation": "flip,rotation15,resizedcrop,colorjitter",
        })

        best_val_acc = 0.0
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            t0 = time.time()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)

            train_loss = running_loss / len(train_ds)

            # validation
            model.eval()
            y_true, y_pred = [], []
            val_loss_sum = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    val_loss_sum += loss.item() * xb.size(0)
                    preds = torch.argmax(logits, dim=1)
                    y_true.extend(yb.cpu().numpy().tolist())
                    y_pred.extend(preds.cpu().numpy().tolist())
            val_loss = val_loss_sum / len(val_ds)
            m = classification_metrics(y_true, y_pred)
            epoch_time = time.time() - t0

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": m["accuracy"],
                "val_precision": m["precision"],
                "val_recall": m["recall"],
                "val_f1": m["f1"],
                "epoch_time_sec": epoch_time,
                "lr": scheduler.get_last_lr()[0],
            }, step=epoch)

            if m["accuracy"] > best_val_acc:
                best_val_acc = m["accuracy"]
                Path(DEFAULT_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), DEFAULT_MODEL_PATH)
                mlflow.log_artifact(DEFAULT_MODEL_PATH)

            scheduler.step()

            print(
                f"Epoch {epoch:02d}/{epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_acc={m['accuracy']:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
                f"({epoch_time:.1f}s)"
            )

        # Save the model as an MLflow model too
        mlflow.pytorch.log_model(model, artifact_path="mlflow_model")

    print(f"Training complete. Best val accuracy={best_val_acc:.4f}. Model saved to {DEFAULT_MODEL_PATH}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--device",     default=_best_device())
    args = parser.parse_args()
    train(data_dir=args.data_dir, epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr, device=args.device)
