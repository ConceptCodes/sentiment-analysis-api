import torch
import time
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from torch.utils.data import DataLoader
try:
    from .dataset import TweetDataset
    from .model import SentimentClassifier
    from .config import Config
except ImportError:  # pragma: no cover - fallback for direct script execution
    from dataset import TweetDataset
    from model import SentimentClassifier
    from config import Config
from tqdm import tqdm

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        return 0.0, 0.0

    return total_loss / len(loader), correct / total


def main():
    parser = argparse.ArgumentParser(description="Train the sentiment classifier.")
    parser.add_argument("--train-path", default="data/processed/train.pt", help="Path to the processed training tensor file.")
    parser.add_argument("--val-path", default="data/processed/val.pt", help="Path to the processed validation tensor file.")
    parser.add_argument("--model-dir", default="models", help="Directory to save checkpoints.")
    parser.add_argument("--epochs", type=int, default=None, help="Override the configured number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override the configured batch size.")
    args = parser.parse_args()

    cfg = Config()
    batch_size = args.batch_size or cfg.batch_size
    epochs = args.epochs or cfg.epochs
    embed_dim = cfg.embed_dim
    hidden_dim = cfg.hidden_dim
    vocab_size = cfg.vocab_size
    output_dim = cfg.output_dim
    pad_idx = cfg.pad_idx
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")
    print("Loading data...")

    train_loader = DataLoader(TweetDataset(args.train_path), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TweetDataset(args.val_path), batch_size=batch_size)
    print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

    print("Initializing model...")
    model = SentimentClassifier(vocab_size, embed_dim, hidden_dim, output_dim, pad_idx).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    best_val_loss = float("inf")
    best_model_path = None
    os.makedirs(args.model_dir, exist_ok=True)

    print("Starting training loop...\n")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        print(f"\nEpoch {epoch + 1}/{epochs}...")
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({"loss": loss.item()})

        train_acc = correct / total if total else 0.0
        train_loss = total_loss / len(train_loader) if len(train_loader) else 0.0
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            date = time.strftime("%Y%m%d_%H%M%S")
            best_model_path = os.path.join(args.model_dir, f"sentiment_model_epoch_{epoch + 1}_{date}.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Saved best model to {best_model_path}")

    if best_model_path is None:
        date = time.strftime("%Y%m%d_%H%M%S")
        best_model_path = os.path.join(args.model_dir, f"sentiment_model_final_{date}.pt")
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Saved final model to {best_model_path}")


if __name__ == "__main__":
    main()
