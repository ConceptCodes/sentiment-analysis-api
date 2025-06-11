import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TweetDataset
from model import SentimentClassifier
from config import Config
from tqdm import tqdm

cfg = Config()
BATCH_SIZE = cfg.batch_size
EPOCHS = cfg.epochs
EMBED_DIM = cfg.embed_dim
HIDDEN_DIM = cfg.hidden_dim
VOCAB_SIZE = cfg.vocab_size
OUTPUT_DIM = cfg.output_dim
PAD_IDX = cfg.pad_idx
DEVICE = torch.device(
  "mps" if torch.mps.is_available() 
  else "cuda" if torch.cuda.is_available() 
  else "cpu"
)

print(f"Using device: {DEVICE}")
print("Loading data...")

train_loader = DataLoader(TweetDataset("data/processed/train.pt"), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TweetDataset("data/processed/val.pt"), batch_size=BATCH_SIZE)
print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

print("Initializing model...")
model = SentimentClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, PAD_IDX).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-4)

print("Starting training loop...\n")
for epoch in range(EPOCHS):
  model.train()
  total_loss = 0
  correct = 0
  total = 0

  print(f"\nEpoch {epoch+1}/{EPOCHS}...")
  pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
  for batch_idx, (inputs, labels) in enumerate(pbar):
    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    pred = outputs.argmax(dim=1)
    correct += (pred == labels).sum().item()
    total += labels.size(0)

    pbar.set_postfix({'loss': loss.item()})

  acc = correct / total
  print(f"Loss: {total_loss:.4f} | Accuracy: {acc:.4f}")

date = time.strftime("%Y%m%d_%H%M%S")
model_name = f"models/sentiment_model_date_{date}.pt"
torch.save(model.state_dict(), model_name)
print(f"✅ Model saved to {model_name}")
