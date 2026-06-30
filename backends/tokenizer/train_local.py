"""
train_local.py - Train the Zora Sign Language Model Locally

This script trains the ST-GCN classification model using the pre-extracted landmarks
in the C:\Akash\Ammus\zora\V2 folder. It does not require any video files or Google Drive downloads.

Usage:
    python train_local.py
"""

import os
import urllib.request
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Import model definition from local model.py
from model import UniSignDownstreamModel

# Config
V2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "V2"))
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
PRETRAINED_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "how2sign_pose_only_slt.pth")
BEST_MODEL_PATH = os.path.join(WEIGHTS_DIR, "model_best.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15
BATCH_SIZE = 8
LEARNING_RATE = 1e-4

# The 13 target sentences
SENTENCES = [
    "Can you help me?",
    "Can I help you",
    "Can you wait?",
    "Can you repeat?",
    "Can you see me?",
    "Please help me",
    "Please wait",
    "Please repeat",
    "I need help",
    "I understand",
    "I don't understand",
    "Do you understand?",
    "Do you need help?",
]
NUM_CLASSES = len(SENTENCES)

class SignDataset(Dataset):
    def __init__(self, data_dir, seq_len=60):
        self.seq_len = seq_len
        self.samples = []
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found at: {data_dir}")
            
        for file in os.listdir(data_dir):
            if file.endswith('.npy') and file.startswith('class'):
                try:
                    # File name format: class<id>_Signer...
                    class_id = int(file.split('_')[0].replace('class', ''))
                    self.samples.append({
                        'path': os.path.join(data_dir, file),
                        'class_id': class_id
                    })
                except Exception as e:
                    print(f"Skipping malformed file {file}: {e}")
                    
        print(f"SignDataset initialized with {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        arr = np.load(sample['path'])
        
        # Pad or truncate to seq_len
        if len(arr) < self.seq_len:
            pad_width = self.seq_len - len(arr)
            arr = np.pad(arr, ((0, pad_width), (0, 0), (0, 0)), mode='constant')
        elif len(arr) > self.seq_len:
            arr = arr[:self.seq_len]
            
        return torch.tensor(arr, dtype=torch.float32), sample['class_id']

def download_pretrained_weights():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    if not os.path.exists(PRETRAINED_WEIGHTS_PATH):
        url = "https://huggingface.co/ZechengLi19/Uni-Sign/resolve/main/how2sign_pose_only_slt.pth"
        print("Pretrained H2S weights not found. Downloading from HuggingFace...")
        print(f"URL: {url}")
        
        # Clean progress callback
        def reporthook(blocknum, blocksize, totalsize):
            readsofar = blocknum * blocksize
            if totalsize > 0:
                percent = readsofar * 1e2 / totalsize
                s = f"\rDownloading: {percent:.1f}% ({readsofar / (1024*1024):.2f}MB of {totalsize / (1024*1024):.2f}MB)"
                print(s, end="")
            else:
                print(f"\rRead {readsofar} bytes", end="")
                
        urllib.request.urlretrieve(url, PRETRAINED_WEIGHTS_PATH, reporthook)
        print("\nDownload complete!")
    else:
        print("Pretrained H2S weights already exist locally.")

def main():
    print("==================================================")
    print("          Zora Local Training Pipeline            ")
    print("==================================================")
    print(f"Device: {DEVICE}")
    print(f"Landmarks source: {V2_DIR}")
    print(f"Number of classes: {NUM_CLASSES}")
    
    # Check dataset existence
    try:
        dataset = SignDataset(V2_DIR)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
        
    if len(dataset) == 0:
        print("❌ Error: No samples found in V2 directory! Please make sure landmarks are extracted.")
        return
        
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Check and download weights
    download_pretrained_weights()
    
    # Initialize model
    print("Initializing ST-GCN classifier model...")
    model = UniSignDownstreamModel(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_pretrained_weights(PRETRAINED_WEIGHTS_PATH)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    best_acc = 0.0
    
    print("\nStarting local training loop...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        for keypoints, labels in train_loader:
            keypoints, labels = keypoints.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(keypoints)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * keypoints.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            
        acc = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/total:.4f} Acc: {acc:.1f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save({'model_state_dict': model.state_dict()}, BEST_MODEL_PATH)
            print(f"  → Saved best model (acc={acc:.1f}%) to: {BEST_MODEL_PATH}")
            
    print(f"\nTraining complete! Best Accuracy: {best_acc:.1f}%")
    print(f"Model saved to: {BEST_MODEL_PATH}")

if __name__ == "__main__":
    main()
