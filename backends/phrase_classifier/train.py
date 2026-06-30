import os
import csv
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import UniSignDownstreamModel

# Set random seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Explicit phrase mapping to match backend SENTENCES order
PHRASE_TO_IDX = {
    "CAN YOU HELP ME": 0,
    "CAN I HELP YOU": 1,
    "CAN YOU WAIT": 2,
    "CAN YOU REPEAT": 3,
    "CAN YOU SEE ME": 4,
    "PLEASE HELP ME": 5,
    "PLEASE WAIT": 6,
    "PLEASE REPEAT": 7,
    "I NEED HELP": 8,
    "I UNDERSTAND": 9,
    "I NOT UNDERSTAND": 10,
    "YOU UNDERSTAND": 11,
    "YOU NEED HELP": 12,
}

class SignDataset(Dataset):
    def __init__(self, x_data, y_data, augment=False):
        self.x_data = x_data
        self.y_data = y_data
        self.augment = augment

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx].copy()
        y = self.y_data[idx]

        if self.augment:
            # 1. Coordinate scaling (zoom in/out)
            scale = np.random.uniform(0.9, 1.1)
            x = x * scale

            # 2. Coordinate shifting (translation)
            shift = np.random.uniform(-0.05, 0.05, size=(1, 1, 3))
            x = x + shift

            # 3. Add small Gaussian noise
            noise = np.random.normal(0, 0.005, size=x.shape)
            x = x + noise

        # Permute to (C, T, V) = (3, 60, 75)
        x_tensor = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return x_tensor, y_tensor

def extract_joints(landmarks_flat):
    # landmarks_flat shape: (num_frames, 1662)
    num_frames = landmarks_flat.shape[0]
    
    # Pose: 1404 to 1536 -> 132 features -> shape (num_frames, 33, 4) -> slice [:, :, :3]
    pose = landmarks_flat[:, 1404:1536].reshape(num_frames, 33, 4)[:, :, :3]
    
    # Left Hand: 1536 to 1599 -> 63 features -> shape (num_frames, 21, 3)
    left_hand = landmarks_flat[:, 1536:1599].reshape(num_frames, 21, 3)
    
    # Right Hand: 1599 to 1662 -> 63 features -> shape (num_frames, 21, 3)
    right_hand = landmarks_flat[:, 1599:1662].reshape(num_frames, 21, 3)
    
    # Concatenate to (num_frames, 75, 3)
    joints = np.concatenate([pose, left_hand, right_hand], axis=1)
    return joints

def preprocess_sequence(raw_coords, target_len=60):
    # raw_coords shape: (T, 75, 3)
    T = raw_coords.shape[0]
    
    preprocessed = []
    prev_mid_shoulder = None
    prev_shoulder_width = None
    
    for t in range(T):
        pose_pts = raw_coords[t, 0:33, :]
        left_pts = raw_coords[t, 33:54, :]
        right_pts = raw_coords[t, 54:75, :]
        
        # Check if pose is detected (not all zeros)
        pose_detected = not np.all(pose_pts == 0.0)
        
        if pose_detected:
            mid_shoulder = (pose_pts[11] + pose_pts[12]) / 2.0
            shoulder_width = np.linalg.norm(pose_pts[11] - pose_pts[12])
            if shoulder_width < 1e-5:
                shoulder_width = 1.0
        else:
            mid_shoulder = prev_mid_shoulder if prev_mid_shoulder is not None else np.zeros(3, dtype=np.float32)
            shoulder_width = prev_shoulder_width if prev_shoulder_width is not None else 1.0
            
        # Center and scale pose
        if pose_detected:
            pose_pts = (pose_pts - mid_shoulder) / shoulder_width
            
        # Center and scale left hand if detected
        if not np.all(left_pts == 0.0):
            left_pts = (left_pts - mid_shoulder) / shoulder_width
        else:
            left_pts = np.zeros((21, 3), dtype=np.float32)
            
        # Center and scale right hand if detected
        if not np.all(right_pts == 0.0):
            right_pts = (right_pts - mid_shoulder) / shoulder_width
        else:
            right_pts = np.zeros((21, 3), dtype=np.float32)
            
        # Assemble frame
        frame_data = np.zeros((75, 3), dtype=np.float32)
        frame_data[0:33] = pose_pts
        frame_data[33:54] = left_pts
        frame_data[54:75] = right_pts
        
        preprocessed.append(frame_data)
        
        if pose_detected:
            prev_mid_shoulder = mid_shoulder
            prev_shoulder_width = shoulder_width
            
    # Resample to 60 frames
    F = len(preprocessed)
    if F == 0:
        return np.zeros((target_len, 75, 3), dtype=np.float32)
        
    preprocessed = np.array(preprocessed, dtype=np.float32)
    indices = np.linspace(0, F - 1, target_len)
    resampled = np.zeros((target_len, 75, 3), dtype=np.float32)
    
    for t in range(target_len):
        s = indices[t]
        s_low = int(np.floor(s))
        s_high = int(np.ceil(s))
        w = s - s_low
        resampled[t] = (1.0 - w) * preprocessed[s_low] + w * preprocessed[s_high]
        
    return resampled

def load_dataset(manifest_path, landmarks_dir):
    print("Loading manifest and pre-loading dataset into memory...")
    x_list = []
    y_list = []
    
    with open(manifest_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    total = len(rows)
    for idx, row in enumerate(rows):
        phrase = row['phrase']
        if phrase not in PHRASE_TO_IDX:
            print(f"Warning: Phrase '{phrase}' not in PHRASE_TO_IDX mapping. Skipping.")
            continue
            
        label = PHRASE_TO_IDX[phrase]
        rel_path = row['landmark_path']
        abs_path = os.path.abspath(os.path.join(landmarks_dir, rel_path))
        
        if not os.path.exists(abs_path):
            print(f"Warning: File {abs_path} not found. Skipping.")
            continue
            
        # Load landmarks
        data = np.load(abs_path)
        landmarks_flat = data['landmarks']  # shape: (num_frames, 1662)
        
        # Extract 75 joints and preprocess
        joints = extract_joints(landmarks_flat)  # (num_frames, 75, 3)
        preprocessed = preprocess_sequence(joints, target_len=60)  # (60, 75, 3)
        
        x_list.append(preprocessed)
        y_list.append(label)
        
        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"Loaded and preprocessed [{idx + 1}/{total}] files.")
            
    return np.array(x_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

def stratified_split(x_data, y_data, train_ratio=0.8, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    # Group indices by class label
    class_indices = {}
    for idx, y in enumerate(y_data):
        class_indices.setdefault(y, []).append(idx)
        
    train_idx = []
    val_idx = []
    
    for label, idxs in class_indices.items():
        random.shuffle(idxs)
        split_pt = int(len(idxs) * train_ratio)
        if split_pt == len(idxs) and len(idxs) > 1:
            split_pt = len(idxs) - 1
        train_idx.extend(idxs[:split_pt])
        val_idx.extend(idxs[split_pt:])
        
    # Shuffle splits
    random.shuffle(train_idx)
    random.shuffle(val_idx)
    
    x_train, y_train = x_data[train_idx], y_data[train_idx]
    x_val, y_val = x_data[val_idx], y_data[val_idx]
    return x_train, y_train, x_val, y_val

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

def main():
    set_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset paths
    manifest_path = r"C:\Akash\Ammus\zora\V2\extracted_landmarks\extraction_manifest.csv"
    landmarks_dir = r"C:\Akash\Ammus\zora\V2\extracted_landmarks"

    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file {manifest_path} not found.")
        return

    # Load dataset directly into memory
    x_data, y_data = load_dataset(manifest_path, landmarks_dir)
    print(f"Total processed samples in memory: X={x_data.shape}, Y={y_data.shape}")

    # Perform stratified split
    x_train, y_train, x_val, y_val = stratified_split(x_data, y_data, train_ratio=0.8, seed=42)
    print(f"Split Summary: Train size = {len(x_train)} | Val size = {len(x_val)}")

    # Create Datasets and DataLoaders
    train_dataset = SignDataset(x_train, y_train, augment=True)
    val_dataset = SignDataset(x_val, y_val, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Instantiate Model
    num_classes = len(PHRASE_TO_IDX)
    model = UniSignDownstreamModel(num_classes=num_classes).to(device)
    print(f"Initialized ST-GCN model with {num_classes} output classes.")

    # Criterion, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Output directory for weights
    weights_dir = r"C:\Akash\Ammus\zora\backend\weights"
    os.makedirs(weights_dir, exist_ok=True)
    best_weights_path = os.path.join(weights_dir, "model.pth")

    # Training Loop
    epochs = 100
    best_val_acc = 0.0
    
    print("Starting training on GPU...")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        # Save model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_weights_path)
            saved_str = " -> SAVED BEST CHECKPOINT"
        else:
            saved_str = ""

        if epoch % 5 == 0 or epoch == 1 or saved_str != "":
            print(f"Epoch {epoch:03d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%{saved_str}")

    print(f"Training complete! Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Best model weights saved to {best_weights_path}")

if __name__ == "__main__":
    main()
