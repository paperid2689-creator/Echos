import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
from collections import defaultdict
import json

# ============================================================================
# DATASET - Load paired images
# ============================================================================

class PairedDataset(Dataset):
    
    def __init__(self, pairs_path, config_a_labels_path):
        self.pairs_path = Path(pairs_path)
        self.config_a_labels_path = Path(config_a_labels_path)
        
        self.image_pairs = []
        self.label_to_class = {}
        self.class_counter = 0
        
        print("="*60)
        print("BUILDING LABEL MAPPING FROM CONFIG A")
        print("="*60)
        self._build_label_mapping()
        
        print("\n" + "="*60)
        print("LOADING PAIRED IMAGES (FOR CONTRASTIVE)")
        print("="*60)
        self._load_pairs()
        
        print(f"\n✓ Total pairs: {len(self.image_pairs)}")
        print(f"✓ Total classes: {self.class_counter}\n")
    
    def _build_label_mapping(self):
        for view_folder in sorted(self.config_a_labels_path.iterdir()):
            if not view_folder.is_dir():
                continue
            
            view_name = view_folder.name
            subfolders = [f for f in view_folder.iterdir() if f.is_dir()]
            
            if len(subfolders) > 0:
                for grade_folder in sorted(subfolders):
                    class_name = f"{view_name}/{grade_folder.name}"
                    self.label_to_class[class_name] = self.class_counter
                    print(f"  Class {self.class_counter}: {class_name}")
                    self.class_counter += 1
            else:
                self.label_to_class[view_name] = self.class_counter
                print(f"  Class {self.class_counter}: {view_name}")
                self.class_counter += 1
    
    def _load_pairs(self):
        for class_folder in sorted(self.pairs_path.iterdir()):
            if not class_folder.is_dir():
                continue
            
            class_name = class_folder.name
            
            if class_name not in self.label_to_class:
                continue
            
            label = self.label_to_class[class_name]
            
            images = sorted([p for p in class_folder.glob("*") 
                           if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
            
            if len(images) >= 2:
                for i in range(0, len(images) - 1, 2):
                    path_a = str(images[i])
                    path_b = str(images[i + 1])
                    self.image_pairs.append((path_a, path_b, label))
            
            print(f"Class '{class_name}': {len(images)//2} pairs")
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        path_a, path_b, label = self.image_pairs[idx]
        
        try:
            img_a = Image.open(path_a).convert("L")
            img_a = transforms.Resize((224, 224))(img_a)
            img_a = transforms.ToTensor()(img_a)
        except:
            img_a = torch.zeros(1, 224, 224)
        
        try:
            img_b = Image.open(path_b).convert("L")
            img_b = transforms.Resize((224, 224))(img_b)
            img_b = transforms.ToTensor()(img_b)
        except:
            img_b = torch.zeros(1, 224, 224)
        
        return img_a, img_b, label


# ============================================================================
# MODEL
# ============================================================================

class DualHeadViT(nn.Module):
    """ViT with two heads: projection (contrastive) and classification"""
    
    def __init__(self, vit_model, output_dim=128, num_classes=9):
        super().__init__()
        
        self.vit = vit_model
        feature_dim = 768
        
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim, bias=False),
            nn.BatchNorm1d(output_dim)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        features = self.vit._process_input(x)
        n, _, c = features.shape
        
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        features = torch.cat([batch_class_token, features], dim=1)
        features = self.vit.encoder(features)
        features = features[:, 0]
        
        embeddings = self.projection(features)
        logits = self.classifier(features)
        
        return embeddings, logits


# ============================================================================
# CONTRASTIVE LOSS / SUPERVISED CONTRASTIVE LOSS
# ============================================================================

class SupervisedContrastiveLoss(nn.Module):
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings_a, embeddings_b, labels):
        device = embeddings_a.device
        batch_size = embeddings_a.shape[0]
        
        embeddings_a = F.normalize(embeddings_a, dim=1)
        embeddings_b = F.normalize(embeddings_b, dim=1)
        
        all_embeddings = torch.cat([embeddings_a, embeddings_b], dim=0)
        all_labels = torch.cat([labels, labels], dim=0)
        
        sim_matrix = torch.mm(all_embeddings, all_embeddings.t())
        
        loss_list = []
        for i in range(2 * batch_size):
            mask = (all_labels == all_labels[i]).float()
            mask[i] = 0
            
            positives_idx = torch.nonzero(mask > 0, as_tuple=True)[0]
            
            if len(positives_idx) == 0:
                continue
            
            sim_i = sim_matrix[i] / self.temperature
            sim_i_max = sim_i.max()
            exp_sim = torch.exp(sim_i - sim_i_max)
            
            pos_exp_sum = exp_sim[positives_idx].sum()
            all_exp_sum = exp_sim.sum() - exp_sim[i]
            
            if all_exp_sum > 0 and pos_exp_sum > 0:
                loss_i = -torch.log(pos_exp_sum / all_exp_sum)
                loss_list.append(loss_i)
        
        if len(loss_list) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return torch.stack(loss_list).mean()


# ============================================================================
# TRAINING FUNCTION ( Train with EQUAL WEIGHT LOSS: 1.0 * contrastive + 1.0 * classification)
# ============================================================================

def train(pairs_path, config_a_labels_path, vit_model_path, label_percent=0.01, 
          epochs=30, batch_size=32, device="cuda"):
    
    print("\n" + "="*80)
    print(f"SEMI-SUPERVISED LEARNING: {label_percent*100:.1f}% LABELED DATA")
    print("="*80)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Loss: 1.0 * contrastive_loss + 1.0 * classification_loss\n")
    
    # 1. Load ViT
    print("Loading ViT-B/32...")
    from torchvision.models import vit_b_32
    vit = vit_b_32(weights=None)
    checkpoint = torch.load(vit_model_path, map_location=device)
    vit.load_state_dict(checkpoint)
    vit = vit.to(device)
    print("✓ ViT loaded\n")
    
    # 2. Load paired dataset
    print("Loading paired dataset (for contrastive learning)...")
    full_dataset = PairedDataset(pairs_path, config_a_labels_path)
    num_classes = full_dataset.class_counter
    
    # 3. Create labeled subset
    print(f"\nCreating {label_percent*100:.1f}% labeled subset...")
    class_to_indices = defaultdict(list)
    for idx, (_, _, label) in enumerate(full_dataset.image_pairs):
        class_to_indices[label].append(idx)
    
    labeled_indices = []
    for class_id, indices in sorted(class_to_indices.items()):
        class_name = [k for k, v in full_dataset.label_to_class.items() if v == class_id][0]
        n_labeled = max(1, int(len(indices) * label_percent))
        selected = np.random.choice(indices, size=n_labeled, replace=False)
        labeled_indices.extend(selected)
        print(f"  Class {class_name:30s}: {n_labeled} labeled samples")
    
    np.random.shuffle(labeled_indices)
    n_train = int(len(labeled_indices) * 0.8)
    train_indices = labeled_indices[:n_train]
    val_indices = labeled_indices[n_train:]
    
    print(f"\n✓ Train samples: {len(train_indices)}")
    print(f"✓ Val samples: {len(val_indices)}")
    print(f"✓ Pairs (all) for contrastive: {len(full_dataset)}\n")
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    # 4. Compute class weights
    print("Computing class weights...")
    class_counts = np.zeros(num_classes)
    for _, _, label in full_dataset.image_pairs:
        class_counts[label] += 1
    
    class_weights = 1.0 / (class_counts + 1e-8)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # 5. Model and losses
    print("Initializing model...")
    model = DualHeadViT(vit, output_dim=128, num_classes=num_classes).to(device)
    
    contrastive_loss_fn = SupervisedContrastiveLoss(temperature=0.07)
    classification_loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
    
    optimizer = torch.optim.Adam([
        {'params': model.vit.parameters(), 'lr': 1e-4},
        {'params': model.projection.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)
    
    print("Optimizer: Adam with fixed LR=1e-4")
    print("Loss weights: 1.0 * contrastive + 1.0 * classification\n")
    
    # 6. Training loop
    print("Starting training...\n")
    print("Epoch | Contrastive | Classification | Train Acc | Val Acc")
    print("-" * 60)
    
    best_val_acc = 0.0
    best_epoch = 0
    
    metrics = {
        "label_percent": label_percent,
        "epochs": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_loss": [],
        "val_loss": [],
        "contrastive_loss": [],
        "classification_loss": []
    }
    
    for epoch in range(epochs):
        model.train()
        total_cont_loss = 0
        total_class_loss = 0
        total_correct = 0
        total_samples = 0
        num_batches = 0
        
        for img_a, img_b, labels in train_dataloader:
            img_a = img_a.to(device)
            img_b = img_b.to(device)
            labels = labels.to(device)
            
            # Forward pass
            embeddings_a, logits_a = model(img_a)
            embeddings_b, logits_b = model(img_b)
            
            # Losses
            cont_loss = contrastive_loss_fn(embeddings_a, embeddings_b, labels)
            class_loss = classification_loss_fn(logits_a, labels)
            
            # EQUAL WEIGHT: 1.0 * contrastive + 1.0 * classification
            loss = 1.0 * cont_loss + 1.0 * class_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_cont_loss += cont_loss.item()
            total_class_loss += class_loss.item()
            
            predictions = logits_a.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.shape[0]
            
            num_batches += 1
        
        train_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        avg_train_loss = total_class_loss / num_batches
        avg_cont_loss = total_cont_loss / num_batches
        
        # Validation
        model.eval()
        val_correct = 0
        val_samples = 0
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for img_a, img_b, labels in val_dataloader:
                img_a = img_a.to(device)
                labels = labels.to(device)
                
                _, logits_a = model(img_a)
                val_loss += classification_loss_fn(logits_a, labels).item()
                
                predictions = logits_a.argmax(dim=1)
                val_correct += (predictions == labels).sum().item()
                val_samples += labels.shape[0]
                val_batches += 1
        
        val_accuracy = val_correct / val_samples if val_samples > 0 else 0.0
        avg_val_loss = val_loss / val_batches
        
        print(f"{epoch+1:5d} | {avg_cont_loss:11.4f} | "
              f"{avg_train_loss:14.4f} | {train_accuracy:9.4f} | {val_accuracy:7.4f}")
        
        metrics["epochs"].append(epoch + 1)
        metrics["train_accuracy"].append(float(train_accuracy))
        metrics["val_accuracy"].append(float(val_accuracy))
        metrics["train_loss"].append(float(avg_train_loss))
        metrics["val_loss"].append(float(avg_val_loss))
        metrics["contrastive_loss"].append(float(avg_cont_loss))
        metrics["classification_loss"].append(float(avg_train_loss))
        
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_epoch = epoch + 1
            model_name = f"model_{label_percent*100:.0f}percent_equal_weight.pth"
            torch.save(model.state_dict(), model_name)
            print(f"  ✓ Best model (val_acc={val_accuracy:.4f}): {model_name}")
    
    print("\n" + "="*60)
    print(f"✓ Training complete ({label_percent*100:.1f}%)")
    print(f"✓ Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"✓ Best epoch: {best_epoch}")
    print("="*60 + "\n")
    
    # Save metrics
    metrics_file = f"metrics_{label_percent*100:.0f}percent_equal_weight.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    return best_val_acc


# ============================================================================
# 5. MAIN
# ============================================================================

if __name__ == "__main__":
    
    pairs_path = "/Contrastive_Learning/Pairs"
    config_a_labels_path = "/Contrastive_Learning/Labeled_data"
    vit_model_path = "/vit_b_32-d86f8d99.pth"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    label_percentages = [0.01, 0.05, 0.10]  # 1%, 5%, 10%
    
    results = {}
    
    for label_percent in label_percentages:
        best_val_acc = train(
            pairs_path=pairs_path,
            config_a_labels_path=config_a_labels_path,
            vit_model_path=vit_model_path,
            label_percent=label_percent,
            epochs=30,
            batch_size=32,
            device=device
        )
        
        results[f"{label_percent*100:.0f}%"] = best_val_acc
    
    print("\n" + "="*80)
    print("BEST VALIDATION ACCURACY")
    print("="*80)
    for label_str in sorted(results.keys(), key=lambda x: float(x.rstrip('%'))):
        acc = results[label_str]
        print(f"  {label_str:5s} labeled: {acc*100:.2f}%")
    print("="*80 + "\n")
    
    # Save summary
    summary = {
        "loss_config": "1.0 * contrastive + 1.0 * classification",
        "label_percentages": [0.01, 0.05, 0.10],
        "results": {k: float(v) for k, v in results.items()}
    }
    
    with open("summary_equal_weight_minimal.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("✓ Summary saved: summary_equal_weight_minimal.json\n")