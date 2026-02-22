import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import os
import numpy as np
from sklearn.metrics import accuracy_score
from torchvision.models import vit_b_32
# ============================================================================
# VIT ENCODER
# ============================================================================

class ViTEncoder(nn.Module):
    
    def __init__(self, vit_model, output_dim=128):
        super().__init__()
        
        # ViT backbone frozen
        self.vit = vit_model
        for param in self.vit.parameters():
            param.requires_grad = False
        
        feature_dim = 768  # For ViT-B/32
        
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim, bias=False),
            nn.BatchNorm1d(output_dim)
        )
    
    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        with torch.no_grad():
            features = self.vit._process_input(x)
            n, _, c = features.shape
            
            batch_class_token = self.vit.class_token.expand(n, -1, -1)
            features = torch.cat([batch_class_token, features], dim=1)
            
            features = self.vit.encoder(features)
            features = features[:, 0]
        
        output = self.projection(features)
        return output


# ============================================================================
# LABELED DATASET
# ============================================================================

class LabeledDataset(Dataset):
    
    def __init__(self, base_path, include_classes=None):
        self.image_paths = []
        self.labels = []
        self.label_to_class = {}
        self.class_counter = 0
        self.include_classes = include_classes
        
        base_path = Path(base_path)
        
        # First level folders
        for first_level_folder in sorted(base_path.iterdir()):
            if not first_level_folder.is_dir():
                continue
            
            first_level_name = first_level_folder.name
            
            # Check if this folder has subfolders with images
            subfolders = [f for f in first_level_folder.iterdir() if f.is_dir()]
            direct_images = list(first_level_folder.glob("*.jpg")) + list(first_level_folder.glob("*.png")) + list(first_level_folder.glob("*.jpeg"))
            
            if len(subfolders) > 0 and len(direct_images) == 0:
                for subfolder in sorted(subfolders):
                    class_name = f"{first_level_name}/{subfolder.name}"
                    
                    if self.include_classes is not None and class_name not in self.include_classes:
                        continue
                    
                    self.label_to_class[class_name] = self.class_counter
                    class_label = self.class_counter
                    self.class_counter += 1
                    
                    # Find images in this subfolder
                    image_count = 0
                    for img_path in subfolder.rglob("*"):
                        if img_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                            self.image_paths.append(str(img_path))
                            self.labels.append(class_label)
                            image_count += 1
                    
                    print(f"✓ Class {class_label}: {class_name:25s} ({image_count} images)")
            else:
                class_name = first_level_name
            
                if self.include_classes is not None and class_name not in self.include_classes:
                    continue
                
                self.label_to_class[class_name] = self.class_counter
                class_label = self.class_counter
                self.class_counter += 1
                
                image_count = 0
                for img_path in first_level_folder.rglob("*"):
                    if img_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                        self.image_paths.append(str(img_path))
                        self.labels.append(class_label)
                        image_count += 1
                
                print(f"✓ Class {class_label}: {class_name:25s} ({image_count} images)")
        
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {base_path}")
        
        print(f"\n Total labeled images: {len(self.image_paths)}")
        print(f"Total classes: {self.class_counter}\n")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("L")
            image = transforms.Resize(224)(image)
            image = transforms.ToTensor()(image)
        except:
            image = torch.zeros(1, 224, 224)
        
        label = self.labels[idx]
        return image, label, self.image_paths[idx]


# ============================================================================
# 3. LINEAR CLASSIFIER HEAD
# ============================================================================

class LinearClassifier(nn.Module):
    
    def __init__(self, embedding_dim=128, num_classes=6):
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, embeddings):
        logits = self.classifier(embeddings)
        return logits


# ============================================================================
# 4. EXTRACT EMBEDDINGS
# ============================================================================

def extract_embeddings(encoder, dataloader, device):
    encoder.eval()
    
    all_embeddings = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
        for images, labels, paths in dataloader:
            images = images.to(device)
            embeddings = encoder(images)
            
            all_embeddings.append(embeddings.cpu())
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)
    
    embeddings = torch.cat(all_embeddings, dim=0)
    labels = np.array(all_labels)
    
    return embeddings, labels, all_paths


# ============================================================================
# 5. TRAIN CLASSIFIER ON LABELED DATASET
# ============================================================================

def train_classifier(embeddings, labels, num_classes, num_epochs=50, learning_rate=1e-3, device="cuda"):
    
    embedding_dim = embeddings.shape[1]
    classifier = LinearClassifier(embedding_dim, num_classes).to(device)
    
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    classifier.train()
    
    print("Training LINEAR CLASSIFIER\n")
    print("Epoch | Loss     | LR")
    print("------|----------|----------")
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        batch_size = 64
        indices = torch.randperm(embeddings_tensor.shape[0])
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            
            batch_embeddings = embeddings_tensor[batch_indices]
            batch_labels = labels_tensor[batch_indices]
            
            logits = classifier(batch_embeddings)
            loss = criterion(logits, batch_labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        
        avg_loss = total_loss / num_batches
        lr = optimizer.param_groups[0]["lr"]
        
        if (epoch + 1) % 10 == 0:
            print(f"{epoch+1:5d} | {avg_loss:8.4f} | {lr:.2e}")
    
    print("\nClassifier training complete\n")
    
    return classifier.to(device)


# ============================================================================
# 6. EVALUATE ON UNLABELED DATA
# ============================================================================

def evaluate_and_copy_predictions(classifier, encoder, test_path, label_to_class, output_path, device):
    
    test_dataset = LabeledDataset(test_path, include_classes=list(label_to_class.keys()))
    
    # Map test dataset classes to labeled dataset's label_to_class
    class_name_to_part_a_label = {}
    for class_name, part_a_label in label_to_class.items():
        class_name_to_part_a_label[class_name] = part_a_label
    
    remapped_labels = []
    for idx in range(len(test_dataset)):
        _, original_label, _ = test_dataset[idx]
        test_class_name = {v: k for k, v in test_dataset.label_to_class.items()}[original_label]
        part_a_label = class_name_to_part_a_label[test_class_name]
        remapped_labels.append(part_a_label)
    
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    classifier.eval()
    encoder.eval()
    
    all_predictions = []
    all_true_labels = []
    all_paths = []
    
    label_idx = 0
    with torch.no_grad():
        for images, _, paths in test_dataloader:
            images = images.to(device)
            embeddings = encoder(images)
            logits = classifier(embeddings)
            predictions = logits.argmax(dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            batch_size = len(paths)
            all_true_labels.extend(remapped_labels[label_idx:label_idx+batch_size])
            all_paths.extend(paths)
            label_idx += batch_size
    
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    
    # Calculate accuracy
    accuracy = accuracy_score(all_true_labels, all_predictions)
    
    print("=" * 60)
    print("EVALUATION ON TEST DATA")
    print("=" * 60)
    print(f"Test Accuracy: {accuracy:.4f}\n")
    
    return accuracy

# ============================================================================
# 8. MAIN PIPELINE
# ============================================================================

def main(trained_model_path, part_a_path, part_b_path, output_path, include_classes=None):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # ========== Load trained ViT encoder ==========
    print("=" * 60)
    print("Loading trained ViT encoder")
    print("=" * 60 + "\n")

    vit = vit_b_32(pretrained=False)
    vit = vit.to(device)
    encoder = ViTEncoder(vit, output_dim=128).to(device)
    
    # Load trained weights
    checkpoint = torch.load(trained_model_path, map_location=device)
    encoder.load_state_dict(checkpoint)
    encoder.eval()  # Set to eval mode (freeze)
    print(f"✓ Loaded encoder from {trained_model_path}\n")
    
    # ========== Load Labeled Data ==========
    print("=" * 60)
    print("STEP 2: Loading labeled data (PART A)")
    print("=" * 60 + "\n")
    
    dataset_a = LabeledDataset(part_a_path, include_classes=include_classes)
    label_to_class = dataset_a.label_to_class
    num_classes = dataset_a.class_counter
    
    dataloader_a = DataLoader(
        dataset_a,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    # ========== STEP 3: Extract embeddings from frozen encoder ==========
    print("=" * 60)
    print("Extracting embeddings from frozen encoder")
    print("=" * 60 + "\n")
    
    embeddings_a, labels_a, paths_a = extract_embeddings(encoder, dataloader_a, device)
    print(f"Extracted embeddings shape: {embeddings_a.shape}\n")
    
    # ========== Train linear classifier on frozen embeddings ==========
    print("=" * 60)
    print("STEP 4: Training linear classifier on frozen embeddings")
    print("=" * 60 + "\n")
    
    classifier = train_classifier(
        embeddings_a.numpy(), 
        labels_a, 
        num_classes, 
        num_epochs=50, 
        learning_rate=1e-3,
        device=device
    )
    
    # ========== Evaluate on Part B (Labeled Test) ==========
    print("=" * 60)
    print("STEP 5: Evaluating and copying predictions on Part B (Test)")
    print("=" * 60 + "\n")
    
    evaluate_and_copy_predictions(classifier, encoder, part_b_path, label_to_class, output_path, device)
    
    print("=" * 60)
    print("✓ PIPELINE COMPLETE")
    print("=" * 60)


# ============================================================================
# 9. RUN
# ============================================================================

if __name__ == "__main__":
    
    trained_model_path = "vit_supervised_contrastive_trained.pth"  #  trained model
    part_a_path = "/labeled_data"  # Training data
    part_b_path = "/unlabeled_data"  # Test data
    output_path = "/output"  # Output directory
    
    include_classes = None  # Use all classes from labeled data
    
    # Check paths
    if not os.path.exists(trained_model_path):
        print(f" Error: {trained_model_path} not found")
        exit(1)
    
    if not os.path.exists(part_a_path):
        print(f" Error: {part_a_path} not found")
        exit(1)
    
    if not os.path.exists(part_b_path):
        print(f" Error: {part_b_path} not found")
        exit(1)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Run pipeline
    main(trained_model_path, part_a_path, part_b_path, output_path, include_classes=include_classes)