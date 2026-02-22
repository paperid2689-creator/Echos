import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from pathlib import Path
import os
import numpy as np
import json
import csv
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from torchvision.models import vit_b_32

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'datasets': {
        'vimedix': '/root/Datasets/vimedix/Train',
        'cactus': '/root/Datasets/Cactus/Train/',
        'human': '/root/Datasets/Human/Train',
    },
    'model_path': '/root/vit/vit_b_32-d86f8d99.pth',
    'output_dir': '/root/final_results/final_results',
    'epochs': 25,
    'batch_size': 128,
    'learning_rate': 1e-4,
    'num_folds': 3,
    'seed': 42,
    'val_interval': 10,
}

FIXED_TEMPS = [0.02, 0.05, 0.07, 0.10, 0.20, 0.50, 1.00]
LEARNABLE_TEMPS = [0.02, 0.05, 0.07, 0.10, 0.20, 0.50, 1.00]
LOSS_TYPES = ['supervised_contrastive', 'npair', 'symmetric_contrastive']

DATASET_COMBINATIONS = {
    'vimedix_only': ['vimedix'],
    'cactus_only': ['cactus'],
    'human_only': ['human'],
    'vimedix_human': ['vimedix', 'human'],
    'vimedix_cactus_human': ['vimedix', 'cactus', 'human'],
}

# ============================================================================
# TRANSFORMS FOR EACH DATASET
# ============================================================================

class CustomCropTransform:
    def __init__(self, crop_params):
        self.crop_params = tuple(int(p) for p in crop_params)

    def __call__(self, img):
        return TF.crop(img, *self.crop_params)


class TransformFactory:
    @staticmethod
    def get_vimedix_transform():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    @staticmethod
    def get_blue_phantom_transform():
        img_initial_height = 1080
        crop_top = int(img_initial_height / 7)
        crop_left = 155
        crop_height = 656
        crop_width = 1145

        return transforms.Compose([
            CustomCropTransform((crop_top, crop_left, crop_height, crop_width)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    @staticmethod
    def get_human_transform():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    @staticmethod
    def get_transform(dataset_name):
        dataset_name = dataset_name.lower()
        if 'vimedix' in dataset_name:
            return TransformFactory.get_vimedix_transform()
        elif 'cactus' in dataset_name:
            return TransformFactory.get_blue_phantom_transform()
        elif 'human' in dataset_name:
            return TransformFactory.get_human_transform()
        else:
            return TransformFactory.get_vimedix_transform()


# ============================================================================
# DATASET CLASS
# ============================================================================

class CardiacUltrasoundDataset(Dataset):
    def __init__(self, base_paths, dataset_names, transforms_dict=None, preload=False):
        self.image_paths = []
        self.labels = []
        self.label_to_class = {}
        self.class_counter = 0
        self.dataset_names = []
        self.preload = preload       
        self.image_cache = {}         

        if isinstance(base_paths, str):
            base_paths = [base_paths]
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]

        if transforms_dict is None:
            transforms_dict = {}
        self.cached_transforms = {} 

        for base_path, dataset_name in zip(base_paths, dataset_names):
            transform = transforms_dict.get(
                dataset_name, 
                TransformFactory.get_transform(dataset_name)
            )

            base_path = Path(base_path)
            if not base_path.exists():
                raise FileNotFoundError(f"Dataset path not found: {base_path}")

            for class_folder in sorted(base_path.iterdir()):
                if not class_folder.is_dir():
                    continue

                class_name = f"{dataset_name}_{class_folder.name}"
                self.label_to_class[class_name] = self.class_counter

                image_count = 0
                for img_path in class_folder.rglob("*.jpg"):
                    self.image_paths.append(str(img_path))
                    self.labels.append(self.class_counter)
                    self.dataset_names.append(dataset_name)
                    image_count += 1

                if image_count > 0:
                    print(f"  Class {self.class_counter}: {class_name} ({image_count} images)")

                self.class_counter += 1

        for dataset_name in set(self.dataset_names):
            if transforms_dict and dataset_name in transforms_dict:
                self.cached_transforms[dataset_name] = transforms_dict[dataset_name]
            else:
                self.cached_transforms[dataset_name] = TransformFactory.get_transform(dataset_name)

        if self.preload:
            print("Preloading images to RAM...")
            for idx in range(len(self.image_paths)):
                self._load_image_cached(idx)
            print(f"Preloaded {len(self.image_cache)} images\n")

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in provided paths")

        print(f"  Total: {len(self.image_paths)} images, {self.class_counter} classes\n")
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def _load_image_cached(self, idx):
        if idx not in self.image_cache:
            try:
                image = Image.open(self.image_paths[idx]).convert("L")
                dataset_name = self.dataset_names[idx]
                transform = self.cached_transforms[dataset_name]
                image = transform(image)
                self.image_cache[idx] = image
            except Exception as e:
                self.image_cache[idx] = torch.zeros(1, 224, 224)
        return self.image_cache[idx]

    def __getitem__(self, idx):
        if self.preload:
            image = self._load_image_cached(idx)
        else:
            try:
                image = Image.open(self.image_paths[idx]).convert("L")
                dataset_name = self.dataset_names[idx]
                transform = self.cached_transforms[dataset_name]
                image = transform(image)
            except Exception as e:
                image = torch.zeros(1, 224, 224)

        label = self.labels[idx]
        return image, label

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, learnable=False):
        super().__init__()
        self.learnable = learnable

        if learnable:
            self.log_temperature = nn.Parameter(torch.tensor(np.log(temperature)))
            self.min_temp = 0.001
            self.max_temp = 1.0
        else:
            self.temperature = temperature

    @property
    def temp(self):
        if self.learnable:
            return torch.exp(self.log_temperature).clamp(min=self.min_temp, max=self.max_temp)
        else:
            return self.temperature

    def forward(self, embeddings, labels):
        device = embeddings.device
        batch_size = embeddings.shape[0]

        embeddings = F.normalize(embeddings, dim=1)
        sim_matrix = torch.mm(embeddings, embeddings.t())
        sim_matrix = torch.clamp(sim_matrix, min=-1.0, max=1.0)

        labels_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
        mask = labels_equal - torch.eye(batch_size, device=device)

        sim_matrix_scaled = sim_matrix / self.temp
        sim_max = torch.max(sim_matrix_scaled, dim=1, keepdim=True)[0]
        exp_sim = torch.exp(sim_matrix_scaled - sim_max)
        exp_sim = torch.clamp(exp_sim, min=1e-10, max=1e10)

        pos_exp_sum = (exp_sim * mask).sum(dim=1, keepdim=True)
        all_exp_sum = exp_sim.sum(dim=1, keepdim=True) - exp_sim.diag().unsqueeze(1)

        ratio = torch.clamp(pos_exp_sum / (all_exp_sum + 1e-10), min=1e-10, max=1.0)
        loss = -torch.log(ratio)
        loss = loss[mask.sum(dim=1) > 0].mean()

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=device, requires_grad=True)

        return loss


class NPairLoss(nn.Module):
    def __init__(self, temperature=0.07, learnable=False):
        super().__init__()
        self.learnable = learnable

        if learnable:
            self.log_temperature = nn.Parameter(torch.tensor(np.log(temperature)))
            self.min_temp = 0.001
            self.max_temp = 1.0
        else:
            self.temperature = temperature

    @property
    def temp(self):
        if self.learnable:
            return torch.exp(self.log_temperature).clamp(min=self.min_temp, max=self.max_temp)
        else:
            return self.temperature

    def forward(self, embeddings, labels):
        device = embeddings.device
        batch_size = embeddings.shape[0]

        embeddings = F.normalize(embeddings, dim=1, p=2)

        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        sim_matrix = torch.mm(embeddings, embeddings.t())
        sim_matrix = torch.clamp(sim_matrix, min=-1.0, max=1.0)

        labels_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
        mask = labels_equal - torch.eye(batch_size, device=device)

        sim_matrix_scaled = sim_matrix / self.temp
        sim_max = torch.max(sim_matrix_scaled, dim=1, keepdim=True)[0]
        exp_sim = torch.exp(sim_matrix_scaled - sim_max)
        exp_sim = torch.clamp(exp_sim, min=1e-10, max=1e10)

        pos_mask = mask > 0.5
        neg_mask = (1 - labels_equal)

        pos_exp_sum = (exp_sim * pos_mask).sum(dim=1, keepdim=True)
        neg_exp_sum = (exp_sim * neg_mask).sum(dim=1, keepdim=True)

        ratio = torch.clamp(pos_exp_sum / (pos_exp_sum + neg_exp_sum + 1e-10), min=1e-10, max=1.0)
        loss = -torch.log(ratio)
        loss = loss[mask.sum(dim=1) > 0].mean()

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=device, requires_grad=True)

        return loss


class SymmetricContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, learnable=False):
        super().__init__()
        self.learnable = learnable

        if learnable:
            self.log_temperature = nn.Parameter(torch.tensor(np.log(temperature)))
            self.min_temp = 0.001
            self.max_temp = 1.0
        else:
            self.temperature = temperature

    @property
    def temp(self):
        if self.learnable:
            return torch.exp(self.log_temperature).clamp(min=self.min_temp, max=self.max_temp)
        else:
            return self.temperature

    def forward(self, embeddings, labels):
        device = embeddings.device
        batch_size = embeddings.shape[0]

        embeddings = F.normalize(embeddings, dim=1, p=2)

        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        sim_matrix = torch.mm(embeddings, embeddings.t())
        sim_matrix = torch.clamp(sim_matrix, min=-1.0, max=1.0)

        labels_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
        mask = labels_equal - torch.eye(batch_size, device=device)

        sim_matrix_scaled = sim_matrix / self.temp
        sim_max = torch.max(sim_matrix_scaled, dim=1, keepdim=True)[0]
        exp_sim = torch.exp(sim_matrix_scaled - sim_max)
        exp_sim = torch.clamp(exp_sim, min=1e-10, max=1e10)

        pos_mask = mask > 0.5
        neg_mask = (1 - labels_equal)

        pos_exp_sum = (exp_sim * pos_mask).sum(dim=1, keepdim=True)
        neg_exp_sum = (exp_sim * neg_mask).sum(dim=1, keepdim=True)

        ratio = torch.clamp(pos_exp_sum / (pos_exp_sum + neg_exp_sum + 1e-10), min=1e-10, max=1.0)
        loss = -torch.log(ratio)
        loss = loss[mask.sum(dim=1) > 0].mean()

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=device, requires_grad=True)

        return loss


# ============================================================================
# VISION TRANSFORMER ENCODER
# ============================================================================

class ViTEncoder(nn.Module):
    def __init__(self, vit_model, output_dim=128):
        super().__init__()
        self.vit = vit_model

        for param in self.vit.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(768, 512, bias=False),
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

        return self.projection(features)


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_epoch(model, criterion, optimizer, train_loader, device, epoch, total_epochs, scaler):
    model.train()
    if hasattr(criterion, 'train'):
        criterion.train()

    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.cuda.amp.autocast():
            embeddings = model(images)
            loss = criterion(embeddings, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.projection.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        batch_loss = loss.item()
        total_loss += batch_loss

    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
    print(f"    Epoch {epoch+1}/{total_epochs}: Loss={avg_loss:.6f}")

    return avg_loss


def evaluate_model(model, val_loader, device):
    model.eval()

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            embeddings = model(images)

            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    knn = KNeighborsClassifier(n_neighbors=5)

    split = int(0.8 * len(embeddings))
    knn.fit(embeddings[:split], labels[:split])
    preds = knn.predict(embeddings[split:])

    accuracy = accuracy_score(labels[split:], preds)
    return accuracy


# ============================================================================
# RESULTS MANAGER
# ============================================================================

class ResultsManager:
    def __init__(self, output_dir='/root/final_results/final_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.results_file = self.output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.csv_file = self.output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.all_results = {}
        self.csv_writer = None
        self.csv_file_handle = None

        self._init_csv()

    def _init_csv(self):
        self.csv_file_handle = open(self.csv_file, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file_handle)
        self.csv_writer.writerow([
            'Timestamp', 'Dataset', 'Loss Type', 'Temperature Type',
            'Temperature Value', 'Fold', 'Final Val Acc',
            'Mean Val Acc', 'Std Val Acc', 'Min Loss', 'Final Loss', 'Epochs Trained'
        ])
        self.csv_file_handle.flush()

    def save_experiment(self, dataset_name, loss_type, temp_type, temp_value,
                       fold_results, model_path=None):

        key = f"{dataset_name}_{loss_type}_{temp_type}_{temp_value}"

        accs = [r['val_acc'] for r in fold_results]
        losses_history = [r['loss_history'] for r in fold_results]
        epochs_trained = [r['epochs_trained'] for r in fold_results]
        min_losses = [min(lh) for lh in losses_history if len(lh) > 0]
        final_losses = [lh[-1] for lh in losses_history if len(lh) > 0]

        result = {
            'timestamp': datetime.now().isoformat(),
            'dataset': dataset_name,
            'loss_type': loss_type,
            'temp_type': temp_type,
            'temp_value': temp_value,
            'fold_results': fold_results,
            'mean_val_acc': float(np.mean(accs)),
            'std_val_acc': float(np.std(accs)),
            'min_loss': float(np.min(min_losses)) if min_losses else 0,
            'final_loss': float(np.mean(final_losses)) if final_losses else 0,
            'mean_epochs_trained': float(np.mean(epochs_trained)),
            'model_path': model_path,
        }

        self.all_results[key] = result

        self.csv_writer.writerow([
            result['timestamp'],
            dataset_name,
            loss_type,
            temp_type,
            temp_value,
            'all',
            np.mean(accs),
            np.mean(accs),
            np.std(accs),
            result['min_loss'],
            result['final_loss'],
            result['mean_epochs_trained'],
        ])
        self.csv_file_handle.flush()

        self._save_json()

        return result

    def _save_json(self):
        with open(self.results_file, 'w') as f:
            json_results = {}
            for key, val in self.all_results.items():
                json_results[key] = {
                    'timestamp': val['timestamp'],
                    'dataset': val['dataset'],
                    'loss_type': val['loss_type'],
                    'temp_type': val['temp_type'],
                    'temp_value': val['temp_value'],
                    'mean_val_acc': float(val['mean_val_acc']),
                    'std_val_acc': float(val['std_val_acc']),
                    'min_loss': float(val['min_loss']),
                    'final_loss': float(val['final_loss']),
                    'mean_epochs_trained': float(val['mean_epochs_trained']),
                    'model_path': val['model_path'],
                }
            json.dump(json_results, f, indent=2)

    def close(self):
        if self.csv_file_handle:
            self.csv_file_handle.close()
        self._save_json()

    def get_results_dir(self):
        return self.output_dir


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_experiment(dataset_paths, dataset_names, loss_type, temp_type, temp_value,
                   fold, config, device):

    torch.manual_seed(config['seed'] + fold)
    np.random.seed(config['seed'] + fold)

    transforms_dict = {name: TransformFactory.get_transform(name) for name in dataset_names}
    dataset = CardiacUltrasoundDataset(dataset_paths, dataset_names, transforms_dict, preload=False)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['seed'] + fold)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=8
    )

    vit = vit_b_32(pretrained=False)
    checkpoint = torch.load(config['model_path'], map_location=device)
    vit.load_state_dict(checkpoint)
    vit = vit.to(device)

    model = ViTEncoder(vit, output_dim=128).to(device)

    learnable = (temp_type == 'learnable')

    if loss_type == 'supervised_contrastive':
        criterion = SupervisedContrastiveLoss(temperature=temp_value, learnable=learnable)
    elif loss_type == 'npair':
        criterion = NPairLoss(temperature=temp_value, learnable=learnable)
    else:
        criterion = SymmetricContrastiveLoss(temperature=temp_value, learnable=learnable)

    criterion = criterion.to(device)

    if learnable:
        optimizer = torch.optim.Adam([
            {'params': model.projection.parameters(), 'lr': config['learning_rate']},
            {'params': [criterion.log_temperature], 'lr': config['learning_rate'] * 10}
        ], weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(
            model.projection.parameters(),
            lr=config['learning_rate'],
            weight_decay=1e-4
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    scaler = torch.cuda.amp.GradScaler()

    loss_history = []
    val_acc_history = []
    temp_history = []

    for epoch in range(config['epochs']):
        loss = train_epoch(model, criterion, optimizer, train_loader, device, epoch, config['epochs'], scaler)
        loss_history.append(loss)

        if (epoch + 1) % config.get('val_interval', 10) == 0 or epoch == config['epochs'] - 1:
            val_acc = evaluate_model(model, val_loader, device)
            val_acc_history.append(val_acc)

            if learnable:
                current_temp = criterion.temp.item()
            else:
                current_temp = temp_value
            temp_history.append(current_temp)

            print(f"Val Acc={val_acc:.4f}, τ={current_temp:.4f}")
        else:
            if val_acc_history:
                val_acc_history.append(val_acc_history[-1])
                temp_history.append(temp_history[-1] if temp_history else temp_value)
            else:
                val_acc_history.append(0)
                temp_history.append(temp_value)

        scheduler.step()

    return {
        'loss_history': loss_history,
        'val_acc_history': val_acc_history,
        'temp_history': temp_history,
        'val_acc': val_acc_history[-1],
        'final_temp': temp_history[-1],
        'epochs_trained': len(loss_history),
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    results_mgr = ResultsManager(CONFIG['output_dir'])

    temp_settings = (
        [('fixed', t) for t in FIXED_TEMPS] +
        [('learnable', t) for t in LEARNABLE_TEMPS]
    )

    combinations = list(DATASET_COMBINATIONS.items())
    total = len(combinations) * len(LOSS_TYPES) * len(temp_settings)

    try:
        idx = 0
        for dataset_combo_name, dataset_keys in combinations:
            for loss_type in LOSS_TYPES:
                for temp_type, temp_value in temp_settings:
                    idx += 1
                    print(f"\n{'='*80}")
                    print(f"[{idx}/{total}] {dataset_combo_name} | {loss_type} | {temp_type}_{temp_value:.2f}")
                    print(f"{'='*80}\n")

                    dataset_paths = [CONFIG['datasets'][key] for key in dataset_keys]
                    fold_results = []

                    for fold in range(CONFIG['num_folds']):
                        try:
                            result = run_experiment(
                                dataset_paths, dataset_keys,
                                loss_type, temp_type, temp_value,
                                fold, CONFIG, device
                            )
                            fold_results.append(result)
                            print(f"  Fold {fold}: Acc={result['val_acc']:.4f}, τ={result['final_temp']:.4f}, Epochs={result['epochs_trained']}/{CONFIG['epochs']}")
                        except Exception as e:
                            print(f"Fold {fold} failed: {e}")
                            import traceback
                            traceback.print_exc()

                    if fold_results:
                        results_mgr.save_experiment(
                            dataset_combo_name, loss_type, temp_type, temp_value, fold_results
                        )
                        mean_acc = np.mean([r['val_acc'] for r in fold_results])
                        std_acc = np.std([r['val_acc'] for r in fold_results])
                        print(f"  Mean Acc: {mean_acc:.4f} ± {std_acc:.4f}\n")

    finally:
        results_mgr.close()
        print(f"Results saved to: {results_mgr.get_results_dir()}")


if __name__ == "__main__":
    main()
