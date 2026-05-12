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

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, average_precision_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.utils import resample

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from torchvision.models import vit_b_32

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'datasets': {
        'vimedix': '/root/Datasets/vimedix/Train',
        'cactus':  '/root/Datasets/Cactus/Train/',
        'human':   '/root/Datasets/Human/Train',
    },
    'model_path':   '/root/vit/vit_b_32-d86f8d99.pth',
    'output_dir':   '/root/final_results/final_results',
    'epochs':       25,
    'batch_size':   128,
    'learning_rate': 1e-4,
    'num_folds':    3,
    'seed':         42,
    'val_interval': 10,
    'tsne_enabled': True,   # set False to skip t-SNE (saves time)
    'bootstrap_n':  1000,   # number of bootstrap iterations for CI
}

FIXED_TEMPS    = [0.02, 0.05, 0.07, 0.10, 0.20, 0.50, 1.00]
LEARNABLE_TEMPS = [0.02, 0.05, 0.07, 0.10, 0.20, 0.50, 1.00]
LOSS_TYPES     = ['supervised_contrastive', 'npair', 'symmetric_contrastive']

DATASET_COMBINATIONS = {
    'vimedix_only':         ['vimedix'],
    'cactus_only':          ['cactus'],
    'human_only':           ['human'],
    'vimedix_human':        ['vimedix', 'human'],
    'vimedix_cactus_human': ['vimedix', 'cactus', 'human'],
}

# ============================================================================
# TRANSFORMS
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
        crop_top    = int(img_initial_height / 7)
        crop_left   = 155
        crop_height = 656
        crop_width  = 1145
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
# DATASET
# ============================================================================

class CardiacUltrasoundDataset(Dataset):
    def __init__(self, base_paths, dataset_names, transforms_dict=None, preload=False):
        self.image_paths    = []
        self.labels         = []
        self.label_to_class = {}
        self.class_counter  = 0
        self.dataset_names  = []
        self.preload        = preload
        self.image_cache    = {}

        if isinstance(base_paths, str):
            base_paths = [base_paths]
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]

        if transforms_dict is None:
            transforms_dict = {}
        self.cached_transforms = {}

        for base_path, dataset_name in zip(base_paths, dataset_names):
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
            raise RuntimeError("No images found in provided paths")

        print(f"  Total: {len(self.image_paths)} images, {self.class_counter} classes\n")
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def _load_image_cached(self, idx):
        if idx not in self.image_cache:
            try:
                image   = Image.open(self.image_paths[idx]).convert("L")
                transform = self.cached_transforms[self.dataset_names[idx]]
                self.image_cache[idx] = transform(image)
            except Exception:
                self.image_cache[idx] = torch.zeros(1, 224, 224)
        return self.image_cache[idx]

    def __getitem__(self, idx):
        if self.preload:
            image = self._load_image_cached(idx)
        else:
            try:
                image   = Image.open(self.image_paths[idx]).convert("L")
                transform = self.cached_transforms[self.dataset_names[idx]]
                image   = transform(image)
            except Exception:
                image = torch.zeros(1, 224, 224)
        return image, self.labels[idx]


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
            return torch.exp(self.log_temperature).clamp(self.min_temp, self.max_temp)
        return self.temperature

    def forward(self, embeddings, labels):
        device     = embeddings.device
        batch_size = embeddings.shape[0]

        embeddings = F.normalize(embeddings, dim=1)
        sim_matrix = torch.clamp(torch.mm(embeddings, embeddings.t()), -1.0, 1.0)

        labels_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
        mask         = labels_equal - torch.eye(batch_size, device=device)

        sim_scaled = sim_matrix / self.temp
        sim_max    = sim_scaled.max(dim=1, keepdim=True)[0]
        exp_sim    = torch.clamp(torch.exp(sim_scaled - sim_max), 1e-10, 1e10)

        pos_sum = (exp_sim * mask).sum(dim=1, keepdim=True)
        all_sum = exp_sim.sum(dim=1, keepdim=True) - exp_sim.diag().unsqueeze(1)

        ratio = torch.clamp(pos_sum / (all_sum + 1e-10), 1e-10, 1.0)
        loss  = -torch.log(ratio)
        loss  = loss[mask.sum(dim=1) > 0].mean()

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
            return torch.exp(self.log_temperature).clamp(self.min_temp, self.max_temp)
        return self.temperature

    def forward(self, embeddings, labels):
        device     = embeddings.device
        batch_size = embeddings.shape[0]

        embeddings = F.normalize(embeddings, dim=1, p=2)
        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        sim_matrix   = torch.clamp(torch.mm(embeddings, embeddings.t()), -1.0, 1.0)
        labels_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
        mask         = labels_equal - torch.eye(batch_size, device=device)

        sim_scaled = sim_matrix / self.temp
        sim_max    = sim_scaled.max(dim=1, keepdim=True)[0]
        exp_sim    = torch.clamp(torch.exp(sim_scaled - sim_max), 1e-10, 1e10)

        pos_mask = mask > 0.5
        neg_mask = (1 - labels_equal)

        pos_sum = (exp_sim * pos_mask).sum(dim=1, keepdim=True)
        neg_sum = (exp_sim * neg_mask).sum(dim=1, keepdim=True)

        ratio = torch.clamp(pos_sum / (pos_sum + neg_sum + 1e-10), 1e-10, 1.0)
        loss  = -torch.log(ratio)
        loss  = loss[mask.sum(dim=1) > 0].mean()

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
            return torch.exp(self.log_temperature).clamp(self.min_temp, self.max_temp)
        return self.temperature

    def forward(self, embeddings, labels):
        device     = embeddings.device
        batch_size = embeddings.shape[0]

        embeddings = F.normalize(embeddings, dim=1, p=2)
        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        sim_matrix   = torch.clamp(torch.mm(embeddings, embeddings.t()), -1.0, 1.0)
        labels_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
        mask         = labels_equal - torch.eye(batch_size, device=device)

        sim_scaled = sim_matrix / self.temp
        sim_max    = sim_scaled.max(dim=1, keepdim=True)[0]
        exp_sim    = torch.clamp(torch.exp(sim_scaled - sim_max), 1e-10, 1e10)

        pos_mask = mask > 0.5
        neg_mask = (1 - labels_equal)

        pos_sum = (exp_sim * pos_mask).sum(dim=1, keepdim=True)
        neg_sum = (exp_sim * neg_mask).sum(dim=1, keepdim=True)

        ratio = torch.clamp(pos_sum / (pos_sum + neg_sum + 1e-10), 1e-10, 1.0)
        loss  = -torch.log(ratio)
        loss  = loss[mask.sum(dim=1) > 0].mean()

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
            features          = self.vit._process_input(x)
            n, _, c           = features.shape
            batch_class_token = self.vit.class_token.expand(n, -1, -1)
            features          = torch.cat([batch_class_token, features], dim=1)
            features          = self.vit.encoder(features)
            features          = features[:, 0]

        return self.projection(features)


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, criterion, optimizer, train_loader, device,
                epoch, total_epochs, scaler):
    model.train()
    if hasattr(criterion, 'train'):
        criterion.train()

    total_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.cuda.amp.autocast():
            embeddings = model(images)
            loss       = criterion(embeddings, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.projection.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader) if train_loader else 0.0
    print(f"    Epoch {epoch+1}/{total_epochs}: Loss={avg_loss:.6f}")
    return avg_loss


# ============================================================================
# EVALUATION  ← UPDATED
# ============================================================================

def evaluate_model(model, val_loader, device, num_classes, return_embeddings=False):
    """
    Runs KNN evaluation on the validation set and returns a comprehensive
    metrics dictionary including accuracy, macro-F1, AUROC, per-class
    sensitivity/specificity, and confusion matrix.

    Parameters
    ----------
    model            : trained ViTEncoder
    val_loader       : DataLoader for validation split
    device           : torch device
    num_classes      : total number of classes in the dataset
    return_embeddings: if True, also return raw embeddings and labels
                       (used for t-SNE on the final epoch)
    """
    model.eval()
    all_embeddings, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            all_embeddings.append(model(images).cpu())
            all_labels.append(labels.cpu())

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    labels     = torch.cat(all_labels,     dim=0).numpy()

    split = int(0.8 * len(embeddings))
    knn   = KNeighborsClassifier(n_neighbors=5)
    knn.fit(embeddings[:split], labels[:split])

    true  = labels[split:]
    preds = knn.predict(embeddings[split:])
    probs = knn.predict_proba(embeddings[split:])   # shape: (N, num_classes)

    # --- confusion matrix & per-class sensitivity / specificity ----------
    present_classes = np.unique(true)
    cm = confusion_matrix(true, preds, labels=list(range(num_classes)))

    per_class_sensitivity, per_class_specificity = [], []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        per_class_sensitivity.append(float(tp / (tp + fn + 1e-10)))
        per_class_specificity.append(float(tn / (tn + fp + 1e-10)))

    # --- AUROC (guard against missing classes in small val splits) --------
    try:
        # knn.predict_proba columns follow knn.classes_ order; we need to
        # pad to full num_classes if some classes are absent from the val set
        full_probs = np.zeros((len(true), num_classes))
        for col_idx, cls in enumerate(knn.classes_):
            full_probs[:, cls] = probs[:, col_idx]

        auroc = roc_auc_score(
            true, full_probs,
            multi_class='ovr',
            average='macro',
            labels=list(range(num_classes))
        )
    except ValueError as e:
        print(f"    [AUROC warning] {e}")
        auroc = float('nan')

    # --- per-class F1 (zero_division=0 avoids warnings) ------------------
    per_class_f1 = f1_score(
        true, preds,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0
    ).tolist()

    metrics = {
        'accuracy':              float(accuracy_score(true, preds)),
        'macro_f1':              float(f1_score(true, preds, average='macro', zero_division=0)),
        'per_class_f1':          per_class_f1,
        'auroc':                 float(auroc),
        'mean_sensitivity':      float(np.mean(per_class_sensitivity)),
        'mean_specificity':      float(np.mean(per_class_specificity)),
        'per_class_sensitivity': per_class_sensitivity,
        'per_class_specificity': per_class_specificity,
        'confusion_matrix':      cm.tolist(),
        # keep true/preds for bootstrap CI in ResultsManager
        '_true':                 true,
        '_preds':                preds,
    }

    if return_embeddings:
        metrics['_embeddings'] = embeddings
        metrics['_labels']     = labels

    return metrics


# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVAL  ← NEW
# ============================================================================

def bootstrap_ci(true, preds, metric_fn, n_bootstrap=1000, ci=95, seed=42):
    """
    Compute mean and (ci)% confidence interval for a metric via bootstrap.

    Parameters
    ----------
    true, preds  : array-like ground truth and predictions
    metric_fn    : callable(true, preds) -> float
    n_bootstrap  : number of resampling iterations
    ci           : confidence level (e.g. 95 → 2.5th–97.5th percentile)
    seed         : random seed for reproducibility

    Returns
    -------
    (mean, lower, upper) as floats
    """
    rng    = np.random.RandomState(seed)
    scores = []
    true   = np.asarray(true)
    preds  = np.asarray(preds)

    for _ in range(n_bootstrap):
        idx = rng.choice(len(true), size=len(true), replace=True)
        try:
            scores.append(metric_fn(true[idx], preds[idx]))
        except Exception:
            pass

    if not scores:
        return float('nan'), float('nan'), float('nan')

    alpha = (100 - ci) / 2
    return (
        float(np.mean(scores)),
        float(np.percentile(scores, alpha)),
        float(np.percentile(scores, 100 - alpha)),
    )


# ============================================================================
# t-SNE VISUALIZATION  ← NEW
# ============================================================================

def save_tsne(embeddings, labels, dataset_source_per_sample,
              output_path, title="", class_names=None):
    """
    Produce a two-panel t-SNE figure:
      Left  — coloured by class label
      Right — coloured by dataset source (domain)

    Parameters
    ----------
    embeddings               : np.ndarray (N, D)
    labels                   : np.ndarray (N,) int class indices
    dataset_source_per_sample: list[str] of length N, e.g. ['vimedix', ...]
    output_path              : str or Path where the PNG is saved
    title                    : figure suptitle prefix
    class_names              : optional dict {int -> str} for legend labels
    """
    print("  Computing t-SNE (this may take a moment)...")
    max_samples = 5000   # cap for speed; remove if you have time to spare
    if len(embeddings) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(embeddings), max_samples, replace=False)
        embeddings              = embeddings[idx]
        labels                  = labels[idx]
        dataset_source_per_sample = [dataset_source_per_sample[i] for i in idx]

    proj = TSNE(
        n_components=2, random_state=42,
        perplexity=30, n_iter=1000, learning_rate='auto', init='pca'
    ).fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=11)

    # --- Panel 1: by class label -----------------------------------------
    unique_labels = np.unique(labels)
    colors_cls    = cm.tab20(np.linspace(0, 1, max(len(unique_labels), 1)))
    for i, lbl in enumerate(unique_labels):
        mask    = labels == lbl
        display = class_names[lbl] if (class_names and lbl in class_names) else str(lbl)
        axes[0].scatter(proj[mask, 0], proj[mask, 1],
                        c=[colors_cls[i % len(colors_cls)]],
                        label=display, s=4, alpha=0.6, linewidths=0)
    axes[0].set_title("Embedding Space — by Class")
    axes[0].legend(markerscale=3, fontsize=7, loc='best',
                   framealpha=0.6, ncol=max(1, len(unique_labels) // 8))
    axes[0].axis('off')

    # --- Panel 2: by domain source ---------------------------------------
    unique_domains = sorted(set(dataset_source_per_sample))
    colors_dom     = cm.Set1(np.linspace(0, 1, max(len(unique_domains), 1)))
    src_arr        = np.array(dataset_source_per_sample)
    for i, dname in enumerate(unique_domains):
        mask = src_arr == dname
        axes[1].scatter(proj[mask, 0], proj[mask, 1],
                        c=[colors_dom[i % len(colors_dom)]],
                        label=dname, s=4, alpha=0.6, linewidths=0)
    axes[1].set_title("Embedding Space — by Domain")
    axes[1].legend(markerscale=3, fontsize=8, loc='best', framealpha=0.6)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  t-SNE saved → {output_path}")


# ============================================================================
# RESULTS MANAGER  ← UPDATED
# ============================================================================

class ResultsManager:
    def __init__(self, output_dir='/root/final_results/final_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_file    = self.output_dir / f"results_{ts}.json"
        self.csv_file        = self.output_dir / f"results_{ts}.csv"
        self.all_results     = {}
        self.csv_writer      = None
        self.csv_file_handle = None

        self._init_csv()

    def _init_csv(self):
        self.csv_file_handle = open(self.csv_file, 'w', newline='')
        self.csv_writer      = csv.writer(self.csv_file_handle)
        # ← expanded columns
        self.csv_writer.writerow([
            'Timestamp', 'Dataset', 'Loss Type', 'Temperature Type', 'Temperature Value',
            'Mean Accuracy', 'Std Accuracy', 'CI95 Lower', 'CI95 Upper',
            'Mean Macro F1', 'Std Macro F1',
            'Mean AUROC', 'Std AUROC',
            'Mean Sensitivity', 'Std Sensitivity',
            'Mean Specificity', 'Std Specificity',
            'Min Loss', 'Final Loss', 'Epochs Trained',
        ])
        self.csv_file_handle.flush()

    # ------------------------------------------------------------------
    def save_experiment(self, dataset_name, loss_type, temp_type, temp_value,
                        fold_results, model_path=None,
                        n_bootstrap=1000):
        """
        Aggregate metrics across folds, compute bootstrap CI on accuracy,
        write to CSV and JSON, and return the result dict.
        """
        key = f"{dataset_name}_{loss_type}_{temp_type}_{temp_value}"

        # ---- per-fold scalars ----------------------------------------
        accs          = [r['val_metrics']['accuracy']         for r in fold_results]
        f1s           = [r['val_metrics']['macro_f1']         for r in fold_results]
        aurocs        = [r['val_metrics']['auroc']            for r in fold_results]
        sensitivities = [r['val_metrics']['mean_sensitivity'] for r in fold_results]
        specificities = [r['val_metrics']['mean_specificity'] for r in fold_results]

        losses_history = [r['loss_history']   for r in fold_results]
        epochs_trained = [r['epochs_trained'] for r in fold_results]
        min_losses     = [min(lh) for lh in losses_history if lh]
        final_losses   = [lh[-1]  for lh in losses_history if lh]

        # ---- bootstrap CI on accuracy (last fold as representative) --
        last_true  = fold_results[-1].get('_true',  np.array([]))
        last_preds = fold_results[-1].get('_preds', np.array([]))
        if len(last_true) > 0:
            _, ci_low, ci_high = bootstrap_ci(
                last_true, last_preds,
                metric_fn=accuracy_score,
                n_bootstrap=n_bootstrap,
            )
        else:
            ci_low, ci_high = float('nan'), float('nan')

        # ---- confusion matrix (average across folds element-wise) ----
        cms = [np.array(r['val_metrics']['confusion_matrix']) for r in fold_results]
        mean_cm = np.mean(cms, axis=0).round(1).tolist() if cms else []

        result = {
            'timestamp':           datetime.now().isoformat(),
            'dataset':             dataset_name,
            'loss_type':           loss_type,
            'temp_type':           temp_type,
            'temp_value':          temp_value,
            # accuracy
            'mean_val_acc':        float(np.mean(accs)),
            'std_val_acc':         float(np.std(accs)),
            'ci95_lower':          ci_low,
            'ci95_upper':          ci_high,
            # F1
            'mean_macro_f1':       float(np.nanmean(f1s)),
            'std_macro_f1':        float(np.nanstd(f1s)),
            # AUROC
            'mean_auroc':          float(np.nanmean(aurocs)),
            'std_auroc':           float(np.nanstd(aurocs)),
            # sensitivity / specificity
            'mean_sensitivity':    float(np.mean(sensitivities)),
            'std_sensitivity':     float(np.std(sensitivities)),
            'mean_specificity':    float(np.mean(specificities)),
            'std_specificity':     float(np.std(specificities)),
            # loss
            'min_loss':            float(np.min(min_losses))   if min_losses  else 0.0,
            'final_loss':          float(np.mean(final_losses)) if final_losses else 0.0,
            'mean_epochs_trained': float(np.mean(epochs_trained)),
            # confusion matrix
            'mean_confusion_matrix': mean_cm,
            'model_path':          model_path,
        }

        self.all_results[key] = result

        self.csv_writer.writerow([
            result['timestamp'], dataset_name, loss_type, temp_type, temp_value,
            result['mean_val_acc'],     result['std_val_acc'],
            ci_low,                     ci_high,
            result['mean_macro_f1'],    result['std_macro_f1'],
            result['mean_auroc'],       result['std_auroc'],
            result['mean_sensitivity'], result['std_sensitivity'],
            result['mean_specificity'], result['std_specificity'],
            result['min_loss'],         result['final_loss'],
            result['mean_epochs_trained'],
        ])
        self.csv_file_handle.flush()
        self._save_json()
        return result

    # ------------------------------------------------------------------
    def _save_json(self):
        with open(self.results_file, 'w') as f:
            # strip internal numpy arrays before serialising
            serialisable = {}
            for key, val in self.all_results.items():
                serialisable[key] = {
                    k: v for k, v in val.items()
                    if not isinstance(v, np.ndarray)
                }
            json.dump(serialisable, f, indent=2)

    def close(self):
        if self.csv_file_handle:
            self.csv_file_handle.close()
        self._save_json()

    def get_results_dir(self):
        return self.output_dir


# ============================================================================
# EXPERIMENT RUNNER  ← UPDATED
# ============================================================================

def run_experiment(dataset_paths, dataset_names, loss_type, temp_type, temp_value,
                   fold, config, device):

    torch.manual_seed(config['seed'] + fold)
    np.random.seed(config['seed'] + fold)

    transforms_dict = {name: TransformFactory.get_transform(name) for name in dataset_names}
    dataset         = CardiacUltrasoundDataset(
        dataset_paths, dataset_names, transforms_dict, preload=False
    )
    num_classes = dataset.class_counter   # ← needed for evaluate_model

    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config['seed'] + fold)
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config['batch_size'],
                              num_workers=8)

    # ---- model ----------------------------------------------------------
    vit = vit_b_32(pretrained=False)
    checkpoint = torch.load(config['model_path'], map_location=device)
    vit.load_state_dict(checkpoint)
    vit   = vit.to(device)
    model = ViTEncoder(vit, output_dim=128).to(device)

    # ---- loss -----------------------------------------------------------
    learnable = (temp_type == 'learnable')
    if loss_type == 'supervised_contrastive':
        criterion = SupervisedContrastiveLoss(temperature=temp_value, learnable=learnable)
    elif loss_type == 'npair':
        criterion = NPairLoss(temperature=temp_value, learnable=learnable)
    else:
        criterion = SymmetricContrastiveLoss(temperature=temp_value, learnable=learnable)
    criterion = criterion.to(device)

    # ---- optimiser ------------------------------------------------------
    if learnable:
        optimizer = torch.optim.Adam([
            {'params': model.projection.parameters(),   'lr': config['learning_rate']},
            {'params': [criterion.log_temperature],     'lr': config['learning_rate'] * 10},
        ], weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(
            model.projection.parameters(),
            lr=config['learning_rate'], weight_decay=1e-4
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    scaler    = torch.cuda.amp.GradScaler()

    loss_history     = []
    val_acc_history  = []
    temp_history     = []
    last_val_metrics = {}

    for epoch in range(config['epochs']):
        loss = train_epoch(model, criterion, optimizer, train_loader,
                           device, epoch, config['epochs'], scaler)
        loss_history.append(loss)

        is_val_epoch = (
            (epoch + 1) % config.get('val_interval', 10) == 0
            or epoch == config['epochs'] - 1
        )

        if is_val_epoch:
            # On the very last epoch, also collect embeddings for t-SNE
            get_embeddings = (epoch == config['epochs'] - 1 and config.get('tsne_enabled', True))
            val_metrics    = evaluate_model(
                model, val_loader, device, num_classes,
                return_embeddings=get_embeddings
            )
            last_val_metrics = val_metrics
            val_acc          = val_metrics['accuracy']
            val_acc_history.append(val_acc)

            current_temp = criterion.temp.item() if learnable else temp_value
            temp_history.append(current_temp)

            print(
                f"    Val Acc={val_acc:.4f}  "
                f"F1={val_metrics['macro_f1']:.4f}  "
                f"AUROC={val_metrics['auroc']:.4f}  "
                f"Sens={val_metrics['mean_sensitivity']:.4f}  "
                f"Spec={val_metrics['mean_specificity']:.4f}  "
                f"τ={current_temp:.4f}"
            )
        else:
            # Carry forward last known values so histories stay aligned
            val_acc_history.append(val_acc_history[-1] if val_acc_history else 0.0)
            temp_history.append(temp_history[-1] if temp_history else temp_value)

        scheduler.step()

    return {
        'loss_history':    loss_history,
        'val_acc_history': val_acc_history,
        'temp_history':    temp_history,
        'val_acc':         val_acc_history[-1],
        'val_metrics':     last_val_metrics,
        # kept at top level for ResultsManager bootstrap CI
        '_true':           last_val_metrics.get('_true',  np.array([])),
        '_preds':          last_val_metrics.get('_preds', np.array([])),
        # embeddings present only on last epoch when tsne_enabled=True
        '_embeddings':     last_val_metrics.get('_embeddings', None),
        '_emb_labels':     last_val_metrics.get('_labels',     None),
        'final_temp':      temp_history[-1],
        'epochs_trained':  len(loss_history),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    results_mgr = ResultsManager(CONFIG['output_dir'])

    temp_settings = (
        [('fixed',     t) for t in FIXED_TEMPS] +
        [('learnable', t) for t in LEARNABLE_TEMPS]
    )

    combinations = list(DATASET_COMBINATIONS.items())
    total        = len(combinations) * len(LOSS_TYPES) * len(temp_settings)

    try:
        idx = 0
        for dataset_combo_name, dataset_keys in combinations:
            for loss_type in LOSS_TYPES:
                for temp_type, temp_value in temp_settings:
                    idx += 1
                    print(f"\n{'='*80}")
                    print(f"[{idx}/{total}] {dataset_combo_name} | {loss_type} | "
                          f"{temp_type}_{temp_value:.2f}")
                    print(f"{'='*80}\n")

                    dataset_paths = [CONFIG['datasets'][key] for key in dataset_keys]
                    fold_results  = []

                    for fold in range(CONFIG['num_folds']):
                        try:
                            result = run_experiment(
                                dataset_paths, dataset_keys,
                                loss_type, temp_type, temp_value,
                                fold, CONFIG, device
                            )
                            fold_results.append(result)
                            vm = result['val_metrics']
                            print(
                                f"  Fold {fold}: "
                                f"Acc={vm['accuracy']:.4f}  "
                                f"F1={vm['macro_f1']:.4f}  "
                                f"AUROC={vm['auroc']:.4f}  "
                                f"τ={result['final_temp']:.4f}  "
                                f"Epochs={result['epochs_trained']}/{CONFIG['epochs']}"
                            )
                        except Exception as e:
                            print(f"  Fold {fold} failed: {e}")
                            import traceback; traceback.print_exc()

                    if not fold_results:
                        continue

                    # ---- save metrics -----------------------------------
                    results_mgr.save_experiment(
                        dataset_combo_name, loss_type, temp_type, temp_value,
                        fold_results, n_bootstrap=CONFIG['bootstrap_n']
                    )

                    accs = [r['val_metrics']['accuracy'] for r in fold_results]
                    f1s  = [r['val_metrics']['macro_f1'] for r in fold_results]
                    print(
                        f"\n  Summary — "
                        f"Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}  |  "
                        f"F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}\n"
                    )

                    # ---- t-SNE (last fold, last epoch embeddings) -------
                    last_fold = fold_results[-1]
                    if (CONFIG.get('tsne_enabled', True)
                            and last_fold.get('_embeddings') is not None):

                        # Build dataset-source list for the full val split
                        # (we don't store per-sample source in run_experiment,
                        #  so we approximate: all from the primary dataset key)
                        emb_labels = last_fold['_emb_labels']
                        n_emb      = len(emb_labels)
                        # Repeat dataset names proportionally across the embedding set
                        # If single dataset: trivial. If combined: approximate equally.
                        n_datasets = len(dataset_keys)
                        sources    = []
                        chunk      = n_emb // n_datasets
                        for i, dk in enumerate(dataset_keys):
                            end = n_emb if i == n_datasets - 1 else (i + 1) * chunk
                            sources.extend([dk] * (end - len(sources)))

                        tsne_path = (
                            results_mgr.get_results_dir()
                            / f"tsne_{dataset_combo_name}_{loss_type}"
                              f"_{temp_type}_{temp_value:.2f}.png"
                        )
                        save_tsne(
                            embeddings=last_fold['_embeddings'],
                            labels=emb_labels,
                            dataset_source_per_sample=sources,
                            output_path=tsne_path,
                            title=(f"{dataset_combo_name} | {loss_type} | "
                                   f"{temp_type} τ={temp_value:.2f}"),
                        )

    finally:
        results_mgr.close()
        print(f"\nAll results saved to: {results_mgr.get_results_dir()}")


if __name__ == "__main__":
    main()
