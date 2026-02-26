#Some of the code is from this github: https://github.com/facebookresearch/dino/tree/main
import argparse
import os
import sys
import datetime
import time
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms

import utils
import vision_transformer as vits
from vision_transformer import DINOHead
from utils import GaussianBlur, Solarization
import torchvision.transforms.functional as TF
import random

class EllipticalConeAwareCrop:

    def __init__(self, output_size=96, crop_scale=(0.2, 0.5)):

        self.output_size = output_size
        self.crop_scale = crop_scale
        
        self.sample_points = []
        
        # LOW WEIGHT 
        self.sample_points.append((0.20, 0.50, 1.0))
        
        # HIGH WEIGHT
        self.sample_points.extend([
            (0.35, 0.40, 3.0),
            (0.35, 0.50, 4.0),
            (0.35, 0.60, 3.0),
        ])
        
        # HIGHEST WEIGHT
        self.sample_points.extend([
            (0.50, 0.30, 3.0),
            (0.50, 0.40, 4.0),
            (0.50, 0.50, 5.0),  # CENTER
            (0.50, 0.60, 4.0),
            (0.50, 0.70, 3.0),
        ])
        
        #  MEDIUM WEIGHT
        self.sample_points.extend([
            (0.65, 0.30, 2.0),
            (0.65, 0.40, 3.0),
            (0.65, 0.50, 3.0),
            (0.65, 0.60, 3.0),
            (0.65, 0.70, 2.0),
        ])
        
        # LOW WEIGHT
        self.sample_points.extend([
            (0.80, 0.35, 1.0),
            (0.80, 0.50, 1.5),
            (0.80, 0.65, 1.0),
        ])
        
        self.weights = [point[2] for point in self.sample_points]
    
    def get_random_point(self):
        idx = random.choices(range(len(self.sample_points)), 
                           weights=self.weights, k=1)[0]
        row_frac, col_frac, weight = self.sample_points[idx]
        return row_frac, col_frac
    
    def __call__(self, img):

        w, h = img.size
        
        # weighted toward center (like PSAX Mercedes is in the center of the image)
        row_frac, col_frac = self.get_random_point()
        
        # Random crop size within scale range
        crop_scale = random.uniform(self.crop_scale[0], self.crop_scale[1])
        crop_size = int(min(h, w) * crop_scale)
        
        # Center crop
        center_y = int(h * row_frac)
        center_x = int(w * col_frac)
        
        # Calculate crop coordinates
        top = max(0, min(center_y - crop_size // 2, h - crop_size))
        left = max(0, min(center_x - crop_size // 2, w - crop_size))
        
        # Crop and resize
        cropped = TF.crop(img, top, left, crop_size, crop_size)
        resized = TF.resize(cropped, (self.output_size, self.output_size), 
                           interpolation=Image.BICUBIC)
        
        return resized


def get_args_parser():
    parser = argparse.ArgumentParser('Temporal DINO Cardiac Fine-tuning', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'],
        help="Vision Transformer architecture.")
    parser.add_argument('--patch_size', default=16, type=int,
        help="Patch size for ViT")
    parser.add_argument('--out_dim', default=65536, type=int,
        help="Output dimension of DINO head")
    parser.add_argument('--norm_last_layer', default=True, type=bool,
        help="Whether to normalize last layer")
    parser.add_argument('--momentum_teacher', default=0.996, type=float,
        help="EMA parameter for teacher")
    parser.add_argument('--use_bn_in_head', default=False, type=bool,
        help="Use batch norm in head")

    # Temperature parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float)
    parser.add_argument('--teacher_temp', default=0.04, type=float)
    parser.add_argument('--warmup_teacher_temp_epochs', default=10, type=int,
        help="Warmup ensures stable training at the start")
    parser.add_argument('--temporal_temperature', default=0.07, type=float,
        help="Temperature for temporal contrastive loss")

    # Temporal parameters
    parser.add_argument('--use_temporal', type=bool, default=True,
        help="Use temporal supervision from consecutive frames")
    parser.add_argument('--temporal_weight', default=0.3, type=float,
        help="Weight for temporal contrastive loss")
    parser.add_argument('--temporal_distance', default=12, type=int,
        help="Distance between frames")
    parser.add_argument('--classes_to_use', type=str, nargs='+', default=None,
        help="Specific classes to use")

    # Training parameters
    parser.add_argument('--use_fp16', type=bool, default=True)
    parser.add_argument('--weight_decay', type=float, default=0.04)
    parser.add_argument('--weight_decay_end', type=float, default=0.4)
    parser.add_argument('--clip_grad', type=float, default=3.0)
    parser.add_argument('--batch_size_per_gpu', default=8, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--freeze_last_layer', default=1, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--warmup_epochs", default=5, type=int)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd'])
    parser.add_argument('--drop_path_rate', type=float, default=0.1)
    
    parser.add_argument('--pretrained_weights', default='dino_deitsmall16_pretrain_full_checkpoint.pth',
    type=str, help='Path to pretrained DINO weights')

    # Data augmentation
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.6 ,1.0))
    parser.add_argument('--local_crops_number', type=int, default=8)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.4, 0.6))

    # Paths
    parser.add_argument('--data_path', default='/data/train', 
        type=str, help='Path to cardiac dataset')
    parser.add_argument('--output_dir', default="./checkpoints_cardiac_temporal", type=str)
    parser.add_argument('--saveckp_freq', default=5, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    
    return parser


class TemporalImageFolder(datasets.DatasetFolder):
    """
    loads consecutive frames for temporal learning.
    
    For each class directory with images [img1.jpg, img2.jpg, img3.jpg, ...],
    creates pairs: (img1, img3), (img2, img4), ... (if temporal_distance=2)
    """
    
    def __init__(self, root, loader, extensions, transform=None, target_transform=None,
                 is_valid_file=None, use_temporal=False, temporal_distance=2, classes_to_use=None):
        super().__init__(root, loader, extensions, transform, target_transform, is_valid_file)
        self.use_temporal = use_temporal
        self.temporal_distance = temporal_distance
        
        if classes_to_use is not None:
            valid_classes = set(classes_to_use)
            self.samples = [s for s in self.samples if self.classes[s[1]] in valid_classes]
            print(f"Filtered to classes: {classes_to_use}")
            print(f"Remaining samples: {len(self.samples)}")
        
        if use_temporal:
            self._build_temporal_pairs()
    
    def _build_temporal_pairs(self):
        self.temporal_pairs = []
        
        # Group samples by class and create temporal pairs
        for class_idx in range(len(self.classes)):
            class_samples = sorted([s for s in self.samples if s[1] == class_idx], 
                                   key=lambda x: x[0])
            
            # Create temporal pairs with specified distance
            for i in range(len(class_samples) - self.temporal_distance):
                frame_t = class_samples[i]
                frame_t_plus_d = class_samples[i + self.temporal_distance]
                self.temporal_pairs.append((frame_t, frame_t_plus_d))
        
        print(f"Built {len(self.temporal_pairs)} temporal pairs (distance={self.temporal_distance})")
    
    def __getitem__(self, idx):
        if self.use_temporal and hasattr(self, 'temporal_pairs') and len(self.temporal_pairs) > 0:
            (path_t, target_t), (path_t_plus_d, target_t_plus_d) = self.temporal_pairs[idx % len(self.temporal_pairs)]
            
            sample_t = self.loader(path_t)
            sample_t_plus_d = self.loader(path_t_plus_d)
            
            if self.transform is not None:
                sample_t = self.transform(sample_t)
                sample_t_plus_d = self.transform(sample_t_plus_d)
            
            return sample_t, sample_t_plus_d, target_t
        else:
            path, target = self.samples[idx]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, target


class DataAugmentationDINO(object):

    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        # Use less aggressive color jittering for this case of cardiac ultrasound images to save the details
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)],
                p=0.5
            ),
        ])
        
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # GLOBAL CROPS
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])
        
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.1),
            Solarization(p=0.05),
            normalize,
        ])
        
        # LOCAL CROPS
        self.local_crops_number = local_crops_number
        self.cone_crop = EllipticalConeAwareCrop(
            output_size=96,
            crop_scale=local_crops_scale
        )
        
        # Post-crop transforms for local crops
        self.local_post_transform = transforms.Compose([
            flip_and_color_jitter,
            GaussianBlur(p=0.1),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        
        # 2 global crops
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        
        # Local crops using cone cropping
        for _ in range(self.local_crops_number):
            crop = self.cone_crop(image)
            crop = self.local_post_transform(crop)
            crops.append(crop)
        
        return crops

'''
class SupervisedTemporalContrastiveLoss(nn.Module):
    """
    Reference: Khosla et al. "Supervised Contrastive Learning" (NeurIPS 2020)
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features_t, features_t_plus_d, labels):

        device = features_t.device

        features = torch.cat([features_t, features_t_plus_d], dim=0)
        features = F.normalize(features, dim=-1)

        labels = torch.cat([labels, labels], dim=0)  # (2B,)

        batch_size = features.shape[0]  # 2B
        sim_matrix = torch.mm(features, features.t()) / self.temperature  # (2B, 2B)

        labels_row = labels.unsqueeze(1)           # (2B, 1)
        labels_col = labels.unsqueeze(0)           # (1, 2B)
        positive_mask = (labels_row == labels_col).float().to(device)  # (2B, 2B)
        self_mask = torch.eye(batch_size, dtype=torch.float, device=device)
        positive_mask = positive_mask - self_mask  # remove diagonal

        sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()

        exp_sim = torch.exp(sim_matrix)

        denom = (exp_sim * (1 - self_mask)).sum(dim=1, keepdim=True)  # (2B, 1)

        log_prob = sim_matrix - torch.log(denom + 1e-8)  # (2B, 2B)
        num_positives = positive_mask.sum(dim=1)  # (2B,)

        valid = num_positives > 0
        loss = -(positive_mask * log_prob).sum(dim=1)  # (2B,)
        loss = loss[valid] / num_positives[valid]

        return loss.mean()
'''

class TemporalContrastiveLoss(nn.Module):

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features_t, features_t_plus_d):
        
        features_t = F.normalize(features_t, dim=-1)
        features_t_plus_d = F.normalize(features_t_plus_d, dim=-1)
        
        sim_matrix = torch.mm(features_t, features_t_plus_d.t()) / self.temperature
        
        batch_size = features_t.shape[0]
        labels = torch.arange(batch_size, device=features_t.device)
        
        loss_t_to_next = F.cross_entropy(sim_matrix, labels)
        loss_next_to_t = F.cross_entropy(sim_matrix.t(), labels)
        
        return (loss_t_to_next + loss_next_to_t) / 2


class DINOLoss(nn.Module):

    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, temporal_weight=0.0, temporal_temperature=0.07):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.temporal_weight = temporal_weight
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        
        self.temporal_contrastive = TemporalContrastiveLoss(temperature=temporal_temperature)

    def forward(self, student_output, teacher_output, epoch, 
                student_output_next=None, teacher_output_next=None):
        
        # ============ DINO Loss ============
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)
        
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        dino_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                dino_loss += loss.mean()
                n_loss_terms += 1
        dino_loss /= n_loss_terms
        
        # ============ Temporal Contrastive Loss (optional) ============
        temporal_loss = torch.tensor(0.0, device=dino_loss.device)
        
        if (student_output_next is not None and teacher_output_next is not None 
            and self.temporal_weight > 0):
            
            batch_size = student_output.shape[0] // self.ncrops
            
            student_global_t = student_output[:2*batch_size].reshape(2, batch_size, -1).mean(dim=0)
            student_global_next = student_output_next[:2*batch_size].reshape(2, batch_size, -1).mean(dim=0)
            
            temporal_loss = self.temporal_contrastive(student_global_t, student_global_next)
        
        # ============ Combine Losses ============
        total_loss = (1.0 - self.temporal_weight) * dino_loss + self.temporal_weight * temporal_loss
        
        self.update_center(teacher_output)
        if teacher_output_next is not None:
            self.update_center(teacher_output_next)
        
        return total_loss, dino_loss.item(), temporal_loss.item() if temporal_loss.item() > 0 else 0.0

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


def train_cardiac(args):
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"Starting training with args:")
    print(f"{'='*60}\n")

    # Data preparation
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )
    
    # Use temporal (optional))
    if args.use_temporal:
        dataset = TemporalImageFolder(
            args.data_path,
            loader=datasets.folder.default_loader,
            extensions=('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'),
            transform=transform,
            use_temporal=True,
            temporal_distance=args.temporal_distance,
            classes_to_use=args.classes_to_use if hasattr(args, 'classes_to_use') else None
        )
        dataset.samples = [s for s in dataset.samples if s[1] < len(dataset.classes)]
        
        original_getitem = dataset.__getitem__
        def temporal_getitem(idx):
            if hasattr(dataset, 'temporal_pairs') and len(dataset.temporal_pairs) > 0:
                (path_t, target_t), (path_t_plus_d, _) = dataset.temporal_pairs[idx % len(dataset.temporal_pairs)]
                sample_t = datasets.folder.default_loader(path_t)
                sample_t_plus_d = datasets.folder.default_loader(path_t_plus_d)
                sample_t = transform(sample_t)
                sample_t_plus_d = transform(sample_t_plus_d)
                return sample_t, sample_t_plus_d, target_t
            else:
                return original_getitem(idx)
        
        dataset.__getitem__ = temporal_getitem
        dataset_len = len(dataset.temporal_pairs) if hasattr(dataset, 'temporal_pairs') else len(dataset)
    else:
        dataset = datasets.ImageFolder(args.data_path, transform=transform)
        if hasattr(args, 'classes_to_use') and args.classes_to_use:
            valid_classes = set(args.classes_to_use)
            dataset.samples = [s for s in dataset.samples if dataset.classes[s[1]] in valid_classes]
            print(f"✓ Filtered to classes: {args.classes_to_use}")
            print(f"✓ Remaining samples: {len(dataset.samples)}")
        dataset_len = len(dataset)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size_per_gpu,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    print(f"Dataset loaded: {dataset_len} samples")
    print(f"Classes: {dataset.classes}")
    print(f"Batches per epoch: {len(dataloader)}")

    # Model building
    print(f"\nBuilding {args.arch} model")
    student = vits.__dict__[args.arch](
        patch_size=args.patch_size,
        drop_path_rate=args.drop_path_rate,
    )
    teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
    embed_dim = student.embed_dim

    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )

    student = student.to(device)
    teacher = teacher.to(device)

    # Load pretrained weights
    if args.pretrained_weights and os.path.isfile(args.pretrained_weights):
        print(f"Loading pretrained weights from: {args.pretrained_weights}")
        
        try:
            checkpoint = torch.load(args.pretrained_weights, map_location='cpu')
            
            if 'teacher' in checkpoint:
                state_dict = checkpoint['teacher']
                print("  Using 'teacher' weights from checkpoint")
            elif 'student' in checkpoint:
                state_dict = checkpoint['student']
                print(" Using 'student' weights from checkpoint")
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
                print(" Using 'model' weights from checkpoint")
            else:
                state_dict = checkpoint
                print(" Using checkpoint")
            
            state_dict = {k.replace("module.", ""): v 
             for k, v in state_dict.items()}
            
            msg = student.load_state_dict(state_dict, strict=False)
            print(f"Loaded weights into student")
            if msg.missing_keys:
                print(f" Missing keys: {len(msg.missing_keys)} keys")
            if msg.unexpected_keys:
                print(f" Unexpected keys: {len(msg.unexpected_keys)} keys")
            
            print("Pretrained weights loaded successfully\n")
            
        except Exception as e:
            print(f"pretrained weights could not be loaded, training from scratch: {e}")
    else:
        print(" No pretrained weights, training from scratch\n")

    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"Models built: {args.arch}")

    # Loss
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        temporal_weight=args.temporal_weight if args.use_temporal else 0.0,
        temporal_temperature=args.temporal_temperature,
    ).to(device)

    # Optimizer
    params_groups = [
        {"params": list(student.parameters()), "lr": args.lr}
    ]
    
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    fp16_scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None

    # Schedulers
    lr_schedule = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        len(dataloader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        len(dataloader),
    )
    momentum_schedule = utils.cosine_scheduler(
        args.momentum_teacher,
        1.0,
        args.epochs,
        len(dataloader),
    )

    # Training loop
    print("\nStarting training...\n")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_dino = 0
        epoch_temporal = 0
        num_batches = 0

        for it, batch in enumerate(dataloader):
            if args.use_temporal and len(batch) == 3:
                images_t, images_t_plus_d, _ = batch
            else:
                images, _ = batch
                images_t = images

            it_global = len(dataloader) * epoch + it
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule[it_global]
                param_group["weight_decay"] = wd_schedule[it_global]

            images_t = [im.to(device, non_blocking=True) for im in images_t]
            if args.use_temporal:
                images_t_plus_d = [im.to(device, non_blocking=True) for im in images_t_plus_d]

            with torch.cuda.amp.autocast(enabled=args.use_fp16):
                teacher_output_t = teacher(images_t[:2])
                student_output_t = student(images_t)
                
                if args.use_temporal:
                    teacher_output_t_plus_d = teacher(images_t_plus_d[:2])
                    student_output_t_plus_d = student(images_t_plus_d)
                else:
                    teacher_output_t_plus_d = None
                    student_output_t_plus_d = None
                
                loss, dino_val, temporal_val = dino_loss(
                    student_output_t, teacher_output_t, epoch,
                    student_output_t_plus_d, teacher_output_t_plus_d
                )

            if not math.isfinite(loss.item()):
                print(f"Loss is {loss.item()}, stopping training")
                sys.exit(1)

            optimizer.zero_grad()
            if fp16_scaler is not None:
                fp16_scaler.scale(loss).backward()
                if args.clip_grad:
                    fp16_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
            else:
                loss.backward()
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad)
                optimizer.step()

            with torch.no_grad():
                m = momentum_schedule[it_global]
                for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            epoch_loss += loss.item()
            epoch_dino += dino_val
            epoch_temporal += temporal_val
            num_batches += 1

            if (it + 1) % max(5, len(dataloader) // 3) == 0:
                msg = f"Epoch [{epoch}/{args.epochs}] Batch [{it+1}/{len(dataloader)}] Loss: {loss.item():.4f}"
                if args.use_temporal:
                    msg += f" | DINO: {dino_val:.4f} | Temporal: {temporal_val:.4f}"
                msg += f" | LR: {optimizer.param_groups[0]['lr']:.2e}"
                print(msg)

        avg_loss = epoch_loss / num_batches
        avg_dino = epoch_dino / num_batches
        avg_temporal = epoch_temporal / num_batches
        
        print(f"\n{'─'*70}")
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Avg Loss: {avg_loss:.4f}", end="")
        if args.use_temporal:
            print(f" | DINO: {avg_dino:.4f} | Temporal: {avg_temporal:.4f}")
        else:
            print()
        print(f"{'─'*70}\n")

        if (epoch + 1) % args.saveckp_freq == 0 or epoch == args.epochs - 1:
            checkpoint = {
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'args': args,
                'dino_loss': dino_loss.state_dict(),
            }
            if fp16_scaler is not None:
                checkpoint['fp16_scaler'] = fp16_scaler.state_dict()
            
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_{epoch:04d}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: checkpoint_{epoch:04d}.pth\n")

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f'Training completed in {datetime.timedelta(seconds=int(total_time))}')
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dino training for cardiac ultrasound images', parents=[get_args_parser()])
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    train_cardiac(args)
