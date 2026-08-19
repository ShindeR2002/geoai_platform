"""Trainer module for TinyCD production framework.

Orchestrates training epochs, validation loops, learning rate updates, and calls
checkpointing, logging, and performance profiling triggers.
"""

import time
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tinycd.trainer.checkpoint import CheckpointManager
from tinycd.trainer.logger import CSVLogger
from tinycd.trainer.runtime import RuntimeProfiler
from tinycd.trainer.metrics import (
    compute_precision,
    compute_recall,
    compute_f1,
    compute_iou
)

logger = logging.getLogger(__name__)

class Trainer:
    """Production Trainer orchestrating the lifecycle of model fitting."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        checkpoint_manager: CheckpointManager,
        csv_logger: CSVLogger,
        runtime_profiler: RuntimeProfiler,
        epochs: int,
        device: torch.device,
        mixed_precision: bool = False,
        early_stopping_patience: int = 5,
        git_commit: Optional[str] = None
    ) -> None:
        """Initialize the Trainer.

        Args:
            model (torch.nn.Module): PyTorch module to fit.
            optimizer (torch.optim.Optimizer): Optimization handler.
            scheduler (torch.optim.lr_scheduler.LRScheduler): LR scheduler.
            train_loader (DataLoader): Loader for training dataset.
            val_loader (Optional[DataLoader]): Loader for validation dataset.
            checkpoint_manager (CheckpointManager): Manager saving weights.
            csv_logger (CSVLogger): Logger saving metric values to CSV.
            runtime_profiler (RuntimeProfiler): Profiler tracking resource usage.
            epochs (int): Number of epochs to train.
            device (torch.device): Device to run training on (CPU/CUDA).
            mixed_precision (bool): If True, use automatic mixed precision (AMP).
            early_stopping_patience (int): Epoch patience before halting.
            git_commit (Optional[str]): Current repository Git commit hash.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_manager = checkpoint_manager
        self.csv_logger = csv_logger
        self.runtime_profiler = runtime_profiler
        self.epochs = epochs
        self.device = device
        self.mixed_precision = mixed_precision
        self.early_stopping_patience = early_stopping_patience
        self.git_commit = git_commit
        
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == "cuda" and self.mixed_precision))

    def train_one_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Run training steps over the train_loader for one epoch.

        Returns:
            Tuple[float, Dict[str, float]]: Average training loss and dictionary of training metrics.
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for x_batch, y_batch in self.train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Autocast mixed precision check
            enabled_amp = (self.device.type == "cuda" and self.mixed_precision)
            with torch.amp.autocast(device_type=self.device.type, enabled=enabled_amp):
                logits = self.model(x_batch)
                H_out, W_out = logits.shape[2], logits.shape[3]
                center_logits = logits[:, :, H_out // 2, W_out // 2]
                loss = F.cross_entropy(center_logits, y_batch)
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item() * x_batch.size(0)
            preds = center_logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            
        avg_loss = total_loss / len(self.train_loader.dataset)
        
        # Calculate training epoch metrics using reusable methods from metrics.py
        preds_np = np.array(all_preds)
        labels_np = np.array(all_labels)
        train_metrics = {
            "precision": compute_precision(labels_np, preds_np),
            "recall": compute_recall(labels_np, preds_np),
            "f1": compute_f1(labels_np, preds_np),
            "iou": compute_iou(labels_np, preds_np)
        }
        return avg_loss, train_metrics

    def validate_one_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Run validation steps over the val_loader.

        Returns:
            Tuple[float, Dict[str, float]]: Average validation loss and dictionary of validation metrics.
        """
        if self.val_loader is None:
            return 0.0, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}
            
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x_batch, y_batch in self.val_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                enabled_amp = (self.device.type == "cuda" and self.mixed_precision)
                with torch.amp.autocast(device_type=self.device.type, enabled=enabled_amp):
                    logits = self.model(x_batch)
                    H_out, W_out = logits.shape[2], logits.shape[3]
                    center_logits = logits[:, :, H_out // 2, W_out // 2]
                    loss = F.cross_entropy(center_logits, y_batch)
                    
                total_loss += loss.item() * x_batch.size(0)
                preds = center_logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
                
        avg_loss = total_loss / len(self.val_loader.dataset)
        
        # Calculate validation epoch metrics using metrics.py functions
        preds_np = np.array(all_preds)
        labels_np = np.array(all_labels)
        val_metrics = {
            "precision": compute_precision(labels_np, preds_np),
            "recall": compute_recall(labels_np, preds_np),
            "f1": compute_f1(labels_np, preds_np),
            "iou": compute_iou(labels_np, preds_np)
        }
        return avg_loss, val_metrics

    def fit(self) -> None:
        """Run the complete multi-epoch fitting process with profiling and log triggers."""
        logger.info("Starting model fitting loop.")
        t0_total = time.time()
        patience_counter = 0
        best_val_iou = -1.0
        
        for epoch in range(1, self.epochs + 1):
            t0_epoch = time.time()
            
            # 1. Run epoch training
            train_loss, train_metrics = self.train_one_epoch()
            
            # 2. Run epoch validation
            t0_val = time.time()
            val_loss, val_metrics = self.validate_one_epoch()
            val_time = time.time() - t0_val
            self.runtime_profiler.update("validation_time", self.runtime_profiler.metrics["validation_time"] + val_time)
            
            epoch_time = time.time() - t0_epoch
            self.scheduler.step()
            
            # 3. Measure resource stats
            self.runtime_profiler.measure_resource_usage()
            
            # 4. Log metrics via csv_logger
            lr = self.optimizer.param_groups[0]["lr"]
            epoch_data = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_precision": train_metrics["precision"],
                "val_precision": val_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "val_recall": val_metrics["recall"],
                "train_f1": train_metrics["f1"],
                "val_f1": val_metrics["f1"],
                "train_iou": train_metrics["iou"],
                "val_iou": val_metrics["iou"],
                "learning_rate": lr,
                "epoch_time": epoch_time,
                "gpu_memory": self.runtime_profiler.metrics["peak_gpu_memory"],
                "ram_usage": self.runtime_profiler.metrics["peak_ram"]
            }
            self.csv_logger.log_epoch(epoch_data)
            
            logger.info(
                "Epoch %d/%d - Train Loss: %.4f, IoU: %.4f | Val Loss: %.4f, IoU: %.4f (Time: %.2fs)",
                epoch, self.epochs, train_loss, train_metrics["iou"], val_loss, val_metrics["iou"], epoch_time
            )
            
            # 5. Call checkpoint manager to serialize states
            val_iou = val_metrics["iou"]
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                val_metric=val_iou,
                git_commit=self.git_commit
            )
            
            # 6. Early stopping patience checks
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info("Early stopping patience reached. Halting training.")
                    break
                    
        total_time = time.time() - t0_total
        self.runtime_profiler.update("training_time", total_time)
        logger.info("Training process finalized.")
