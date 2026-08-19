"""Checkpoint Manager for TinyCD production framework.

Saves weights, state manifests, optimizer states, and scheduler states for complete
reproducibility and fault-tolerant restarts.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import torch

class CheckpointManager:
    """Manages model checkpoints and training state metadata."""

    def __init__(self, checkpoint_dir: Path, random_seed: int) -> None:
        """Initialize CheckpointManager.

        Args:
            checkpoint_dir (Path): Output directory for checkpoints.
            random_seed (int): Experiment random seed.
        """
        self.checkpoint_dir = checkpoint_dir
        self.random_seed = random_seed
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = -1.0

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        epoch: int,
        val_metric: float,
        git_commit: Optional[str] = None
    ) -> None:
        """Save the epoch state and determine if it's the best checkpoint.

        Args:
            model (torch.nn.Module): The model to serialize.
            optimizer (torch.optim.Optimizer): The optimizer state.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The scheduler state.
            epoch (int): Current epoch number.
            val_metric (float): Validation metric (IoU) to evaluate performance.
            git_commit (Optional[str]): Current Git commit hash.
        """
        # 1. Save last model state
        last_path = self.checkpoint_dir / "last_model.pt"
        torch.save(model.state_dict(), last_path)

        # 2. Save optimizer and scheduler states independently
        opt_path = self.checkpoint_dir / "optimizer_state.pt"
        sch_path = self.checkpoint_dir / "scheduler_state.pt"
        torch.save(optimizer.state_dict(), opt_path)
        torch.save(scheduler.state_dict(), sch_path)

        # 3. Determine if best model
        is_best = False
        if val_metric > self.best_metric:
            self.best_metric = val_metric
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(model.state_dict(), best_path)
            is_best = True

        # 4. Save training_state.json metadata
        state_data = {
            "epoch": epoch,
            "best_metric": self.best_metric,
            "optimizer_state_path": str(opt_path.resolve()),
            "scheduler_state_path": str(sch_path.resolve()),
            "random_seed": self.random_seed,
            "git_commit_hash": git_commit or "unknown",
            "is_best_epoch": is_best
        }
        
        state_json = self.checkpoint_dir / "training_state.json"
        with open(state_json, "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=2)
            
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        load_best: bool = False
    ) -> Dict[str, Any]:
        """Load states from checkpoints directory.

        Args:
            model (torch.nn.Module): Model to load parameters into.
            optimizer (torch.optim.Optimizer): Optimizer to load state into.
            scheduler (torch.optim.lr_scheduler.LRScheduler): Scheduler to load state into.
            load_best (bool): If True, load the best model instead of the last one.

        Returns:
            Dict[str, Any]: Loaded training state metadata.
        """
        state_json = self.checkpoint_dir / "training_state.json"
        if not state_json.exists():
            return {}

        with open(state_json, "r", encoding="utf-8") as f:
            state_data = json.load(f)

        # Load weights
        weight_name = "best_model.pt" if load_best else "last_model.pt"
        weight_path = self.checkpoint_dir / weight_name
        if weight_path.exists():
            model.load_state_dict(torch.load(weight_path, map_location="cpu"))

        # Load optimizer
        opt_path = Path(state_data["optimizer_state_path"])
        if opt_path.exists():
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))

        # Load scheduler
        sch_path = Path(state_data["scheduler_state_path"])
        if sch_path.exists():
            scheduler.load_state_dict(torch.load(sch_path, map_location="cpu"))

        self.best_metric = state_data.get("best_metric", -1.0)
        return state_data
