from dataclasses import dataclass

import json
from pathlib import Path
import logging
import numpy as np
import time
import torch
from torch.utils.data import DataLoader

from dataset import SkipGramDataset
from gmm_word_embedding import GMMWordEmbedding
import metrics_logger
from preprocess import id_dtype
from sampler import NegativeSampler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

HEARTBEAT_INTERVAL_SEC = 300  # How often _run_epoch logs a liveness line during a long epoch

@dataclass
class TrainingConfig:
    processed_dir: str = "data/processed"
    output_path: str = "data/model"
    log_dir: str = "logs/"
    embedding_dim: int = 50
    K: int = 2
    window_size: int = 5
    batch_size: int = 256
    epochs: int = 5
    optimizer: str = "adam"  # "adam" or "adagrad"
    lr: float = 0.01  # Initial learning rate; decays linearly to lr_final over the whole run. 
    lr_final: float = 1e-5
    margin: float = 1.0
    num_negatives: int = 1  # DO NOT CHANGE THIS VALUE. Code to support multiple negative samples is not yet implemented.
    checkpoint_every: int = 0  # 0 disables checkpointing; e.g. 10 saves a snapshot every 10 epochs into output_path/epoch_<N>/
    resume_from: str = None  # Directory of a previous checkpoint (e.g. output_path/epoch_20) to continue training from; None starts fresh.
    num_workers: int = 0  # Background processes to prepare batches in parallel; 0 means the main process does it
    var_lower: float = 0.05  # Every variance is clamped to [var_lower, var_upper] after each optimizer step
    var_upper: float = 5.0
    max_mean_norm: float = 8.0  # Each component's mean vector is scaled back to this L2 norm if it grows past it

# One batch's worth of numbers the epoch loop needs back from _run_batch: bundled into one
# object instead of 4+ separate return values or 8+ separate arguments.
@dataclass
class BatchStats:
    loss: float
    active_fraction: float
    pos_energy: float
    neg_energy: float
    pair_count: int
    grad_norm: float
    mix_weight: float
    var_min: float
    var_max: float
    var_mean: float

"""
GmmTrainer runs through the .memmap file created by gmm_word_embedding.py, passes its data into PyTorch, and trains a probablistic
word embedding model using the energy math from gmm_model.py.
It outputs a .npz file containing the fully-trained model.
"""
class GmmTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
        Main training loop. This method orchestrates the entire training process, including validation of the processed directory, setup of the model and data pipeline, running epochs, and saving the final trained model.
    """
    def train(self):
        self._validate_processed_dir()
        self._setup()

        # Create MetricsLogger object for logging
        with metrics_logger.MetricsLogger(self.config.log_dir) as metrics:
            logger.info(f"Starting training for {self.config.epochs} epochs on device: {self.device}")
            for epoch in range(self.start_epoch, self.config.epochs):
                self._run_epoch(epoch, metrics)

                # Periodic checkpoint: Saves a model after every N epochs
                if self.config.checkpoint_every and (epoch + 1) % self.config.checkpoint_every == 0:
                    self.save_model(subdir=f"epoch_{epoch + 1}", next_epoch=epoch + 1)
                    logger.info(f"Checkpoint saved for epoch {epoch + 1} at '{self.config.output_path}/epoch_{epoch + 1}'.")

        # Save the final trained model
        self.save_model(next_epoch=self.config.epochs)


    """ Validation method to verify that the processed directory contains the files preprocess.py produces """
    def _validate_processed_dir(self):
        if not Path(self.config.processed_dir).exists():
            raise FileNotFoundError(f"Processed directory '{self.config.processed_dir}' does not exist. Please run the preprocess command first.")
        if not Path(self.config.processed_dir, "sorted_vocab.json").exists():
            raise FileNotFoundError(f"Vocabulary file 'sorted_vocab.json' not found in '{self.config.processed_dir}'. Please run the preprocess command first.")
        if not Path(self.config.processed_dir, "corpus_index.bin").exists():
            raise FileNotFoundError(f"Corpus index file 'corpus_index.bin' not found in '{self.config.processed_dir}'. Please run the preprocess command first.")


    """ Setup method to load the vocab and build the model/data/optimizer pipeline, storing it all on self so
        _run_epoch/_run_batch can use it without a long parameter list."""
    def _setup(self):
        with open(Path(self.config.processed_dir) / "sorted_vocab.json", "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.vocab_size = len(self.vocab)
        word_count_list = [count for rank, count in sorted(self.vocab.values(), key=lambda x: x[0])]

        self.model = GMMWordEmbedding(
            self.vocab_size, self.config.embedding_dim, self.config.K,
            var_lower=self.config.var_lower, var_upper=self.config.var_upper, max_mean_norm=self.config.max_mean_norm,
        ).to(self.device)
        self.dataset = SkipGramDataset(
            bin_file=Path(self.config.processed_dir) / "corpus_index.bin",
            window_size=self.config.window_size, dtype=id_dtype(self.vocab_size),
        )
        self.sampler = NegativeSampler(word_counts=word_count_list)
        # pin_memory only helps when there's a GPU to transfer pinned batches to; harmless but pointless on CPU
        dataloader_kwargs = dict(
            batch_size=self.config.batch_size, shuffle=True,
            num_workers=self.config.num_workers, pin_memory=(self.device.type == "cuda"),
        )
        if self.config.num_workers > 0:
            # Always spawn rather than fork workers, on every device -- forking a process that has
            # already initialized CUDA is unsafe and can duplicate GPU-driver memory per worker.
            # multiprocessing_context is only a valid kwarg when num_workers > 0 at all, hence the guard.
            dataloader_kwargs["multiprocessing_context"] = "spawn"
        self.dataloader = DataLoader(self.dataset, **dataloader_kwargs)


        self.optimizer = self.make_optimizer(self.model.parameters())
        total_steps = self.config.epochs * len(self.dataloader)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=self.config.lr_final / self.config.lr, total_iters=total_steps
        )

        self.start_epoch = self._load_checkpoint() if self.config.resume_from else 0

    """ Loads model/optimizer/scheduler state from a previous checkpoint directory and returns the epoch
        to resume at. Requires the checkpoint's vocab to exactly match the one we just built in _setup(),
        since model rows are indexed by word id: silently loading a mismatched vocab would assign the
        wrong word to every embedding row without raising an error. """
    def _load_checkpoint(self) -> int:
        resume_dir = Path(self.config.resume_from)

        with open(resume_dir / "sorted_vocab.json", "r", encoding="utf-8") as f:
            resumed_vocab = json.load(f)
        if resumed_vocab != self.vocab:
            raise ValueError(
                f"Vocabulary saved in '{resume_dir}' does not match the vocabulary built from "
                f"'{self.config.processed_dir}'. Resuming with a different vocab would silently "
                f"assign the wrong word to every embedding row."
            )

        self.model.load_state_dict(torch.load(resume_dir / "gmm_embeddings.pt", map_location=self.device))

        training_state = torch.load(resume_dir / "training_state.pt", map_location=self.device)
        self.optimizer.load_state_dict(training_state["optimizer_state"])
        self.scheduler.load_state_dict(training_state["scheduler_state"])

        logger.info(f"Resumed from '{resume_dir}', continuing at epoch {training_state['next_epoch'] + 1}.")
        return training_state["next_epoch"]


    """ Runs every batch in the dataloader once, accumulating and logging epoch-level metrics."""
    def _run_epoch(self, epoch: int, metrics: metrics_logger.MetricsLogger):
        epoch_start_time = time.perf_counter()
        # Weight each batch's stats by its pair count so the epoch average is exact
        epoch_loss_sum = 0.0
        epoch_pair_count = 0
        epoch_active_pairs = 0
        last_heartbeat_sec = 0.0  # Seconds into this epoch when we last logged a liveness line

        for batch_idx, (centers, contexts) in enumerate(self.dataloader):
            batch_start_time = time.perf_counter()

            # non_blocking only actually overlaps with compute when the source tensor is pinned
            # (see pin_memory above); it's a harmless no-op otherwise.
            centers = centers.to(self.device, non_blocking=True)
            contexts = contexts.to(self.device, non_blocking=True)

            # Reshape the output of SkipGramDataset to match input shape of (target, context)
            target_ids, ctx_ids = self.reshape(centers, contexts)
            # Get our negative samples for this batch
            negative_samples = self.sampler.sample(self.config.num_negatives * ctx_ids.size(0)).to(self.device)

            stats = self._run_batch(target_ids, ctx_ids, negative_samples)

            epoch_loss_sum += stats.loss * stats.pair_count
            epoch_active_pairs += stats.active_fraction * stats.pair_count
            epoch_pair_count += stats.pair_count

            batch_elapsed_sec = time.perf_counter() - batch_start_time
            epoch_elapsed_sec = time.perf_counter() - epoch_start_time  # This aggregates over each batch until we finish the epoch.

            metrics.writeRow(
                epoch=epoch, batch=batch_idx, epoch_sec=epoch_elapsed_sec,
                batch_sec=batch_elapsed_sec, loss=stats.loss, pos_energy=stats.pos_energy,
                neg_energy=stats.neg_energy, active_pair=stats.active_fraction, grad_norm=stats.grad_norm,
                mix_weight=stats.mix_weight, var_min=stats.var_min, var_max=stats.var_max, var_mean=stats.var_mean
            )

            # Liveness line for long epochs, gated by elapsed time rather than batch count so it
            # doesn't need retuning for a different corpus size or batch size.
            if epoch_elapsed_sec - last_heartbeat_sec >= HEARTBEAT_INTERVAL_SEC:
                logger.info(f"Epoch {epoch + 1}/{self.config.epochs}: batch {batch_idx + 1}/{len(self.dataloader)} ({epoch_elapsed_sec:.0f}s elapsed)")
                last_heartbeat_sec = epoch_elapsed_sec

        logger.info(
            f"Epoch {epoch + 1}/{self.config.epochs} completed. "
            f"Avg loss: {epoch_loss_sum / epoch_pair_count:.4f}, "
            f"Active pairs: {epoch_active_pairs / epoch_pair_count:.4f}"
        )


    """ Method that runs the forward/backward/step for one batch of (target, context, negative) ids and
        returns the numbers _run_epoch needs for logging."""
    def _run_batch(self, target_ids: torch.Tensor, ctx_ids: torch.Tensor, negative_samples: torch.Tensor) -> BatchStats:
        # Get positive and negative energies for this batch
        e_pos = self.model.forward(target_ids, ctx_ids)
        e_neg = self.model.forward(target_ids, negative_samples)

        # Get the means, variances, and mixture weights for the target words in this batch
        with (torch.no_grad()):
            mu, var, mix_weights = self.model.get_word_params(target_ids)
            var_min = var.min().item()
            var_max = var.max().item()
            var_mean = var.mean().item()
            mix_weight_spread = mix_weights.max(dim=-1).values.mean().item()

        # Compute the max-margin ranking loss
        loss, active_fraction = self.model.max_margin_ranking(e_pos, e_neg, self.config.margin)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float("inf"))  # No clipping, but returns the norm for logging
        self.optimizer.step()
        self.model.project_parameters_()  # Pull any variance or mean the step pushed out of bounds back in
        self.scheduler.step()
        self.optimizer.zero_grad()

        return BatchStats(
            loss=loss.item(), active_fraction=active_fraction.item(),
            pos_energy=e_pos.mean().item(), neg_energy=e_neg.mean().item(),
            pair_count=e_pos.numel(), grad_norm=grad_norm.item(),
            mix_weight=mix_weight_spread, var_min=var_min, var_max=var_max, var_mean=var_mean
        )


    """ Method to create the optimizer based on the configuration. Supports 'adam' and 'adagrad'. """
    def make_optimizer(self, parameters):
        if self.config.optimizer == "adam":
            return torch.optim.Adam(parameters, lr=self.config.lr)
        if self.config.optimizer == "adagrad":
            return torch.optim.Adagrad(parameters, lr=self.config.lr)
        raise ValueError(f"Unknown optimizer '{self.config.optimizer}'. Use 'adagrad' or 'adam'.")


    """
    Method to save the trained GMM embeddings and the vocabulary to disk. The embeddings are saved in both .npz and .pt formats.
    Parameters:
        subdir - optional subdirectory to store mid-training checkpoints (e.g. "epoch_10").
                 If None, we save in the main output_path directory (the final training run passes no subdir, since we want to save the final model).
        next_epoch - the 0-based epoch index a --resume_from run should start at. Saved alongside the
                     model so a later run can continue the same optimizer/scheduler trajectory instead
                     of just warm-starting from these weights.
    """
    def save_model(self, subdir: str = None, next_epoch: int = None):
        output_dir = Path(self.config.output_path)
        if subdir is not None:
            output_dir = output_dir / subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        # get_word_params applies softplus/softmax, so the saved arrays are usable variances and mixture weights, not raw logits
        with torch.no_grad():
            all_ids = torch.arange(self.vocab_size, device=self.device)
            mu, var, mix_weights = self.model.get_word_params(all_ids)

        np.savez(
            output_dir / "gmm_embeddings.npz",
            mu=mu.cpu().numpy(),
            var=var.cpu().numpy(),
            mix_weights=mix_weights.cpu().numpy(),
        )
        torch.save(self.model.state_dict(), output_dir / "gmm_embeddings.pt")

        # Keep a copy of the vocab with the model it was trained on so the two can't drift apart
        with open(output_dir / "sorted_vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)

        # Separate from gmm_embeddings.pt: this file is only for --resume_from, not for downstream
        # users of the model, so it carries optimizer/scheduler state instead of just weights.
        torch.save(
            {
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "next_epoch": next_epoch,
            },
            output_dir / "training_state.pt",
        )


    @staticmethod
    def reshape(centers: torch.Tensor, contexts: torch.Tensor) -> torch.Tensor:
        """
        Reshapes the center/context word embeddings from SkipGramDataset to match the expected input shape for the GMMWordEmbedding model.
        Parameters:
            centers - Tensor of shape (batch_size,)
            contexts - Tensor of shape (batch_size, 2*window_size)
        Returns:
            reshaped_tensor - Tensor of shape (2, batch_size * 2*window_size), meant to be unpacked as
                target_ids, ctx_ids = reshape(centers, contexts)
                each of shape (batch_size * 2*window_size,), aligned so target_ids[i] is paired with ctx_ids[i]
        """
        repeated_centers = torch.repeat_interleave(centers, contexts.size(1))
        flat_ctx = contexts.flatten()
        reshaped_tensor = torch.stack((repeated_centers, flat_ctx), dim=0)
        return reshaped_tensor