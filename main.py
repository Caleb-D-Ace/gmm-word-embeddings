"""
This is the main entry point for training the GMM word embedding model.
It expects data/processed/ to already exist (run the preprocess command on a corpus first).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from gmm_training import GmmTrainer, TrainingConfig


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train GMM Word Embeddings")

    parser.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed",
        help="Directory containing the preprocessed corpus (default: data/processed)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/model",
        help="Where to save the trained model (default: data/model/gmm_embeddings.[filetype])\n\tFiles are saved in both a .npz and a .pt format."
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=50,
        help="Dimensionality of each Gaussian component (default: 50)"
    )
    parser.add_argument(
        "--k", dest="K",
        type=int,
        default=2,
        help="Number of mixture components per word (default: 2)"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=5,
        help="Context window size on each side of the center word (default: 5)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training (default: 256)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "adagrad"],
        default="adam",
        help="Optimizer (default: adam)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Initial learning rate (default: 0.01); decays linearly to --lr_final over the whole run"
    )
    parser.add_argument(
        "--lr_final",
        type=float,
        default=1e-5,
        help="Learning rate reached at the end of training (default: 1e-5)"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.0,
        help="Margin for the max-margin ranking loss (default: 1.0)"
    )
    parser.add_argument(
        "--num_negatives",
        type=int,
        default=1,
        help="Negative samples drawn per positive pair (default: 1)"
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=0,
        help="Save a checkpoint every N epochs into output_path/epoch_<N>/, in addition to the final save (default: 0, disabled)"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/",
        help="Where to save the training logs (default: data/logs)"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Directory of a previous checkpoint (e.g. output_path/epoch_20) to continue training from. "
             "Must be run with the same --processed_dir, --embedding_dim, --k, and --epochs as the original run."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Background processes to prepare batches in parallel (default: 0, single-process). "
             "On a GPU machine, set this above 0 so the CPU can prepare the next batch while the GPU works on the current one."
    )
    parser.add_argument(
        "--shuffle_buffer_size",
        type=int,
        default=1_000_000,
        help="Window size (in token positions) for approximate per-epoch shuffling (default: 1,000,000). "
             "Bigger means closer to a true random shuffle but more memory; see sampler.py's ShuffleBufferSampler."
    )
    parser.add_argument(
        "--var_lower",
        type=float,
        default=0.05,
        help="Smallest variance any Gaussian component may have, per dimension; enforced after every optimizer step (default: 0.05)"
    )
    parser.add_argument(
        "--var_upper",
        type=float,
        default=5.0,
        help="Largest variance any Gaussian component may have, per dimension; enforced after every optimizer step (default: 5.0)"
    )
    parser.add_argument(
        "--max_mean_norm",
        type=float,
        default=8.0,
        help="Largest L2 norm any Gaussian component's mean may have; longer means are scaled back (default: 8.0)"
    )

    args = parser.parse_args()
    return TrainingConfig(**vars(args))


def main():
    config = parse_args()
    trainer = GmmTrainer(config)
    print(f"Starting training with config: {config}")
    trainer.train()


if __name__ == "__main__":
    main()
