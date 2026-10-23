import pytest
import torch

import gmm_training
import preprocess
from gmm_training import GmmTrainer, TrainingConfig


def test_training_config_defaults():
    config = TrainingConfig()

    assert config.processed_dir == "data/processed"
    assert config.output_path == "data/model"  # directory: gmm_embeddings.npz and .pt are both written here
    assert config.embedding_dim == 50
    assert config.K == 2
    assert config.window_size == 5
    assert config.batch_size == 256
    assert config.epochs == 5
    assert config.optimizer == "adam"
    assert config.lr == 0.01
    assert config.lr_final == 1e-5
    assert config.margin == 1.0
    assert config.num_negatives == 1  # pinned until multi-negative training is implemented


@pytest.mark.parametrize("name, expected", [("adagrad", torch.optim.Adagrad), ("adam", torch.optim.SparseAdam)])
def test_make_optimizer_builds_the_requested_optimizer(name, expected):
    trainer = GmmTrainer(TrainingConfig(optimizer=name))
    params = list(torch.nn.Linear(2, 2).parameters())

    assert isinstance(trainer.make_optimizer(params), expected)


def test_make_optimizer_rejects_unknown_names():
    trainer = GmmTrainer(TrainingConfig(optimizer="sgd"))

    with pytest.raises(ValueError):
        trainer.make_optimizer(list(torch.nn.Linear(2, 2).parameters()))


def test_trainer_reads_ids_with_the_width_preprocessing_wrote(tmp_path, monkeypatch):
    # Over 65,535 words means 4-byte ids. Reading them as 2-byte ids used to double the token
    # count and scramble every id without raising an error.
    monkeypatch.setattr(preprocess, "MIN_FREQ", 1)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "corpus.txt").write_text(" ".join(f"w{i}" for i in range(70000)), encoding="utf-8")
    preprocess.preprocess_corpus(raw_dir=str(raw_dir), processed_dir=str(tmp_path / "processed"), stopwords="none")

    seen = {}

    class SpyDataset(gmm_training.SkipGramDataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            seen["dataset"] = self

    monkeypatch.setattr(gmm_training, "SkipGramDataset", SpyDataset)
    config = TrainingConfig(
        processed_dir=str(tmp_path / "processed"), output_path=str(tmp_path / "model"), log_dir=str(tmp_path / "logs"),
        embedding_dim=2, K=1, window_size=2, batch_size=4096, epochs=1,
    )
    GmmTrainer(config).train()

    dataset = seen["dataset"]
    assert len(dataset) == 70000 - 4
    assert dataset[10][0].item() == 12  # the center of window 10 is corpus position 12, whose id is 12


def test_trainer_init_stores_config_and_resolves_device():
    config = TrainingConfig()
    trainer = GmmTrainer(config)

    assert trainer.config is config
    assert isinstance(trainer.device, torch.device)
    assert trainer.device.type in ("cpu", "cuda")


def test_train_raises_when_processed_dir_missing(tmp_path):
    config = TrainingConfig(processed_dir=str(tmp_path / "does_not_exist"))
    trainer = GmmTrainer(config)

    with pytest.raises(FileNotFoundError):
        trainer.train()


def test_train_raises_when_vocab_file_missing(tmp_path):
    (tmp_path / "corpus_index.bin").touch()
    config = TrainingConfig(processed_dir=str(tmp_path))
    trainer = GmmTrainer(config)

    with pytest.raises(FileNotFoundError):
        trainer.train()


def test_train_raises_when_corpus_bin_missing(tmp_path):
    (tmp_path / "sorted_vocab.json").write_text("{}", encoding="utf-8")
    config = TrainingConfig(processed_dir=str(tmp_path))
    trainer = GmmTrainer(config)

    with pytest.raises(FileNotFoundError):
        trainer.train()


def test_reshape_pairs_each_center_with_its_flattened_context():
    # Spec agreed on separately: repeat_interleave the centers so each one
    # lines up with every word in its own context window after flattening.
    centers = torch.tensor([5, 9])
    contexts = torch.tensor([
        [1, 2, 3, 4],
        [6, 7, 8, 9],
    ])

    target_ids, ctx_ids = GmmTrainer.reshape(centers, contexts)

    assert torch.equal(target_ids, torch.tensor([5, 5, 5, 5, 9, 9, 9, 9]))
    assert torch.equal(ctx_ids, torch.tensor([1, 2, 3, 4, 6, 7, 8, 9]))


def test_run_batch_keeps_parameters_inside_bounds():
    trainer = GmmTrainer(TrainingConfig(lr=10.0, var_lower=0.5, var_upper=1.5, max_mean_norm=0.1))
    trainer.device = torch.device("cpu")
    trainer.model = gmm_training.GMMWordEmbedding(
        20, 4, 2, var_lower=0.5, var_upper=1.5, max_mean_norm=0.1
    )
    trainer.optimizer = trainer.make_optimizer(trainer.model.parameters())
    trainer.scheduler = torch.optim.lr_scheduler.ConstantLR(trainer.optimizer, factor=1.0)

    # A huge learning rate pushes parameters well past these tight bounds in a single step
    for _ in range(3):
        trainer._run_batch(torch.arange(10), torch.arange(10, 20), torch.randint(0, 20, (10,)))

    mu, var, _ = trainer.model.get_word_params(torch.arange(20))
    assert var.min() >= 0.5 - 1e-5 and var.max() <= 1.5 + 1e-5
    assert mu.norm(dim=-1).max() <= 0.1 + 1e-5
