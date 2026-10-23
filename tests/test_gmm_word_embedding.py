import math

import pytest
import torch

from gmm_word_embedding import GMMWordEmbedding


def test_get_word_params_shapes_and_constraints():
    model = GMMWordEmbedding(vocab_size=10, embedding_dim=4, K=3)
    word_ids = torch.tensor([0, 1, 2])

    mu, var, mix_weights = model.get_word_params(word_ids)

    assert mu.shape == (3, 3, 4)
    assert var.shape == (3, 3, 4)
    assert torch.all(var > 0)
    assert mix_weights.shape == (3, 3)
    assert torch.allclose(mix_weights.sum(dim=-1), torch.ones(3), atol=1e-6)


def test_log_overlap_matches_closed_form_for_identical_unit_gaussians():
    # D=1, mu1=mu2=0, var1=var2=1 has a hand-computable closed form:
    # -0.5 * log(var1+var2) - 0.5*log(2*pi) == -0.5 * log(4*pi)
    mu = torch.zeros(1, 1, 1)
    var = torch.ones(1, 1, 1)

    overlap = GMMWordEmbedding.log_overlap(mu, mu, var, var)

    expected = -0.5 * math.log(4 * math.pi)
    assert torch.allclose(overlap, torch.tensor([[[expected]]]), atol=1e-5)


def test_gmm_energy_reduces_to_log_overlap_for_single_component():
    # With K=1 and mixture weights of 1, log(exp(x) * 1 * 1) == x
    log_overlap_matrix = torch.tensor([[[-1.5]]])
    p1 = torch.ones(1, 1)
    p2 = torch.ones(1, 1)

    energy = GMMWordEmbedding.gmm_energy(log_overlap_matrix, p1, p2)

    assert torch.allclose(energy, torch.tensor([-1.5]), atol=1e-5)


def test_gmm_energy_is_exact_for_very_negative_log_overlaps():
    # At D=50 real log-overlaps sit around -60. The energy must track them rather than
    # bottoming out at a fixed floor (the old exp -> sum + 1e-8 -> log version returned -18.42 here).
    log_overlap_matrix = torch.tensor([[[-60.0]]])
    p1 = torch.ones(1, 1)
    p2 = torch.ones(1, 1)

    energy = GMMWordEmbedding.gmm_energy(log_overlap_matrix, p1, p2)

    assert torch.allclose(energy, torch.tensor([-60.0]), atol=1e-4)


def test_gmm_energy_matches_direct_computation_with_multiple_components():
    log_overlap_matrix = torch.tensor([[[-2.0, -3.0], [-4.0, -5.0]]])
    p1 = torch.tensor([[0.7, 0.3]])
    p2 = torch.tensor([[0.4, 0.6]])

    energy = GMMWordEmbedding.gmm_energy(log_overlap_matrix, p1, p2)

    weights = torch.tensor([[0.7 * 0.4, 0.7 * 0.6], [0.3 * 0.4, 0.3 * 0.6]], dtype=torch.float64)
    expected = torch.log((weights * torch.exp(log_overlap_matrix[0].double())).sum())
    assert torch.allclose(energy.double(), expected.unsqueeze(0), atol=1e-5)


def test_high_dimensional_model_still_receives_gradients():
    # Regression: at the default embedding_dim=50 the energies used to be pinned at an
    # epsilon floor, giving gradients around 1e-19, so training silently never learned.
    torch.manual_seed(0)
    model = GMMWordEmbedding(vocab_size=20, embedding_dim=50, K=2)

    energy = model.forward(torch.arange(10), torch.arange(10, 20))
    energy.sum().backward()

    grad_norm = sum(p.grad.norm() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 1e-3


def test_forward_output_shape_and_gradients_flow():
    model = GMMWordEmbedding(vocab_size=10, embedding_dim=4, K=2)
    target_ids = torch.tensor([0, 1])
    ctx_ids = torch.tensor([2, 3])

    energy = model.forward(target_ids, ctx_ids)
    assert energy.shape == (2,)

    energy.sum().backward()
    assert model.mu_embeddings.weight.grad is not None
    # Sparse gradients don't support plain fancy indexing the way a dense tensor does;
    # to_dense() is fine at this toy vocab_size -- the point here is just "is it nonzero".
    touched_rows = model.mu_embeddings.weight.grad.to_dense()[[0, 1, 2, 3]]
    assert torch.any(touched_rows != 0)


def test_max_margin_ranking_zero_when_margin_already_satisfied():
    E_pos = torch.tensor([5.0, 5.0])
    E_neg = torch.tensor([0.0, 0.0])

    loss, active_fraction = GMMWordEmbedding.max_margin_ranking(E_pos, E_neg, margin=1.0)

    assert loss.item() == 0.0
    assert active_fraction.item() == 0.0  # margin already satisfied on both pairs, so neither is active


def test_max_margin_ranking_equals_margin_when_energies_tied():
    E_pos = torch.zeros(3)
    E_neg = torch.zeros(3)

    loss, active_fraction = GMMWordEmbedding.max_margin_ranking(E_pos, E_neg, margin=2.0)

    assert math.isclose(loss.item(), 2.0)
    assert active_fraction.item() == 1.0  # all three pairs violate the margin


def test_max_margin_ranking_active_fraction_counts_only_violating_pairs():
    # First pair already satisfies the margin (hinge == 0); other two don't.
    E_pos = torch.tensor([5.0, 0.0, 0.0])
    E_neg = torch.tensor([0.0, 0.0, 0.0])

    _, active_fraction = GMMWordEmbedding.max_margin_ranking(E_pos, E_neg, margin=1.0)

    assert math.isclose(active_fraction.item(), 2 / 3, rel_tol=1e-6)


def test_inverse_var_transform_round_trips_through_get_word_params():
    model = GMMWordEmbedding(vocab_size=1, embedding_dim=1, K=1)
    with torch.no_grad():
        model.var_embeddings.weight.fill_(GMMWordEmbedding.inverse_var_transform(0.3))

    _, var, _ = model.get_word_params(torch.tensor([0]))

    assert torch.allclose(var, torch.tensor(0.3), atol=1e-5)


def test_project_parameters_clamps_variances_into_bounds():
    model = GMMWordEmbedding(vocab_size=3, embedding_dim=4, K=2, var_lower=0.05, var_upper=5.0)
    with torch.no_grad():
        model.var_embeddings.weight[0].fill_(-20.0)  # softplus(-20) ~ 2e-9, far below var_lower
        model.var_embeddings.weight[1].fill_(20.0)   # softplus(20) ~ 20, far above var_upper
        model.var_embeddings.weight[2].fill_(0.5)    # var ~ 0.97, already inside the bounds

    model.project_parameters_(torch.arange(3))
    _, var, _ = model.get_word_params(torch.arange(3))

    assert torch.allclose(var[0], torch.tensor(0.05), atol=1e-5)
    assert torch.allclose(var[1], torch.tensor(5.0), atol=1e-4)
    assert torch.allclose(model.var_embeddings.weight[2], torch.tensor(0.5))  # in-bounds rows are left alone


def test_project_parameters_caps_each_component_mean_norm_separately():
    model = GMMWordEmbedding(vocab_size=1, embedding_dim=2, K=2, max_mean_norm=8.0)
    with torch.no_grad():
        # Component 0 has norm 50 (over the cap); component 1 has norm 5 (under it). Capping the whole
        # 4-long row instead would wrongly shrink component 1 too.
        model.mu_embeddings.weight[0] = torch.tensor([30.0, 40.0, 3.0, 4.0])

    model.project_parameters_(torch.tensor([0]))
    mu, _, _ = model.get_word_params(torch.tensor([0]))

    assert torch.allclose(mu[0, 0], torch.tensor([4.8, 6.4]))  # same direction, rescaled to norm 8
    assert torch.allclose(mu[0, 1], torch.tensor([3.0, 4.0]))


def test_project_parameters_leaves_a_zero_mean_unchanged():
    model = GMMWordEmbedding(vocab_size=1, embedding_dim=3, K=1)
    with torch.no_grad():
        model.mu_embeddings.weight.zero_()

    model.project_parameters_(torch.tensor([0]))

    assert torch.all(model.mu_embeddings.weight == 0)  # max_norm / 0 = inf must not turn into NaN


def test_project_parameters_leaves_untouched_rows_alone():
    model = GMMWordEmbedding(vocab_size=3, embedding_dim=2, K=1, var_lower=0.05, var_upper=5.0, max_mean_norm=1.0)
    with torch.no_grad():
        model.var_embeddings.weight.fill_(20.0)  # every row starts out of bounds
        model.mu_embeddings.weight.fill_(10.0)   # every row's mean norm starts over the cap

    model.project_parameters_(torch.tensor([1]))  # only row 1 was in this batch
    _, var, _ = model.get_word_params(torch.arange(3))

    assert torch.allclose(var[1], torch.tensor(5.0), atol=1e-4)  # row 1: projected
    assert torch.allclose(model.mu_embeddings.weight[1].norm(), torch.tensor(1.0), atol=1e-4)
    for row in (0, 2):  # rows 0 and 2: left exactly as they were, still out of bounds
        assert torch.allclose(var[row], torch.tensor(20.0), atol=1e-2)
        assert torch.allclose(model.mu_embeddings.weight[row], torch.tensor([10.0, 10.0]))


def test_rejects_variance_bounds_out_of_order():
    with pytest.raises(ValueError):
        GMMWordEmbedding(vocab_size=1, var_lower=5.0, var_upper=0.05)
