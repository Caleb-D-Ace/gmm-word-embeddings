import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
gmm_word_embedding contains the GMM energy function implementation for training the word embedding model.
It uses the PyTorch.nn module to create a neural network trained using this energy function. 
The model is trained using a max-margin ranking loss function, which is implemented in the max_margin_ranking method.
"""
class GMMWordEmbedding(nn.Module):
    VAR_FLOOR = 1e-4  # Added after softplus in get_word_params so a variance can never reach exactly 0

    # Time: O(V*K*D), Space: O(V*K*D)   (V = vocab_size, K = components, D = embedding_dim; one-time model allocation)
    def __init__(self, vocab_size: int, embedding_dim: int = 50, K: int = 2,
                 var_lower: float = 0.05, var_upper: float = 5.0, max_mean_norm: float = 8.0):
        super().__init__()
        # Set up the basic parameters for the equation. These are used in the log_overlap and gmm_energy functions.
        self.vocab_size = vocab_size    # Size of the vocabulary (number of unique words)
        self.D = embedding_dim          # Dimension of the embedding space
        self.K = K                      # How many modes there are

        # Bounds enforced by project_parameters_() after every optimizer step. These are plain Python
        # attributes rather than register_buffer() buffers on purpose: buffers get saved in the state_dict,
        # so adding them would make every checkpoint saved before this change fail to load.
        if not self.VAR_FLOOR < var_lower < var_upper:
            raise ValueError(f"Need {self.VAR_FLOOR} < var_lower < var_upper, got var_lower={var_lower}, var_upper={var_upper}.")
        self.var_lower = var_lower
        self.var_upper = var_upper
        self.max_mean_norm = max_mean_norm
        # The bounds are given in variance space, but the projection clamps the raw (pre-softplus) weights, so convert them once here
        self.raw_var_lower = self.inverse_var_transform(var_lower)
        self.raw_var_upper = self.inverse_var_transform(var_upper)

        # sparse=True: each batch only touches a few thousand of (potentially) millions of rows,
        # but a dense Embedding still runs backward/optimizer.step() over the WHOLE table every
        # batch regardless -- the cost scales with vocab_size, not batch_size. Sparse gradients
        # (paired with SparseAdam in make_optimizer, and row-scoped project_parameters_ below)
        # make the cost scale with how many distinct words actually appear in the batch instead.
        self.mu_embeddings = nn.Embedding(vocab_size, K * self.D, sparse=True)
        self.var_embeddings = nn.Embedding(vocab_size, K * self.D, sparse=True)
        self.mix_logits = nn.Embedding(vocab_size, K, sparse=True)

        # Initialize weights with standard distributions
        nn.init.uniform_(self.mu_embeddings.weight, -0.5 / self.D, 0.5 / self.D)
        nn.init.uniform_(self.var_embeddings.weight, 0.1, 1.0) 
        nn.init.zeros_(self.mix_logits.weight)

    # Time: O(n*K*D), Space: O(n*K*D)   (n = word_ids.shape[0]; n=P=batch_size*2*window_size in this codebase's usage)
    def get_word_params(self, word_ids: torch.Tensor):
        """
        Given a batch of word ids, this method retrieves the corresponding means, variances, and mixture weights.
        It reshapes the means and variances to have shape (batch_size, K, D) for further calculations.
        """
        batch_size = word_ids.shape[0]

        # Get mu and var for word_ids as a 1D representation
        mu = self.mu_embeddings(word_ids)
        var = self.var_embeddings(word_ids)

        # Ensure variance is strictly positive using Exponential or Softplus
        var = F.softplus(var) + self.VAR_FLOOR

        # Reshape from (batch_size, K*D) to (batch_size, K, D)
        mu = mu.view(batch_size, self.K, self.D)
        var = var.view(batch_size, self.K, self.D)

        # Use softmax to convert logits to probabilities for mixture weights
        mix_logits = self.mix_logits(word_ids)
        mix_weights = torch.softmax(mix_logits, dim=-1)  # Convert logits to probabilities

        return mu, var, mix_weights

    @classmethod
    def inverse_var_transform(cls, var: float) -> float:
        """
        Inverts the variance transform in get_word_params, turning a variance into the raw weight that produces it.

        Math:
            var = softplus(raw) + floor = log(1 + e^raw) + floor   =>   raw = log(e^(var - floor) - 1)
        math.expm1(x) computes e^x - 1 accurately even when x is tiny, where e^x - 1 would lose precision.
        """
        return math.log(math.expm1(var - cls.VAR_FLOOR))

    # Time: O(T*K*D), Space: O(T*K*D)   (T = len(word_ids), the batch's touched rows -- not V, thanks to
    # sparse embeddings + SparseAdam; see __init__ and make_optimizer)
    @torch.no_grad()
    def project_parameters_(self, word_ids: torch.Tensor):
        """
        Pulls the given words' variances and means back inside their bounds. Call this right after
        optimizer.step(), passing the (deduplicated) word ids involved in this batch's forward pass --
        under SparseAdam, a row is only ever touched by a step if it appeared in that step's gradient,
        so these are the only rows that could have just drifted out of bounds. (With dense Adam, every
        row's momentum could carry it out of bounds even on a batch that didn't touch it, which is why
        this used to project the whole table; that's no longer true now that the embeddings and the
        optimizer are both sparse-gradient-aware -- see their setup in __init__/make_optimizer.)

        @torch.no_grad() turns off autograd for the whole method. These edits are corrections to the weights, not
        part of the computation being differentiated, and autograd refuses in-place edits to a leaf tensor that
        requires grad anyway.
        """
        # Variances: softplus is monotonic, so clamping the raw weight to [raw_lower, raw_upper] is the same as
        # clamping the variance to [var_lower, var_upper]. Advanced indexing (weight[word_ids]) returns a copy,
        # not a view, so the clamped result has to be written back explicitly rather than clamped in place.
        self.var_embeddings.weight[word_ids] = self.var_embeddings.weight[word_ids].clamp(
            self.raw_var_lower, self.raw_var_upper
        )

        # Means: cap the L2 norm of each component's mean, not each word's whole row.
        mu_rows = self.mu_embeddings.weight[word_ids].view(-1, self.K, self.D)
        norms = mu_rows.norm(dim=-1, keepdim=True)  # (num_words, K, 1): keepdim broadcasts the scale over D
        # Scale factor is max_norm / norm for components over the cap and 1 for the rest. A zero norm gives
        # inf, which the clamp also turns into 1.
        scale = (self.max_mean_norm / norms).clamp(max=1.0)
        self.mu_embeddings.weight[word_ids] = (mu_rows * scale).view(-1, self.K * self.D)

    # Time: O(n*K^2*D), Space: O(n*K^2*D)   (n = mu1.shape[0] = P; the K*K broadcast below is the single
    # largest compute block per batch -- everything else in the model is O(n*K*D) or smaller)
    @staticmethod
    def log_overlap(mu1, mu2, var1, var2):
        """
        Loss function: Step 1
        Computes the closed-form Expected Likelihood Kernel (log-overlap) between 
        Gaussian components across D dimensions for all batch items and component pairs.

        Math:
            log_Δ(f_k, g_m) = -0.5 * Σ_d [ log(var1 + var2) + (mu1 - mu2)^2 / (var1 + var2) ] - (D / 2) * log(2π)

        Args:
            mu1 (Tensor): Target word means of shape (batch_size, K, D)
            mu2 (Tensor): Context word means of shape (batch_size, K, D)
            var1 (Tensor): Target word variances (σ²) of shape (batch_size, K, D)
            var2 (Tensor): Context word variances (σ²) of shape (batch_size, K, D)

        Returns:
            Tensor: Pairwise component log-overlaps of shape (batch_size, K, K)
        """
        # Expand the dimensions for our input tensors for calculation
        mu1_expand = mu1.unsqueeze(2)   # Shape: (batch_size, K, 1, D)
        var1_expand = var1.unsqueeze(2) # Shape: (batch_size, K, 1, D)
        mu2_expand = mu2.unsqueeze(1)   # Shape: (batch_size, 1, K, D)
        var2_expand = var2.unsqueeze(1) # Shape: (batch_size, 1, K, D)

        # Get vector dimension D dynamically from the last axis
        D = mu1.shape[-1]

        # Calculate combined variance
        var_combined = var1_expand + var2_expand    # Shape: (batch_size, K, K, D)

        # Calculate normalized distance
        norm_dist = ((mu1_expand - mu2_expand)**2) / var_combined

        # Calculate combined variance penalty
        var_penalty = torch.log(var_combined)

        # Inner equation
        overlap = var_penalty + norm_dist

        # Sum across D dimensions (you can find the Dth dimension using dim=-1)
        dimensional_sum = torch.sum(overlap, dim=-1)

        log_constant = math.log(2 * math.pi)

        return -0.5 * dimensional_sum - (0.5 * D * log_constant)

    # Time: O(n*K^2), Space: O(n*K^2)   (n = log_overlap_matrix.shape[0] = P; no D factor here -- log_overlap
    # already reduced that axis)
    @staticmethod
    def gmm_energy(log_overlap_matrix, p1, p2):
        """
        Loss function: Step 2
        Computes the GMM energy for each batch item based on the log-overlap matrix and mixture weights.

        Math:
            E(f, g) = log(Σ_k Σ_m [ w_k * w_m * exp(log_Δ(f_k, g_m)) ])
        Computed in log space with logsumexp, because exp() of typical log-overlaps (around -60 at D=50) underflows.
        Args:
            log_overlap_matrix (Tensor): Pairwise component log-overlaps of shape (batch_size, K, K)
            p1 (Tensor): Target word mixture weights of shape (batch_size, K)
            p2 (Tensor): Context word mixture weights of shape (batch_size, K)
        """

        # Log of the combined mixture weight for each pair of components: log(w_k * w_m) = log(w_k) + log(w_m)
        log_p1_expand = torch.log(p1).unsqueeze(2)   # Shape: (batch_size, K, 1)
        log_p2_expand = torch.log(p2).unsqueeze(1)   # Shape: (batch_size, 1, K)
        log_mix_weights = log_p1_expand + log_p2_expand  # Shape: (batch_size, K, K)

        # log(Σ w_k * w_m * exp(log_Δ)) == logsumexp(log(w_k * w_m) + log_Δ), summed across both K axes
        return torch.logsumexp(log_mix_weights + log_overlap_matrix, dim=(-2, -1))  # Shape: (batch_size,)


    # Time: O(P*K^2*D), Space: O(P*K^2*D)   (P = target_ids.shape[0] = batch_size*2*window_size;
    # dominated by log_overlap -- see its comment)
    def forward(self, target_ids: torch.Tensor, ctx_ids: torch.Tensor):
        """
        Forward pass execution method for PyTorch.
        Takes a batch of target and context word ids and returns the energy score for the whole batch.
        """
        # Get target word parameters. These will be compiled for log_overlap and gmm_energy.
        # torch.compile doesn't work for sparse gradient tensors, so we can't use this for the backward-pass optimizer.
        # However, mu/var/mix_weights are plain dense tensors, so they compile fine.
        mu1, var1, p1 = self.get_word_params(target_ids)
        mu2, var2, p2 = self.get_word_params(ctx_ids)

        # Compiled versions of log_overlap/gmm_energy (defined at module scope below the class),
        # not the plain staticmethods -- those stay uncompiled so tests can call them directly as
        # a readable reference implementation. Verified numerically identical to eager; measured
        # ~1.33x faster at batch_size=2560 by fusing this sequential chain of small elementwise
        # ops into fewer kernel launches.
        overlap = _compiled_log_overlap(mu1, mu2, var1, var2)

        # Calculate total GMM energy based on the overlaps and the probability weights
        gmm_energy = _compiled_gmm_energy(overlap, p1, p2)

        return gmm_energy


    # Time: O(P), Space: O(P)   (P = E_pos.shape[0])
    @staticmethod
    def max_margin_ranking(E_pos: torch.Tensor, E_neg: torch.Tensor, margin: float = 1.0):
        """
        Loss function: Step 3
        Computes hinged margin ranking loss on paired energies. 
        This ensures that backpropogation pulls positive pairs closer together and pushes negative pairs away.

        Math:
            L = max(0, margin - E(word, context) + E(word, negative_context))
        """
        # margin - E_pos + E_neg
        losses = margin - E_pos + E_neg

        hinge_loss = torch.clamp(losses, min=0.0)

        # Calculate the fraction of active pairs (those that contributed to the loss)
        active_pairs = (hinge_loss > 0).float().mean()

        return (torch.mean(hinge_loss), active_pairs)


# Compiled separately from the class body, not decorated onto the staticmethods directly, so
# GMMWordEmbedding.log_overlap/gmm_energy stay plain and directly callable for tests/reference use.
# torch.compile here is lazy -- defining these doesn't trigger compilation, only the first
# real call with concrete shapes does (and triggers again if a new shape shows up, e.g. a
# shorter final batch at the end of an epoch).
_compiled_log_overlap = torch.compile(GMMWordEmbedding.log_overlap)
_compiled_gmm_energy = torch.compile(GMMWordEmbedding.gmm_energy)
