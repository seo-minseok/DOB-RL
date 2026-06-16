"""
ensemble_model.py — PyTorch Probabilistic Ensemble (Hopper-v5)
원본 MBPO BNN (TensorFlow) 구조를 PyTorch로 재구현.

Architecture:
  Input  : obs(11) + act(3) = 14D
  Hidden : 4 × 200 (Swish)
  Output : reward(1) + delta_obs(11) = 12D  →  mean + log_var (24D)
  Networks: 7 ensembles, 5 elites

Weight decays (per layer, matching mbpo/mbpo/models/constructor.py):
  L1: 2.5e-5, L2: 5e-5, L3: 7.5e-5, L4: 7.5e-5, Out: 1e-4
"""
import numpy as np
import torch
import torch.nn as nn


LOG_STD_MIN = -10.0
LOG_STD_MAX = 0.5


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class EnsembleLinear(nn.Module):
    """FC layer shared across E networks (vectorized)."""

    def __init__(self, in_features: int, out_features: int, num_networks: int):
        super().__init__()
        self.num_networks = num_networks
        # (E, in, out)  – stored transposed for bmm efficiency
        std = 1.0 / (2.0 * np.sqrt(in_features))
        w = torch.empty(num_networks, in_features, out_features)
        nn.init.trunc_normal_(w, std=std, a=-2*std, b=2*std)
        self.weight = nn.Parameter(w)
        self.bias   = nn.Parameter(torch.zeros(num_networks, 1, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, in) or (E, B, in)
        → (E, B, out)
        """
        if x.dim() == 2:
            # (B, in) → (E, B, out)
            return torch.einsum('bi,eio->ebo', x, self.weight) + self.bias
        # (E, B, in) → (E, B, out)
        return torch.bmm(x, self.weight) + self.bias


class InputNormalizer:
    """Running mean/std normalizer (CPU numpy)."""

    def __init__(self):
        self.mean: np.ndarray = None
        self.std:  np.ndarray = None

    def fit(self, x: np.ndarray):
        self.mean = x.mean(axis=0, keepdims=True).astype(np.float32)
        self.std  = x.std(axis=0, keepdims=True).astype(np.float32)
        self.std  = np.where(self.std < 1e-8, np.ones_like(self.std), self.std)

    def transform(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        if self.mean is None:
            return x
        mean = torch.tensor(self.mean, device=device)
        std  = torch.tensor(self.std,  device=device)
        return (x - mean) / std


class EnsembleModel(nn.Module):
    """
    Probabilistic ensemble dynamics model.
    Output: (mean, log_var) for [reward, delta_obs] = 12D
    """

    WEIGHT_DECAYS = [2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4]

    def __init__(self, obs_dim: int = 11, act_dim: int = 3, rew_dim: int = 1,
                 hidden_dim: int = 200, num_networks: int = 7, num_elites: int = 5):
        super().__init__()
        in_dim  = obs_dim + act_dim
        out_dim = rew_dim + obs_dim   # 12D for Hopper

        self.out_dim      = out_dim
        self.num_networks = num_networks
        self.num_elites   = num_elites

        self.fc1 = EnsembleLinear(in_dim,     hidden_dim,      num_networks)
        self.fc2 = EnsembleLinear(hidden_dim, hidden_dim,      num_networks)
        self.fc3 = EnsembleLinear(hidden_dim, hidden_dim,      num_networks)
        self.fc4 = EnsembleLinear(hidden_dim, hidden_dim,      num_networks)
        self.out = EnsembleLinear(hidden_dim, out_dim * 2,     num_networks)  # mean + log_var

        # Learnable logvar bounds
        self.max_logvar = nn.Parameter( 0.5 * torch.ones(1, out_dim))
        self.min_logvar = nn.Parameter(-10.  * torch.ones(1, out_dim))

        self.normalizer  = InputNormalizer()
        self.elite_inds: list = list(range(num_elites))   # updated after training

    # ------------------------------------------------------------------ #
    def _forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in) or (E, B, in) → (E, B, 2*out_dim) raw output."""
        device = x.device
        x = self.normalizer.transform(x, device)
        x = swish(self.fc1(x))
        x = swish(self.fc2(x))
        x = swish(self.fc3(x))
        x = swish(self.fc4(x))
        return self.out(x)   # (E, B, 2*out_dim)

    def forward(self, x: torch.Tensor):
        """Returns (mean, log_var), each (E, B, out_dim)."""
        raw     = self._forward_raw(x)                          # (E, B, 2*out_dim)
        mean        = raw[..., :self.out_dim]                   # first half
        log_var_raw = raw[..., self.out_dim:]                   # second half
        # Bounded log_var
        log_var = self.max_logvar - torch.nn.functional.softplus(self.max_logvar - log_var_raw)
        log_var = self.min_logvar + torch.nn.functional.softplus(log_var - self.min_logvar)
        return mean, log_var

    # ------------------------------------------------------------------ #
    def _nll_loss(self, inputs: torch.Tensor,
                  targets: torch.Tensor) -> torch.Tensor:
        """
        Gaussian NLL loss (per network).
        inputs  : (E, B, in)
        targets : (E, B, out_dim)
        Returns : (E,) loss per network
        """
        mean, log_var = self.forward(inputs)
        inv_var = torch.exp(-log_var)
        mse     = ((mean - targets) ** 2) * inv_var
        nll     = mse + log_var   # (E, B, out_dim)
        return nll.mean(dim=-1).mean(dim=-1)   # (E,)

    def _mse_loss(self, inputs: torch.Tensor,
                  targets: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(inputs)
        return ((mean - targets) ** 2).mean(dim=-1).mean(dim=-1)   # (E,)

    def _weight_decay_loss(self) -> torch.Tensor:
        layers = [self.fc1, self.fc2, self.fc3, self.fc4, self.out]
        loss   = torch.tensor(0.0, device=self.fc1.weight.device)
        for layer, wd in zip(layers, self.WEIGHT_DECAYS):
            loss = loss + wd * (layer.weight ** 2).sum() / 2.0
        return loss

    # ------------------------------------------------------------------ #
    def train_ensemble(self, inputs: np.ndarray, targets: np.ndarray,
                       batch_size: int = 256, max_epochs: int = None,
                       max_epochs_since_update: int = 5,
                       holdout_ratio: float = 0.2, max_t: float = None,
                       device: torch.device = None) -> dict:
        """
        Train ensemble with NLL loss + weight decay.
        Updates self.elite_inds based on holdout losses.

        inputs  : (N, in_dim)  — [obs, act]
        targets : (N, out_dim) — [rew, delta_obs]

        Returns dict with 'val_loss'.
        """
        import time
        if device is None:
            device = next(self.parameters()).device

        self.normalizer.fit(inputs)

        num_holdout = min(int(inputs.shape[0] * holdout_ratio), 5000)
        perm        = np.random.permutation(inputs.shape[0])
        train_in  = inputs[perm[num_holdout:]]
        train_tgt = targets[perm[num_holdout:]]
        hold_in   = inputs[perm[:num_holdout]]
        hold_tgt  = targets[perm[:num_holdout]]

        N = train_in.shape[0]
        # Tile holdout for all networks: (E, num_holdout, dim)
        hold_in_t  = torch.tensor(
            np.tile(hold_in[None],  [self.num_networks, 1, 1]), device=device)
        hold_tgt_t = torch.tensor(
            np.tile(hold_tgt[None], [self.num_networks, 1, 1]), device=device)

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        snapshots    = {i: (None, 1e10) for i in range(self.num_networks)}
        states       = {}
        epochs_since = 0
        t0 = time.time()

        import itertools
        epoch_iter = range(max_epochs) if max_epochs else itertools.count()

        # Independent row-shuffle per network
        idxs = np.stack([np.random.permutation(N) for _ in range(self.num_networks)])

        for epoch in epoch_iter:
            num_batches = int(np.ceil(N / batch_size))
            for b in range(num_batches):
                batch_idxs = idxs[:, b * batch_size:(b + 1) * batch_size]
                # (E, B, dim)
                in_b  = torch.tensor(
                    np.stack([train_in[batch_idxs[e]]  for e in range(self.num_networks)]),
                    device=device)
                tgt_b = torch.tensor(
                    np.stack([train_tgt[batch_idxs[e]] for e in range(self.num_networks)]),
                    device=device)

                optimizer.zero_grad()
                nll   = self._nll_loss(in_b, tgt_b).sum()
                loss  = nll + self._weight_decay_loss()
                loss += 0.01 * self.max_logvar.sum() - 0.01 * self.min_logvar.sum()
                loss.backward()
                optimizer.step()

            # Shuffle per network
            for e in range(self.num_networks):
                idxs[e] = np.random.permutation(N)

            # Holdout evaluation
            with torch.no_grad():
                hold_losses = self._mse_loss(hold_in_t, hold_tgt_t).cpu().numpy()

            updated = False
            for i in range(self.num_networks):
                cur  = float(hold_losses[i])
                _, best = snapshots[i]
                if best < 1e-9:
                    improvement = 1.0
                else:
                    improvement = (best - cur) / best
                if improvement > 0.01:
                    snapshots[i] = (epoch, cur)
                    states[i]    = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                    updated      = True

            if updated:
                epochs_since = 0
            else:
                epochs_since += 1

            if epochs_since > max_epochs_since_update:
                break
            if max_t and (time.time() - t0) > max_t:
                break

        # Restore best states per network (use best overall if per-network restore not feasible)
        # For simplicity, restore from the network with lowest final holdout loss
        if states:
            best_net = min(snapshots, key=lambda i: snapshots[i][1])
            if best_net in states:
                self.load_state_dict(states[best_net])

        # Update elite indices
        final_hold_losses = [snapshots[i][1] for i in range(self.num_networks)]
        self.elite_inds   = sorted(range(self.num_networks),
                                   key=lambda i: final_hold_losses[i])[:self.num_elites]
        val_loss = np.mean([final_hold_losses[i] for i in self.elite_inds])
        print(f'[ Ensemble ] Holdout losses: {[f"{v:.4f}" for v in final_hold_losses]} '
              f'| Elites: {self.elite_inds}')
        return {'val_loss': float(val_loss)}

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, inputs: np.ndarray, factored: bool = False,
                device: torch.device = None):
        """
        inputs : (B, in_dim) numpy array

        factored=True  → (E, B, out_dim) mean, (E, B, out_dim) var
        factored=False → (B, out_dim) mean, (B, out_dim) var  (ensemble aggregated)
        """
        if device is None:
            device = next(self.parameters()).device
        x = torch.tensor(inputs, dtype=torch.float32, device=device)
        mean, log_var = self.forward(x)
        var           = torch.exp(log_var)

        mean_np = mean.cpu().numpy()    # (E, B, out_dim)
        var_np  = var.cpu().numpy()

        if factored:
            return mean_np, var_np

        agg_mean = mean_np.mean(axis=0)
        agg_var  = var_np.mean(axis=0) + np.square(mean_np - agg_mean).mean(axis=0)
        return agg_mean, agg_var

    def random_elite_inds(self, batch_size: int) -> np.ndarray:
        return np.random.choice(self.elite_inds, size=batch_size)
