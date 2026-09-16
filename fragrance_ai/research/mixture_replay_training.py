"""Resumable real MLP training branches for the prefix-only replay controller."""
from __future__ import annotations

from copy import deepcopy
import time

import numpy as np
import torch

from fragrance_ai.research.mixture_mlp import MixtureMLP
from fragrance_ai.research.physmix_comparison import FrozenAggregation, PairComparison
from fragrance_ai.research.r2_physsim import combined_loss
from fragrance_ai.research.replay_search import Observation
from scripts.compare_physmix_v83 import mixture_key


class PairBank:
    def __init__(self, core, datasets, device):
        self.device = torch.device(device)
        self.arrays = core.arrays
        molecules = sorted({s for pairs in datasets.values() for p in pairs for s in p.molecules})
        raw = core.features(molecules)
        normalized = np.clip((raw - core.arrays["feature_mean"]) / core.arrays["feature_scale"], -12, 12)
        hidden = np.maximum(0, normalized @ core.arrays["molecule_in.weight"].T + core.arrays["molecule_in.bias"])
        hidden = np.maximum(0, hidden @ core.arrays["molecule_out.weight"].T + core.arrays["molecule_out.bias"]).astype(np.float32)
        if not np.isfinite(hidden).all():
            raise ValueError("nonfinite molecular features")
        self.molecular_embeddings = hidden
        lookup = dict(zip(molecules, hidden))
        mixtures = sorted({mixture_key(m) for pairs in datasets.values() for p in pairs for m in (p.mixture_a, p.mixture_b)})
        index = {m: i for i, m in enumerate(mixtures)}
        n = max(len(m) for m in mixtures)
        features, amounts = np.zeros((len(mixtures), n, 256), np.float32), np.zeros((len(mixtures), n), np.float32)
        for i, mixture in enumerate(mixtures):
            features[i, :len(mixture)] = np.stack([lookup[s] for s, _ in mixture])
            counts = np.asarray([c for _, c in mixture], np.float32)
            amounts[i, :len(mixture)] = counts / counts.sum()
        e, w = torch.as_tensor(features, device=device), torch.as_tensor(amounts, device=device)
        context = torch.zeros((len(mixtures), 64), device=device)
        context[:, 12] = 2.
        with torch.no_grad():
            backbone = FrozenAggregation(core.arrays).to(device)
            buffers = [[], [], []]
            for offset in range(0, len(e), 16):
                for sink, value in zip(buffers, backbone(e[offset:offset+16], w[offset:offset+16], context[offset:offset+16])):
                    sink.append(value)
        self.cache = (e, w, *[torch.cat(buffer, 0) for buffer in buffers])
        self.datasets = {name: (
            torch.tensor([index[mixture_key(p.mixture_a)] for p in pairs], device=device),
            torch.tensor([index[mixture_key(p.mixture_b)] for p in pairs], device=device),
            torch.tensor([p.similarity for p in pairs], device=device),
        ) for name, pairs in datasets.items()}

    def batch(self, name, indices):
        first, second, labels = self.datasets[name]
        return (tuple(x[first[indices]] for x in self.cache), tuple(x[second[indices]] for x in self.cache), labels[indices])

    @torch.no_grad()
    def predict(self, model, name, indices):
        model.eval()
        values = []
        for offset in range(0, len(indices), 16):
            a, b, _ = self.batch(name, indices[offset:offset+16])
            values.extend(model.forward_precomputed(a, b).cpu().tolist())
        return np.asarray(values)


class TrainingBranch:
    def __init__(self, bank, mode, train, validation, seed, *, epochs=60, patience=8):
        if not train or not validation or set(train) & set(validation):
            raise ValueError("disjoint nonempty fitting and selection indices required")
        self.bank, self.mode, self.train, self.validation = bank, mode, list(train), list(validation)
        self.seed, self.epochs, self.patience = seed, epochs, patience
        torch.manual_seed(seed)
        common_head = deepcopy(PairComparison(bank.arrays, "current").head.state_dict())
        torch.manual_seed(seed)
        self.model = MixtureMLP(bank.arrays, mode).to(bank.device)
        self.model.head.load_state_dict(common_head)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=.0003, weight_decay=.0001)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs, eta_min=.00003)
        self.step, self.stale, self.best, self.selected = 0, 0, float("inf"), None
        self.history = []

    def advance(self):
        if self.step >= self.epochs or self.stale >= self.patience:
            raise ValueError("training branch has already terminated")
        device = self.bank.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        self.step += 1
        order = np.random.default_rng(self.seed + self.step * 9137).permutation(self.train).tolist()
        self.model.train()
        losses = []
        # Branch-local randomness: interleaving decisions cannot change a
        # branch's weights at a given epoch. This makes the replay executable.
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed + self.step * 10007)
            for offset in range(0, len(order), 16):
                a, b, labels = self.bank.batch("snitz", order[offset:offset+16])
                self.optimizer.zero_grad(set_to_none=True)
                loss = combined_loss(self.model.forward_precomputed(a, b), labels)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite real training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
                self.optimizer.step()
                losses.append(float(loss.detach()))
        self.scheduler.step()
        prediction = self.bank.predict(self.model, "snitz", self.validation)
        target = self.bank.datasets["snitz"][2][self.validation].cpu().numpy()
        rmse = float(np.sqrt(np.mean((prediction - target)**2)))
        if rmse < self.best - 1e-5:
            self.best, self.stale = rmse, 0
            self.selected = {"epoch": self.step, "rmse": rmse, "validation_prediction": prediction,
                             "state": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}}
        else:
            self.stale += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        observation = Observation(self.mode, self.step, -rmse, elapsed,
                                  terminal=self.step >= self.epochs or self.stale >= self.patience)
        self.history.append({"observation": observation, "training_loss": float(np.mean(losses)),
                             "stale_epochs": self.stale, "optimizer_updates": (len(order) + 15) // 16})
        return observation

    def load_selected(self):
        if self.selected is None:
            raise ValueError("no evaluated checkpoint in this branch")
        self.model.load_state_dict(self.selected["state"])
        self.model.eval()
        return self.model


def validation_ensemble(predictions, targets, *, regularization=.001):
    """Small convex stack, fit only to selection data, not outer labels."""
    from scipy.optimize import minimize
    values = np.asarray(predictions, float).T
    target = np.asarray(targets, float)
    if values.ndim != 2 or values.shape[0] != len(target) or not np.isfinite(values).all() or not np.isfinite(target).all():
        raise ValueError("finite matched validation predictions required")
    n = values.shape[1]
    if n < 1:
        raise ValueError("at least one trained model required")
    prior = np.full(n, 1 / n)

    def loss(w):
        error = values @ w - target
        return float(np.mean(error**2) + regularization * np.sum((w - prior)**2)), 2 * values.T @ error / len(target) + 2 * regularization * (w - prior)

    fit = minimize(loss, prior, jac=True, method="SLSQP", bounds=[(0., 1.)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1., "jac": lambda w: np.ones(n)}],
                   options={"maxiter": 200, "ftol": 1e-12})
    best = int(np.argmin(np.mean((values - target[:, None])**2, axis=0)))
    fallback = np.eye(n)[best]
    if not fit.success or not np.isfinite(fit.x).all() or np.min(fit.x) < -1e-8 or abs(fit.x.sum() - 1.) > 1e-7:
        return fallback, {"status": "single_model_solver_fallback", "best_single": best}
    weight = np.maximum(fit.x, 0.)
    weight /= weight.sum()
    if np.mean((values @ weight - target)**2) > np.mean((values[:, best] - target)**2):
        return fallback, {"status": "single_model_validation_retained", "best_single": best}
    return weight, {"status": "validation_regularized_convex_stack", "best_single": best}
