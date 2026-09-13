"""Training definition for ONE shared, dose-aware formulation network.

PyTorch is a training dependency only. The matching NumPy implementation lives
in recommender.formulation_core. No pretrained encoders are imported here.
"""
from __future__ import annotations

import torch
from torch import nn


class Residual(nn.Module):
    def __init__(self, width=256, expansion=384):
        super().__init__()
        self.norm = nn.LayerNorm(width, eps=1e-5)
        self.up = nn.Linear(width, expansion)
        self.down = nn.Linear(expansion, width)

    def forward(self, x):
        return x + self.down(torch.relu(self.up(self.norm(x))))


class FormulationNetwork(nn.Module):
    """Mass-weighted set moments + attention + ordered process recurrence.

All tasks traverse the same fusion/residual trunk. Heads are readouts of that
trunk, not separately loaded networks. Ingredient count is not fixed by weights.
"""
    def __init__(self, feature_width, fine_outputs, quantitative_outputs, actions):
        super().__init__()
        self.molecule_in = nn.Linear(feature_width, 384)
        self.molecule_out = nn.Linear(384, 256)
        self.context_in = nn.Linear(64, 128)
        self.attention_key = nn.Linear(256, 64, bias=False)
        self.attention_query = nn.Linear(128, 64, bias=False)
        self.step_embedding = nn.Embedding(actions + 1, 48, padding_idx=0)
        self.process_input = nn.Linear(60, 96)
        self.process_state = nn.Linear(96, 96, bias=False)
        self.fusion = nn.Linear(992, 384)
        self.shared_in = nn.Linear(384, 256)
        self.shared = nn.ModuleList([Residual(), Residual()])
        self.fine_head = nn.Linear(256, fine_outputs)
        self.quantitative_head = nn.Linear(256, quantitative_outputs)
        self.transport_head = nn.Linear(256, 10)
        self.action_head = nn.Linear(256, actions)
        self.check_head = nn.Linear(256, 8)
        self.revision_head = nn.Linear(256, 19)
        self.emulsion_head = nn.Linear(256, 101)
        nn.init.zeros_(self.emulsion_head.weight)
        nn.init.zeros_(self.emulsion_head.bias)

    def forward(self, molecules, masses, context, step_ids, step_values):
        mol = torch.relu(self.molecule_out(torch.relu(self.molecule_in(molecules))))
        weights = masses / masses.sum(-1, keepdim=True).clamp_min(1e-30)
        mean = (weights[..., None] * mol).sum(1)
        variance = (weights[..., None] * (mol - mean[:, None])**2).sum(1)
        ctx = torch.relu(self.context_in(context))
        attention = (self.attention_key(mol) * self.attention_query(ctx)[:, None]).sum(-1) / 8
        # Zero-mass rows, including padding, have exactly no contribution.
        attention = attention + masses.clamp_min(1e-30).log()
        attention = attention.masked_fill(masses <= 0, -1e30)
        aw = torch.softmax(attention, -1) * (masses > 0)
        aw = aw / aw.sum(-1, keepdim=True).clamp_min(1e-30)
        attended = (aw[..., None] * mol).sum(1)
        state = torch.zeros((len(molecules), 96), device=molecules.device)
        for j in range(step_ids.shape[1]):
            embedded = torch.cat((self.step_embedding(step_ids[:, j]), step_values[:, j]), -1)
            updated = torch.tanh(self.process_input(embedded) + self.process_state(state))
            state = torch.where((step_ids[:, j] != 0)[:, None], updated, state)
        fused = torch.cat((mean, variance, attended, ctx, state), -1)
        h = torch.relu(self.shared_in(torch.relu(self.fusion(fused))))
        for block in self.shared:
            h = block(h)
        return {name: getattr(self, name + '_head')(h) for name in
                ('fine', 'quantitative', 'transport', 'action', 'check', 'revision', 'emulsion')}


def export_arrays(model):
    return {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}
