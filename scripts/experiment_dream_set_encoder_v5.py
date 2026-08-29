#!/usr/bin/env python
"""Outcome-aware attention-set encoder search for DREAM mixtures.

The search uses already-public external outcomes and is therefore diagnostic
only. It cannot authorize a 90% claim or a production model. The purpose is to
test whether learned component interactions beat fixed mean/set embeddings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from ogb.utils import smiles2graph
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import benchmark_dream_mixture_2025 as base  # noqa: E402
from scripts import benchmark_dream_pair_ensemble_v2 as pair  # noqa: E402


SCHEMA = "dream-attention-set-outcome-aware-search/v5"
SEEDS = (20_260_828, 20_260_829, 20_260_830)
MAX_COMPONENTS = 43
EPOCHS = 80
BATCH_SIZE = 256
PCA_DIMENSIONS = 32
V4_REPORT_SHA256 = "60db21db55ea775fe75103e6f8264d87cbbc0b061c40e0f735524d78f7d241f6"


@dataclass(frozen=True)
class Config:
    hidden: int
    layers: int
    dropout: float
    design_weight: float
    descriptor_loss_weight: float
    learning_rate: float = 0.001
    weight_decay: float = 0.003


CONFIGS = (
    Config(64, 1, 0.10, 1.0, 0.02),
    Config(64, 2, 0.15, 2.0, 0.02),
    Config(96, 1, 0.15, 3.0, 0.05),
    Config(96, 2, 0.20, 3.0, 0.02),
)


def sha256(path: Path) -> str:
    return base._sha256(path)


def git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    ).stdout.strip()


def required_files(root: Path) -> dict[str, Path]:
    return {
        "training_pairs": root / "Training_Dataset" / "TrainingData_mixturedist.csv",
        "training_definitions": root
        / "Training_Dataset"
        / "Mixure_Definitions_Training_set.csv",
        "test_pairs": root / "Test_Dataset" / "Test_set_mixturedist.csv",
        "test_definitions": root
        / "Test_Dataset"
        / "Test_set_Mixure_Definitions.csv",
        "validation_definitions": root
        / "Validation_Dataset"
        / "Mixure_Definitions_Validation_set.csv",
        "public_validation_predictions": root
        / "Predictions"
        / "Validation_set_Prediction_top6_Teams.csv",
        "pom_profiles": root
        / "PostChallenge_Model"
        / "Dataset"
        / "openpom_ensemble_predictions_results.csv",
        "cid_to_smiles": root
        / "PostChallenge_Model"
        / "Dataset"
        / "cid_to_smiles.csv",
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


class SetEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, config: Config, pom_dim: int):
        super().__init__()
        self.project = torch.nn.Sequential(
            torch.nn.Linear(input_dim, config.hidden),
            torch.nn.GELU(),
            torch.nn.LayerNorm(config.hidden),
        )
        layer = torch.nn.TransformerEncoderLayer(
            d_model=config.hidden,
            nhead=4,
            dim_feedforward=config.hidden * 2,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=config.layers)
        self.attention = torch.nn.Linear(config.hidden, 1)
        self.mix_norm = torch.nn.LayerNorm(config.hidden * 2)
        pair_width = config.hidden * 8 + 5
        self.head = torch.nn.Sequential(
            torch.nn.Linear(pair_width, config.hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(config.hidden, 1),
        )
        self.descriptor_head = torch.nn.Linear(config.hidden * 2, pom_dim)

    def mixture(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.project(values)
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        scores = self.attention(hidden).squeeze(-1).masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        attended = torch.sum(hidden * weights.unsqueeze(-1), dim=1)
        masked = hidden.masked_fill(~mask.unsqueeze(-1), -1e9)
        maximum = torch.max(masked, dim=1).values
        return self.mix_norm(torch.cat([attended, maximum], dim=1))

    def forward(
        self,
        first: torch.Tensor,
        first_mask: torch.Tensor,
        second: torch.Tensor,
        second_mask: torch.Tensor,
        design: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        left = self.mixture(first, first_mask)
        right = self.mixture(second, second_mask)
        cosine = torch.nn.functional.cosine_similarity(left, right).unsqueeze(1)
        pair_values = torch.cat(
            [
                torch.abs(left - right),
                left * right,
                left + right,
                torch.minimum(left, right),
                cosine,
                design,
            ],
            dim=1,
        )
        prediction = torch.sigmoid(self.head(pair_values).squeeze(1))
        return prediction, self.descriptor_head(left), self.descriptor_head(right)


def design_for_pair(first: Sequence[int], second: Sequence[int]) -> np.ndarray:
    left = set(first)
    right = set(second)
    overlap = len(left & right)
    smaller = max(1, min(len(left), len(right)))
    return np.asarray(
        [
            len(left) / 43.0,
            len(right) / 43.0,
            overlap / 30.0,
            overlap / smaller,
        ],
        dtype=np.float32,
    )


def build_pair_rows(
    pairs: pd.DataFrame,
    definitions: Mapping[Any, Sequence[int]],
    key_to_index: Mapping[Any, int],
    *,
    training: bool,
    target_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    first_indices = []
    second_indices = []
    designs = []
    targets = []
    for _, row in pairs.iterrows():
        if training:
            source = str(row["Dataset"]).strip()
            first_key = (source, base._label(row["Mixture 1"]))
            second_key = (source, base._label(row["Mixture 2"]))
        else:
            first_key = base._label(row["Mixture 1"])
            second_key = base._label(row["Mixture 2"])
        first_indices.append(key_to_index[first_key])
        second_indices.append(key_to_index[second_key])
        designs.append(design_for_pair(definitions[first_key], definitions[second_key]))
        targets.append(float(row[target_column]))
    return (
        np.asarray(first_indices, dtype=np.int64),
        np.asarray(second_indices, dtype=np.int64),
        np.asarray(designs, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


def fit_model(
    *,
    config: Config,
    seed: int,
    mixture_values: torch.Tensor,
    mixture_masks: torch.Tensor,
    mixture_pom: torch.Tensor,
    train_rows: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    device: torch.device,
) -> SetEncoder:
    seed_everything(seed)
    model = SetEncoder(mixture_values.shape[-1], config, mixture_pom.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    first, second, design, target = train_rows
    exact10 = (
        (np.isclose(design[:, 0], 10.0 / 43.0))
        & (np.isclose(design[:, 1], 10.0 / 43.0))
        & (design[:, 2] == 0.0)
    )
    sample_weight = np.where(exact10, config.design_weight, 1.0).astype(np.float32)
    generator = np.random.default_rng(seed)
    model.train()
    for _epoch in range(EPOCHS):
        order = generator.permutation(len(target))
        for start in range(0, len(order), BATCH_SIZE):
            index = order[start : start + BATCH_SIZE]
            batch_first = torch.as_tensor(first[index], device=device)
            batch_second = torch.as_tensor(second[index], device=device)
            batch_design = torch.as_tensor(design[index], device=device)
            batch_target = torch.as_tensor(target[index], device=device)
            batch_weight = torch.as_tensor(sample_weight[index], device=device)
            prediction, left_descriptor, right_descriptor = model(
                mixture_values[batch_first],
                mixture_masks[batch_first],
                mixture_values[batch_second],
                mixture_masks[batch_second],
                batch_design,
            )
            regression = torch.sum(batch_weight * (prediction - batch_target) ** 2)
            regression /= torch.sum(batch_weight)
            descriptor = torch.nn.functional.mse_loss(
                torch.sigmoid(left_descriptor), mixture_pom[batch_first]
            ) + torch.nn.functional.mse_loss(
                torch.sigmoid(right_descriptor), mixture_pom[batch_second]
            )
            loss = regression + config.descriptor_loss_weight * descriptor
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    return model.eval()


@torch.inference_mode()
def predict(
    model: SetEncoder,
    rows: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    mixture_values: torch.Tensor,
    mixture_masks: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    first, second, design, _target = rows
    output = []
    for start in range(0, len(first), 256):
        left = torch.as_tensor(first[start : start + 256], device=device)
        right = torch.as_tensor(second[start : start + 256], device=device)
        batch_design = torch.as_tensor(design[start : start + 256], device=device)
        prediction, _left_descriptor, _right_descriptor = model(
            mixture_values[left],
            mixture_masks[left],
            mixture_values[right],
            mixture_masks[right],
            batch_design,
        )
        output.append(prediction.detach().cpu().numpy())
    return np.clip(np.concatenate(output), 0.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dream-root", type=Path, required=True)
    parser.add_argument("--pair-source-root", type=Path, required=True)
    parser.add_argument(
        "--v4-report",
        type=Path,
        default=ROOT / "benchmarks" / "dream_accuracy_search_v4.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks" / "dream_set_encoder_search_v5.json",
    )
    args = parser.parse_args()
    dream_root = args.dream_root.expanduser().resolve(strict=True)
    pair_root = args.pair_source_root.expanduser().resolve(strict=True)
    v4_report_path = args.v4_report.expanduser().resolve(strict=True)
    if sha256(v4_report_path) != V4_REPORT_SHA256:
        raise RuntimeError("frozen DREAM v4 report changed")
    v4_report = json.loads(v4_report_path.read_text(encoding="utf-8"))
    v4_test = v4_report["selected_diagnostic"]["test"]
    v4_validation = v4_report["selected_diagnostic"]["validation"]
    if git_commit(dream_root) != base.DREAM_COMMIT:
        raise RuntimeError("unsupported DREAM source commit")
    required = required_files(dream_root)
    changed = [
        name
        for name, path in required.items()
        if not path.is_file() or sha256(path) != base.DREAM_FILE_SHA256[name]
    ]
    if changed:
        raise RuntimeError("DREAM source changed: " + ", ".join(changed))

    pom, rdkit, _morgan, _scaffolds, smiles, pom_names, _rdkit_names = (
        base._component_features(dream_root)
    )
    train_definitions = base._definitions(required["training_definitions"], training=True)
    test_definitions = base._definitions(required["test_definitions"], training=False)
    validation_definitions = base._definitions(
        required["validation_definitions"], training=False
    )
    all_cids = sorted(
        {
            cid
            for definitions in (
                train_definitions,
                test_definitions,
                validation_definitions,
            )
            for components in definitions.values()
            for cid in components
        }
    )
    model_root = dream_root / "SOTA" / "3-Pair_Model" / "finetuned_model"
    pair_model, pair_data, pairdata, _pair_config = pair._load_pair_model(
        pair_root, model_root / "config.json", model_root / "model.pt"
    )
    pair_smiles_path = dream_root / "SOTA" / "data" / "raw" / "cid_to_smiles.json"
    if sha256(pair_smiles_path) != pair.PAIR_SMILES_SHA256:
        raise RuntimeError("odor-pair SMILES changed")
    exact_smiles = {
        int(cid): text
        for cid, text in json.loads(pair_smiles_path.read_text(encoding="utf-8")).items()
    }
    graphs = {
        cid: pairdata.to_torch(smiles2graph(exact_smiles[cid])) for cid in all_cids
    }
    single_embeddings = pair._generated_embeddings(
        {cid: [cid] for cid in all_cids}, pair_model, pair_data, pairdata, graphs
    )
    single_rows = np.asarray([single_embeddings[cid] for cid in all_cids], dtype=float)
    pom_rows = np.asarray([pom[cid] for cid in all_cids], dtype=float)
    rdkit_rows = np.asarray([rdkit[cid] for cid in all_cids], dtype=float)
    rdkit_pca = PCA(n_components=PCA_DIMENSIONS, random_state=SEEDS[0]).fit_transform(
        StandardScalerLike.fit_transform(rdkit_rows)
    )
    component_values = np.column_stack([single_rows, pom_rows, rdkit_pca])
    component_values = StandardScalerLike.fit_transform(component_values).astype(
        np.float32
    )
    cid_to_row = {cid: index for index, cid in enumerate(all_cids)}

    mixture_definitions: list[tuple[Any, Sequence[int]]] = []
    for name, definitions in (
        ("train", train_definitions),
        ("test", test_definitions),
        ("validation", validation_definitions),
    ):
        mixture_definitions.extend(definitions.items())
    mixture_array = np.zeros(
        (len(mixture_definitions), MAX_COMPONENTS, component_values.shape[1]),
        dtype=np.float32,
    )
    mixture_mask = np.zeros((len(mixture_definitions), MAX_COMPONENTS), dtype=bool)
    mixture_pom_target = np.zeros(
        (len(mixture_definitions), len(pom_names)), dtype=np.float32
    )
    key_maps: dict[str, dict[Any, int]] = {}
    cursor = 0
    for name, definitions in (
        ("train", train_definitions),
        ("test", test_definitions),
        ("validation", validation_definitions),
    ):
        current_map = {}
        for key, components in definitions.items():
            if not 1 <= len(components) <= MAX_COMPONENTS:
                raise RuntimeError("mixture component count exceeds encoder contract")
            rows = [cid_to_row[cid] for cid in components]
            mixture_array[cursor, : len(rows)] = component_values[rows]
            mixture_mask[cursor, : len(rows)] = True
            mixture_pom_target[cursor] = np.mean(pom_rows[rows], axis=0)
            current_map[key] = cursor
            cursor += 1
        key_maps[name] = current_map

    train_pairs = pd.read_csv(required["training_pairs"])
    test_pairs = pd.read_csv(required["test_pairs"])
    validation_pairs = pd.read_csv(required["public_validation_predictions"])
    train_rows = build_pair_rows(
        train_pairs,
        train_definitions,
        key_maps["train"],
        training=True,
        target_column="Experimental Values",
    )
    test_rows = build_pair_rows(
        test_pairs,
        test_definitions,
        key_maps["test"],
        training=False,
        target_column="Experimental values",
    )
    validation_rows = build_pair_rows(
        validation_pairs,
        validation_definitions,
        key_maps["validation"],
        training=False,
        target_column="ExpMean_combined",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    values_tensor = torch.as_tensor(mixture_array, device=device)
    mask_tensor = torch.as_tensor(mixture_mask, device=device)
    pom_tensor = torch.as_tensor(mixture_pom_target, device=device)

    candidates = []
    for config_index, config in enumerate(CONFIGS):
        test_members = []
        validation_members = []
        for seed in SEEDS:
            model = fit_model(
                config=config,
                seed=seed,
                mixture_values=values_tensor,
                mixture_masks=mask_tensor,
                mixture_pom=pom_tensor,
                train_rows=train_rows,
                device=device,
            )
            test_members.append(
                predict(model, test_rows, values_tensor, mask_tensor, device)
            )
            validation_members.append(
                predict(model, validation_rows, values_tensor, mask_tensor, device)
            )
        test_prediction = np.mean(test_members, axis=0)
        validation_prediction = np.mean(validation_members, axis=0)
        name = f"set_encoder_config{config_index}"
        candidates.append(
            {
                "name": name,
                "config": asdict(config),
                "test": base.metrics(test_prediction, test_rows[3]),
                "validation": base.metrics(validation_prediction, validation_rows[3]),
                "test_member_disagreement_mean": float(
                    np.mean(np.std(test_members, axis=0))
                ),
                "validation_member_disagreement_mean": float(
                    np.mean(np.std(validation_members, axis=0))
                ),
            }
        )
    for candidate in candidates:
        checks = {
            "test_pearson": candidate["test"]["pearson"] > v4_test["pearson"],
            "test_spearman": candidate["test"]["spearman"] > v4_test["spearman"],
            "test_rmse": candidate["test"]["rmse"] < v4_test["rmse"],
            "test_mae": candidate["test"]["mae"] < v4_test["mae"],
            "validation_pearson": candidate["validation"]["pearson"]
            > v4_validation["pearson"],
            "validation_spearman": candidate["validation"]["spearman"]
            > v4_validation["spearman"],
            "validation_rmse": candidate["validation"]["rmse"]
            < v4_validation["rmse"],
            "validation_mae": candidate["validation"]["mae"]
            < v4_validation["mae"],
        }
        candidate["v4_point_checks"] = checks
        candidate["v4_point_pareto"] = all(checks.values())
    selected = max(
        candidates,
        key=lambda row: (
            float(row["test"]["pearson"])
            + float(row["validation"]["pearson"]),
            -float(row["test"]["rmse"]) - float(row["validation"]["rmse"]),
        ),
    )
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "attention_set_encoder_rejected_below_v4",
        "source": {
            "dream_commit": base.DREAM_COMMIT,
            "pair_commit": pair.PAIR_SOURCE_COMMIT,
            "pair_weights_sha256": pair.PAIR_WEIGHTS_SHA256,
            "single_embedding_rows_sha256": hashlib.sha256(
                single_rows.astype(np.float32).tobytes()
            ).hexdigest(),
            "v4_report_sha256": sha256(v4_report_path),
        },
        "timing": {
            "test_outcomes_visible": True,
            "validation_outcomes_visible": True,
            "eligible_for_selection_or_promotion": False,
        },
        "implementation": {
            "script_sha256": sha256(Path(__file__).resolve()),
            "component_feature_dimensions": int(component_values.shape[1]),
            "mixtures": len(mixture_definitions),
            "training_pairs": len(train_rows[3]),
            "device": str(device),
            "epochs": EPOCHS,
            "seeds": list(SEEDS),
            "unlabeled_component_preprocessing_scope": (
                "PCA_and_standardization_fit_on_train_test_validation_component_structures"
            ),
        },
        "candidates": candidates,
        "v4_baseline": {"test": v4_test, "validation": v4_validation},
        "selected_diagnostic": selected,
        "gates": {
            "point_pareto_above_v4": {
                "passed": any(row["v4_point_pareto"] for row in candidates)
            },
            "human_ceiling_90_percent": {"passed": False, "not_recomputed": True},
            "production": {"passed": False, "runtime_primary_score_weight": 0.0},
        },
        "claim_boundary": {
            "outcome_aware_hyperparameter_search": True,
            "human_olfactory_90_percent_certified": False,
            "natural_language_recipe_accuracy_measured": False,
        },
    }
    base._write_json(args.output.expanduser().resolve(), report)
    print(json.dumps({"selected": selected, "output": str(args.output)}, indent=2))
    return 0


class StandardScalerLike:
    """Tiny deterministic array standardizer to keep component preprocessing explicit."""

    @staticmethod
    def fit_transform(values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=float)
        mean = np.mean(matrix, axis=0)
        scale = np.std(matrix, axis=0)
        scale = np.where(scale < 1e-12, 1.0, scale)
        return (matrix - mean) / scale


if __name__ == "__main__":
    raise SystemExit(main())
