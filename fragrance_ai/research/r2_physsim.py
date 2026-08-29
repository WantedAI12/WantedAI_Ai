"""JCIM R2 PhysSim-Core architecture and reproducible data utilities.

This module implements the parsimonious three-interaction specification in
the user's R2 manuscript.  It deliberately keeps the two extended-model
projections/scalars in the state dictionary so that the parameter count and
checkpoint layout remain compatible with the reported 162,059-parameter
reference architecture, but only the five core constants are active.

The implementation is a mixture-similarity model.  It is not a molecular
odor-character model and its learned latent properties are not literal mass,
charge, radius, position, or velocity measurements.
"""

from __future__ import annotations

import csv
from collections import defaultdict
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:  # Optional production dependency; required by the training script.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset
except ImportError as error:  # pragma: no cover - exercised by minimal installs
    raise ImportError(
        "R2 PhysSim research support requires the 'physsim' optional dependencies"
    ) from error


MAX_MOLECULES = 50
DESCRIPTOR_DIM = 217
LATENT_DIM = 128
FINGERPRINT_DIM = 134
PROJECTION_DIM = 128
SIMILARITY_HEAD_DIM = 257
N_STEPS = 16
SOFT_CORE_DELTA = 0.5
TIME_HORIZON = 0.1
EXPECTED_PARAMETER_COUNT = 162_059
MODEL_SPEC_VERSION = "jcim-r2-physsim-core-1.0"
PU_LABEL_CONTRACT_VERSION = "positive-unlabeled-odor-labels-1.0"
STRICT_SPLIT_CONTRACT_VERSION = "all-components-held-out-1.0"


@dataclass(frozen=True)
class MixturePair:
    mixture_a: tuple[str, ...]
    mixture_b: tuple[str, ...]
    similarity: float
    record_id: str

    @property
    def molecules(self) -> frozenset[str]:
        return frozenset((*self.mixture_a, *self.mixture_b))


@dataclass(frozen=True)
class OdorDescriptorRecord:
    """A canonical molecule with standardized monomolecular odor labels."""

    smiles: str
    labels: tuple[float, ...]
    # Public odor archives are predominantly positive assertion catalogues.
    # A missing word is not a measured negative, so this mask marks only
    # source-backed positive assertions.  It intentionally must not be used
    # as a conventional all-label observation mask.
    positive_observation_mask: tuple[float, ...]
    # Per-label source lineage makes the positive-only contract auditable
    # after source unions, rather than losing it in a single binary vector.
    label_sources: tuple[tuple[str, ...], ...]
    sources: tuple[str, ...]


@dataclass(frozen=True)
class DescriptorNormalizer:
    mean: np.ndarray
    std: np.ndarray
    descriptor_names: tuple[str, ...]

    def transform(self, value: np.ndarray) -> np.ndarray:
        standardized = (
            value.astype(np.float64) - self.mean.astype(np.float64)
        ) / self.std.astype(np.float64)
        # External chemistry archives occasionally contain polymer-like or
        # malformed structures whose otherwise finite descriptor values are
        # many orders of magnitude outside volatile odorant chemistry.  A
        # wide z-score guard prevents one such value from poisoning inference
        # without affecting ordinary molecules.
        standardized = np.nan_to_num(
            standardized, nan=0.0, posinf=100.0, neginf=-100.0
        )
        return np.clip(standardized, -100.0, 100.0).astype(np.float32)

    def as_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean.astype(float).tolist(),
            "std": self.std.astype(float).tolist(),
            "descriptor_names": list(self.descriptor_names),
        }


def require_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as error:  # pragma: no cover - dependency guard
        raise RuntimeError("RDKit is required to featurize or split R2 data") from error
    if len(Descriptors.descList) != DESCRIPTOR_DIM:
        raise RuntimeError(
            f"R2 descriptor contract requires {DESCRIPTOR_DIM} RDKit descriptors; "
            f"this runtime exposes {len(Descriptors.descList)}"
        )
    return Chem, Descriptors, MurckoScaffold


def descriptor_contract() -> tuple[tuple[str, object], ...]:
    _, descriptors, _ = require_rdkit()
    return tuple((name, function) for name, function in descriptors.descList)


def canonical_smiles(smiles: str) -> str:
    chem, _, _ = require_rdkit()
    molecule = chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    return chem.MolToSmiles(molecule, isomericSmiles=True, canonical=True)


def bemis_murcko_scaffold(smiles: str) -> str:
    chem, _, murcko = require_rdkit()
    molecule = chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    # The empty scaffold is the standard Bemis-Murcko result for acyclic
    # molecules.  It remains one group rather than being silently split into
    # molecule-identical pseudo-scaffolds.
    return murcko.MurckoScaffoldSmiles(
        mol=molecule, includeChirality=False
    )


def smiles_to_descriptors(smiles: str) -> np.ndarray:
    chem, _, _ = require_rdkit()
    molecule = chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    values: list[float] = []
    for _, function in descriptor_contract():
        try:
            value = function(molecule)
            numeric = float(value) if value is not None else 0.0
            if not math.isfinite(numeric):
                numeric = 0.0
            # Stay well inside float32 while preserving descriptor ordering.
            values.append(max(-1e30, min(1e30, numeric)))
        except Exception:
            values.append(0.0)
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (DESCRIPTOR_DIM,):
        raise RuntimeError(f"unexpected descriptor shape: {result.shape}")
    return result


def build_raw_descriptor_cache(smiles: Iterable[str]) -> dict[str, np.ndarray]:
    return {value: smiles_to_descriptors(value) for value in sorted(set(smiles))}


def fit_normalizer(
    raw_cache: dict[str, np.ndarray], training_molecules: Iterable[str]
) -> DescriptorNormalizer:
    molecules = sorted(set(training_molecules))
    if not molecules:
        raise ValueError("cannot fit descriptor normalizer without training molecules")
    matrix = np.asarray([raw_cache[value] for value in molecules], dtype=np.float64)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    # Constant descriptors carry no fold-specific information and map to zero.
    std = np.where(std < 1e-8, 1.0, std)
    names = tuple(name for name, _ in descriptor_contract())
    return DescriptorNormalizer(mean.astype(np.float32), std, names)


def normalized_cache(
    raw_cache: dict[str, np.ndarray], normalizer: DescriptorNormalizer
) -> dict[str, np.ndarray]:
    return {
        smiles: normalizer.transform(value) for smiles, value in raw_cache.items()
    }


def _read_cid_smiles(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        for row in csv.DictReader(handle):
            cid = str(row.get("CID", "")).strip()
            smiles = str(row.get("IsomericSMILES", "")).strip()
            if not cid or not smiles:
                continue
            try:
                result[cid] = canonical_smiles(smiles)
            except ValueError:
                continue
    return result


def _identifier(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def _read_flexible_molecule_map(
    path: Path, allowed_identifiers: set[str] | None = None
) -> dict[str, str]:
    """Read the small schema variants used by the Pyrfume archives."""

    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        for row in csv.DictReader(handle):
            identifier = _identifier(row.get("CID", row.get("Unnamed: 0", "")))
            if allowed_identifiers is not None and identifier not in allowed_identifiers:
                continue
            smiles = str(
                row.get("IsomericSMILES", row.get("CanonicalSMILES", ""))
            ).strip()
            if not identifier or not smiles:
                continue
            try:
                result[identifier] = canonical_smiles(smiles)
            except ValueError:
                continue
    return result


def _standard_descriptor_tokens(value: object, vocabulary: set[str]) -> set[str]:
    if value is None:
        return set()
    text = str(value).strip().lower()
    if not text or text in {"nan", "none"}:
        return set()
    tokens: set[str] = set()
    for separator in (",", "/", "|"):
        text = text.replace(separator, ";")
    for token in text.split(";"):
        normalized = " ".join(token.strip().split())
        if normalized in vocabulary:
            tokens.add(normalized)
    return tokens


def load_odor_descriptor_records(
    pyrfume_root: str | Path,
) -> tuple[tuple[str, ...], list[OdorDescriptorRecord], dict[str, int]]:
    """Merge six Pyrfume monomolecular archives into one 113-label corpus.

    The Leffingwell label columns define the vocabulary.  Text descriptors
    from GoodScents, FlavorNet, AromaDB, IFRA, and FlavorDB are retained only
    on exact normalized vocabulary matches.  Molecules shared across sources
    are canonicalized and their positive labels are unioned.  This avoids
    inventing ontology mappings while still providing a sizeable pretraining
    corpus for the R2 chemical encoder.
    """

    root = Path(pyrfume_root)
    leffingwell_behavior = root / "leffingwell" / "behavior.csv"
    with leffingwell_behavior.open(
        "r", encoding="utf-8", errors="ignore", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        vocabulary = tuple(
            " ".join(str(value).strip().lower().split())
            for value in (reader.fieldnames or [])[1:]
        )
        leffingwell_rows = list(reader)
    if len(vocabulary) != 113:
        raise RuntimeError(
            f"expected the 113-label Leffingwell ontology, found {len(vocabulary)}"
        )
    vocabulary_set = set(vocabulary)
    labels_by_smiles: dict[str, set[str]] = defaultdict(set)
    sources_by_smiles: dict[str, set[str]] = defaultdict(set)
    label_sources_by_smiles: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    source_record_counts: dict[str, int] = defaultdict(int)

    def record_positive_assertions(
        smiles: str, positive: set[str], source: str
    ) -> None:
        """Store assertions without manufacturing negative labels.

        Each upstream table is an incomplete descriptor catalogue.  This
        helper centralizes the positive-only/PU data contract so a future
        source cannot accidentally convert absent descriptor text into a
        supervised negative by writing a dense zero target.
        """

        labels_by_smiles[smiles].update(positive)
        sources_by_smiles[smiles].add(source)
        for label in positive:
            label_sources_by_smiles[smiles][label].add(source)

    leffingwell_map = _read_flexible_molecule_map(
        root / "leffingwell" / "molecules.csv"
    )
    for row in leffingwell_rows:
        smiles = leffingwell_map.get(_identifier(row.get("Stimulus", "")))
        if not smiles:
            continue
        positive = {
            label
            for label in vocabulary
            if float(row.get(label, 0.0) or 0.0) > 0.0
        }
        if positive:
            record_positive_assertions(smiles, positive, "leffingwell")
            source_record_counts["leffingwell"] += 1

    text_archives = (
        ("flavornet", ("Descriptors",)),
        ("aromadb", ("Filtered Descriptors",)),
        ("ifra_2019", ("Descriptor 1", "Descriptor 2", "Descriptor 3")),
        ("flavordb", ("Odor Percepts",)),
    )
    for archive, descriptor_columns in text_archives:
        prepared_rows: list[tuple[str, set[str]]] = []
        with (root / archive / "behavior.csv").open(
            "r", encoding="utf-8", errors="ignore", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                positive: set[str] = set()
                for column in descriptor_columns:
                    positive.update(
                        _standard_descriptor_tokens(row.get(column), vocabulary_set)
                    )
                if positive:
                    prepared_rows.append(
                        (_identifier(row.get("Stimulus", "")), positive)
                    )
        molecule_map = _read_flexible_molecule_map(
            root / archive / "molecules.csv",
            {identifier for identifier, _ in prepared_rows},
        )
        for identifier, positive in prepared_rows:
            smiles = molecule_map.get(identifier)
            if not smiles:
                continue
            record_positive_assertions(smiles, positive, archive)
            source_record_counts[archive] += 1

    goodscents_root = root / "goodscents"
    cas_to_cid = json.loads(
        (goodscents_root / "cas_to_cid.json").read_text(encoding="utf-8")
    )
    prepared_goodscents: list[tuple[str, set[str]]] = []
    with (goodscents_root / "behavior.csv").open(
        "r", encoding="utf-8", errors="ignore", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            cid = _identifier(cas_to_cid.get(str(row.get("Stimulus", "")), ""))
            positive = _standard_descriptor_tokens(
                row.get("Descriptors"), vocabulary_set
            )
            if cid and positive:
                prepared_goodscents.append((cid, positive))
    molecule_map = _read_flexible_molecule_map(
        goodscents_root / "molecules.csv",
        {identifier for identifier, _ in prepared_goodscents},
    )
    for identifier, positive in prepared_goodscents:
        smiles = molecule_map.get(identifier)
        if not smiles:
            continue
        record_positive_assertions(smiles, positive, "goodscents")
        source_record_counts["goodscents"] += 1

    label_index = {label: index for index, label in enumerate(vocabulary)}
    records: list[OdorDescriptorRecord] = []
    for smiles in sorted(labels_by_smiles):
        target = [0.0] * len(vocabulary)
        positive_mask = [0.0] * len(vocabulary)
        label_sources: list[tuple[str, ...]] = [()] * len(vocabulary)
        for label in labels_by_smiles[smiles]:
            index = label_index[label]
            target[index] = 1.0
            positive_mask[index] = 1.0
            label_sources[index] = tuple(
                sorted(label_sources_by_smiles[smiles][label])
            )
        records.append(
            OdorDescriptorRecord(
                smiles=smiles,
                labels=tuple(target),
                positive_observation_mask=tuple(positive_mask),
                label_sources=tuple(label_sources),
                sources=tuple(sorted(sources_by_smiles[smiles])),
            )
        )
    return vocabulary, records, dict(sorted(source_record_counts.items()))


def load_snitz_pairs(data_root: str | Path) -> list[MixturePair]:
    root = Path(data_root) / "snitz_2013"
    cid_to_smiles = _read_cid_smiles(root / "molecules.csv")
    pairs: list[MixturePair] = []
    with (root / "behavior.csv").open(
        "r", encoding="utf-8", errors="ignore", newline=""
    ) as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            mixture_a = tuple(
                cid_to_smiles[cid.strip()]
                for cid in str(row["StimulusA"]).split(",")
                if cid.strip() in cid_to_smiles
            )
            mixture_b = tuple(
                cid_to_smiles[cid.strip()]
                for cid in str(row["StimulusB"]).split(",")
                if cid.strip() in cid_to_smiles
            )
            if not mixture_a or not mixture_b:
                continue
            pairs.append(
                MixturePair(
                    mixture_a,
                    mixture_b,
                    float(row["Similarity"]) / 100.0,
                    f"snitz:{index}",
                )
            )
    return pairs


def load_ravia_pairs(data_root: str | Path) -> list[MixturePair]:
    root = Path(data_root) / "ravia_2020"
    cid_to_smiles = _read_cid_smiles(root / "molecules.csv")
    stimulus_components: dict[str, tuple[str, ...]] = {}
    with (root / "stimuli.csv").open(
        "r", encoding="utf-8", errors="ignore", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            components = tuple(
                cid_to_smiles[cid.strip()]
                for cid in str(row.get("CID", "")).split(";")
                if cid.strip() in cid_to_smiles
            )
            if components:
                stimulus_components[str(row["Stimulus"]).strip()] = components
    pairs: list[MixturePair] = []
    with (root / "behavior_2.csv").open(
        "r", encoding="utf-8", errors="ignore", newline=""
    ) as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            first = str(row["Stimulus 1"]).strip()
            second = str(row["Stimulus 2"]).strip()
            if first == second or first not in stimulus_components or second not in stimulus_components:
                continue
            pairs.append(
                MixturePair(
                    stimulus_components[first],
                    stimulus_components[second],
                    float(row["RatedSimilarity"]) / 100.0,
                    f"ravia:{index}",
                )
            )
    return pairs


class MixturePairDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[MixturePair],
        cache: dict[str, np.ndarray],
        *,
        max_molecules: int | None = None,
    ) -> None:
        self.pairs = list(pairs)
        self.cache = cache
        observed_maximum = max(
            (
                max(len(pair.mixture_a), len(pair.mixture_b))
                for pair in self.pairs
            ),
            default=1,
        )
        # Masked zero-padding is mathematically inert.  Padding only to the
        # largest mixture in the current dataset preserves the R2 equations
        # while avoiding an unnecessary 50x50 interaction tensor for Snitz,
        # whose mixtures contain at most 10 components.
        requested = observed_maximum if max_molecules is None else max_molecules
        self.max_molecules = min(MAX_MOLECULES, max(1, requested))

    def __len__(self) -> int:
        return len(self.pairs)

    def _pad(self, mixture: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        values = np.zeros(
            (self.max_molecules, DESCRIPTOR_DIM), dtype=np.float32
        )
        mask = np.zeros(self.max_molecules, dtype=np.float32)
        for index, smiles in enumerate(mixture[: self.max_molecules]):
            descriptor = self.cache.get(smiles)
            if descriptor is not None:
                values[index] = descriptor
                mask[index] = 1.0
        return values, mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        pair = self.pairs[index]
        mixture_a, mask_a = self._pad(pair.mixture_a)
        mixture_b, mask_b = self._pad(pair.mixture_b)
        return {
            "mixture_a": torch.from_numpy(mixture_a),
            "mask_a": torch.from_numpy(mask_a),
            "mixture_b": torch.from_numpy(mixture_b),
            "mask_b": torch.from_numpy(mask_b),
            "similarity": torch.tensor(pair.similarity, dtype=torch.float32),
        }


def symmetric_augmentation(pairs: Sequence[MixturePair]) -> list[MixturePair]:
    original = list(pairs)
    swapped = [
        MixturePair(
            pair.mixture_b,
            pair.mixture_a,
            pair.similarity,
            f"{pair.record_id}:swap",
        )
        for pair in original
    ]
    return original + swapped


class R2PhysSimCore(nn.Module):
    """Exact parsimonious R2 latent-dynamics architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.chemical_encoder = nn.Sequential(
            nn.Linear(DESCRIPTOR_DIM, LATENT_DIM),
            nn.LayerNorm(LATENT_DIM),
            nn.GELU(),
            nn.Linear(LATENT_DIM, LATENT_DIM),
            nn.LayerNorm(LATENT_DIM),
            nn.GELU(),
        )
        self.mass_mapper = nn.Linear(LATENT_DIM, 1)
        self.charge_mapper = nn.Linear(LATENT_DIM, 1)
        self.sigma_mapper = nn.Linear(LATENT_DIM, 1)
        self.position_mapper = nn.Linear(LATENT_DIM, LATENT_DIM)
        self.velocity_mapper = nn.Linear(LATENT_DIM, LATENT_DIM)
        # Retained for checkpoint/parameter-count compatibility; inactive in
        # the preferred three-interaction core specification.
        self.spin_mapper = nn.Linear(LATENT_DIM, LATENT_DIM)

        # Five active dimensionless core constants, all initialized to 1.0.
        self.log_attraction = nn.Parameter(torch.tensor(0.0))
        self.log_velocity_limit = nn.Parameter(torch.tensor(0.0))
        self.log_mass_decay = nn.Parameter(torch.tensor(0.0))
        self.log_charge_coupling = nn.Parameter(torch.tensor(0.0))
        self.log_lj_well_depth = nn.Parameter(torch.tensor(0.0))
        # Extended-model constants retained but inactive.
        self.log_nonlinear_distance = nn.Parameter(torch.tensor(0.0))
        self.log_spin_coupling = nn.Parameter(torch.tensor(0.0))

        self.fingerprint_projection = nn.Sequential(
            nn.Linear(FINGERPRINT_DIM, PROJECTION_DIM),
            nn.LayerNorm(PROJECTION_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(PROJECTION_DIM, PROJECTION_DIM),
        )
        self.similarity_head = nn.Sequential(
            nn.Linear(SIMILARITY_HEAD_DIM, PROJECTION_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(PROJECTION_DIM, 1),
        )
        actual = sum(parameter.numel() for parameter in self.parameters())
        if actual != EXPECTED_PARAMETER_COUNT:
            raise RuntimeError(
                f"R2 architecture parameter drift: {actual} != {EXPECTED_PARAMETER_COUNT}"
            )

    def _mixture_fingerprint(
        self, descriptors: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        eps = 1e-8
        mask_expanded = mask.unsqueeze(-1)
        atoms = self.chemical_encoder(descriptors)
        masses = F.softplus(self.mass_mapper(atoms)) * mask_expanded
        charges = torch.tanh(self.charge_mapper(atoms)) * mask_expanded
        sigmas = F.softplus(self.sigma_mapper(atoms)) * mask_expanded + 0.1
        positions = self.position_mapper(atoms) * mask_expanded
        velocities = self.velocity_mapper(atoms) * mask_expanded

        attraction = torch.exp(self.log_attraction)
        velocity_limit = torch.exp(self.log_velocity_limit)
        mass_decay = torch.exp(self.log_mass_decay)
        charge_coupling = torch.exp(self.log_charge_coupling)
        lj_well_depth = torch.exp(self.log_lj_well_depth)
        time_step = TIME_HORIZON / N_STEPS
        trajectory = [positions]
        mass_history = [masses]

        for _ in range(N_STEPS):
            _, molecule_count, _ = positions.shape
            difference = positions.unsqueeze(2) - positions.unsqueeze(1)
            raw_distance_squared = difference.square().sum(dim=-1, keepdim=True)
            soft_distance = torch.sqrt(
                raw_distance_squared + SOFT_CORE_DELTA**2
            )
            direction = difference / soft_distance
            mass_i = masses.unsqueeze(2)
            mass_j = masses.unsqueeze(1)
            force = (
                -attraction
                * mass_i
                * mass_j
                / soft_distance.square()
                * direction
            )
            charge_i = charges.unsqueeze(2)
            charge_j = charges.unsqueeze(1)
            force = (
                force
                + charge_coupling
                * charge_i
                * charge_j
                / soft_distance.square()
                * direction
            )
            sigma_i = sigmas.unsqueeze(2)
            sigma_j = sigmas.unsqueeze(1)
            sigma_pair = (sigma_i + sigma_j) / 2.0
            sigma_over_r_6 = (sigma_pair / soft_distance).pow(6)
            force = (
                force
                + 24.0
                * lj_well_depth
                / soft_distance
                * (2.0 * sigma_over_r_6.square() - sigma_over_r_6)
                * direction
            )

            pair_mask = (
                mask.unsqueeze(1).unsqueeze(-1)
                * mask.unsqueeze(2).unsqueeze(-1)
            )
            off_diagonal = (
                1.0
                - torch.eye(
                    molecule_count,
                    device=descriptors.device,
                    dtype=descriptors.dtype,
                )
            ).unsqueeze(0).unsqueeze(-1)
            acceleration = (force * pair_mask * off_diagonal).sum(dim=2) / (
                masses + eps
            )
            speed = velocities.norm(dim=-1, keepdim=True)
            speed_ratio = (speed / (velocity_limit + eps)).clamp(max=0.999)
            gamma = 1.0 / torch.sqrt(1.0 - speed_ratio.square() + eps)
            acceleration = acceleration / (gamma + eps)

            velocities = (velocities + acceleration * time_step) * mask_expanded
            velocities = torch.nan_to_num(
                velocities, nan=0.0, posinf=0.0, neginf=0.0
            )
            positions = (positions + velocities * time_step) * mask_expanded
            positions = torch.nan_to_num(
                positions, nan=0.0, posinf=0.0, neginf=0.0
            )
            masses = (
                masses
                - mass_decay / (masses.square() + eps) * time_step * mask_expanded
            ).clamp(min=eps)
            masses = masses * mask_expanded
            trajectory.append(positions)
            mass_history.append(masses)

        trajectory_tensor = torch.stack(trajectory, dim=1)
        mass_tensor = torch.stack(mass_history, dim=1)
        final_position = trajectory_tensor[:, -1]
        position_variance = trajectory_tensor.var(dim=1).sum(
            dim=-1, keepdim=True
        )
        displacement = trajectory_tensor[:, 1:] - trajectory_tensor[:, :-1]
        speed_history = displacement.norm(dim=-1)
        maximum_speed = speed_history.max(dim=1).values.unsqueeze(-1)
        speed_variance = speed_history.var(dim=1).unsqueeze(-1)
        final_mass = mass_tensor[:, -1]
        mass_ratio = final_mass / (mass_tensor[:, 0] + eps)
        molecule_features = torch.cat(
            [
                final_position,
                position_variance,
                maximum_speed,
                speed_variance,
                final_mass,
                mass_ratio,
                charges.abs(),
            ],
            dim=-1,
        ) * mask_expanded
        return molecule_features.sum(dim=1) / (
            mask.sum(dim=1, keepdim=True) + eps
        )

    def forward(
        self,
        mixture_a: torch.Tensor,
        mask_a: torch.Tensor,
        mixture_b: torch.Tensor,
        mask_b: torch.Tensor,
    ) -> torch.Tensor:
        # Both branches share weights and have the same padded shape. Running
        # them as one batch removes duplicate kernel launches without changing
        # the equations or pair symmetry.
        combined_descriptors = torch.cat([mixture_a, mixture_b], dim=0)
        combined_mask = torch.cat([mask_a, mask_b], dim=0)
        projected = self.fingerprint_projection(
            self._mixture_fingerprint(combined_descriptors, combined_mask)
        )
        projected_a, projected_b = projected.chunk(2, dim=0)
        absolute_difference = (projected_a - projected_b).abs()
        product = projected_a * projected_b
        cosine = F.cosine_similarity(
            projected_a, projected_b, dim=-1, eps=1e-8
        ).unsqueeze(-1)
        features = torch.cat([absolute_difference, product, cosine], dim=-1)
        return torch.sigmoid(self.similarity_head(features).squeeze(-1))

    def learned_constants(self) -> dict[str, float]:
        return {
            "attraction_G": float(torch.exp(self.log_attraction).detach().cpu()),
            "charge_k_e": float(
                torch.exp(self.log_charge_coupling).detach().cpu()
            ),
            "lj_epsilon": float(
                torch.exp(self.log_lj_well_depth).detach().cpu()
            ),
            "velocity_limit": float(
                torch.exp(self.log_velocity_limit).detach().cpu()
            ),
            "mass_decay_kappa": float(
                torch.exp(self.log_mass_decay).detach().cpu()
            ),
        }


def differentiable_spearman_loss(
    prediction: torch.Tensor, target: torch.Tensor, alpha: float = 10.0
) -> torch.Tensor:
    if prediction.shape[0] < 3:
        prediction_centered = prediction - prediction.mean()
        target_centered = target - target.mean()
        correlation = (prediction_centered * target_centered).sum() / (
            torch.sqrt(prediction_centered.square().sum() + 1e-8)
            * torch.sqrt(target_centered.square().sum() + 1e-8)
        )
        return 1.0 - correlation
    prediction_rank = (
        torch.sigmoid(
            alpha * (prediction.unsqueeze(1) - prediction.unsqueeze(0))
        ).sum(dim=1)
        + 1.0
    )
    target_rank = (
        torch.sigmoid(alpha * (target.unsqueeze(1) - target.unsqueeze(0))).sum(
            dim=1
        )
        + 1.0
    )
    prediction_centered = prediction_rank - prediction_rank.mean()
    target_centered = target_rank - target_rank.mean()
    correlation = (prediction_centered * target_centered).sum() / (
        torch.sqrt(prediction_centered.square().sum() + 1e-8)
        * torch.sqrt(target_centered.square().sum() + 1e-8)
    )
    return 1.0 - correlation


def combined_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 0.7 * F.mse_loss(prediction, target) + 0.3 * differentiable_spearman_loss(
        prediction, target
    )


def positive_unlabeled_descriptor_loss(
    logits: torch.Tensor,
    positive_observation_mask: torch.Tensor,
    class_prior: torch.Tensor,
) -> torch.Tensor:
    """Non-negative PU risk for incomplete monomolecular odor catalogues.

    ``positive_observation_mask`` contains only source-backed positive
    assertions.  Entries equal to zero are *unlabeled*, never asserted
    negatives.  The unlabeled population is used as a mixture distribution
    in the standard non-negative PU-risk correction; no individual absent
    descriptor is passed to BCE as target zero.  This is deliberately a
    separate loss from ordinary multi-label BCE to make accidental regression
    to dense-negative training easy to audit.
    """

    if logits.ndim != 2 or positive_observation_mask.shape != logits.shape:
        raise ValueError("logits and positive observation mask must have equal [N, L] shape")
    if class_prior.ndim != 1 or class_prior.shape[0] != logits.shape[1]:
        raise ValueError("class_prior must have one entry per descriptor label")
    if not torch.isfinite(logits).all():
        raise ValueError("PU loss received non-finite logits")

    positive_mask = positive_observation_mask.to(
        dtype=logits.dtype, device=logits.device
    ).clamp(0.0, 1.0)
    unlabeled_mask = 1.0 - positive_mask
    positive_count = positive_mask.sum(dim=0).clamp_min(1.0)
    unlabeled_count = unlabeled_mask.sum(dim=0).clamp_min(1.0)
    prior = class_prior.to(dtype=logits.dtype, device=logits.device).clamp(
        1e-4, 1.0 - 1e-4
    )

    # softplus(-z) is logistic loss for a positive assertion and softplus(z)
    # is the negative-class risk term.  The latter is corrected at the
    # population level, then clipped non-negative, rather than assigning
    # every unmentioned source label a negative target.
    positive_risk = (F.softplus(-logits) * positive_mask).sum(dim=0) / positive_count
    positive_as_negative_risk = (
        F.softplus(logits) * positive_mask
    ).sum(dim=0) / positive_count
    unlabeled_as_negative_risk = (
        F.softplus(logits) * unlabeled_mask
    ).sum(dim=0) / unlabeled_count
    negative_risk = torch.relu(
        unlabeled_as_negative_risk - prior * positive_as_negative_risk
    )
    return (prior * positive_risk + negative_risk).mean()


def audit_external_source_disjointness(
    external_molecules: Iterable[str],
    supervised_populations: dict[str, Iterable[str]],
) -> dict[str, object]:
    """Report molecule/scaffold overlap before calling a result external.

    The report intentionally carries counts and stable hashes rather than raw
    molecule names, so it can be committed with public archive provenance
    without creating a second ungoverned copy of source identifiers.
    """

    external = set(external_molecules)
    external_scaffolds = {bemis_murcko_scaffold(value) for value in external}
    populations: dict[str, dict[str, object]] = {}
    for name, values in sorted(supervised_populations.items()):
        supervised = set(values)
        supervised_scaffolds = {bemis_murcko_scaffold(value) for value in supervised}
        molecule_overlap = sorted(external & supervised)
        scaffold_overlap = sorted(external_scaffolds & supervised_scaffolds)
        populations[name] = {
            "molecule_count": len(supervised),
            "scaffold_count": len(supervised_scaffolds),
            "molecule_overlap_count": len(molecule_overlap),
            "scaffold_overlap_count": len(scaffold_overlap),
            "molecule_overlap_sha256": sha256_json(molecule_overlap),
            "scaffold_overlap_sha256": sha256_json(scaffold_overlap),
            "molecule_disjoint": not molecule_overlap,
            "scaffold_disjoint": not scaffold_overlap,
        }
    passed = all(
        bool(row["molecule_disjoint"]) and bool(row["scaffold_disjoint"])
        for row in populations.values()
    )
    return {
        "audit_contract_version": "external-source-disjointness-1.0",
        "external_molecule_count": len(external),
        "external_scaffold_count": len(external_scaffolds),
        "populations": populations,
        "passed": passed,
    }


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms are deliberately not forced: CUDA reductions
    # used by this architecture can otherwise become unavailable.  Seeds and
    # artifact hashes still make each published run auditable.


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
