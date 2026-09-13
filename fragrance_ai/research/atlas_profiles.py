"""Quantitative Atlas descriptor prediction, not concentration or human accuracy.

Applicability and use are kept distinct from intensity. No numeric dilution
or solvent is inferred from this source's ordinal high/low condition labels.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .conditional_profiles import molecule_features
from .fine_odor_features import validate_features

SCHEMA = "atlas-quantitative-profiles/v1"
FEATURES = "atlas-graph-native-fine-ordinal/v1"


def read_table(path, key):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames
        if not names or len(names) != len(set(names)) or key not in names:
            raise ValueError("invalid Atlas table columns")
        rows = list(reader)
    if any(None in row or not row[key] for row in rows) or len(
        {r[key] for r in rows}
    ) != len(rows):
        raise ValueError("invalid or duplicate Atlas identity")
    return names, {r[key]: r for r in rows}


def load_atlas(source):
    """Join only explicit CID/graph identities; unresolved stimuli remain rows."""
    from rdkit import Chem

    source = Path(source)
    manifest = json.loads((source / "source_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "atlas-source-snapshot/v1":
        raise ValueError("Atlas source provenance required")
    for name, record in manifest["files"].items():
        if (
            Path(name).name != name
            or hashlib.sha256((source / name).read_bytes()).hexdigest()
            != record["sha256"]
        ):
            raise ValueError("Atlas source drift")
    names, app = read_table(source / "behavior_1.csv", "Stimulus")
    use_names, use = read_table(source / "behavior_2.csv", "Stimulus")
    _, stimuli = read_table(source / "stimuli.csv", "Stimulus")
    _, molecules = read_table(source / "molecules.csv", "CID")
    if names != use_names or set(app) != set(use) or set(app) != set(stimuli):
        raise ValueError("Atlas endpoint/stimulus order mismatch")
    endpoints = names[1:]
    rows = []
    for identifier, row in app.items():
        stimulus = stimuli[identifier]
        raw_cid = stimulus["CID"]
        cid = (
            str(int(float(raw_cid)))
            if raw_cid and float(raw_cid) > 0 and float(raw_cid).is_integer()
            else None
        )
        molecule = molecules.get(cid)
        graph = None
        if molecule:
            parsed = Chem.MolFromSmiles(molecule["IsomericSMILES"])
            if parsed is not None and "." not in molecule["IsomericSMILES"]:
                graph = Chem.MolToSmiles(parsed, isomericSmiles=True)
        level = stimulus["Conc"]
        if level not in ("high", "low"):
            raise ValueError("unknown Atlas ordinal condition")
        applicability = np.array([float(row[name]) for name in endpoints])
        usage = np.array([float(use[identifier][name]) for name in endpoints])
        if (
            not np.isfinite([applicability, usage]).all()
            or np.any(applicability < 0)
            or np.any(usage < 0)
        ):
            raise ValueError("invalid quantitative Atlas profile")
        rows.append(
            {
                "id": identifier,
                "cid": cid,
                "graph": graph,
                "level": level,
                "applicability": applicability,
                "use": usage,
                "identity_status": "explicit_source_graph"
                if graph
                else "unresolved_not_zero_odor",
            }
        )
    return rows, endpoints, manifest


def atlas_features(graphs, levels, native_profiles, fine):
    """Explicit graph features and known categorical descriptions, no outcomes."""
    if len(graphs) != len(levels):
        raise ValueError("Atlas feature lengths differ")
    vocabulary = fine["vocabulary"]
    width = 1063 + len(vocabulary)
    output = []
    for graph, level in zip(graphs, levels):
        if level not in ("high", "low"):
            raise ValueError("ordinal Atlas reference required")
        if graph is None:
            output.append(np.full(width, np.nan))
            continue
        molecule = molecule_features(graph)
        bits = np.zeros(1024)
        bits[molecule["fingerprint_bits"]] = 1.0
        profile = native_profiles.get(graph)
        native = np.zeros(19) if profile is None else np.asarray(profile, float)
        if native.shape != (19,) or not np.isfinite(native).all() or np.any(native < 0):
            raise ValueError("invalid native Atlas feature")
        terms = fine["by_structure"].get(graph, [])
        extra = np.zeros(len(vocabulary))
        extra[terms] = 1.0
        output.append(
            np.r_[
                bits,
                molecule["physical"],
                native,
                float(profile is not None),
                extra,
                float(bool(terms)),
                float(level == "high"),
                float(level == "low"),
            ]
        )
    return np.asarray(output)


def _kernel(a, b, scale, fine_weight):
    def jaccard(left, right):
        dot = left @ right.T
        union = left.sum(1)[:, None] + right.sum(1)[None, :] - dot
        return np.divide(dot, union, out=np.zeros_like(dot), where=union > 0)

    def radial(start, end):
        left, right = (
            a[:, start:end] / scale[start - 1024 : end - 1024],
            b[:, start:end] / scale[start - 1024 : end - 1024],
        )
        distance = np.maximum(
            0.0,
            np.mean(left * left, axis=1)[:, None]
            + np.mean(right * right, axis=1)[None, :]
            - 2 * (left @ right.T) / (end - start),
        )
        return np.exp(-distance)

    structural = (
        0.5 * jaccard(a[:, :1024], b[:, :1024])
        + 0.25 * radial(1024, 1040)
        + 0.25 * radial(1040, 1060)
    )
    fine = jaccard(a[:, 1060:-3], b[:, 1060:-3])
    condition = 0.75 + 0.25 * (a[:, -2:] @ b[:, -2:].T)
    return ((1.0 - fine_weight) * structural + fine_weight * fine) * condition


def _valid_features(x):
    return (
        x.ndim == 2
        and x.shape[1] > 1063
        and np.isfinite(x).all()
        and all(np.all((v == 0) | (v == 1)) for v in (x[:, :1024], x[:, 1059:]))
        and np.all(x[:, -2:].sum(1) == 1)
    )


def _kernel_diagonal(x, fine_weight):
    """Exact self-similarity; missing annotation is not a zero-odor target."""
    structural = 0.5 * (x[:, :1024].sum(axis=1) > 0) + 0.5
    fine = x[:, 1060:-3].sum(axis=1) > 0
    return (1 - fine_weight) * structural + fine_weight * fine


def atlas_kernel(a, b, scale, fine_weight, *, normalize=False):
    value = _kernel(a, b, scale, fine_weight)
    if normalize:
        denominator = np.sqrt(
            _kernel_diagonal(a, fine_weight)[:, None]
            * _kernel_diagonal(b, fine_weight)[None, :]
        )
        value = np.divide(
            value, denominator, out=np.zeros_like(value), where=denominator > 0
        )
    return value


def fit_atlas(x, y, alpha, fine_weight):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if (
        not _valid_features(x)
        or y.ndim != 2
        or len(x) != len(y)
        or len(x) < 2
        or not np.isfinite(y).all()
        or np.any(y < 0)
        or y.shape[1] < 2
        or not np.isfinite([alpha, fine_weight]).all()
        or alpha <= 0
        or not 0 <= fine_weight <= 1
    ):
        raise ValueError("invalid Atlas model training input")
    scale = x[:, 1024:1060].std(0)
    scale[scale < 1e-10] = 1.0
    center = y.mean(0)
    weights = np.linalg.solve(
        _kernel(x, x, scale, fine_weight) + alpha * np.eye(len(x)), y - center
    )
    return {
        "kind": SCHEMA,
        "features": FEATURES,
        "support": x.tolist(),
        "weights": weights.tolist(),
        "scale": scale.tolist(),
        "intercept": center.tolist(),
        "alpha": float(alpha),
        "fine_weight": float(fine_weight),
    }


def fit_atlas_structured(
    x, y, alpha, fine_weight, *, target_transform="sqrt", output_coupling=0.0
):
    """Train V66 on all endpoint values, retaining their original semantics.

    Ordinal high/low labels remain categorical. No concentration or environment
    is invented from these labels. Descriptor correlation is a statistical
    regularizer, never a claim of receptor, chemical or perceptual interaction.
    """
    from .structured_kernel import fit_structured_kernel

    x, y = np.asarray(x, float), np.asarray(y, float)
    if (
        not _valid_features(x)
        or y.ndim != 2
        or len(x) != len(y)
        or len(x) < 2
        or not np.isfinite(y).all()
        or np.any(y < 0)
        or y.shape[1] < 2
        or not np.isfinite(fine_weight)
        or not 0 <= fine_weight <= 1
        or target_transform not in ("identity", "sqrt", "log1p")
    ):
        raise ValueError("invalid structured Atlas training input")
    scale = x[:, 1024:1060].std(axis=0)
    scale[scale < 1e-10] = 1.0
    target = (
        np.sqrt(y)
        if target_transform == "sqrt"
        else np.log1p(y)
        if target_transform == "log1p"
        else y
    )
    solution = fit_structured_kernel(
        atlas_kernel(x, x, scale, fine_weight, normalize=True),
        target,
        alpha=alpha,
        output_coupling=output_coupling,
    )
    return {
        "kind": "atlas-quantitative-profiles/v3",
        "features": FEATURES,
        "support": x.tolist(),
        "weights": solution["coefficients"].tolist(),
        "scale": scale.tolist(),
        "intercept": solution["intercept"].tolist(),
        "alpha": float(alpha),
        "fine_weight": float(fine_weight),
        "kernel_normalization": "unit_diagonal",
        "target_transform": target_transform,
        "training_solver": solution["diagnostics"],
    }


def predict_atlas(model, x):
    if model.get("kind") == "atlas-quantitative-profiles/v4":
        return predict_atlas_scientific(model, x)
    kind = model.get("kind")
    transform = model.get("target_transform", "identity")
    if (
        kind
        not in (
            SCHEMA,
            "atlas-quantitative-profiles/v2",
            "atlas-quantitative-profiles/v3",
        )
        or transform not in ("identity", "sqrt", "log1p")
        or (kind == SCHEMA and transform != "identity")
    ):
        raise ValueError("invalid Atlas target transform contract")
    x, support, weights, scale, intercept = [
        np.asarray(v, float)
        for v in (
            x,
            model["support"],
            model["weights"],
            model["scale"],
            model["intercept"],
        )
    ]
    if (
        model.get("features") != FEATURES
        or not _valid_features(x)
        or not _valid_features(support)
        or x.shape[1] != support.shape[1]
        or scale.shape != (36,)
        or np.any(scale <= 0)
        or intercept.ndim != 1
        or weights.shape != (len(support), len(intercept))
        or not all(np.isfinite(v).all() for v in (weights, scale, intercept))
        or not np.isfinite(model["fine_weight"])
        or not 0 <= model["fine_weight"] <= 1
    ):
        raise ValueError("invalid portable Atlas model")
    normalized = kind == "atlas-quantitative-profiles/v3"
    if normalized and model.get("kernel_normalization") != "unit_diagonal":
        raise ValueError("structured Atlas requires explicit kernel normalization")
    latent = np.maximum(
        0.0,
        atlas_kernel(x, support, scale, model["fine_weight"], normalize=normalized)
        @ weights
        + intercept,
    )
    with np.errstate(over="raise", invalid="raise"):
        prediction = (
            latent**2
            if transform == "sqrt"
            else np.expm1(latent)
            if transform == "log1p"
            else latent
        )
    if not np.isfinite(prediction).all():
        raise ValueError("nonfinite inverse Atlas target transform")
    return prediction


def fit_atlas_scientific(
    x,
    y,
    alpha,
    alignment_regularization,
    *,
    target_transform="sqrt",
    output_coupling=0.9,
):
    """Learn molecular geometry mixing inside the current training fold only."""
    from .scientific_kernel import (
        MolecularGeometry,
        ScientificKernel,
        fit_geometry,
        fit_alignment,
        kernel_prior,
    )
    from .structured_kernel import fit_structured_kernel

    x, y = np.asarray(x, float), np.asarray(y, float)
    if (
        not _valid_features(x)
        or len(x) < 2
        or y.ndim != 2
        or len(y) != len(x)
        or y.shape[1] < 2
        or not np.isfinite(y).all()
        or np.any(y < 0)
        or target_transform not in ("identity", "sqrt", "log1p")
    ):
        raise ValueError("invalid scientific Atlas training input")
    target = (
        np.sqrt(y)
        if target_transform == "sqrt"
        else np.log1p(y)
        if target_transform == "log1p"
        else y
    )
    geometry = fit_geometry(x)
    blocks = MolecularGeometry.from_dict(geometry).blocks(x, x)
    alignment = fit_alignment(
        blocks, target, regularization=alignment_regularization, prior=kernel_prior()
    )
    specification = {"geometry": geometry, "alignment": alignment}
    kernel = ScientificKernel(specification).combine(blocks, x, x)
    solution = fit_structured_kernel(
        kernel, target, alpha=alpha, output_coupling=output_coupling
    )
    return {
        "kind": "atlas-quantitative-profiles/v4",
        "features": FEATURES,
        "support": x.tolist(),
        "weights": solution["coefficients"].tolist(),
        "intercept": solution["intercept"].tolist(),
        "alpha": float(alpha),
        "target_transform": target_transform,
        "kernel_normalization": "unit_diagonal",
        "kernel_specification": specification,
        "training_solver": solution["diagnostics"],
    }


def inverse_atlas_target(latent, transform):
    """One finite inverse transform shared by portable and compiled inference."""
    if transform not in ("identity", "sqrt", "log1p"):
        raise ValueError("invalid Atlas inverse target transform")
    latent = np.maximum(0.0, latent)
    with np.errstate(over="raise", invalid="raise"):
        prediction = (
            latent**2
            if transform == "sqrt"
            else np.expm1(latent)
            if transform == "log1p"
            else latent
        )
    if not np.isfinite(prediction).all():
        raise ValueError("nonfinite inverse Atlas target transform")
    return prediction


def predict_atlas_scientific(model, x):
    from .scientific_kernel import ScientificKernel

    x, support, weights, intercept = (
        np.asarray(v, float)
        for v in (x, model["support"], model["weights"], model["intercept"])
    )
    if (
        model.get("kind") != "atlas-quantitative-profiles/v4"
        or model.get("features") != FEATURES
        or model.get("kernel_normalization") != "unit_diagonal"
        or not _valid_features(x)
        or not _valid_features(support)
        or x.shape[1] != support.shape[1]
        or intercept.ndim != 1
        or len(intercept) < 2
        or weights.shape != (len(support), len(intercept))
        or not np.isfinite(weights).all()
        or not np.isfinite(intercept).all()
        or not np.isfinite(model["alpha"])
        or model["alpha"] <= 0
    ):
        raise ValueError("invalid scientific Atlas checkpoint")
    kernel = ScientificKernel(model["kernel_specification"])(x, support)
    return inverse_atlas_target(kernel @ weights + intercept, model["target_transform"])


class AtlasProfilePredictor:
    """Local, source-bound quantitative descriptor model for known graphs."""

    def __init__(self, path, *, sha256, experimental=False):
        if not experimental:
            raise ValueError("Atlas model requires research opt-in")
        self.path = Path(path).resolve(strict=True)
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha256:
            raise ValueError("Atlas model hash mismatch")
        artifact = json.loads(raw)
        if (
            artifact.get("schema") != "atlas-profile-candidate/v1"
            or artifact.get("runtime_promotion_allowed") is not False
            or artifact.get("data_redistribution_authorized") is not False
        ):
            raise ValueError("Atlas model provenance mismatch")
        self.sha256, self.endpoints = sha256, tuple(artifact["endpoints"])
        self.artifact_version = artifact.get(
            "quantitative_target_model_version", SCHEMA
        )
        self.parent_checkpoint_sha256 = artifact.get("parent_checkpoint_sha256")
        self.development_nonregression_passed = (
            artifact.get("development_nonregression_passed") is True
        )
        self.source_digest = hashlib.sha256(
            json.dumps(artifact.get("source"), sort_keys=True).encode()
        ).hexdigest()
        stat = self.path.stat()
        self._file_metadata = (stat.st_size, stat.st_mtime_ns)
        self.fine, self.native = artifact["fine_features"], artifact["native_profiles"]
        validate_features(self.fine)
        models = artifact["models"]
        if set(models) != {"applicability", "use"} or len(set(self.endpoints)) != len(
            self.endpoints
        ):
            raise ValueError("distinct Atlas measurement models required")
        for model in models.values():
            if len(model["intercept"]) != len(self.endpoints):
                raise ValueError("Atlas endpoints mismatch")
        from .atlas_inference import CompiledAtlasHeads

        self._compiled = CompiledAtlasHeads(models)
        self.models = self._compiled.models

    def assert_current(self):
        stat = self.path.stat()
        if (stat.st_size, stat.st_mtime_ns) != self._file_metadata:
            raise ValueError("Atlas checkpoint changed; reload the local runtime")

    def _query_features(self, graphs, *, reference_level):
        self.assert_current()
        from rdkit import Chem

        canonical = []
        for graph in graphs:
            parsed = (
                Chem.MolFromSmiles(graph)
                if isinstance(graph, str) and "." not in graph
                else None
            )
            if parsed is None:
                raise ValueError("one explicit molecular graph required")
            canonical.append(Chem.MolToSmiles(parsed, isomericSmiles=True))
        x = atlas_features(
            canonical, [reference_level] * len(canonical), self.native, self.fine
        )
        return x

    def predict(self, graphs, *, reference_level="high"):
        return self._compiled.predict(
            self._query_features(graphs, reference_level=reference_level)
        )

    def predict_with_diagnostics(self, graphs, *, reference_level="high"):
        """Profile predictions plus train-domain facts, not calibrated confidence."""
        return self._compiled.predict_with_diagnostics(
            self._query_features(graphs, reference_level=reference_level)
        )
