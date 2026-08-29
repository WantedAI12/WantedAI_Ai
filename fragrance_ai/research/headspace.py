"""Read-only physical headspace calculations over the research data hub.

The implementation deliberately separates three evidence classes:

* measured EPA OPERA endpoint rows;
* a labelled Trouton-rule fallback from a measured normal boiling point; and
* missing values, which remain missing instead of receiving an invented value.

Raoult and Henry calculations describe ideal equilibrium reference systems.
They are not dynamic evaporation, supplied-lot headspace GC-MS, or human
olfactory validation.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


ATMOSPHERIC_PRESSURE_PA = 101_325.0
GAS_CONSTANT_PA_M3_MOL_K = 8.314_462_618
MMHG_TO_PA = 133.322_368_421_052_63
REFERENCE_TEMPERATURE_K = 298.15
EXPECTED_HUB_SCHEMA = "headspace-sensory-hub/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    lower_ok = result >= 0.0 if allow_zero else result > 0.0
    if isinstance(value, bool) or not math.isfinite(result) or not lower_ok:
        comparator = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {comparator}")
    return result


@dataclass(frozen=True)
class PropertyEvidence:
    endpoint: str
    value: float
    unit: str
    observation_count: int
    robust_log_sigma: float
    source_paths: tuple[str, ...]
    splits: tuple[str, ...]
    evidence_class: str = "measured_endpoint_median"


@dataclass(frozen=True)
class VaporPressureEvidence:
    pressure_pa: float
    log_sigma: float
    evidence_class: str
    source_endpoint: str
    observation_count: int


@dataclass(frozen=True)
class HeadspaceInput:
    cid: int
    liquid_mole_fraction: float
    activity_coefficient: float = 1.0


@dataclass(frozen=True)
class AqueousHeadspaceInput:
    cid: int
    liquid_concentration_mol_m3: float


@dataclass(frozen=True)
class HeadspaceComponent:
    cid: int
    liquid_input: float
    partial_pressure_pa: float | None
    gas_ppmv: float | None
    gas_mol_m3: float | None
    headspace_signal: float | None
    headspace_share: float | None
    odor_threshold_ppmv: float | None
    odor_activity_value: float | None
    partition_evidence_class: str


@dataclass(frozen=True)
class HeadspaceResult:
    status: str
    method: str
    temperature_k: float
    total_pressure_pa: float
    response_exponent: float
    resolved_liquid_input: float
    vapor_pressure_coverage_percent: float
    odor_threshold_coverage_percent: float
    components: tuple[HeadspaceComponent, ...]
    flags: tuple[str, ...]
    human_olfactory_accuracy_claimed: bool = False


class HeadspaceSensoryHub:
    """Verified read-only access to a built headspace/sensory SQLite hub."""

    REQUIRED_TABLES = frozenset(
        {
            "hub_metadata",
            "source_files",
            "molecules",
            "physchem_observations",
            "molecule_physchem_links",
            "stimuli",
            "stimulus_components",
            "stimulus_dilutions",
            "sensory_observations",
        }
    )

    def __init__(
        self,
        database: str | Path,
        *,
        report: str | Path | None = None,
        verify_hash: bool = True,
    ) -> None:
        self.path = Path(database).expanduser().resolve(strict=True)
        self.report_path = (
            Path(report).expanduser().resolve(strict=True) if report is not None else None
        )
        if self.report_path is not None:
            document = json.loads(self.report_path.read_text(encoding="utf-8"))
            if document.get("schema") != EXPECTED_HUB_SCHEMA:
                raise ValueError("unsupported headspace sensory hub report schema")
            expected = document.get("database", {}).get("sha256")
            if verify_hash and (not isinstance(expected, str) or _sha256(self.path) != expected):
                raise RuntimeError("headspace sensory hub database hash mismatch")
        uri = self.path.as_uri() + "?mode=ro"
        self.connection = sqlite3.connect(uri, uri=True)
        tables = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if not self.REQUIRED_TABLES.issubset(tables):
            self.connection.close()
            raise ValueError("headspace sensory hub schema is incomplete")
        schema_row = self.connection.execute(
            "SELECT value FROM hub_metadata WHERE key='schema'"
        ).fetchone()
        if schema_row is None or schema_row[0] != EXPECTED_HUB_SCHEMA:
            self.connection.close()
            raise ValueError("unsupported headspace sensory hub database schema")
        if self.connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            self.connection.close()
            raise RuntimeError("headspace sensory hub integrity check failed")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "HeadspaceSensoryHub":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def molecule(self, cid: int) -> Mapping[str, object] | None:
        row = self.connection.execute(
            "SELECT cid, inchi_key, canonical_smiles, molecular_weight, "
            "preferred_name, casrn FROM molecules WHERE cid=?",
            (int(cid),),
        ).fetchone()
        if row is None:
            return None
        names = (
            "cid",
            "inchi_key",
            "canonical_smiles",
            "molecular_weight",
            "preferred_name",
            "casrn",
        )
        return dict(zip(names, row, strict=True))

    def property(self, cid: int, endpoint: str) -> PropertyEvidence | None:
        rows = self.connection.execute(
            "SELECT p.value, p.unit, p.source_path, p.split "
            "FROM molecule_physchem_links l "
            "JOIN physchem_observations p ON p.observation_id=l.observation_id "
            "WHERE l.cid=? AND p.endpoint=? ORDER BY p.observation_id",
            (int(cid), str(endpoint)),
        ).fetchall()
        if not rows:
            return None
        units = {str(row[1]) for row in rows}
        if len(units) != 1:
            raise RuntimeError(f"mixed units for {endpoint} and CID {cid}")
        values = np.asarray([float(row[0]) for row in rows], dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"non-finite physical data for {endpoint} and CID {cid}")
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        # The endpoints stored in logarithmic units already express spread in
        # log10 units. Other endpoints use a conservative relative-log proxy.
        if endpoint.startswith("log10_"):
            log_sigma = max(0.0, math.log(10.0) * 1.4826 * mad)
        else:
            relative_mad = 1.4826 * mad / max(abs(median), 1e-12)
            log_sigma = math.log1p(max(0.0, relative_mad))
        return PropertyEvidence(
            endpoint=endpoint,
            value=median,
            unit=next(iter(units)),
            observation_count=len(rows),
            robust_log_sigma=log_sigma,
            source_paths=tuple(sorted({str(row[2]) for row in rows})),
            splits=tuple(sorted({str(row[3]) for row in rows})),
        )

    @staticmethod
    def _boiling_point_to_vapor_pressure(boiling_point_c: float) -> float:
        boiling_k = float(boiling_point_c) + 273.15
        if boiling_k <= 0.0:
            raise ValueError("boiling point is below absolute zero")
        if boiling_k <= REFERENCE_TEMPERATURE_K:
            return ATMOSPHERIC_PRESSURE_PA
        enthalpy_j_mol = 88.0 * boiling_k
        log_pressure = math.log(ATMOSPHERIC_PRESSURE_PA) - (
            enthalpy_j_mol / GAS_CONSTANT_PA_M3_MOL_K
        ) * (1.0 / REFERENCE_TEMPERATURE_K - 1.0 / boiling_k)
        return min(ATMOSPHERIC_PRESSURE_PA, max(1e-12, math.exp(log_pressure)))

    def vapor_pressure(
        self, cid: int, *, allow_boiling_point_fallback: bool = True
    ) -> VaporPressureEvidence | None:
        measured = self.property(cid, "log10_vapor_pressure_mmhg")
        if measured is not None:
            return VaporPressureEvidence(
                pressure_pa=10.0**measured.value * MMHG_TO_PA,
                log_sigma=max(0.20, measured.robust_log_sigma),
                evidence_class="measured_opera_vapor_pressure_median",
                source_endpoint=measured.endpoint,
                observation_count=measured.observation_count,
            )
        if not allow_boiling_point_fallback:
            return None
        boiling = self.property(cid, "boiling_point_c")
        if boiling is None:
            return None
        return VaporPressureEvidence(
            pressure_pa=self._boiling_point_to_vapor_pressure(boiling.value),
            log_sigma=max(0.55, boiling.robust_log_sigma),
            evidence_class="measured_boiling_point_trouton_fallback",
            source_endpoint=boiling.endpoint,
            observation_count=boiling.observation_count,
        )

    def henry_constant_pa_m3_mol(self, cid: int) -> PropertyEvidence | None:
        measured = self.property(cid, "log10_henry_atm_m3_mol")
        if measured is None:
            return None
        return PropertyEvidence(
            endpoint="henry_pa_m3_mol",
            value=10.0**measured.value * ATMOSPHERIC_PRESSURE_PA,
            unit="Pa*m^3/mol",
            observation_count=measured.observation_count,
            robust_log_sigma=max(0.20, measured.robust_log_sigma),
            source_paths=measured.source_paths,
            splits=measured.splits,
            evidence_class="measured_opera_henry_median_converted_to_si",
        )

    def odor_threshold_ppmv(self, cid: int) -> float | None:
        rows = self.connection.execute(
            "SELECT so.value, so.unit FROM stimulus_components sc "
            "JOIN sensory_observations so "
            "ON so.dataset=sc.dataset AND so.stimulus_a=sc.stimulus_id "
            "WHERE sc.dataset='abraham_2012' AND sc.cid=? "
            "AND so.endpoint='log10_inverse_odor_detection_threshold'",
            (int(cid),),
        ).fetchall()
        if not rows:
            return None
        if {str(row[1]) for row in rows} != {"log10_inverse_ppmv"}:
            raise RuntimeError("Abraham odor-threshold unit contract changed")
        log_inverse = float(np.median([float(row[0]) for row in rows]))
        return 10.0 ** (-log_inverse)

    @staticmethod
    def _validate_common(
        temperature_k: float, total_pressure_pa: float, response_exponent: float
    ) -> tuple[float, float, float]:
        return (
            _finite_positive(temperature_k, "temperature_k"),
            _finite_positive(total_pressure_pa, "total_pressure_pa"),
            _finite_positive(response_exponent, "response_exponent"),
        )

    @staticmethod
    def _finish(
        raw: list[dict[str, object]],
        *,
        method: str,
        temperature_k: float,
        total_pressure_pa: float,
        response_exponent: float,
        total_liquid_input: float,
        flags: list[str],
    ) -> HeadspaceResult:
        resolved_signal = sum(
            float(row["headspace_signal"])
            for row in raw
            if row["headspace_signal"] is not None
        )
        resolved_input = sum(
            float(row["liquid_input"])
            for row in raw
            if row["partial_pressure_pa"] is not None
        )
        threshold_input = sum(
            float(row["liquid_input"])
            for row in raw
            if row["odor_threshold_ppmv"] is not None
            and row["partial_pressure_pa"] is not None
        )
        components = []
        for row in raw:
            signal = row["headspace_signal"]
            share = (
                float(signal) / resolved_signal
                if signal is not None and resolved_signal > 0.0
                else None
            )
            components.append(HeadspaceComponent(**row, headspace_share=share))
        coverage = 100.0 * resolved_input / max(total_liquid_input, 1e-12)
        threshold_coverage = 100.0 * threshold_input / max(total_liquid_input, 1e-12)
        if resolved_input <= 0.0:
            status = "insufficient_physical_data"
        elif resolved_input + 1e-12 < total_liquid_input:
            status = "partial_equilibrium_reference"
            flags.append("one_or_more_components_missing_vapor_partition_data")
        else:
            status = "complete_equilibrium_reference"
        if threshold_coverage + 1e-12 < coverage:
            flags.append("odor_activity_is_partial_and_not_used_for_headspace_share")
        return HeadspaceResult(
            status=status,
            method=method,
            temperature_k=temperature_k,
            total_pressure_pa=total_pressure_pa,
            response_exponent=response_exponent,
            resolved_liquid_input=resolved_input,
            vapor_pressure_coverage_percent=min(100.0, coverage),
            odor_threshold_coverage_percent=min(100.0, threshold_coverage),
            components=tuple(components),
            flags=tuple(dict.fromkeys(flags)),
        )

    def raoult_headspace(
        self,
        inputs: Iterable[HeadspaceInput],
        *,
        temperature_k: float = REFERENCE_TEMPERATURE_K,
        total_pressure_pa: float = ATMOSPHERIC_PRESSURE_PA,
        response_exponent: float = 1.0,
        allow_boiling_point_fallback: bool = True,
    ) -> HeadspaceResult:
        temperature_k, total_pressure_pa, response_exponent = self._validate_common(
            temperature_k, total_pressure_pa, response_exponent
        )
        values = tuple(inputs)
        if not values:
            raise ValueError("at least one headspace input is required")
        if len({int(item.cid) for item in values}) != len(values):
            raise ValueError("headspace inputs contain duplicate CIDs")
        fractions = [
            _finite_positive(
                item.liquid_mole_fraction,
                "liquid_mole_fraction",
                allow_zero=True,
            )
            for item in values
        ]
        if sum(fractions) > 1.0 + 1e-9:
            raise ValueError("liquid mole fractions cannot sum above one")
        raw: list[dict[str, object]] = []
        for item, fraction in zip(values, fractions, strict=True):
            activity = _finite_positive(item.activity_coefficient, "activity_coefficient")
            vapor = self.vapor_pressure(
                item.cid,
                allow_boiling_point_fallback=allow_boiling_point_fallback,
            )
            threshold = self.odor_threshold_ppmv(item.cid)
            if vapor is None:
                partial = gas_ppmv = gas_mol_m3 = signal = oav = None
                evidence = "missing"
            else:
                partial = fraction * activity * vapor.pressure_pa
                gas_ppmv = partial / total_pressure_pa * 1_000_000.0
                gas_mol_m3 = partial / (
                    GAS_CONSTANT_PA_M3_MOL_K * temperature_k
                )
                signal = max(gas_ppmv, 0.0) ** response_exponent
                oav = gas_ppmv / threshold if threshold is not None else None
                evidence = vapor.evidence_class
            raw.append(
                {
                    "cid": int(item.cid),
                    "liquid_input": fraction,
                    "partial_pressure_pa": partial,
                    "gas_ppmv": gas_ppmv,
                    "gas_mol_m3": gas_mol_m3,
                    "headspace_signal": signal,
                    "odor_threshold_ppmv": threshold,
                    "odor_activity_value": oav,
                    "partition_evidence_class": evidence,
                }
            )
        total_partial_pressure = sum(
            float(row["partial_pressure_pa"])
            for row in raw
            if row["partial_pressure_pa"] is not None
        )
        if total_partial_pressure > total_pressure_pa * (1.0 + 1e-9):
            raise ValueError(
                "ideal Raoult partial pressures exceed total pressure; "
                "the requested activity/fraction state is outside the model domain"
            )
        return self._finish(
            raw,
            method="ideal_raoult_equilibrium",
            temperature_k=temperature_k,
            total_pressure_pa=total_pressure_pa,
            response_exponent=response_exponent,
            total_liquid_input=sum(fractions),
            flags=[
                "ideal_liquid_activity_assumption",
                "static_equilibrium_not_dynamic_evaporation",
                "not_supplied_lot_headspace_gc_ms",
                "not_human_olfactory_accuracy",
            ],
        )

    def aqueous_headspace(
        self,
        inputs: Iterable[AqueousHeadspaceInput],
        *,
        temperature_k: float = REFERENCE_TEMPERATURE_K,
        total_pressure_pa: float = ATMOSPHERIC_PRESSURE_PA,
        response_exponent: float = 1.0,
    ) -> HeadspaceResult:
        temperature_k, total_pressure_pa, response_exponent = self._validate_common(
            temperature_k, total_pressure_pa, response_exponent
        )
        values = tuple(inputs)
        if not values:
            raise ValueError("at least one aqueous headspace input is required")
        if len({int(item.cid) for item in values}) != len(values):
            raise ValueError("aqueous headspace inputs contain duplicate CIDs")
        concentrations = [
            _finite_positive(
                item.liquid_concentration_mol_m3,
                "liquid_concentration_mol_m3",
                allow_zero=True,
            )
            for item in values
        ]
        raw: list[dict[str, object]] = []
        for item, concentration in zip(values, concentrations, strict=True):
            henry = self.henry_constant_pa_m3_mol(item.cid)
            threshold = self.odor_threshold_ppmv(item.cid)
            if henry is None:
                partial = gas_ppmv = gas_mol_m3 = signal = oav = None
                evidence = "missing"
            else:
                partial = henry.value * concentration
                if partial > total_pressure_pa:
                    raise ValueError(
                        "ideal Henry calculation exceeds total pressure; dilute-law "
                        "assumption is outside its domain"
                    )
                gas_ppmv = partial / total_pressure_pa * 1_000_000.0
                gas_mol_m3 = partial / (
                    GAS_CONSTANT_PA_M3_MOL_K * temperature_k
                )
                signal = max(gas_ppmv, 0.0) ** response_exponent
                oav = gas_ppmv / threshold if threshold is not None else None
                evidence = henry.evidence_class
            raw.append(
                {
                    "cid": int(item.cid),
                    "liquid_input": concentration,
                    "partial_pressure_pa": partial,
                    "gas_ppmv": gas_ppmv,
                    "gas_mol_m3": gas_mol_m3,
                    "headspace_signal": signal,
                    "odor_threshold_ppmv": threshold,
                    "odor_activity_value": oav,
                    "partition_evidence_class": evidence,
                }
            )
        total_partial_pressure = sum(
            float(row["partial_pressure_pa"])
            for row in raw
            if row["partial_pressure_pa"] is not None
        )
        if total_partial_pressure > total_pressure_pa * (1.0 + 1e-9):
            raise ValueError(
                "ideal Henry partial pressures exceed total pressure; dilute-law "
                "assumption is outside its domain"
            )
        return self._finish(
            raw,
            method="dilute_aqueous_henry_equilibrium",
            temperature_k=temperature_k,
            total_pressure_pa=total_pressure_pa,
            response_exponent=response_exponent,
            total_liquid_input=sum(concentrations),
            flags=[
                "dilute_aqueous_solution_assumption",
                "static_equilibrium_not_dynamic_evaporation",
                "not_supplied_lot_headspace_gc_ms",
                "not_human_olfactory_accuracy",
            ],
        )


def load_calibrated_response_exponent(path: str | Path) -> float:
    """Load the held-out Keller exponent from a research-only artifact."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema") != "concentration-headspace-calibration/v1":
        raise ValueError("unsupported concentration-headspace calibration schema")
    if not document.get("gates", {}).get("molecule_holdout_transfer", {}).get("passed"):
        raise RuntimeError("concentration-headspace calibration gate is closed")
    if float(document.get("runtime_primary_score_weight", -1.0)) != 0.0:
        raise RuntimeError("research calibration unexpectedly has production weight")
    return _finite_positive(
        document["calibration"]["response_exponent"], "response_exponent"
    )


def predict_portable_log_vapor_pressure(
    runtime: Mapping[str, object],
    descriptors: np.ndarray,
    descriptor_names: Iterable[str],
) -> np.ndarray:
    """Execute a validated pickle-free OPERA logVP ridge artifact."""
    if runtime.get("schema") != "opera-vp-portable-ridge/v1":
        raise ValueError("unsupported OPERA vapor-pressure runtime schema")
    names = list(descriptor_names)
    expected_names = runtime.get("descriptor_names")
    if not isinstance(expected_names, list) or expected_names != names:
        raise ValueError("OPERA vapor-pressure descriptor names do not match")
    if runtime.get("descriptor_contract_sha256") != _canonical_json_sha256(names):
        raise ValueError("OPERA vapor-pressure descriptor contract hash mismatch")
    width = len(names)

    def vector(key: str) -> np.ndarray:
        raw = runtime.get(key)
        if (
            not isinstance(raw, list)
            or len(raw) != width
            or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in raw
            )
        ):
            raise ValueError(f"invalid OPERA vapor-pressure runtime vector: {key}")
        value = np.asarray(raw, dtype=float)
        if value.shape != (width,) or not np.isfinite(value).all():
            raise ValueError(f"invalid OPERA vapor-pressure runtime vector: {key}")
        return value

    median = vector("feature_median")
    mean = vector("feature_mean")
    scale = vector("feature_scale")
    coefficients = vector("coefficients")
    if np.any(scale <= 0.0):
        raise ValueError("OPERA vapor-pressure feature scale must be positive")
    try:
        raw_intercept = runtime["intercept"]
        if isinstance(raw_intercept, bool) or not isinstance(
            raw_intercept, (int, float)
        ):
            raise TypeError
        intercept = float(raw_intercept)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid OPERA vapor-pressure intercept") from error
    if not math.isfinite(intercept):
        raise ValueError("invalid OPERA vapor-pressure intercept")
    matrix = np.asarray(descriptors, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != width:
        raise ValueError("OPERA vapor-pressure descriptor matrix width does not match")
    if np.isinf(matrix).any():
        raise ValueError("OPERA vapor-pressure descriptors contain infinity")
    imputed = np.where(np.isnan(matrix), median, matrix)
    prediction = ((imputed - mean) / scale) @ coefficients + intercept
    clip = runtime.get("prediction_clip")
    if (
        not isinstance(clip, list)
        or len(clip) != 2
        or not all(
            not isinstance(value, bool) and isinstance(value, (int, float))
            for value in clip
        )
        or not math.isfinite(float(clip[0]))
        or not math.isfinite(float(clip[1]))
        or float(clip[0]) >= float(clip[1])
    ):
        raise ValueError("invalid OPERA vapor-pressure prediction clip")
    return np.clip(prediction, float(clip[0]), float(clip[1]))
