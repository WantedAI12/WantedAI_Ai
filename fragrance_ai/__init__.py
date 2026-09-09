"""Focused, safety-constrained perfumery AI core.

The natural-language recommender stays lightweight.  Legacy Torch/DEAP
components are imported only when explicitly requested.
"""

from .recommender import (
    CalibrationArtifact,
    ManufacturingPlan,
    NaturalLanguagePerfumeryAI,
    PerceptionGuidance,
    StockAliquot,
    StockMass,
    StockControlAliquot,
    StockMixturePredictor,
    OdorProfileStore,
    QualityEvidenceStore,
    ActivatedIngredientPromotion,
    PromotionActivationBundle,
    EvidenceTrustRoot,
    VerifiedArtifact,
    AppendOnlyAuditLog,
    ManufacturingProfileRegistry,
    PackagingProfile,
    ProductBaseProfile,
    TechnicalEvidence,
    ReferenceComponent,
    ReferenceEvidence,
    ReferenceTarget,
    ReferenceTargetStore,
    ResolvedReferenceTarget,
    RecipeConstraints,
    RecipeResult,
    SensoryEvaluationStore,
    SimulatedSensoryEngine,
    SimulatedSensoryResult,
    ScientificPropertyStore,
    TemporalMixtureSimulator,
    ScientificTwinResult,
    ConcentrationAwarePhysSim,
    PhysSimResult,
    CommercialReleaseStore,
    RegulatorySignoff,
    ReleaseEvidenceAssessment,
    VerifiedRegulatorySignoff,
    ReleaseSpec,
    NonHumanDataHub,
    EPACompToxStore,
    SupplierRegistry,
)

__version__ = "1.4.0"

__all__ = [
    "StockAliquot",
    "StockMass",
    "StockControlAliquot",
    "StockMixturePredictor",
    "UnifiedTransportModel",
    "UnifiedProductPredictor",
    "NaturalLanguagePerfumeryAI",
    "PerceptionGuidance",
    "RecipeConstraints",
    "RecipeResult",
    "ManufacturingPlan",
    "SupplierRegistry",
    "SensoryEvaluationStore",
    "QualityEvidenceStore",
    "ActivatedIngredientPromotion",
    "PromotionActivationBundle",
    "EvidenceTrustRoot",
    "VerifiedArtifact",
    "AppendOnlyAuditLog",
    "ManufacturingProfileRegistry",
    "PackagingProfile",
    "ProductBaseProfile",
    "TechnicalEvidence",
    "ReferenceComponent",
    "ReferenceEvidence",
    "ReferenceTarget",
    "ReferenceTargetStore",
    "ResolvedReferenceTarget",
    "CalibrationArtifact",
    "OdorProfileStore",
    "SimulatedSensoryEngine",
    "SimulatedSensoryResult",
    "ScientificPropertyStore",
    "TemporalMixtureSimulator",
    "ScientificTwinResult",
    "ConcentrationAwarePhysSim",
    "PhysSimResult",
    "CommercialReleaseStore",
    "RegulatorySignoff",
    "ReleaseEvidenceAssessment",
    "VerifiedRegulatorySignoff",
    "ReleaseSpec",
    "NonHumanDataHub",
    "EPACompToxStore",
    "UnifiedAIConfig",
    "UnifiedFragranceAI",
    "create_unified_ai_system",
]


def __getattr__(name: str):
    if name == 'UnifiedTransportModel':
        from .recommender.unified_transport import UnifiedTransportModel
        return UnifiedTransportModel
    if name == 'UnifiedProductPredictor':
        from .recommender.unified_product import UnifiedProductPredictor
        return UnifiedProductPredictor
    if name in {"UnifiedAIConfig", "UnifiedFragranceAI", "create_unified_ai_system"}:
        from .ai.unified_ai_system import (
            UnifiedAIConfig,
            UnifiedFragranceAI,
            create_unified_ai_system,
        )

        return {
            "UnifiedAIConfig": UnifiedAIConfig,
            "UnifiedFragranceAI": UnifiedFragranceAI,
            "create_unified_ai_system": create_unified_ai_system,
        }[name]
    raise AttributeError(name)
