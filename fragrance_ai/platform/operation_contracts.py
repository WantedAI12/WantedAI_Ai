"""Machine-readable routing and units for the existing backend integration."""


def operation_contracts(app, legacy_products, *, unified_available, evidence_configured):
    registered = {(method, route.path) for route in app.routes
                  for method in getattr(route, "methods", ())}

    def operation(path, *, products=(), available=True, prerequisites=(), kind="json"):
        present = ("POST", path) in registered
        return {"method": "POST", "path": path, "registered": present,
                "runtime_available": bool(present and available), "transport": kind,
                "product_codes": list(products), "prerequisites": list(prerequisites),
                "request_schema_source": "/openapi.json#/paths/" + path.replace("/", "~1") + "/post/requestBody"}

    operations = {
        "generate": operation("/v1/formulas", products=sorted(legacy_products)),
        "generate_stream": operation("/v1/formulas/stream", products=sorted(legacy_products), kind="sse"),
        "prepare": operation("/v2/briefs/prepare", products=sorted(legacy_products)),
        "clarify": operation("/v2/briefs/clarify", prerequisites=["current_prepared_result_id"]),
        "evaluate": operation("/v2/formulas/evaluate", products=sorted(legacy_products),
                              prerequisites=["confirmed_review_id", "applicable_registered_evidence"]),
        "reassess": operation("/v2/formulas/reassess", products=sorted(legacy_products),
                              prerequisites=["confirmed_review_id", "exact_fixed_lines", "applicable_registered_evidence"]),
        "compare_saved": operation("/v2/formulas/compare", prerequisites=["backend_owned_rd_evaluation_snapshots"]),
        "revise_saved_intent": operation("/v2/briefs/revise", prerequisites=["backend_owned_rd_evaluation_snapshot"]),
        "assess_evidence": operation("/v2/formulas/assess-evidence"),
        "change_impact": operation("/v2/formulas/change-impact", prerequisites=["registered_previous_evidence_version"]),
        "lotion_design": operation("/v1/applications/body-lotion/design", products=["body_lotion"]),
        "lotion_prepare": operation("/v1/applications/body-lotion/prepare", products=["body_lotion"],
                                    prerequisites=["lotion_simulation_context", "scoped_transport_coefficients"]),
        "lotion_optimize": operation("/v1/applications/body-lotion/optimize", products=["body_lotion"],
                                     prerequisites=["lotion_simulation_context", "scoped_transport_coefficients"]),
        "lotion_simulate": operation("/v1/applications/body-lotion/simulate", products=["body_lotion"],
                                     prerequisites=["fixed_composition", "lotion_simulation_context", "scoped_transport_coefficients"]),
        "conditioned_product_prediction": operation("/v1/applications/unified/predict",
            products=["perfume", "body_lotion", "body_wash"], available=unified_available,
            prerequisites=["fixed_composition", "parameter_context_id", "per_material_per_stage_coefficients",
                           "body_wash_requires_rinse_retention"]),
        "audit_stream": operation("/v1/audit-logs/stream", kind="sse", prerequisites=["backend_history_snapshot"]),
        "audit_report": operation("/v1/reports/audit", prerequisites=["backend_history_snapshot"]),
        "audit_report_stream": operation("/v1/reports/audit/stream", kind="sse", prerequisites=["backend_history_snapshot"]),
    }
    return {"schema_version": "ai-operation-contracts-1", "operations": operations,
        "units": {"concentrate_percent": "concentrate_w/w_percent",
                  "finished_product_percent": "finished_product_w/w_percent",
                  "product_concentration_percent": "fragrance_in_finished_product_w/w_percent",
                  "max_formula_cost_per_kg": "USD/kg_concentrate", "max_ingredient_price_per_kg": "USD/kg_ingredient",
                  "finished_batch_mass_g": "g_finished_product", "maximum_purchase_cost_usd": "USD",
                  "maximum_lead_time_days": "day", "times_minutes": "minute",
                  "target_match_score": "model_points_0_100_not_human_accuracy"},
        "conversion": {"finished_product_percent": "concentrate_percent * product_concentration_percent / 100",
                       "volume_budget_requires": ["product_density_g_ml", "per_volume_ml", "scope",
                           "krw_per_catalog_price_unit", "price_basis_reference", "price_as_of"],
                       "missing_values_are_zero": False},
        "product_routing": {
            "perfume": {"generation": "generate", "review": "prepare", "fixed_reassessment": "reassess"},
            "body_lotion": {"generation": "lotion_design", "review": "lotion_prepare",
                            "optimization": "lotion_optimize", "fixed_prediction": "lotion_simulate",
                            "legacy_formula_request_supported": False},
            "body_wash": {"fragrance_concentrate_generation": "generate",
                          "finished_product_prediction": "conditioned_product_prediction",
                          "rinse_deposition_in_legacy_formula_model": False,
                          "automatic_full_base_formulation": False}},
        "evidence": {"bundle_configured": evidence_configured,
                     "configuration_does_not_imply_request_evidence_passed": True},
        "workflow": {"job_storage_owner": "backend", "approval_owner": "authorized_backend_workflow",
                     "pdf_rendering_owner": "backend", "ai_job_cancel_endpoint": None,
                     "sse_retry": "new_call_from_start", "comparison_storage_owner": "backend",
                     "comparison_requires_active_cache": False,
                     "generation_progress_is_checklist_progress": False},
        "scope": "current_process_routes_and_requirements_not_remote_deployment_verification"}
