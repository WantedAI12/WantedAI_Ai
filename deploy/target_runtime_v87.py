"""Target-90 release factory. Calculated scores are never rewritten."""

from deploy.system_runtime_v76 import create_local_app


TARGET_POINTS = 90.0


def create_release_app(*, language_backend=None, registry_path=None):
    return create_local_app(language_backend=language_backend, registry_path=registry_path,
                            minimum_profile_target=TARGET_POINTS)


def verify_target_contract(app):
    """Inspect the exact served schemas, without running recipe inference."""
    schema = app.openapi()

    def resolve(value):
        while "$ref" in value:
            ref = value["$ref"]
            if not ref.startswith("#/components/schemas/"):
                raise ValueError("unsupported request schema reference")
            value = schema["components"]["schemas"][ref.rsplit("/", 1)[1]]
        return value

    checked = {}
    paths = {
        "/v1/formulas": (),
        "/v1/formulas/stream": (),
        "/v1/briefs/prepare": ("formula",),
        "/v1/formulas/evaluate": ("formula",),
        "/v2/briefs/prepare": ("request", "formula"),
        "/v2/formulas/evaluate": ("request", "formula"),
        "/v2/formulas/reassess": ("request", "formula"),
        "/v1/applications/body-lotion/design": (),
        "/v1/applications/body-lotion/prepare": (),
        "/v1/applications/body-lotion/optimize": (),
    }
    for path, nested in paths.items():
        value = resolve(schema["paths"][path]["post"]["requestBody"]["content"]["application/json"]["schema"])
        for field in nested:
            value = resolve(value["properties"][field])
        default = value["properties"]["target_similarity"]["default"]
        if default != TARGET_POINTS:
            raise ValueError("release target default mismatch: " + path)
        checked[path] = default
    return {"target_points": TARGET_POINTS, "request_defaults": checked,
            "recipe_inference_performed": False}
