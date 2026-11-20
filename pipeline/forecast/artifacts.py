"""Model artifacts in object storage at models/demand/<version>/."""

from __future__ import annotations

import json
import os

from pipeline.forecast.model import DemandModel
from pipeline.storage import make_store, sha256

MODEL_PREFIX = "models/demand"
MODEL_FILE = "model.ubj"
METADATA_FILE = "metadata.json"


class ArtifactIntegrityError(RuntimeError):
    """Raised when a stored model does not match the checksum in its metadata."""


def default_store():
    """Local or S3 store chosen by the STORE variable."""
    return make_store(os.getenv("LOCAL_STORE_ROOT", "data"))


def model_key(version: str) -> str:
    """Object key of a version's model.ubj."""
    return f"{MODEL_PREFIX}/{version}/{MODEL_FILE}"


def metadata_key(version: str) -> str:
    """Object key of a version's metadata.json."""
    return f"{MODEL_PREFIX}/{version}/{METADATA_FILE}"


def save_artifacts(store, version: str, model: DemandModel, metadata: dict) -> str:
    """Write model.ubj and metadata.json with a SHA-256 of the model file."""
    model_bytes = model.to_bytes()
    document = {
        **metadata,
        "model_version": version,
        "model": model.describe(),
        "model_sha256": sha256(model_bytes),
    }
    store.put(model_key(version), model_bytes)
    store.put(metadata_key(version), json.dumps(document, indent=2, default=str).encode())
    return model_key(version)


def load_artifacts(store, version: str) -> tuple[DemandModel, dict]:
    """Load a model and its metadata, verifying the checksum first."""
    metadata = json.loads(store.get(metadata_key(version)))
    model_bytes = store.get(model_key(version))
    if sha256(model_bytes) != metadata["model_sha256"]:
        raise ArtifactIntegrityError(f"model artifact for {version} does not match its checksum")
    return DemandModel.from_bytes(model_bytes, metadata["model"]), metadata
