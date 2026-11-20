"""Model artifacts in object storage and checksum verification."""

import json

import numpy as np
import pytest
from forecast_fixtures import CATEGORIES, FAST_PARAMS, WAREHOUSES, synthetic_history

from pipeline.forecast.artifacts import (
    ArtifactIntegrityError,
    load_artifacts,
    metadata_key,
    model_key,
    save_artifacts,
)
from pipeline.forecast.model import recursive_forecast, train
from pipeline.storage import LocalStore


@pytest.fixture(scope="module")
def model():
    return train(synthetic_history(days=500), CATEGORIES, WAREHOUSES, 450, FAST_PARAMS, 20)


def test_artifacts_use_the_shared_object_storage_layout(tmp_path, model):
    store = LocalStore(tmp_path)
    key = save_artifacts(store, "20251231-abc", model, {"data_fingerprint": "abc"})
    assert key == "models/demand/20251231-abc/model.ubj"
    assert (tmp_path / "models/demand/20251231-abc/model.ubj").exists()
    metadata = json.loads((tmp_path / metadata_key("20251231-abc")).read_text())
    assert metadata["model_version"] == "20251231-abc"
    assert metadata["data_fingerprint"] == "abc"
    assert metadata["model"]["feature_names"][0] == "lag_7"


def test_loaded_model_predicts_like_the_saved_one(tmp_path, model):
    store = LocalStore(tmp_path)
    save_artifacts(store, "v1", model, {})
    restored, metadata = load_artifacts(store, "v1")
    history = synthetic_history(days=500)
    np.testing.assert_allclose(
        recursive_forecast(restored, history, 470), recursive_forecast(model, history, 470)
    )
    assert metadata["model"]["categories"] == list(CATEGORIES)


def test_tampered_model_is_rejected(tmp_path, model):
    store = LocalStore(tmp_path)
    save_artifacts(store, "v1", model, {})
    path = tmp_path / model_key("v1")
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ArtifactIntegrityError):
        load_artifacts(store, "v1")
