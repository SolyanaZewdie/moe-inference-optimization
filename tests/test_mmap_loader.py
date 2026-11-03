import numpy as np
import pytest

from pathlib import Path

from src.memory.mmap_loader import (
    save_weights_mmap,
    load_weights_mmap
)


def create_sample_weights():
    rng = np.random.default_rng(42)

    return {
        "embedding": rng.standard_normal((10, 4)).astype(np.float32),

        "layers": [
            {
                "router": rng.standard_normal((4, 3)).astype(np.float32),

                "experts": [
                    {
                        "W_gate": rng.standard_normal((4, 6)).astype(np.float32),
                        "W_up": rng.standard_normal((4, 6)).astype(np.float32),
                        "W_down": rng.standard_normal((6, 4)).astype(np.float32)
                    }
                ]
            }
        ],

        "output": rng.standard_normal((4, 10)).astype(np.float32)
    }


def test_save_and_load_weights(tmp_path):
    original_weights = create_sample_weights()

    save_weights_mmap(original_weights, tmp_path)

    loaded_weights = load_weights_mmap(tmp_path)

    assert np.array_equal(
        original_weights["embedding"],
        loaded_weights["embedding"]
    )

    assert np.array_equal(
        original_weights["output"],
        loaded_weights["output"]
    )


def test_weights_are_memory_mapped(tmp_path):
    original_weights = create_sample_weights()

    save_weights_mmap(original_weights, tmp_path)

    loaded_weights = load_weights_mmap(tmp_path)

    embedding = loaded_weights["embedding"]

    assert isinstance(embedding, np.memmap)


def test_nested_weights_are_preserved(tmp_path):
    original_weights = create_sample_weights()

    save_weights_mmap(original_weights, tmp_path)

    loaded_weights = load_weights_mmap(tmp_path)

    original_expert = original_weights["layers"][0]["experts"][0]
    loaded_expert = loaded_weights["layers"][0]["experts"][0]

    assert np.array_equal(
        original_expert["W_gate"],
        loaded_expert["W_gate"]
    )

    assert np.array_equal(
        original_expert["W_up"],
        loaded_expert["W_up"]
    )

    assert np.array_equal(
        original_expert["W_down"],
        loaded_expert["W_down"]
    )


def test_manifest_is_created(tmp_path):
    weights = create_sample_weights()

    save_weights_mmap(weights, tmp_path)

    manifest = Path(tmp_path) / "manifest.json"

    assert manifest.exists()


def test_missing_manifest():
    with pytest.raises(FileNotFoundError):
        load_weights_mmap("nonexistent_mmap_directory")