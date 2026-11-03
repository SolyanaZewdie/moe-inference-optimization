import json
import numpy as np

from pathlib import Path


def save_node(node, directory, counter):
    """
    Save nested model weights.

    Arrays are saved as separate .npy files.
    Dictionaries and lists retain their original structure.
    """

    if isinstance(node, np.ndarray):

        filename = f"weight_{counter[0]}.npy"
        file_path = directory / filename

        np.save(file_path, node)

        counter[0] += 1

        return {
            "type": "array",
            "file": filename
        }

    elif isinstance(node, dict):

        items = {}

        for key, value in node.items():
            items[key] = save_node(value, directory, counter)

        return {
            "type": "dict",
            "items": items
        }

    elif isinstance(node, list):

        items = []

        for value in node:
            items.append(save_node(value, directory, counter))

        return {
            "type": "list",
            "items": items
        }

    else:
        raise TypeError(f"Unsupported weight type: {type(node)}")


def save_weights_mmap(weights, directory):
    """
    Save model weights in separate .npy files
    and create a JSON manifest describing their structure.
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    counter = [0]

    structure = save_node(weights, directory, counter)

    manifest_path = directory / "manifest.json"

    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(structure, file, indent=4)


def load_node(structure, directory):
    """
    Reconstruct the original weight structure.

    Arrays are opened using NumPy memory mapping.
    """

    node_type = structure["type"]

    if node_type == "array":

        file_path = directory / structure["file"]

        return np.load(file_path, mmap_mode="r")

    elif node_type == "dict":

        result = {}

        for key, value in structure["items"].items():
            result[key] = load_node(value, directory)

        return result

    elif node_type == "list":

        result = []

        for value in structure["items"]:
            result.append(load_node(value, directory))

        return result

    else:
        raise ValueError(f"Unknown structure type: {node_type}")


def load_weights_mmap(directory):
    """
    Load model weights using memory mapping.
    """

    directory = Path(directory)
    manifest_path = directory / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError("Weight manifest not found")

    with open(manifest_path, "r", encoding="utf-8") as file:
        structure = json.load(file)

    return load_node(structure, directory)