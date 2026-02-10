from typing import Optional, Dict, List
from pathlib import Path
import numpy as np
from ..resources import DEFAULT_RESOURCES_BASE, DEFAULT_SBERT_SUBDIR
from ..scenegraph import SceneGraph

_SENTENCE_MODEL = None


def get_default_fcnn_model_path() -> Path:
    """Get the packaged default FCNN model path."""
    return Path(__file__).resolve().parents[1] / "resources" / "models" / "procthor_obj_prob_net.pt"


def _get_embedding_cache_dir() -> Path:
    """Return embedding cache directory, creating it if needed."""
    cache_dir = Path(DEFAULT_RESOURCES_BASE) / DEFAULT_SBERT_SUBDIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def prepare_fcnn_input(
    graph: "SceneGraph",
    containers: List[int],
    objects_to_find: List[str]
) -> Dict[str, np.ndarray]:
    graph = graph.copy()

    objs_idx = []
    for obj in objects_to_find:
        idx = graph.add_node({
            "name": obj,
            "type": [0, 0, 0, 1],
        })
        objs_idx.append(idx)

    node_features = compute_node_features(graph.nodes)

    node_features_dict = {}
    for idx, obj in zip(objs_idx, objects_to_find):
        node_feats_input = []
        for container in containers:
            feats = []
            room_idx = graph.get_parent_node_idx(container)
            if room_idx is None:
                raise ValueError(f"Container {container} does not have a parent room in the scene graph.")
            feats.extend(node_features[room_idx])
            feats.extend(node_features[container])
            feats.extend(node_features[idx])
            node_feats_input.append(feats)
        node_features_dict[obj] = np.array(node_feats_input)

    return node_features_dict


def compute_node_features(nodes: Dict[int, Dict[str, str]]) -> np.ndarray:
    """Get node features for all nodes."""
    features = []
    for node in nodes.values():
        node_feature = np.concatenate((
            get_sentence_embedding(node["name"]), node["type"]
        ))
        features.append(node_feature)
    return np.array(features)


def load_sentence_embedding(target_file_name: str) -> Optional[np.ndarray]:
    file_path = _get_embedding_cache_dir() / target_file_name
    if file_path.exists():
        return np.load(file_path)
    return None


def get_sentence_embedding(sentence: str) -> np.ndarray:
    loaded_embedding = load_sentence_embedding(f"{sentence}.npy")
    if loaded_embedding is not None:
        return loaded_embedding

    from sentence_transformers import SentenceTransformer
    model_dir = Path(DEFAULT_RESOURCES_BASE) / DEFAULT_SBERT_SUBDIR
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        _SENTENCE_MODEL = SentenceTransformer(model_dir.as_posix())
    sentence_embedding = _SENTENCE_MODEL.encode([sentence])[0]
    file_path = _get_embedding_cache_dir() / f"{sentence}.npy"
    np.save(file_path, sentence_embedding)
    return sentence_embedding
