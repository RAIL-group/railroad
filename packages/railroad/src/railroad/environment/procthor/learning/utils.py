
from typing import Optional, Dict, List
import os
import numpy as np
from ..resources import DEFAULT_RESOURCES_BASE, DEFAULT_SBERT_SUBDIR
from ..scenegraph import SceneGraph


def prepare_fcnn_input(
    graph: SceneGraph,
    containers: List[int],
    objects_to_find: List[str]
) -> Dict[str, np.ndarray]:
    graph = graph.copy()

    objs_idx = []
    for obj in objects_to_find:
        idx = graph.add_node({
            'name': obj,
            'type': [0, 0, 0, 1],
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
            get_sentence_embedding(node['name']), node['type']
        ))
        features.append(node_feature)
    return np.array(features)


def load_sentence_embedding(target_file_name: str) -> Optional[np.ndarray]:
    target_dir = os.path.join(DEFAULT_RESOURCES_BASE, DEFAULT_SBERT_SUBDIR, 'cache')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # Walk through all directories and files in target_dir
    for root, dirs, files in os.walk(target_dir):
        if target_file_name in files:
            file_path = os.path.join(root, target_file_name)
            if os.path.exists(file_path):
                return np.load(file_path)
    return None


def get_sentence_embedding(sentence: str) -> np.ndarray:
    loaded_embedding = load_sentence_embedding(sentence + '.npy')
    if loaded_embedding is None:
        from sentence_transformers import SentenceTransformer
        model_path = os.path.join(DEFAULT_RESOURCES_BASE, DEFAULT_SBERT_SUBDIR)
        model = SentenceTransformer(model_path)
        sentence_embedding = model.encode([sentence])[0]
        file_name = os.path.join(model_path, 'cache', sentence + '.npy')
        np.save(file_name, sentence_embedding)
        return sentence_embedding
    else:
        return loaded_embedding
