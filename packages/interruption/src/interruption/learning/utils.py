from typing import Any
from interruption.learning.data import CSVPickleDataset
from torch.utils.data import Subset, random_split
import torch
import numpy as np
from torch_geometric.data import Data
from railroad.environment.procthor.scenegraph import SceneGraph
from railroad.environment.procthor.learning.utils import compute_node_features


def prepare_gcn_input(
    datum: tuple[SceneGraph, float]
) -> Data:
    """
    Helper function for converting data of the form (scene_graph, ev)
    into a form that can be passed as input to a GCN.
    """
    graph = datum[0].copy()
    expected_value = datum[1]

    node_features = compute_ap_node_features(graph.nodes)
    node_features_t = torch.tensor(node_features, dtype=torch.float)

    edge_indices_t, edge_features_t = compute_edge_features(graph.nodes, graph.edges)

    return Data(
        node_features_t,
        edge_indices_t,
        edge_features_t,
        expected_value
    )


def compute_ap_node_features(nodes: dict[int, dict[str, str]]) -> np.ndarray:
    """
    Wrapper function for railroad's compute_node_features that includes the position of a
    node in the input features vector. This function relies on the iteration order of nodes
    being consistent.
    """
    return np.concat(
        (
            compute_node_features(nodes),
            [node["position"] for idx, node in nodes.items()]
        ),
        axis=1
    )


def compute_edge_features(
    nodes: dict[int, dict[str, Any]],
    edges: list[tuple[int, int]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Helper function that takes the edges from a SceneGraph object and converts them
    into the format expected by a GCN.
    """
    # euclidean distance between source node and dest node
    edge_costs = []
    src_indices = []
    dest_indices = []
    for (src_idx, dest_idx) in edges:
        edge_costs.append(np.linalg.norm(
            [
                x1 - x2
                for x1, x2 in zip(nodes[src_idx]["position"], nodes[dest_idx]["position"])
            ]
        ))
        src_indices.append(src_idx)
        dest_indices.append(dest_idx)

    # scale distances
    # 600 because engineering things
    scaled_edge_costs = 1 - np.expand_dims(edge_costs, axis=1) / 600

    # graph should be undirected, account for edges in the reversed direction as well
    return (
        torch.tensor([src_indices + dest_indices, dest_indices + src_indices], dtype=torch.long),
        torch.tensor(np.tile(scaled_edge_costs, (2, 1)), dtype=torch.float)
    )


def split_dataset(
    dataset: CSVPickleDataset,
    train_test_split: float = 0.8,
    seed: int = 8616
) -> list[Subset]:
    """
    Helper function for splitting a dataset into a test and train dataset. Wrapper function
    of torch.utils.data random_split function.
    """
    g = torch.Generator().manual_seed(seed)

    total_length = len(dataset)
    train_length = int(total_length * train_test_split)
    test_length = total_length - train_length

    return random_split(dataset, [train_length, test_length], g)


def convert_batch_format(batch) -> dict[str, torch.Tensor]:
    """
    Helper function for converting a batch returned by a DataLoader
    to the data format expected by forward method of AnticipateGCN.
    """
    return {
        "batch_index": batch.batch,
        "edge_data": batch.edge_index,
        "edge_features": batch.edge_attr,
        "latent_features": batch.x
    }


def get_torch_device() -> torch.device:
    """
    Helper function for getting the device a torch model should run on.
    """
    # load model
    device_str = "cpu"
    if torch.cuda.is_available():
        device_str = "cuda"
    elif torch.backends.mps.is_available():
        device_str = "mps"
    return torch.device(device_str)
