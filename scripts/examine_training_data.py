"""
Inspect the scene graphs in a GCN training dataset (procthor_data_*.csv), to
check for data sparsity around specific object-placement combinations.

Motivating case: the trained GCN predicts a *higher* expected value for a
scene where both the knife and the pan are at the fridge than for a scene
where only the knife is at the fridge (the current task) -- suspicious, since
having satisfied more of the upcoming task should lower expected remaining
cost, not raise it. This script buckets every training example by whether
the target/other object are at the given container, so you can see whether
the "both at container" bucket is under-represented (or absent) relative to
"target only" -- and render a few examples from each bucket to sanity-check
the bucketing itself.

Usage:
    uv run python scripts/examine_training_data.py
    uv run python scripts/examine_training_data.py --target-object pan --other-object knife
    uv run python scripts/examine_training_data.py --render-per-bucket 3
    uv run python scripts/examine_training_data.py --render-per-bucket 3 --save-dir /tmp/scenegraphs
"""
import argparse
import os
import statistics
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path

from interruption.learning.data import CSVPickleDataset
from railroad.environment.procthor.resources import get_procthor_10k_dir
from railroad.environment.procthor.scenegraph import SceneGraph

TRAIN_DATASET_PATH = get_procthor_10k_dir() / "procthor_data_201.csv"

# node type index -> (label, color), matching SceneGraph's one-hot type encoding
NODE_TYPE_STYLE = {
    0: ("apartment", "#4a3aa7"),
    1: ("robot", "#e34948"),
    2: ("room", "#1baf7a"),
    3: ("container", "#2a78d6"),
    4: ("object", "#eda100"),
}


# # --- Option A: bucket every datum by object/container membership --------------

# def containers_holding(graph: SceneGraph, object_name: str) -> set[str]:
#     """Generic container names holding any object with the given generic name."""
#     containers = set()
#     for obj_idx in graph.get_node_indices_by_name(object_name):
#         parent_idx = graph.get_parent_node_idx(obj_idx)
#         if parent_idx is not None:
#             containers.add(graph.nodes[parent_idx]["name"])
#     return containers


# def bucket_for(
#     graph: SceneGraph, target_object: str, other_object: str, container: str
# ) -> str:
#     target_at = container in containers_holding(graph, target_object)
#     other_at = container in containers_holding(graph, other_object)
#     if target_at and other_at:
#         return "both"
#     if target_at:
#         return "target-only"
#     if other_at:
#         return "other-only"
#     return "neither"


# def summarize(
#     dataset: CSVPickleDataset, target_object: str, other_object: str, container: str
# ) -> tuple[dict[str, list[float]], dict[str, list[int]]]:
#     """Bucket every (scene_graph, actual_ev) pair; returns (ev_by_bucket, dataset_index_by_bucket)."""
#     ev_by_bucket: dict[str, list[float]] = defaultdict(list)
#     index_by_bucket: dict[str, list[int]] = defaultdict(list)
#     for i in range(len(dataset)):
#         scene_graph, actual_ev = dataset[i]
#         bucket = bucket_for(scene_graph, target_object, other_object, container)
#         ev_by_bucket[bucket].append(actual_ev)
#         index_by_bucket[bucket].append(i)
#     return ev_by_bucket, index_by_bucket


# def print_summary(
#     ev_by_bucket: dict[str, list[float]], target_object: str, other_object: str, container: str
# ) -> None:
#     total = sum(len(evs) for evs in ev_by_bucket.values())
#     print(f"\n{total} scene graphs total. Bucketed by '{target_object}'/'{other_object}' at '{container}':\n")
#     order = ["both", "target-only", "other-only", "neither"]
#     labels = {
#         "both": f"{target_object} AND {other_object} at {container}",
#         "target-only": f"{target_object} only at {container}",
#         "other-only": f"{other_object} only at {container}",
#         "neither": f"neither at {container}",
#     }
#     for bucket in order:
#         evs = ev_by_bucket.get(bucket, [])
#         if not evs:
#             print(f"  {labels[bucket]:45s} n=0")
#             continue
#         print(
#             f"  {labels[bucket]:45s} n={len(evs):4d}  "
#             f"actual_ev min={min(evs):.2f} mean={statistics.mean(evs):.2f} "
#             f"max={max(evs):.2f}"
        # )


# --- Option B: render individual scene graphs with networkx -------------------

def _object_layout(
    graph: SceneGraph,
) -> tuple[dict[int, tuple[float, float]], dict[int, tuple[float, float]]]:
    """Position apartment/robot/room/container nodes at their stored position;
    fan objects out around their parent container so same-container objects
    (which all share their container's position) don't render on top of one
    another. Returns (node_pos, label_pos) -- label_pos pushes labels a bit
    further out than the node markers so they don't collide with the
    container's own label.
    """
    import math

    pos: dict[int, tuple[float, float]] = {}
    objects_by_parent: dict[int | None, list[int]] = defaultdict(list)
    for idx, node in graph.nodes.items():
        if node["type"][4] == 1:  # object
            parent_idx = graph.get_parent_node_idx(idx)
            objects_by_parent[parent_idx].append(idx)
        else:
            pos[idx] = tuple(node["position"])

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0) if pos else 1.0
    jitter_radius = max(span * 0.035, 3.0)

    label_pos: dict[int, tuple[float, float]] = dict(pos)
    for parent_idx, obj_indices in objects_by_parent.items():
        base = pos.get(parent_idx, (0.0, 0.0)) if parent_idx is not None else (0.0, 0.0)
        n = len(obj_indices)
        for i, obj_idx in enumerate(obj_indices):
            angle = 2 * math.pi * i / n
            offset = (jitter_radius * math.cos(angle), jitter_radius * math.sin(angle))
            pos[obj_idx] = (base[0] + offset[0], base[1] + offset[1])
            label_pos[obj_idx] = (base[0] + 1.3 * offset[0], base[1] + 1.3 * offset[1])
    return pos, label_pos


def render_scene_graph(graph: SceneGraph, title: str, ax=None):
    import networkx as nx

    g = nx.DiGraph()
    for idx, node in graph.nodes.items():
        g.add_node(idx)
    g.add_edges_from(graph.edges)

    pos, label_pos = _object_layout(graph)

    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(figsize=(8, 8))

    for type_idx, (label, color) in NODE_TYPE_STYLE.items():
        node_idxs = graph.get_node_indices_by_type(type_idx)
        if not node_idxs:
            continue
        nx.draw_networkx_nodes(
            g, pos, nodelist=node_idxs, node_color=color, node_size=180,
            label=label, ax=ax,
        )
    nx.draw_networkx_edges(g, pos, ax=ax, arrows=False, edge_color="#c3c2b7", width=0.8)

    # Only label container/object nodes -- apartment/robot/room labels add clutter.
    labels = {
        idx: node["name"]
        for idx, node in graph.nodes.items()
        if node["type"][3] == 1 or node["type"][4] == 1
    }
    nx.draw_networkx_labels(g, label_pos, labels=labels, font_size=6, ax=ax)

    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, loc="upper right")
    ax.set_axis_off()
    return ax

def main():
    dataset = CSVPickleDataset(os.fspath(TRAIN_DATASET_PATH), preprocess_function=None)
    

    for i in range(12, 20):
        render_scene_graph(dataset[i][0], title="hello")
        plt.show()
        plt.close()




# def render_bucket_examples(
#     dataset: CSVPickleDataset,
#     index_by_bucket: dict[str, list[int]],
#     ev_by_bucket: dict[str, list[float]],
#     buckets: list[str],
#     n_per_bucket: int,
#     save_dir: Path | None,
# ) -> None:
#     import matplotlib.pyplot as plt

#     for bucket in buckets:
#         indices = index_by_bucket.get(bucket, [])[:n_per_bucket]
#         if not indices:
#             print(f"No examples in bucket '{bucket}' to render.")
#             continue
#         for dataset_idx in indices:
#             scene_graph, actual_ev = dataset[dataset_idx]
#             fig, ax = plt.subplots(figsize=(8, 8))
#             render_scene_graph(
#                 scene_graph,
#                 f"bucket={bucket}  dataset_idx={dataset_idx}  actual_ev={actual_ev:.2f}",
#                 ax=ax,
#             )
#             if save_dir is not None:
#                 save_dir.mkdir(parents=True, exist_ok=True)
#                 out_path = save_dir / f"{bucket}_{dataset_idx}.png"
#                 fig.savefig(out_path, dpi=150, bbox_inches="tight")
#                 print(f"Saved {out_path}")
#                 plt.close(fig)
#             else:
#                 plt.show()


# def main() -> None:
#     parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
#     parser.add_argument("--dataset", type=str, default=os.fspath(TRAIN_DATASET_PATH))
#     parser.add_argument("--target-object", type=str, default="knife")
#     parser.add_argument("--other-object", type=str, default="pan")
#     parser.add_argument("--container", type=str, default="fridge")
#     parser.add_argument(
#         "--render-per-bucket", type=int, default=0,
#         help="If > 0, render this many scene graphs from each bucket with networkx.",
#     )
#     parser.add_argument(
#         "--render-buckets", type=str, default="both,target-only",
#         help="Comma-separated buckets to render from: both, target-only, other-only, neither.",
#     )
#     parser.add_argument(
#         "--save-dir", type=str, default=None,
#         help="If set, save rendered figures here instead of showing them interactively.",
#     )
#     args = parser.parse_args()

#     dataset = CSVPickleDataset(args.dataset, preprocess_function=None)

#     ev_by_bucket, index_by_bucket = summarize(
#         dataset, args.target_object, args.other_object, args.container
#     )
#     print_summary(ev_by_bucket, args.target_object, args.other_object, args.container)

#     if args.render_per_bucket > 0:
#         render_bucket_examples(
#             dataset,
#             index_by_bucket,
#             ev_by_bucket,
#             args.render_buckets.split(","),
#             args.render_per_bucket,
#             Path(args.save_dir) if args.save_dir else None,
#         )


if __name__ == "__main__":
    main()
