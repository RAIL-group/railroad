import pytest
from railroad.environment.procthor.scenegraph import SceneGraph


def test_scenegraph_basic_operations():
    """Test SceneGraph node and edge operations."""
    sg = SceneGraph()

    # Add apartment node
    idx_apt = sg.add_node({
        'id': 'apartment0',
        'type': [1, 0, 0, 0],
        'position': [0, 0],
        'name': 'apartment'
    })

    # Add room
    idx_room = sg.add_node({
        'id': 'bedroom0',
        'type': [0, 1, 0, 0],
        'position': [0, 1],
        'name': 'bedroom'
    })
    sg.add_edge(idx_apt, idx_room)

    # Add container
    idx_container = sg.add_node({
        'id': 'bed0',
        'type': [0, 0, 1, 0],
        'position': [0, 1],
        'name': 'bed'
    })
    sg.add_edge(idx_room, idx_container)

    # Add object
    idx_obj = sg.add_node({
        'id': 'pillow0',
        'type': [0, 0, 0, 1],
        'position': [0, 1],
        'name': 'pillow'
    })
    sg.add_edge(idx_container, idx_obj)

    assert len(sg.nodes) == 4
    assert len(sg.edges) == 3
    assert set(sg.room_indices) == {idx_room}
    assert set(sg.container_indices) == {idx_container}
    assert set(sg.object_indices) == {idx_obj}


def test_scenegraph_adjacency():
    """Test get_adjacent_nodes_idx and get_parent_node_idx."""
    sg = SceneGraph()
    idx_apt = sg.add_node({'id': 'apt', 'type': [1, 0, 0, 0], 'position': [0, 0], 'name': 'apt'})
    idx_room = sg.add_node({'id': 'room', 'type': [0, 1, 0, 0], 'position': [1, 0], 'name': 'room'})
    idx_cnt = sg.add_node({'id': 'cnt', 'type': [0, 0, 1, 0], 'position': [2, 0], 'name': 'cnt'})
    sg.add_edge(idx_apt, idx_room)
    sg.add_edge(idx_room, idx_cnt)

    assert set(sg.get_adjacent_nodes_idx(idx_room)) == {idx_apt, idx_cnt}
    assert sg.get_parent_node_idx(idx_room) == idx_apt
    assert sg.get_parent_node_idx(idx_cnt) == idx_room
    assert sg.get_parent_node_idx(idx_apt) is None


def test_scenegraph_object_free_copy():
    """Test get_object_free_graph removes objects."""
    sg = SceneGraph()
    idx_apt = sg.add_node({'id': 'apt', 'type': [1, 0, 0, 0], 'position': [0, 0], 'name': 'apt'})
    idx_cnt = sg.add_node({'id': 'cnt', 'type': [0, 0, 1, 0], 'position': [1, 0], 'name': 'cnt'})
    idx_obj = sg.add_node({'id': 'obj', 'type': [0, 0, 0, 1], 'position': [1, 0], 'name': 'obj'})
    sg.add_edge(idx_apt, idx_cnt)
    sg.add_edge(idx_cnt, idx_obj)

    sg_free = sg.get_object_free_graph()
    assert len(sg_free.nodes) == 2
    assert len(sg_free.object_indices) == 0
