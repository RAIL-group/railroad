from procthor.scenegraph import SceneGraph


def test_scenegraph():
    sg = SceneGraph()
    # add apartment node
    idx_apartment0 = sg.add_node({'id': 'apartment0', 'type': [1, 0, 0, 0], 'position': [0, 0], 'name': 'apartment'})
    # add rooms to apartment
    idx_bedroom0 = sg.add_node({'id': 'bedroom0', 'type': [0, 1, 0, 0], 'position': [0, 1], 'name': 'bedroom'})
    idx_bedroom1 = sg.add_node({'id': 'bedroom1', 'type': [0, 1, 0, 0], 'position': [0, 2], 'name': 'bedroom'})
    idx_kitchen0 = sg.add_node({'id': 'kitchen0', 'type': [0, 1, 0, 0], 'position': [1, 1], 'name': 'kitchen'})
    sg.add_edge(idx_apartment0, idx_bedroom0)
    sg.add_edge(idx_apartment0, idx_bedroom1)
    sg.add_edge(idx_apartment0, idx_kitchen0)
    # add containers to rooms
    idx_bed0 = sg.add_node({'id': 'bed0', 'type': [0, 0, 1, 0], 'position': [0, 1], 'name': 'bed'})
    idx_bed1 = sg.add_node({'id': 'bed1', 'type': [0, 0, 1, 0], 'position': [0, 2], 'name': 'bed'})
    idx_table0 = sg.add_node({'id': 'table0', 'type': [0, 0, 1, 0], 'position': [1, 1], 'name': 'table'})
    sg.add_edge(idx_bedroom0, idx_bed0)
    sg.add_edge(idx_bedroom1, idx_bed1)
    sg.add_edge(idx_kitchen0, idx_table0)
    # add objects to containers
    idx_pillow0 = sg.add_node({'id': 'pillow0', 'type': [0, 0, 0, 1], 'position': [0, 1], 'name': 'pillow'})
    idx_pillow1 = sg.add_node({'id': 'pillow1', 'type': [0, 0, 0, 1], 'position': [0, 2], 'name': 'pillow'})
    idx_plate0 = sg.add_node({'id': 'plate0', 'type': [0, 0, 0, 1], 'position': [1, 1], 'name': 'plate'})
    sg.add_edge(idx_bed0, idx_pillow0)
    sg.add_edge(idx_bed1, idx_pillow1)
    sg.add_edge(idx_table0, idx_plate0)
    # check nodes, edges and indices
    assert len(sg.nodes) == 10
    assert len(sg.edges) == 9
    assert set(sg.room_indices) == {idx_bedroom0, idx_bedroom1, idx_kitchen0}
    assert set(sg.container_indices) == {idx_bed0, idx_bed1, idx_table0}
    assert set(sg.object_indices) == {idx_pillow0, idx_pillow1, idx_plate0}
    # check get_adjacent_nodes_idx
    assert set(sg.get_adjacent_nodes_idx(idx_apartment0)) == {idx_bedroom0, idx_bedroom1, idx_kitchen0}
    assert set(sg.get_adjacent_nodes_idx(idx_bedroom0)) == {idx_apartment0, idx_bed0}
    assert set(sg.get_adjacent_nodes_idx(idx_bedroom1)) == {idx_apartment0, idx_bed1}
    assert set(sg.get_adjacent_nodes_idx(idx_kitchen0)) == {idx_apartment0, idx_table0}
    assert set(sg.get_adjacent_nodes_idx(idx_bed0)) == {idx_bedroom0, idx_pillow0}
    assert set(sg.get_adjacent_nodes_idx(idx_bed1)) == {idx_bedroom1, idx_pillow1}
    assert set(sg.get_adjacent_nodes_idx(idx_table0)) == {idx_kitchen0, idx_plate0}
    assert set(sg.get_adjacent_nodes_idx(idx_pillow0)) == {idx_bed0}
    assert set(sg.get_adjacent_nodes_idx(idx_plate0)) == {idx_table0}
    # check get_parent_node_idx
    assert sg.get_parent_node_idx(idx_apartment0) is None
    assert sg.get_parent_node_idx(idx_bedroom0) == idx_apartment0
    assert sg.get_parent_node_idx(idx_bedroom1) == idx_apartment0
    assert sg.get_parent_node_idx(idx_kitchen0) == idx_apartment0
    assert sg.get_parent_node_idx(idx_bed0) == idx_bedroom0
    assert sg.get_parent_node_idx(idx_bed1) == idx_bedroom1
    assert sg.get_parent_node_idx(idx_table0) == idx_kitchen0
    assert sg.get_parent_node_idx(idx_pillow0) == idx_bed0
    assert sg.get_parent_node_idx(idx_pillow1) == idx_bed1
    assert sg.get_parent_node_idx(idx_plate0) == idx_table0
    # check get_node_name_by_idx
    assert sg.get_node_name_by_idx(idx_apartment0) == 'apartment'
    assert sg.get_node_name_by_idx(idx_bedroom0) == 'bedroom'
    assert sg.get_node_name_by_idx(idx_bedroom1) == 'bedroom'
    assert sg.get_node_name_by_idx(idx_kitchen0) == 'kitchen'
    assert sg.get_node_name_by_idx(idx_bed0) == 'bed'
    assert sg.get_node_name_by_idx(idx_bed1) == 'bed'
    assert sg.get_node_name_by_idx(idx_table0) == 'table'
    assert sg.get_node_name_by_idx(idx_pillow0) == 'pillow'
    assert sg.get_node_name_by_idx(idx_pillow1) == 'pillow'
    assert sg.get_node_name_by_idx(idx_plate0) == 'plate'
    # check check_if_node_exists_by_id
    assert sg.check_if_node_exists_by_id('apartment0')
    assert sg.check_if_node_exists_by_id('bedroom0')
    assert sg.check_if_node_exists_by_id('bedroom1')
    assert sg.check_if_node_exists_by_id('kitchen0')
    # check get_node_name_by_idx
    assert sg.get_node_name_by_idx(0) == 'apartment'
    assert sg.get_node_name_by_idx(1) == 'bedroom'
    assert sg.get_node_name_by_idx(2) == 'bedroom'
    assert sg.get_node_name_by_idx(3) == 'kitchen'
    # check get_node_position_by_idx
    assert sg.get_node_position_by_idx(0) == [0, 0]
    assert sg.get_node_position_by_idx(1) == [0, 1]
    assert sg.get_node_position_by_idx(2) == [0, 2]
    assert sg.get_node_position_by_idx(3) == [1, 1]
    # get_object_free_graph
    sg_obj_free = sg.get_object_free_graph()
    assert len(sg_obj_free.nodes) == 7
    assert len(sg_obj_free.edges) == 6
    assert set(sg_obj_free.room_indices) == {idx_bedroom0, idx_bedroom1, idx_kitchen0}
    assert set(sg_obj_free.container_indices) == {idx_bed0, idx_bed1, idx_table0}
    assert len(sg_obj_free.object_indices) == 0
    assert set(sg_obj_free.get_adjacent_nodes_idx(idx_apartment0)) == {idx_bedroom0, idx_bedroom1, idx_kitchen0}
    assert set(sg_obj_free.get_adjacent_nodes_idx(idx_bed0)) == {idx_bedroom0}
    assert set(sg_obj_free.get_adjacent_nodes_idx(idx_bed1)) == {idx_bedroom1}
    assert set(sg_obj_free.get_adjacent_nodes_idx(idx_table0)) == {idx_kitchen0}
    # check delete_node
    sg.delete_node(idx_bed0)
    assert not sg.check_if_node_exists_by_id('bed0')
    assert len(sg.nodes) == 9
    assert len(sg.edges) == 7
    assert set(sg.container_indices) == {idx_bed1, idx_table0}
    assert set(sg.get_adjacent_nodes_idx(idx_bedroom0)) == {idx_apartment0}
    # check delete_edge
    sg.delete_edge(idx_apartment0, idx_bedroom0)
    assert len(sg.edges) == 6
    assert set(sg.get_adjacent_nodes_idx(idx_apartment0)) == {idx_bedroom1, idx_kitchen0}
