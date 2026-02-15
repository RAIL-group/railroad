from railroad.environment.pyrobosim import PyRoboSimEnvironment, PyRoboSimScene, get_default_pyrobosim_world_file_path
from railroad.environment import SkillStatus
from railroad._bindings import Fluent, State
from pathlib import Path


def test_pyrobosim_demo():
    world_file = get_default_pyrobosim_world_file_path()
    scene = PyRoboSimScene(world_file)

    initial_fluents = set()
    robot_names = [r.name for r in scene.world.robots]
    for robot in robot_names:
        robot_loc = f"{robot}_loc"
        initial_fluents.add(Fluent("at", robot, robot_loc))
        initial_fluents.add(Fluent("free", robot))
        initial_fluents.add(Fluent("revealed", robot_loc))
    state = State(0.0, initial_fluents)

    env = PyRoboSimEnvironment(
        scene=scene,
        state=state,
        objects_by_type={
            "robot": set(robot_names),
            "location": set(scene.locations.keys()),
            "object": scene.objects,
        },
        operators=[],
        show_plot=False,
        record_plots=True,
    )
    r_names = list(env.robots.keys())
    action_list = {
        r_names[0]: [
            ("move", None, "my_desk"),
            ("pick", "my_desk", "apple0"),
            ("move", "my_desk", "table0"),
            ("place", "table0", "apple0"),
        ],
        r_names[1]: [
            ("move", None, "table0"),
            ("pick", "table0", "banana0"),
            ("move", "table0", "counter0"),
            ("place", "counter0", "banana0"),
            ("move", "counter0", "table0"),
        ],
    }
    action_done = {r: True for r in r_names}

    while sum([len(action_list[r]) for r in r_names]) != 0:
        for r in r_names:
            if action_done[r]:
                try:
                    action_name = action_list[r].pop(0)
                    skill, arg1, arg2 = action_name
                    env.execute_skill(r, skill, arg1, arg2)
                    action_done[r] = False
                except Exception:
                    pass
            status = env.get_executed_skill_status(r, skill)
            if status == SkillStatus.DONE:
                action_done[r] = True
        if env.canvas:
            env.canvas.update()

    save_dir = Path("./data/test_logs")
    save_dir.mkdir(parents=True, exist_ok=True)
    if env.canvas:
        env.canvas.save_animation(save_dir / "pyrobosim_demo.mp4")
        env.canvas.wait_for_close()
