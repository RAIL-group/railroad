import pytest

from environments.pyrobosim import PyRoboSimEnv
from railroad.experimental.environment import SkillStatus
from pathlib import Path


@pytest.mark.xfail(reason="Resource path needs fixing")
def test_pyrobosim_demo():
    env = PyRoboSimEnv(world_file='./resources/pyrobosim_worlds/test_world.yaml',
                       show_plot=False,
                       record_plots=True)

    r_names = list(env.robots.keys())
    action_list = {
        r_names[0]: [
            ('move', None, 'my_desk'),
            ('pick', 'my_desk', 'apple0'),
            ('move', 'my_desk', 'table0'),
            ('place', 'table0', 'apple0'),
        ],
        r_names[1]: [
            ('move', None, 'table0'),
            ('pick', 'table0', 'banana0'),
            ('move', 'table0', 'counter0'),
            ('place', 'counter0', 'banana0'),
            ('move', 'counter0', 'table0'),
        ]
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
        env.canvas.update()

    save_dir = Path("./data/test_logs")
    save_dir.mkdir(parents=True, exist_ok=True)
    env.canvas.save_animation(save_dir / "pyrobosim_demo.mp4")
    env.canvas.wait_for_close()
