import argparse
from environments.pyrobosim import PyRoboSimEnv
from environments import SkillStatus


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--world_file", type=str, default='./pyrobosim_worlds/test_world.yaml',
                      help="Path to the world YAML file.")
    args = args.parse_args()

    env = PyRoboSimEnv(args.world_file)

    for loc in env.locations:
        objects = env.get_objects_at_location(loc)
        print(f"Objects at {loc}: {objects}")

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
                except:
                    pass
            status = env.get_executed_skill_status(r, skill)
            if status == SkillStatus.DONE:
                action_done[r] = True

        env.canvas.update()
    print("All robots task complete !!!")
