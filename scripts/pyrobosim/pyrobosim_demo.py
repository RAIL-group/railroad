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

    action_list = [
        ('move', 'robot', None, 'my_desk'),
        ('search', 'robot', 'my_desk', None),
        ('pick', 'robot', 'my_desk', 'apple0'),
        ('move', 'robot', 'my_desk', 'counter0'),
        ('place', 'robot', 'counter0', 'apple0'),
        ('no_op', 'robot', None, None),  # by design no_op makes the robot wait
        ('move', 'robot', None, 'kitchen')
    ]

    for action in action_list:
        skill, robot, arg1, arg2 = action
        print(f"Executing skill: {skill} with args: {arg1}, {arg2}")
        env.execute_skill(robot, skill, arg1, arg2)

        while True:
            status = env.get_executed_skill_status(robot, skill)
            env.canvas.update()
            if status == SkillStatus.DONE:
                break
