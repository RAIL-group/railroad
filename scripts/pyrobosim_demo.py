import argparse
from environments.pyrobosim import PyRoboSimEnv
from pyrobosim.gui import start_gui
import threading


def main(env):
    locations = env.locations
    print(locations)
    for loc in locations:
        objects = env.get_objects_at_location(loc)
        print(f"Objects at {loc}: {objects}")

    # move robot to my_desk
    env.execute_skill('robot', 'move', None, 'my_desk')
    # pick apple0 from my_desk
    env.execute_skill('robot', 'pick', 'my_desk', 'apple0')
    # move robot from my_desk to counter0
    env.execute_skill('robot', 'move', 'my_desk', 'counter0')
    # place apple0 at counter0
    env.execute_skill('robot', 'place', 'counter0', 'apple0')


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--world_file", type=str, default='./pyrobosim_worlds/test_world.yaml', help="Path to the world YAML file.")
    args = args.parse_args()

    env = PyRoboSimEnv(args.world_file)

    # Run planning demo in a separate thread
    threading.Thread(target=main, args=(env,)).start()

    # Run GUI in main thread
    start_gui(env.world)
