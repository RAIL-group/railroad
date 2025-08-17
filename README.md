
# Quickstart

Running `make` will output the available targets. 

If not using C++, `make build` is sufficient to start running code. Running `make build` again is *not* necessary, since the `mrppddl` package is installed in "edit" mode and will auto detect changes.

If C++ is being used, `make build-cpp` is required to build those bindings and then use them. If the C++ code is changed, `make build-cpp` must be rerun.

Everything else is run via `uv run`: e.g., `uv run mrppddl_astar.py` will run the relevant script.

# TODO Items

- [ ] I can use the FF heuristic (or similar) to determine which actions I never execute and prune those from the set of available actions.
- [ ] I still need to look over the other hashing functions to avoid using a "bare" XOR, which makes the likelihood of a collision problematically high.


# Organization of simulator code

Here is how the simulator needs to work:
- It provides the state of the world to the planner, which includes what objects exist and the state and upcoming effects for each robot.
- When all robots are given an action, the simulator's job is to execute those actions until a robot needs a new action.
- This means that the simulator needs to forward propagate the state of the world and keep track of what each robot is doing and until when.

For example: imagine a scenario in which you have two robots and a very minimal state:
```
(free r1) (at r1 r1_start)
(free r2) (at r2 r2_start)
```
with no upcoming effects.


Step 1: the simulator asks the planner for actions for the robots and the planner returns `(move r1 r1_start bedroom)`. The state of the world includes 
```
(free r2) (at r2 r2_start)
```
With the following upcoming effects:
```
after time_move_r1_start_bedroom
- (free r1) (at r1 bedroom)
```
This upcoming effect is very important, since without it `r1` will never be free again.

Step 2 is much like the first, where the second robot is given an action `(move r2 r2_start kitchen)`. In this case, the set of active fluents is empty, since neither robot is free nor at any of the key locations, but there are multiple upcoming effects.
```
after time_move_r1_start_bedroom
- (free r1) (at r1 bedroom)
after time_move_r2_start_kitchen
- (free r2) (at r2 kitchen)
```

Step 3: the simulator executes these actions. The robots have been told to move to their respective locations, so that's what happens. Each robot is told to move and so it should know to move to those locations. When one of them reaches the location, that robot is free and needs a new action.

This process loops until the goal is met. Remember: the simulator should mimic what happens in the world, where the robots each have some "assignment" and proceed until it's done or interrupted in some way.

Abhish: the way you have currently implemented things is nothing like this in mutiple ways. The robots should really be keeping track of their current "assignment" (and probably whether or not they can be interrupted in some way). It is from these assignments that you should be able to compute whether the robot is free and when it will be done (from which the upcoming effects are determined).

My idea is something like this: you have a Robot class. When you get some action like `(move r1 start somewhere)` the robot now has some sort of move skill its executing to move and the simulator can basically ask: (1) are you currently executing an action and (2) when will you be done? Based on the current state of the environment and the robots, you should be able to compute the "current state" of the world for the purpose of passing that to the planner. Propagating the state of the world then involves something similar to what you have: the first task to be done is "popped off the stack" in some way.

