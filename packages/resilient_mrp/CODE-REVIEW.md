# Planner design: what I changed, and what I still need you to look at

I went back through the feedback you have given me on how these four planners should be built, and
checked each point against what I actually implemented. This is that check.

Section A is the points I believe I have addressed, with the code that does it, so you can confirm
I read them the way you meant. Section B is the five places where I do not think the implementation
matches what you asked for, or where I made a call I want you to overrule if I got it wrong.
Section C is what I already know is outstanding, so you do not spend time trying to find it again.

Paths are relative to `src/resilient_mrp/`.

The four planners are `optimistic`, `cautious`, `failure_aware_ff` and `failure_aware_split`. The
first two are deterministic replanners. The last two run MCTS over the same probabilistic model and
differ in one argument, the leaf estimate.

---

## A. What you asked for, and what I did

| What you asked for | What I did | Status |
|---|---|---|
| the baselines were letting each robot race to the nearest goal and then reassign, which is not a fair baseline. Build a simplified robot-and-goal space with precomputed motion costs, plan in that space, and let the planner decide which robot takes which goal. | I precompute a route table with Dijkstra backwards from each goal, per robot (`planning/baselines.py:43`). That is the simplified space. `best_assignment` (`:75`) then searches it over (where each robot stands, which goals are covered, what each robot has already accumulated) and returns the robot-to-goal pairs. I do not assign anything by hand anywhere. | done |
| robots should not be pulled off a goal partway through. | I cache the assignment on (robots still operational, goals done) at `planning/baselines.py:155`, so it only rebuilds when a robot fails or a goal completes. | done |
| model the multi-robot system explicitly. The state has to include all robots and which goals are finished. | The state is the joint fluent state: `at`, `free` and `operational` for every robot, plus `safely_visited` per goal. `parse_state` (`planning/core.py:40`) reads those two things back out. | done |
| losing a robot should be part of the transition function, not a separate construction. | I moved it into the operator. It is now a probability branch inside `risk_move` (`planning/core.py:144-155`) that clears `free` and `operational`, and there is no failure handling anywhere outside it. A robot that fails is filtered out of the team at `experiments/mission.py:85`, which gives the smaller-team state you described. | done |
| a robot breaking down should not close the edge behind it. Take that out of the operator, it makes the problem much harder to plan and adds complexity for no benefit. | Removed. The failure branch of `risk_move` clears `free` and `operational` and nothing else (`planning/core.py:150-153`). `path_available` is asserted once over every edge when the instance is built (`planning/core.py:82`) and nothing retracts it, so an edge a robot broke down on stays open to the rest of the team for the whole mission. | done |
| optimistic should use durations and edge lengths. Cautious should use probabilities or log-probabilities. Mine looked misaligned. | I rewrote both as one class with the weight passed in. `optimistic_weight` returns the travel cost (`planning/baselines.py:30`) and `cautious_weight` returns `-log(survival)` (`:35`). Nothing else separates the two baselines. | done |
| do not compute a single-robot number and then repeat or scale it across the team. | `best_assignment` charges each robot its own accumulated load and prices a joint state by `max(carried.values())` (`planning/baselines.py:111`), so it is a makespan over the team rather than a sum or a scaled single-robot figure. Putting every goal on one robot loses on its own. | done for the baselines, see B for the estimate |
| 2000 MCTS iterations is really low, raise it to 10000 or more once the modelling is fixed. | The default is now 10000 (`experiments/instance.py:36`). | done |
| one unified way to compute failure cost. Stop using separate inconsistent constants and drop the time-dependent penalty that parked a failed robot. | I removed the 200-unit time effect. The failure branch of `risk_move` now carries no cost at all (`planning/core.py:147-153`), and `C_fail` enters in exactly two places: the trial score (`experiments/instance.py:96`) and the leaf estimate (`planning/value_function.py:51`). | done, except the cap, see B |
| apply a single failure cost in the analysis rather than coding it in several places, so every failed run scores the same. | `trial_cost` returns the makespan when every goal was visited and a flat `C_fail` otherwise (`experiments/instance.py:95-96`). When it fails, when it failed does not enter the number. | done |
| the dashboard only averages the successful runs, so the failed ones are missing from the number that matters. Average over all runs, failed and successful, and show the two as separate lines. | Partly. The total average now includes the failed runs and prices each of them at `C_fail` (`railroad/bench/dashboard/helpers.py:106`), so both numbers exist and are correct. But they are only printed as hover text on the case row (`figures.py:221-225`). There are no two separate lines on the chart yet. | partly done |
| use a heuristic value function rather than a rollout policy. | I build the leaf in `planner_setup` (`experiments/instance.py:176`) and it reaches the search as `heuristic_fn` (`experiments/mission.py:130`). There is no rollout policy. | done |
| stop planning inside a leaf node. [strong NO against plan within a plan] That compute should be going to the search. | The leaf no longer searches. I precompute a fast and a safe route table per (robot, goal) once, when the planner is built (`planning/value_function.py:16-21`), and `estimate` is then table lookups plus one pass over the outstanding goals. | done |
| do not modify the transition function to carry the failure cost. | There is one operator list, built in `experiments/instance.py:106`, and both execution and search use it. The model the planner searches is the model it acts in. | done |
| the estimate should be the cost of success plus the failure probability times the cost of failure, with failure as one large global constant. | `estimate` returns `max(load) + (1 - survival) * C_fail` (`planning/value_function.py:51`). Both terms are in makespan units, so they add. | done |
| our planner should not be worse than a basic cautious baseline, and the comparison has to isolate what the guidance contributes. | `planner_setup` (`experiments/instance.py:172-177`) hands `failure_aware_ff` and `failure_aware_split` the same operators, the same multiplier, the same unreachable penalty, the same budget and the same execution path. Split passes the estimate where ff passes nothing and falls back to the relaxed plan. Any difference between the two is the guidance. | done as an experiment design; the behaviour itself is in C |

---

## B. Where I think my code may not match what you asked for

### + The estimate combines the survival and cost

there was a feedback you gave about not taking a single-robot quantity and aggregate it in a way that does not reflect what
the team would actually do. I fixed that in `best_assignment`, which uses a per-robot load and a
makespan over the team. I did not fix it in the leaf estimate.

The estimate multiplies survival across goals (`planning/value_function.py:46`). If two robots each
take one goal, it treats either robot failing as the whole mission failing. That overstates the
risk, because the second robot may still be able to reach the goal the failed robot was carrying.
It is the same kind of aggregation you objected to in the baselines.

I left it in because the error is in the safer direction: the estimate says a state is more
dangerous than it is, rather than less. Tell me whether that is good enough, or whether the estimate
has to account for the second robot taking over. This is part of the main contribution, so changing
it means changing the estimate and re-running every result.

### + The estimate reads travel and survival off two different routes

For one robot and one goal, I take the travel term from the fast table and the survival term from
the safe table (`planning/value_function.py:40` and `:46`). Those are two different paths through
from graph (cautious graph, optimistic graph). So the travel time I report is measured on one path and the survival probability is
measured on another, and no single path has both properties.

I did this since the estimate is meant to combine an optimistic cost with a cautious
risk. Inside either table the two are never mixed: `RouteToGoal` carries the weight, the travel cost
and the survival for one path only (`planning/baselines.py:17-23`), so a cost from one path can
never be paired with a risk from another. Line 46 of the estimate is the one place the two tables
meet.

The whole estimate depends on this being valid, so need to confirm that combining the two
relaxations this way is what you meant.

### + Only failed runs are capped at C_fail

I think you also mentione no run should ever score above `C_fail`. For failed runs this already holds.
`trial_cost` (`experiments/instance.py:95-96`) does not add `C_fail` to the time a failed mission
spent. It throws that time away and returns `C_fail` on its own, so every failure scores exactly
`C_fail` and none of them can go above it. That is why the failed trials all sit on a single value
in the dashboard.

What is not capped is a successful run. If a mission visits every goal but takes longer than
`C_fail` to do it, `trial_cost` returns that makespan unchanged, and a slow success then scores
worse than a failure. Nothing in the code stops this. Whether it ever actually happens depends on
whether any successful makespan exceeds `C_fail`, which the trial records answer directly.

I have not added a cap, because there are two ways to close it and they do not mean the same thing.
The first might be to limit the reported number, so any trial computing above `C_fail` is recorded as
`C_fail`. The second is to choose a larger `C_fail` per instance, big enough that no successful
mission can reach it. The first changes what the metric reports. The second changes what `C_fail`
means. Tell me which one you had in mind.

### + Every planner but one treats an in-transit robot as already arrived

The mission loop advances until any one robot is free and then commits an action for it
(`experiments/mission.py:139`). The others are partway across an edge, and whether they arrive or
fail has not been decided yet.

A robot in that position has no `at` fluent, because `risk_move` retracts it on departure. So
`parse_state` reads its destination out of the upcoming effects and reports it there
(`planning/core.py:47-50`). Without that, the assignment and the estimate would both see a smaller
team than there is, and would plan using only the robots standing still.

The consequence is that a robot in transit is counted as though it has already arrived safely. Both
baselines get this through `experiments/mission.py:84` and the split leaf gets it through
`planning/value_function.py:25`. `failure_aware_ff` is the only one that does not, since the relaxed
plan reads the state itself.

You said the actions happen concurrently but still have to be planned in a coordinated way. The
The coordination is there, but it assumes every robot will safely_arrive. Is that the right
approximation, or should I weigh those robots by the survival probability of the edge they are on?
`optimistic`, `cautious` and `failure_aware_split` all share the assumption, so it does not change
how those three compare against each other.

### + Two constants I am still trying to justify

You objected to extra free parameters that add complexity without a conceptual reason. Two are left.

The wait action is built with two numbers I picked (`experiments/instance.py:109`). A robot waits
when it is free but has nowhere useful to go, usually because another robot is still crossing an
edge.

+ `no_op_time=5.0` is how long the robot stays busy waiting. It advances the clock, so it does enter
  the makespan, which is the reported cost.
+ `extra_cost=100.0` is a flat penalty on the action inside the search, so MCTS does not sit still
  instead of acting. It does not enter the makespan and it does not enter travel, which only sums
  `risk_move` edges. It only changes what the search prefers. I adopted this frpm the railroad examples. Let me know if you want me to remove them or make any adjustments. 


All four planners get the same operator, so neither number favours one over another. I did not
derive either one.

The MCTS exploration constant is hardcoded to 100 at the single call site
(`experiments/instance.py:199`) against a default `C_fail` of 500. I think you mentioend that this should
be on the order of the failure cost, so I have that change queued up. 

---

## C. Already outstanding, so you do not need to find it

+ **Terrain generation.** Terrain is still inferred by ranking the generator's blockage probability
  within each graph, which you objected to. Having the generator set the terrain directly is what I will look into.
+ **`risk_scale` in the code.** I renamed this to the failure-probability multiplier in the paper but not yet in the code.
+ **Cautious beating the split planner.** I measured this separately. The split planner is cheaper
  when it survives but takes its risk too early. I am still investigatin this which is why I need the code reveiew.

+ **The paired blocking experiment.** You asked for one run with the edge closing on failure and one
  without, expecting the open-edge version to be easier - I have not run it yet, still pending.

---

## D. Running it

```bash
uv run python packages/resilient_mrp/scripts/playground.py
```

The block under `__main__` selects the planner, graph type and seed, and can show the simulation
video alongside the terminal run.
