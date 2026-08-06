import json
import os
import re


def update_probability(task):
    refined_task = (''.join([i for i in task[0] if not i.isdigit()]),
                    ''.join([i for i in task[1] if not i.isdigit()]))
    total_items = sum(1 for item in GOAL_CONDITIONS
                      if item[0] == refined_task[0])
    item_to_cont = sum(1 for item in GOAL_CONDITIONS if item == refined_task)
    ratio = round(item_to_cont / total_items, 2) if total_items > 0 else 1
    return ratio


GOAL_CONDITIONS = list()
root = "./resources/alfredtasks/"
json_files = []

for path, _, files in os.walk(root):
    for name in files:
        if "test" not in name:
            json_files.append(os.path.join(path, name))

for file in json_files:
    datum = json.load(open(file))
    if "plan" in datum:
        tasks_in_datum = datum["plan"]["high_pddl"]
        for task in tasks_in_datum:
            if task["discrete_action"]["action"] == "PutObject":
                GOAL_CONDITIONS.append(tuple(task[
                    "discrete_action"]["args"]))

UNIQUE_GOALS = list(set(GOAL_CONDITIONS))
UNIQUE_GOALS = sorted(UNIQUE_GOALS, key=lambda x: x[0])
TASK_PROB = dict()
for goal in UNIQUE_GOALS:
    if goal[0] not in TASK_PROB:
        TASK_PROB[goal[0]] = dict()
    TASK_PROB[goal[0]][goal[1]] = update_probability(goal)


def get_task_probability(task):
    obj = ''.join([i for i in task[0] if not i.isdigit()])
    cnt = ''.join([i for i in task[1] if not i.isdigit()])
    if obj not in TASK_PROB:
        return 0
    if cnt not in TASK_PROB[obj]:
        return 0
    return TASK_PROB[obj][cnt]


def get_feasible_conditions():
    """
    returns:  considers all the tasks present in alfredtasks list and returns only the feasible goal conditions filtering from all the goal conditions in task list
    """
    movables = set([obj[0] for obj in GOAL_CONDITIONS])
    feasible_conditions = [
        condition for condition in GOAL_CONDITIONS
        if condition[1] not in movables
    ]
    return feasible_conditions


def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def categorize_strings(str_list):
    result = {}
    for string in str_list:
        match = re.match(r"(.+?)(\d+)$", string)
        if match:
            base_string, _ = match.groups()
            if base_string not in result:
                result[base_string] = []
            result[base_string].append(string)
    return result


def add_tuples(base_tuples, mapping_dict):
    result = base_tuples.copy()
    for item, furniture in base_tuples:
        if furniture in mapping_dict:
            for mapped_furniture in mapping_dict[furniture]:
                new_tuple = (item, mapped_furniture)
                result.append(new_tuple)
    return result


def add_more_tuples(base_tuples, mapping_dict):
    result = base_tuples.copy()
    for item, furniture in base_tuples:
        if item in mapping_dict:
            for mapped_item in mapping_dict[item]:
                new_tuple = (mapped_item, furniture)
                result.append(new_tuple)
    return result


def get_task_list(available_locations, available_objects):
    """
    Parameters: location present in the environment, seed (env_num)
    returns:
    task distribution: select task if both unique movable objects and fixed object (location) in that task is present in the sampled objects (currently randomly sampling from a range (3, available locations - 1) and location present in the environment
    available objects: list of unique objects in the task distribution
    """
    goal_conditions = get_feasible_conditions()
    task_list = [
        condition
        for condition in goal_conditions
        if condition[0] in available_objects and
        condition[1] in available_locations
    ]
    maped_dict_cnt = categorize_strings(available_locations)
    maped_dict_obj = categorize_strings(available_objects)
    task_list = list(set(add_tuples(task_list, maped_dict_cnt)))
    task_list = list(set(add_more_tuples(task_list, maped_dict_obj)))
    task_list = sorted(task_list, key=lambda x: x[0])
    final_list = list()
    for task in task_list:
        if round(get_task_probability(task), 3) >= 0.1:
            final_list.append(task)
    # print(get_task_probability(task_list[0]))
    # print(get_task_probability(task_list[1]))
    # print(get_task_probability(task_list[2]))
    return final_list
