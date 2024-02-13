import itertools
import numpy as np
from queue import Queue
from tqdm import tqdm


def get_nodes_to_explore(network, exploration_functions, not_included_nodes):
    state_to_explore = sorted(list(set([i for k, v in exploration_functions.items()
                                        for f, _ in v for i in f.keys() if k in not_included_nodes])))
    updated = True
    while updated:
        updated = False
        for node_id in state_to_explore:
            new_nodes = [k for f, _ in exploration_functions[node_id] for k in f.keys() if k not in state_to_explore]

            if len(new_nodes) > 0:
                updated = True
                [state_to_explore.append(x) for x in new_nodes]

    state_to_explore = [i for i in state_to_explore if i < network.state_size]
    print("state_to_explore: ", len(state_to_explore))
    return state_to_explore


def get_included_solutions_states(included_solution, state_to_explore):
    included_solutions_states = np.array([included_solution.stable_nodes.get(i, -1) for i in state_to_explore]).reshape((1, -1))

    for i, k in enumerate(state_to_explore):
        if k not in included_solution.stable_nodes:
            included_solutions_states = \
                np.tile(included_solutions_states, 2).reshape(-1, len(state_to_explore))
            included_solutions_states[np.arange(0, len(included_solutions_states), 2), i] = 0
            included_solutions_states[np.arange(1, len(included_solutions_states), 2), i] = 1
    return included_solutions_states


def check_if_reachable(solution, included_solutions, exploration_functions):
    state_to_explore = sorted(list(exploration_functions.keys()))
    if len(state_to_explore) > 23:
        print(f"state_to_explore: {len(state_to_explore)}")
        return -1

    func = {k: [[func.get(v, 0) for v in state_to_explore] for func, _ in funcs]
            for k, funcs in exploration_functions.items() if k in state_to_explore}
    tau = {k: [t for _, t in funcs] for k, funcs in exploration_functions.items() if k in state_to_explore}

    exploration_queue = Queue()
    exploration_states = {np.array(s).tobytes(): 0 for s in itertools.product((0, 1), repeat=len(state_to_explore))}
    reachable_states = 0
    for s in included_solutions:
        for state in get_included_solutions_states(s, state_to_explore):
            exploration_queue.put(state)
            exploration_states[state.tobytes()] = 1
            reachable_states += 1
    pbar = tqdm(total=2 ** len(state_to_explore))
    pbar.update(reachable_states)

    while not exploration_queue.empty():
        state = exploration_queue.get()
        reachable = np.logical_xor(state, np.eye(len(state_to_explore))).astype(int)
        res = np.array([all(np.inner(func[k], reachable[i]) >= tau[k]) for i, k in enumerate(state_to_explore)])
        for s in reachable[state == res]:
            tmp = s.tobytes()
            if exploration_states[tmp] == 1:
                continue
            exploration_states[tmp] = 1
            exploration_queue.put(s)
            reachable_states += 1
            pbar.update(1)
        if reachable_states == 2 ** len(state_to_explore):
            break

    if reachable_states < 2 ** len(state_to_explore):
        print("solution: ", solution.solution_id, len(included_solutions))
        print(len(exploration_states) - reachable_states)
        return len(exploration_states) - reachable_states
    else:
        return 0


def get_reduced_threshold_functions(network, stable_nodes, external, included_solutions):
    functions = {k: [f for f, _ in v] for k, v in network.threshold_functions.items()
                 if k < network.state_size and k not in stable_nodes}
    thresholds = {k: [t for _, t in v] for k, v in network.threshold_functions.items()
                  if k < network.state_size and k not in stable_nodes}
    default_states_values = {k: [np.sum([stable_nodes.get(i, 0) * v for i, v in ff.items()]) for ff in f]
                             for k, f in functions.items()}
    defaults_external_values = {k: [np.sum([external[i - network.state_size] * v for i, v in ff.items()
                                            if i >= network.state_size]) for ff in f] for k, f in functions.items()}
    updated_threshold = {k: [thresholds[k][j] - default_states_values[k][j] - defaults_external_values[k][j]
                             for j in range(len(thresholds[k]))] for k in thresholds}
    updated_functions = {k: [{i: v for i, v in ff.items()
                              if i not in stable_nodes and i < network.state_size} for ff in f]
                         for k, f in functions.items()}
    new_updated_functions = get_trivial_nodes(updated_functions)
    updated_thresholds_func = {i: [(new_updated_functions[i][j], updated_threshold[i][j])
                                   for j in range(len(new_updated_functions[i]))]
                               for i in new_updated_functions}

    not_included_nodes = list(set([i for s in included_solutions for i in s.stable_nodes if i not in stable_nodes]))
    nodes_to_explore = get_nodes_to_explore(network, updated_thresholds_func, not_included_nodes)

    return {i: v for i, v in updated_thresholds_func.items() if i in nodes_to_explore}


def get_trivial_nodes(f):
    node_to_parents = {k: list(set(p for v in val for p in v.keys())) for k, val in f.items()}
    node_to_children = {v: [] for v in f.keys()}

    for k, children in node_to_parents.items():
        for v in children:
            node_to_children[v].append(k)

    updated = True
    new_f = f.copy()
    while updated:
        updated = False
        for node_id, children in list(node_to_children.items()):
            if len(children) > 1:
                continue
            if len(children) == 0:
                updated = True
                node_to_children.pop(node_id)
                new_f.pop(node_id)
                if node_id not in node_to_parents:
                    continue
                parent = node_to_parents.pop(node_id)
                for pp in parent:
                    if pp in node_to_children:
                        node_to_children[pp].remove(node_id)
            elif node_id not in node_to_parents:
                continue
            elif len(node_to_parents[node_id]) == 1 and node_to_parents[node_id][0] not in children:
                parent = node_to_parents[node_id][0]
                child = node_to_children[node_id][0]
                if new_f[node_id][0][parent] < 1 or new_f[child][0][node_id] < 1:
                    continue
                updated = True
                parent = node_to_parents.pop(node_id)[0]
                child = node_to_children.pop(node_id)[0]
                node_to_children[parent].remove(node_id)
                for ff in new_f[child]:
                    ff[parent] = ff.get(parent, 0) + ff.pop(node_id)
                node_to_parents[child].remove(node_id)
                node_to_parents[child].append(parent)
                node_to_children[parent].append(child)
                new_f.pop(node_id)
    return new_f
