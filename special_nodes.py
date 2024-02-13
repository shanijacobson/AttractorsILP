import itertools
from collections import defaultdict
from itertools import product
import numpy as np
import gurobipy as gp
from tqdm import tqdm

M = 50_000


def find_external_assignments(network, solutions):
    if network.external_size == 0:
        external_id = solutions.add_externals_assignments(np.array([np.array([])]))
        for solution_id in solutions.solutions:
            solutions.update_external_to_solution(solution_id, external_id)
        return solutions

    keys_to_constraints, solutions_to_constraints_keys = get_all_externals_constraints(network, solutions)
    keys_to_external_assignments = {}
    for key, constraints_list in keys_to_constraints.items():
        all_possible_assignments = find_all_possible_assignments_for_constraints(network, constraints_list)
        keys_to_external_assignments[key] = np.unique(all_possible_assignments, axis=0).tobytes()
    solutions = add_externals_to_solutions(network.external_size, solutions, keys_to_external_assignments, solutions_to_constraints_keys)

    return solutions


def get_all_externals_constraints(network, solutions):
    keys_to_constraints = {}
    solutions_to_constraints_keys = {}
    for solution in solutions.solutions.values():
        constraints_groups = get_external_constraints_for_solution(network, solution.stable_nodes)
        constraints_keys = []
        for c in constraints_groups:
            external_constraints = sorted(c)
            key = str(external_constraints)
            keys_to_constraints[key] = external_constraints
            constraints_keys.append(key)
        solutions_to_constraints_keys[solution.solution_id] = constraints_keys
    return keys_to_constraints, solutions_to_constraints_keys


def get_external_constraints_for_solution(network, stable_states):
    all_region_constraints = []
    for s_idx in range(network.state_size):
        threshold_funcs = network.threshold_functions[s_idx]
        empty_constraint = False
        if s_idx in stable_states:
            if stable_states[s_idx] == 0:
                constraints = []
                for weights, threshold_val in threshold_funcs:
                    externals_max = np.sum([max(w, 0) for p_idx, w in weights.items() if p_idx >= network.state_size])
                    states_max = np.sum([w * stable_states[p_idx] if p_idx in stable_states else max(w, 0)
                                         for p_idx, w in weights.items() if p_idx < network.state_size])
                    if states_max + externals_max < threshold_val:
                        empty_constraint = True
                        break
                    if all(weights[i] == 0 for i in weights.keys() if i >= network.state_size):
                        continue
                    external_weights_list = [weights.get(i, 0) for i in
                                             range(network.state_size, network.state_size + network.external_size)]
                    new_constraint = (external_weights_list, round(threshold_val - states_max))
                    if new_constraint not in constraints:
                        constraints.append(new_constraint)
                if len(constraints) > 0 and not empty_constraint:
                    new_constraints = (0, constraints)
                    if new_constraints not in all_region_constraints:
                        all_region_constraints.append(new_constraints)
            elif stable_states[s_idx] == 1:
                for weights, threshold_val in threshold_funcs:
                    externals_min = np.sum([min(w, 0) for p_idx, w in weights.items() if p_idx >= network.state_size])
                    states_min = np.sum([w * stable_states[p_idx] if p_idx in stable_states else min(w, 0)
                                         for p_idx, w in weights.items() if p_idx < network.state_size])
                    if states_min + externals_min < threshold_val:
                        external_weights_list = [weights.get(i, 0) for i in
                                                 range(network.state_size, network.state_size + network.external_size)]
                        new_constraint = (1, [(external_weights_list, round(threshold_val - states_min))])
                        if new_constraint not in all_region_constraints:
                            all_region_constraints.append(new_constraint)
        else:
            constraints = []
            # empty_constraint = False
            for weights, threshold_val in threshold_funcs:
                # add constraint that it not 0 stable
                if all(weights[i] == 0 for i in weights.keys() if i >= network.state_size):
                    # empty_constraint = True
                    continue
                external_weights_list = [weights.get(i, 0) for i in
                                         range(network.state_size, network.state_size + network.external_size)]
                states_max = np.sum([w * stable_states[p_idx] if p_idx in stable_states
                                     else (0 if p_idx == s_idx else max(w, 0))
                                     for p_idx, w in weights.items() if p_idx < network.state_size])
                new_constraint = (2, [(external_weights_list, round(threshold_val - states_max))])
                if new_constraint not in all_region_constraints:
                    all_region_constraints.append(new_constraint)

                # add constraint that it not 1 stable
                states_min = np.sum([w * stable_states[p_idx] if p_idx in stable_states
                                     else (w if p_idx == s_idx else min(w, 0))
                                     for p_idx, w in weights.items() if p_idx < network.state_size])
                new_constraint = (external_weights_list, round(threshold_val - states_min))
                if new_constraint not in constraints:
                    constraints.append(new_constraint)
            if len(constraints) > 0:  # and not empty_constraint:
                new_constraints = (3, constraints)
                if new_constraints not in all_region_constraints:
                    all_region_constraints.append(new_constraints)

    constraints_to_groups = split_externals_constraints_to_groups(all_region_constraints)
    return constraints_to_groups


def split_externals_constraints_to_groups(constraint_list):
    external_per_constraints = [[j for i in range(len(c[1])) for j in range(len(c[1][i][0]))
                                 if c[1][i][0][j] != 0] for c in constraint_list]
    sublist_in_group = []
    groups = []
    for i in range(len(external_per_constraints)):
        if i in sublist_in_group:
            continue
        updated = True
        group_elements = [x for x in external_per_constraints[i]]
        group_lists_idx = [i]
        sublist_in_group.append(i)
        while updated:
            updated = False
            for j in range(len(external_per_constraints)):
                if j in group_lists_idx:
                    continue
                if any(e in group_elements for e in external_per_constraints[j]):
                    group_lists_idx.append(j)
                    group_elements = list(set(group_elements + external_per_constraints[j]))
                    sublist_in_group.append(j)
                    updated = True
        groups.append(group_lists_idx)

    constraints_for_group = [[constraint_list[i] for i in g] for g in groups]
    return constraints_for_group


def find_all_possible_assignments_for_constraints(network, constraints_list):
    external_size = network.external_size
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    external_vars = model.addVars(external_size, vtype=gp.GRB.BINARY, name=f"external")
    # add stable states constraints
    for i, (val, c) in enumerate(constraints_list):
        if val == 0:
            if len(c) > 1:
                zero_constraint = model.addVars(len(c), vtype=gp.GRB.BINARY, name=f"c0_{i}")
                for j in range(len(c)):
                    w, t = c[j]
                    expr = gp.quicksum(w[i] * external_vars[i] for i in range(external_size) if w[i] != 0)
                    model.addConstr(expr - t + 1 - M * (1 - zero_constraint[j]) <= 0)
                    model.addConstr(expr - t + M * zero_constraint[j] >= 0)
                res = model.addVar(vtype=gp.GRB.BINARY, name=f"res_{i}")
                model.addConstr(res == gp.or_(zero_constraint))
                model.addConstr(res >= 1)
            if len(c) == 1:
                w, t = c[0]
                model.addConstr(
                    gp.quicksum(w[i] * external_vars[i] for i in range(external_size) if w[i] != 0) <= t - 1)
        elif val == 1:
            for w, t in c:
                model.addConstr(gp.quicksum(w[i] * external_vars[i] for i in range(external_size) if w[i] != 0) >= t)
        elif val == 2:
            for w, t in c:
                model.addConstr(gp.quicksum(w[i] * external_vars[i] for i in range(external_size) if w[i] != 0) >= t)
        elif val == 3:
            if len(c) > 1:
                zero_constraint = model.addVars(len(c), vtype=gp.GRB.BINARY, name=f"c0_{i}")
                for j in range(len(c)):
                    w, t = c[j]
                    expr = gp.quicksum(w[i] * external_vars[i] for i in range(external_size) if w[i] != 0)
                    model.addConstr(expr - t + 1 - M * (1 - zero_constraint[j]) <= 0)
                    model.addConstr(expr - t + M * zero_constraint[j] >= 0)
                res = model.addVar(vtype=gp.GRB.BINARY, name=f"res_{i}")
                model.addConstr(res == gp.or_(zero_constraint))
                model.addConstr(res >= 1)
            if len(c) == 1:
                w, t = c[0]
                model.addConstr(
                    gp.quicksum(w[i] * external_vars[i] for i in range(external_size) if w[i] != 0) <= t - 1)

    not_null_externals = list(set([i for _, c in constraints_list for i in range(external_size)
                                   if any(cc[0][i] != 0 for cc in c)]))
    model.addConstrs(external_vars[i] <= 0 for i in range(external_size) if i not in not_null_externals)
    possible_external_list = []
    model.setParam("PoolSearchMode", 2)
    model.setParam("PoolSolutions", 2 ** 30)
    model.optimize()
    if model.status != gp.GRB.OPTIMAL:
        [print(x) for x in constraints_list]
        print("wrong!!")
    if model.status == gp.GRB.OPTIMAL:
        for i in range(model.solCount):
            all_vars = model.getVars()
            model.setParam("SolutionNumber", i)
            possible_external_list.append(np.array([round(all_vars[i].Xn) if i in not_null_externals else -1
                                                    for i in range(external_size)]))
    model.dispose()
    return possible_external_list


def add_externals_to_solutions(external_size, solutions, keys_to_external_assignments, solutions_to_constraints_keys):
    externals_assignment_to_id = {v: i for i, v in enumerate(list(set(keys_to_external_assignments.values())))}
    external_assignment_to_idx = {}
    solution_to_externals = {}
    for solution_id in solutions.solutions:
        key = tuple(sorted([externals_assignment_to_id[keys_to_external_assignments[k]]
                            for k in solutions_to_constraints_keys[solution_id]]))
        if key not in external_assignment_to_idx:
            external_assignment_to_idx[key] = len(external_assignment_to_idx)
        solution_to_externals[solution_id] = external_assignment_to_idx[key]

    id_to_externals_assignment = {v: k for k, v in externals_assignment_to_id.items()}
    external_assignments = {}
    for external_list, external_id in external_assignment_to_idx.items():
        total_assignment = np.array([-1 for _ in range(external_size)]).reshape(-1, external_size)
        for idx in external_list:
            external_assignment = np.frombuffer(id_to_externals_assignment[idx], dtype=int).reshape(-1, external_size)
            broadcast_assignments = np.tile(external_assignment, (total_assignment.shape[0], 1))
            total_assignment = np.tile(total_assignment, len(external_assignment)).reshape(-1, external_size)
            total_assignment[broadcast_assignments >= 0] = broadcast_assignments[broadcast_assignments >= 0]
        external_assignments[external_id] = total_assignment
    curr_external_assignment = {e_id: np.unique(assignment, axis=0).tobytes()
                                for e_id, assignment in external_assignments.items()}
    unique_external = list(set(curr_external_assignment.values()))
    new_external_id = {external: i for i, external in enumerate(unique_external)}
    new_solution_to_external = {}
    for s_id, e_id in tqdm(solution_to_externals.items()):
        new_solution_to_external[s_id] = new_external_id[curr_external_assignment[e_id]]

    new_external_assignment = {i: np.frombuffer(new_external, dtype=int).reshape(-1, external_size)
                               for i, new_external in enumerate(unique_external)}

    solutions.solution_to_externals = new_solution_to_external
    solutions.external_assignments = new_external_assignment
    solutions.external_id_counter = len(solutions.external_assignments)
    return solutions


def validate_all_possible_externals(network, states_nodes, all_possible_externals):
    stable_nodes_assignment = {network.index_to_name[idx]: val for idx, val in states_nodes.items()}
    for assignment in product((0, 1), repeat=len(all_possible_externals[0])):
        not_null_external_assignment = {n: assignment[i] for i, n in enumerate(all_possible_externals[0])}
        null_external_assignment = {network.index_to_name[i + network.state_size]: 0
                                    for i in range(network.external_size) if
                                    network.index_to_name[i + network.state_size] not in all_possible_externals[0]}
        all_assignment = {**stable_nodes_assignment, **not_null_external_assignment, **null_external_assignment}
        is_solution = True
        for s_idx in states_nodes.keys():
            res = network.nodes[network.index_to_name[s_idx]].assign_values_to_boolean_function(all_assignment)
            if res != states_nodes[s_idx]:
                is_solution = False
                break
        if not_null_external_assignment in all_possible_externals:
            if not is_solution:
                print("Problem 1")
        else:
            if is_solution:
                tmp = [[x for x in not_null_external_assignment
                        if not_null_external_assignment[x] != all_possible_externals[i][x]]
                       for i in range(len(all_possible_externals))]
                print(tmp)
                print(f"Problem 2")


def add_external_only_dependent_to_solutions(network, solutions):
    all_external = sorted(list(set([k for node_name in network.external_only_depended_nodes_names
                                    for k in network.threshold_functions[network.nodes[node_name].index][0][0].keys()])))
    all_external_idx = np.array([e - network.state_size for e in all_external])
    external_to_results = {}

    all_funcs = [[([func.get(v, 0) for v in all_external], t)
                  for func, t in network.threshold_functions[network.nodes[name].index]]
                 for name in network.external_only_depended_nodes_names]

    for assignment in itertools.product((0, 1), repeat=len(all_external)):
        external_only_dependent_vals = np.array([1 if all(np.sum(val * f[idx] for idx, val in enumerate(assignment)) >= t
                                                          for f, t in funcs) else 0
                                                 for funcs in all_funcs])
        external_to_results[np.array(assignment).tobytes()] = external_only_dependent_vals

    for external_id in tqdm(list(solutions.external_assignments.keys())):
        sol_idx = [s for s in solutions.solutions if solutions.solution_to_externals[s] == external_id]
        if len(sol_idx) == 0:
            continue
        external_assignment = solutions.external_assignments[external_id]
        full_external_list = list(set([idx for idx, v in enumerate(external_assignment[0]) if v != -1] +
                                      [idx for idx in all_external_idx]))
        full_external_assignment = get_full_external_assignment(network, solutions, sol_idx[0], full_external_list)
        dependent_val_to_external = defaultdict(list)
        for e in full_external_assignment:
            dependant_val = external_to_results[e[all_external_idx].tobytes()]
            dependent_val_to_external[dependant_val.tobytes()].append(e)

        for k, val in dependent_val_to_external.items():
            external_depend_assignment = {network.state_size + network.external_size + i: int(v)
                                          for i, v in enumerate(np.frombuffer(k, dtype=int))}
            new_external_id = solutions.add_externals_assignments(val)
            for s_id in sol_idx:
                new_stable_nodes = {**solutions.solutions[s_id].stable_nodes.copy(), **external_depend_assignment}
                new_solution_id = solutions.add_solution(new_stable_nodes)
                solutions.update_external_to_solution(new_solution_id, new_external_id)

        for s_id in sol_idx:
            solutions.remove_solution(s_id)
        solutions.remove_external(external_id)
    print(f"size: {len(solutions.solutions)}")
    return solutions


def add_hole_to_solutions(network, solutions):
    hole_state_parents = [np.array([network.nodes[p].index for i, p in enumerate(network.nodes[node_name].get_parents())
                                    if p in network.state_nodes_names]) for node_name in network.hole_nodes_names]
    hole_external_parents = [np.array([network.nodes[p].index - network.state_size
                                       for i, p in enumerate(network.nodes[node_name].get_parents())
                                       if p in network.external_nodes_names]) for node_name in network.hole_nodes_names]
    hole_tt = [{np.array(t[0]).tobytes(): 1 if t[1] else 0 for t in network.nodes[node_name].get_node_truth_table()}
               for node_name in network.hole_nodes_names]
    index_bias = network.state_size + network.external_size + len(network.external_only_depended_nodes_names)
    hole_external_list = list(set([network.nodes[p].index - network.state_size
                                   for node_name in network.hole_nodes_names
                                   for p in network.nodes[node_name].get_parents()
                                   if p in network.external_nodes_names]))
    for solution_id in tqdm(list(solutions.solutions.keys())):
        sol = solutions.solutions[solution_id]
        stable_nodes_val = np.array([sol.stable_nodes.get(k, -1) for k in range(network.state_size)])
        full_external_list = hole_external_list + solutions.get_not_null_externals_for_solution(solution_id)
        full_external_assignments = get_full_external_assignment(network, solutions, solution_id, full_external_list)
        external_to_hole = defaultdict(list)
        for e in full_external_assignments:
            assignments = [np.concatenate(
                (stable_nodes_val[hole_state_parents[i]] if len(hole_state_parents[i]) > 0 else np.array([]),
                 e[hole_external_parents[i]] if len(hole_external_parents[i]) > 0 else np.array([]))).astype(np.int64)
                           for i in range(len(network.hole_nodes_names))]
            dict_assignments = {network.index_to_name[i]: v for i, v in enumerate(stable_nodes_val) if v != -1}
            res = np.array([hole_tt[i][a.tobytes()] if all(aa != -1 for aa in a) else -1
                            for i, a in enumerate(assignments)])
            unstable_states = np.arange(len(network.hole_nodes_names))[res == -1]
            if len(unstable_states) > 0:
                tmp = [network.nodes[network.index_to_name[i]].assign_values_to_boolean_function(dict_assignments)
                       for i in unstable_states]
                res[unstable_states] = [t if isinstance(t, int) else -1 for t in tmp]
            external_to_hole[res.tobytes()].append(e)

        for k, val in external_to_hole.items():
            hole_value = np.frombuffer(k, dtype=full_external_assignments.dtype)
            external_id = solutions.add_externals_assignments(val)
            stable_nodes = {**sol.stable_nodes,
                            **{i + index_bias: v for i, v in enumerate(hole_value) if hole_value[i] != -1}}
            new_solution_id = solutions.add_solution(stable_nodes)
            solutions.update_external_to_solution(new_solution_id, external_id)
        solutions.remove_solution(solution_id)

    return solutions


def get_full_external_assignment(network, solutions, solution_id, external_list):
    not_null_externals = solutions.get_not_null_externals_for_solution(solution_id)
    full_external_assignment = solutions.external_assignments[solutions.solution_to_externals[solution_id]]
    if len(external_list) == 0:
        return np.array([np.array([-1])])
    external_size = network.external_size
    if len(external_list) > len(not_null_externals):
        null_externals = [e for e in external_list if e not in not_null_externals]
        for i in null_externals:
            full_external_assignment = \
                np.tile(full_external_assignment, 2).reshape(-1, external_size)
            full_external_assignment[np.arange(0, len(full_external_assignment), 2), i] = 0
            full_external_assignment[np.arange(1, len(full_external_assignment), 2), i] = 1
    return full_external_assignment
