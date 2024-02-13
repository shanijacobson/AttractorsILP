import pickle
import gurobipy as gp
from tqdm import tqdm
import special_nodes
from SolutionObject import SolutionObjects
import including_solutions
from collections import Counter

M = 50_000


def build_ilp_model(network, size):
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    state_size = network.state_size
    states_vars = model.addVars(state_size, vtype=gp.GRB.BINARY, name="states")
    externals_vars = model.addVars(network.external_size, vtype=gp.GRB.BINARY, name="external")
    fixed_vars = model.addVars(state_size, vtype=gp.GRB.BINARY, name="fixed")

    for s_idx in range(network.state_size):
        node_thresholds_funcs = network.threshold_functions[s_idx]
        threshold_order = len(node_thresholds_funcs)
        always_over = model.addVars(threshold_order, vtype=gp.GRB.BINARY, name=f"s{s_idx}_always_over")
        always_under = model.addVars(threshold_order, vtype=gp.GRB.BINARY, name=f"s{s_idx}_always_under")
        for order in range(threshold_order):
            weights_dict, func_threshold = node_thresholds_funcs[order]
            func_weights = {k_id: weight for k_id, weight in weights_dict.items() if weight != 0}
            # add always over constraint:
            x_min = model.addVars(len(func_weights), vtype=gp.GRB.BINARY, name=f"s{s_idx}_{order}_min")
            model.addConstrs((fixed_vars[p_idx] == 1) >> (x_min[i] == states_vars[p_idx])
                             for i, p_idx in enumerate(func_weights.keys()) if p_idx < network.state_size)
            model.addConstrs((fixed_vars[p_idx] == 0) >>
                             (x_min[i] == (1 if func_weights[p_idx] < 0 or p_idx == s_idx else 0))
                             for i, p_idx in enumerate(func_weights.keys()) if p_idx < network.state_size)
            y_min = gp.quicksum(x_min[i] * func_weights[p_idx] if p_idx < network.state_size
                                else externals_vars[p_idx - network.state_size] * func_weights[p_idx]
                                for i, p_idx in enumerate(func_weights.keys()))

            model.addConstr(y_min <= func_threshold - 1 + M * always_over[order],
                            name=f"s{s_idx}_o{order}_always_over_1")
            model.addConstr(y_min >= func_threshold - M * (1 - always_over[order]),
                            name=f"s{s_idx}_o{order}_always_over_2")

            # add always under constraint:
            x_max = model.addVars(len(func_weights), vtype=gp.GRB.BINARY, name=f"s{s_idx}_{order}_max")
            model.addConstrs((fixed_vars[p_idx] == 1) >> (x_max[i] == states_vars[p_idx])
                             for i, p_idx in enumerate(func_weights.keys()) if p_idx < network.state_size)
            model.addConstrs((fixed_vars[p_idx] == 0) >>
                             (x_max[i] == (0 if func_weights[p_idx] < 0 or p_idx == s_idx else 1))
                             for i, p_idx in enumerate(func_weights.keys()) if p_idx < network.state_size)
            y_max = gp.quicksum(x_max[i] * func_weights[p_idx] if p_idx < network.state_size
                                else externals_vars[p_idx - network.state_size] * func_weights[p_idx]
                                for i, p_idx in enumerate(func_weights.keys()))

            model.addConstr(y_max <= func_threshold - 1 + M * (1 - always_under[order]),
                            name=f"s{s_idx}_o{order}_always_under_1")
            model.addConstr(y_max >= func_threshold - M * always_under[order],
                            name=f"s{s_idx}_o{order}_always_under_2")

        if threshold_order == 1:
            model.addConstr(fixed_vars[s_idx] == gp.or_(always_over, always_under), name=f"s{s_idx}_fixed")
            model.addConstr(states_vars[s_idx] == always_over[0], name=f"s{s_idx}_state_val")
        else:
            all_always_over = model.addVar(vtype=gp.GRB.BINARY, name=f"s{s_idx}_all_always_over")
            model.addConstr(all_always_over == gp.and_(always_over), name=f"s{s_idx}_all_always_over")
            model.addConstr(fixed_vars[s_idx] == gp.or_(all_always_over, always_under), name=f"s{s_idx}_fixed")
            model.addConstr(states_vars[s_idx] == all_always_over, name=f"s{s_idx}_state_val")

    for node_name in network.state_nodes_names:
        node = network.nodes[node_name]
        if node.static:
            model.addConstr(fixed_vars[node.index] >= 1)

    model.addConstr(gp.quicksum(fixed_vars) == size)
    model.update()
    return model, states_vars, externals_vars, fixed_vars


def add_stable_state_constraint(model, states_vars, fixed_vars, stable_states, stable_state):
    expr_1 = gp.quicksum(1 - states_vars[i] if stable_states[i] == 1 else states_vars[i]
                         for i in stable_states)
    if stable_state:
        model.addConstr(expr_1 >= 1, name='compair')
        return
    expr_2 = gp.quicksum(1 - fixed_vars[i] if i in stable_states else fixed_vars[i]
                         for i in range(len(states_vars)))
    model.addConstr(expr_1 + expr_2 >= 1, name='compair')


def get_stable_states(model, state_size, external_size):
    all_vars = model.getVars()
    states_val = [round(v.Xn) for v in all_vars[0: state_size]]
    fixed_val = [round(v.Xn) for v in all_vars[state_size + external_size: 2 * state_size + external_size]]
    stable_states = {idx: states_val[idx] for idx in range(len(states_val)) if fixed_val[idx] == 1}
    return stable_states


def find_stable_states(network):
    solutions = SolutionObjects()
    for i in tqdm(range(network.state_size, min(1, network.state_size - 1), -1)):
        model, states_vars, externals_vars, fixed_vars = build_ilp_model(network, i)
        fix_attractor = i == network.state_size
        while True:
            model.optimize()
            if model.status != gp.GRB.OPTIMAL:
                break
            stable_states = get_stable_states(model, network.state_size, network.external_size)
            add_stable_state_constraint(model, states_vars, fixed_vars, stable_states, fix_attractor)
            solutions.add_solution(stable_nodes=stable_states)
        model.dispose()
    if len(solutions.solutions) == 0:
        solutions.add_solution(stable_nodes={})
    return solutions


def find_stable_states_and_external(network, is_verify_sub_solutions=False, is_save=False):
    print(f"Network: {network.name}, state size: {network.state_size}, external: {network.external_size}, "
          f"hole: {len(network.hole_nodes_names)}, external dependent: {len(network.external_only_depended_nodes_names)}")
    solutions = find_stable_states(network)
    solutions = special_nodes.find_external_assignments(network, solutions) #find_external_assignments(network, solutions)

    print("size one solutions: ", len([s for s in solutions.solutions.values() if len(s.stable_nodes) == network.state_size]))
    print("bigger than one solutions: ", len([s for s in solutions.solutions.values() if len(s.stable_nodes) < network.state_size]))
    solutions = including_solutions.remove_included_solutions(network, solutions, is_verify_sub_solutions)
    # if network.external_size > 0:
    #     solutions = including_solutions.add_null_external_solutions(network, solutions)
    if len(network.external_only_depended_nodes_names) > 0:
        solutions = special_nodes.add_external_only_dependent_to_solutions(network, solutions)
    if len(network.hole_nodes_names) > 0:
        solutions = special_nodes.add_hole_to_solutions(network, solutions)
        print(f"size: {len(solutions.solutions)}")
    print(f"Counter: {Counter(len(a.stable_nodes) for a in solutions.solutions.values())}")

    if is_save:
        print("save solution")
        with open(f'tmp/final_solution_{network.network_name}.pkl', 'wb') as f:
            pickle.dump(solutions, f)
    return solutions


def validate_attractor(network, attractor):
    states_nodes = attractor.stable_nodes
    external_nodes = attractor.external_nodes
    stable_nodes_assignment = {network.index_to_name[idx]: val for idx, val in states_nodes.items() if val != "-"}
    external_nodes_assignment = {network.external_nodes_names[i]: external_nodes[i] for i in
                                 range(network.external_size)}
    assignment = {**stable_nodes_assignment, **external_nodes_assignment}
    oscillated_nodes = [name for name in network.state_nodes_names if name not in stable_nodes_assignment.keys()]
    is_attractor = True
    for name in stable_nodes_assignment:
        node = network.nodes[name]
        parents = node.get_parents()
        oscillated_parents = [p for p in parents if p in oscillated_nodes]
        if len(oscillated_parents) == 0:
            res = 1 if node.boolean_function.subs(assignment) else 0
            if not stable_nodes_assignment[name] == res:
                print(node.index, node.boolean_function.subs(assignment), stable_nodes_assignment[name])
                is_attractor = False
        else:
            for i in range(2 ** len(oscillated_parents)):
                i_binary = format(i, "0{}b".format(len(oscillated_parents)))
                oscillated_assignment = {oscillated_parents[p]: int(i_binary[p]) for p in
                                         range(len(oscillated_parents))}
                res = 1 if node.boolean_function.subs({**assignment, **oscillated_assignment}) else 0
                if not stable_nodes_assignment[name] == res:
                    is_attractor = False
                    break
    return is_attractor

#
# def find_external_assignments(network, solutions):
#     if network.external_size == 0:
#         external_id = solutions.add_externals_assignments(np.array([np.array([])]))
#         for solution_id in solutions.solutions:
#             solutions.update_external_to_solution(solution_id, external_id)
#         return solutions
#     constraint_list_to_constraints = {}
#     external_constraint_per_solution = defaultdict(list)
#     for sol in solutions.solutions.values():
#         constraints_groups = get_all_external_constraints2(network, sol.stable_nodes)
#         for c in constraints_groups:
#             external_constraints = sorted(c)
#             key = str(external_constraints)
#             constraint_list_to_constraints[key] = external_constraints
#             external_constraint_per_solution[sol.solution_id].append(key)
#
#     external_assignments_for_constraints = {}
#     for key in constraint_list_to_constraints:
#         constraint = constraint_list_to_constraints[key]
#         all_external_assignments = find_all_possible_externals(network, constraint)
#         external_assignments_for_constraints[key] = np.unique(all_external_assignments, axis=0).tobytes()
#
#     unique_external_to_idx = {v: idx for idx, v in enumerate(list(set(external_assignments_for_constraints.values())))}
#     external_assignment_to_idx = {}
#     for solution_id in tqdm(list(solutions.solutions)):
#         key = tuple(sorted([unique_external_to_idx[external_assignments_for_constraints[e]]
#                             for e in external_constraint_per_solution[solution_id]]))
#         if key not in external_assignment_to_idx:
#             external_assignment_to_idx[key] = len(external_assignment_to_idx)
#         solutions.solution_to_externals[solution_id] = external_assignment_to_idx[key]
#
#     idx_to_unique_external = {v: k for k, v in unique_external_to_idx.items()}
#     for external_list, external_id in external_assignment_to_idx.items():
#         total_assignment = np.array([-1 for _ in range(network.external_size)]).reshape(-1, network.external_size)
#         for idx in external_list:
#             external_assignment = np.frombuffer(idx_to_unique_external[idx], dtype=int).reshape(-1, network.external_size)
#             broadcast_assignments = np.tile(external_assignment, (total_assignment.shape[0], 1))
#             total_assignment = np.tile(total_assignment, len(external_assignment)).reshape(-1, network.external_size)
#             total_assignment[broadcast_assignments >= 0] = broadcast_assignments[broadcast_assignments >= 0]
#         solutions.external_assignments[external_id] = total_assignment
#     curr_external_assignment = {e_id: np.unique(assignment, axis=0).tobytes()
#                                 for e_id, assignment in solutions.external_assignments.items()}
#     unique_external = list(set(curr_external_assignment.values()))
#     new_external_id = {external: i for i, external in enumerate(unique_external)}
#     new_solution_to_external = {}
#     for s_id, e_id in tqdm(solutions.solution_to_externals.items()):
#         new_solution_to_external[s_id] = new_external_id[curr_external_assignment[e_id]]
#
#     new_external_assignment = {i: np.frombuffer(new_external, dtype=int).reshape(-1, network.external_size)
#                                for i, new_external in enumerate(unique_external)}
#
#     solutions.solution_to_externals = new_solution_to_external
#     solutions.external_assignments = new_external_assignment
#     solutions.external_id_counter = len(new_external_assignment)
#     solutions.solution_id_counter = len(solutions.solutions)
#     return solutions
#
# # def get_all_external_constraints(network, stable_states):
# #     all_region_constraints = []
# #     for s_idx in stable_states.keys():
# #         threshold_funcs = network.threshold_functions[s_idx]
# #         constraints = []
# #         if stable_states[s_idx] == 0:
# #             empty_constraint = False
# #             for weights, threshold_val in threshold_funcs:
# #                 externals_max = np.sum([max(w, 0) for p_idx, w in weights.items() if p_idx >= network.state_size])
# #                 states_max = np.sum([w * stable_states[p_idx] if p_idx in stable_states else max(w, 0)
# #                                      for p_idx, w in weights.items() if p_idx < network.state_size])
# #                 if states_max + externals_max < threshold_val:
# #                     empty_constraint = True
# #                     break
# #                 if all(weights[i] == 0 for i in weights.keys() if i >= network.state_size):
# #                     continue
# #                 external_weights_list = [weights.get(i, 0) for i in
# #                                          range(network.state_size, network.state_size + network.external_size)]
# #                 new_constraint = (external_weights_list, round(threshold_val - states_max))
# #                 if new_constraint not in constraints:
# #                     constraints.append(new_constraint)
# #             if len(constraints) > 0 and not empty_constraint:
# #                 new_constraints = (0, constraints)
# #                 if new_constraints not in all_region_constraints:
# #                     all_region_constraints.append(new_constraints)
# #         elif stable_states[s_idx] == 1:
# #             for weights, threshold_val in threshold_funcs:
# #                 externals_min = np.sum([min(w, 0) for p_idx, w in weights.items()
# #                                         if p_idx >= network.state_size])
# #                 states_min = np.sum([w * stable_states[p_idx] if p_idx in stable_states else min(w, 0)
# #                                      for p_idx, w in weights.items() if p_idx < network.state_size])
# #                 if states_min + externals_min < threshold_val:
# #                     external_weights_list = [weights.get(i, 0) for i in
# #                                              range(network.state_size, network.state_size + network.external_size)]
# #                     new_constraint = (1, [(external_weights_list, round(threshold_val - states_min))])
# #                     if new_constraint not in all_region_constraints:
# #                         all_region_constraints.append(new_constraint)
# #
# #     constraints_to_groups = split_externals_constraints_to_groups(all_region_constraints)
# #     return constraints_to_groups
#
#
# def get_all_external_constraints2(network, stable_states):
#     all_region_constraints = []
#     for s_idx in range(network.state_size):
#         threshold_funcs = network.threshold_functions[s_idx]
#         empty_constraint = False
#         if s_idx in stable_states:
#             if stable_states[s_idx] == 0:
#                 constraints = []
#                 for weights, threshold_val in threshold_funcs:
#                     externals_max = np.sum([max(w, 0) for p_idx, w in weights.items() if p_idx >= network.state_size])
#                     states_max = np.sum([w * stable_states[p_idx] if p_idx in stable_states else max(w, 0)
#                                          for p_idx, w in weights.items() if p_idx < network.state_size])
#                     if states_max + externals_max < threshold_val:
#                         empty_constraint = True
#                         break
#                     if all(weights[i] == 0 for i in weights.keys() if i >= network.state_size):
#                         continue
#                     external_weights_list = [weights.get(i, 0) for i in
#                                              range(network.state_size, network.state_size + network.external_size)]
#                     new_constraint = (external_weights_list, round(threshold_val - states_max))
#                     if new_constraint not in constraints:
#                         constraints.append(new_constraint)
#                 if len(constraints) > 0 and not empty_constraint:
#                     new_constraints = (0, constraints)
#                     if new_constraints not in all_region_constraints:
#                         all_region_constraints.append(new_constraints)
#             elif stable_states[s_idx] == 1:
#                 for weights, threshold_val in threshold_funcs:
#                     externals_min = np.sum([min(w, 0) for p_idx, w in weights.items() if p_idx >= network.state_size])
#                     states_min = np.sum([w * stable_states[p_idx] if p_idx in stable_states else min(w, 0)
#                                          for p_idx, w in weights.items() if p_idx < network.state_size])
#                     if states_min + externals_min < threshold_val:
#                         external_weights_list = [weights.get(i, 0) for i in
#                                                  range(network.state_size, network.state_size + network.external_size)]
#                         new_constraint = (1, [(external_weights_list, round(threshold_val - states_min))])
#                         if new_constraint not in all_region_constraints:
#                             all_region_constraints.append(new_constraint)
#         else:
#             constraints = []
#             # empty_constraint = False
#             for weights, threshold_val in threshold_funcs:
#                 # add constraint that it not 0 stable
#                 if all(weights[i] == 0 for i in weights.keys() if i >= network.state_size):
#                     # empty_constraint = True
#                     continue
#                 external_weights_list = [weights.get(i, 0) for i in
#                                          range(network.state_size, network.state_size + network.external_size)]
#                 states_max = np.sum([w * stable_states[p_idx] if p_idx in stable_states
#                                      else (0 if p_idx == s_idx else max(w, 0))
#                                      for p_idx, w in weights.items() if p_idx < network.state_size])
#                 new_constraint = (2, [(external_weights_list, round(threshold_val - states_max))])
#                 if new_constraint not in all_region_constraints:
#                     all_region_constraints.append(new_constraint)
#
#                 # add constraint that it not 1 stable
#                 states_min = np.sum([w * stable_states[p_idx] if p_idx in stable_states
#                                      else (w if p_idx == s_idx else min(w, 0))
#                                      for p_idx, w in weights.items() if p_idx < network.state_size])
#                 new_constraint = (external_weights_list, round(threshold_val - states_min))
#                 if new_constraint not in constraints:
#                     constraints.append(new_constraint)
#             if len(constraints) > 0:  # and not empty_constraint:
#                 new_constraints = (3, constraints)
#                 if new_constraints not in all_region_constraints:
#                     all_region_constraints.append(new_constraints)
#
#     constraints_to_groups = split_externals_constraints_to_groups(all_region_constraints)
#     return constraints_to_groups
#
#
# def find_all_possible_externals(network, constraint_list):
#     external_size = network.external_size
#     model = gp.Model()
#     model.setParam('OutputFlag', 0)
#     external_vars = model.addVars(external_size, vtype=gp.GRB.BINARY, name=f"external")
#     # add stable states constraints
#     for i, (val, c) in enumerate(constraint_list):
#         if val == 0:
#             if len(c) > 1:
#                 zero_constraint = model.addVars(len(c), vtype=gp.GRB.BINARY, name=f"c0_{i}")
#                 for j in range(len(c)):
#                     w, t = c[j]
#                     expr = gp.quicksum(w[i] * external_vars[i] for i in range(external_size) if w[i] != 0)
#                     model.addConstr(expr - t + 1 - M * (1 - zero_constraint[j]) <= 0)
#                     model.addConstr(expr - t + M * zero_constraint[j] >= 0)
#                 res = model.addVar(vtype=gp.GRB.BINARY, name=f"res_{i}")
#                 model.addConstr(res == gp.or_(zero_constraint))
#                 model.addConstr(res >= 1)
#             if len(c) == 1:
#                 w, t = c[0]
#                 model.addConstr(
#                     gp.quicksum(w[i] * external_vars[i] for i in range(external_size) if w[i] != 0) <= t - 1)
#         elif val == 1:
#             for w, t in c:
#                 model.addConstr(gp.quicksum(w[i] * external_vars[i] for i in range(external_size) if w[i] != 0) >= t)
#         elif val == 2:
#             for w, t in c:
#                 model.addConstr(gp.quicksum(w[i] * external_vars[i] for i in range(external_size) if w[i] != 0) >= t)
#         elif val == 3:
#             if len(c) > 1:
#                 zero_constraint = model.addVars(len(c), vtype=gp.GRB.BINARY, name=f"c0_{i}")
#                 for j in range(len(c)):
#                     w, t = c[j]
#                     expr = gp.quicksum(w[i] * external_vars[i] for i in range(external_size) if w[i] != 0)
#                     model.addConstr(expr - t + 1 - M * (1 - zero_constraint[j]) <= 0)
#                     model.addConstr(expr - t + M * zero_constraint[j] >= 0)
#                 res = model.addVar(vtype=gp.GRB.BINARY, name=f"res_{i}")
#                 model.addConstr(res == gp.or_(zero_constraint))
#                 model.addConstr(res >= 1)
#             if len(c) == 1:
#                 w, t = c[0]
#                 model.addConstr(
#                     gp.quicksum(w[i] * external_vars[i] for i in range(external_size) if w[i] != 0) <= t - 1)
#
#     not_null_externals = list(set([i for _, c in constraint_list for i in range(external_size)
#                                    if any(cc[0][i] != 0 for cc in c)]))
#     model.addConstrs(external_vars[i] <= 0 for i in range(external_size) if i not in not_null_externals)
#     possible_external_list = []
#     model.setParam("PoolSearchMode", 2)
#     model.setParam("PoolSolutions", 2 ** 30)
#     model.optimize()
#     if model.status != gp.GRB.OPTIMAL:
#         [print(x) for x in constraint_list]
#         print("wrong!!")
#     if model.status == gp.GRB.OPTIMAL:
#         for i in range(model.solCount):
#             all_vars = model.getVars()
#             model.setParam("SolutionNumber", i)
#             possible_external_list.append(np.array([round(all_vars[i].Xn) if i in not_null_externals else -1
#                                                     for i in range(external_size)]))
#     model.dispose()
#     return possible_external_list
#
#     # external_size = network.external_size
#     # total_assignments = []
#     # for i, constraint_list in enumerate(constraints_to_groups):
#     #     group_assignments = find_all_possible_externals_per_constraint(external_size, constraints_to_groups[i])
#     #     total_assignments.append(group_assignments)
#
#     # total_possible_external_list = np.array([-1 for _ in range(external_size)])
#     # for i, assignments in enumerate(total_assignments):
#     #     size = len(assignments)
#     #     total_possible_external_list = np.tile(total_possible_external_list, size).reshape(-1, external_size)
#     #     for j, a in enumerate(assignments):
#     #         for k, e in enumerate(external_groups[i]):
#     #             total_possible_external_list[np.arange(j, len(total_possible_external_list), size), e] = a[k]
#     # a=1
#     # return total_possible_external_list
#
#
# def split_externals_constraints_to_groups(constraint_list):
#     external_per_constraints = [[j for i in range(len(c[1])) for j in range(len(c[1][i][0]))
#                                  if c[1][i][0][j] != 0] for c in constraint_list]
#     sublist_in_group = []
#     groups = []
#     for i in range(len(external_per_constraints)):
#         if i in sublist_in_group:
#             continue
#         updated = True
#         group_elements = [x for x in external_per_constraints[i]]
#         group_lists_idx = [i]
#         sublist_in_group.append(i)
#         while updated:
#             updated = False
#             for j in range(len(external_per_constraints)):
#                 if j in group_lists_idx:
#                     continue
#                 if any(e in group_elements for e in external_per_constraints[j]):
#                     group_lists_idx.append(j)
#                     group_elements = list(set(group_elements + external_per_constraints[j]))
#                     sublist_in_group.append(j)
#                     updated = True
#         groups.append(group_lists_idx)
#
#     constraints_for_group = [[constraint_list[i] for i in g] for g in groups]
#     return constraints_for_group
#
#
# def validate_all_possible_externals(network, states_nodes, all_possible_externals):
#     stable_nodes_assignment = {network.index_to_name[idx]: val for idx, val in states_nodes.items()}
#     for assignment in product((0, 1), repeat=len(all_possible_externals[0])):
#         not_null_external_assignment = {n: assignment[i] for i, n in enumerate(all_possible_externals[0])}
#         null_external_assignment = {network.index_to_name[i + network.state_size]: 0
#                                     for i in range(network.external_size) if
#                                     network.index_to_name[i + network.state_size] not in all_possible_externals[0]}
#         all_assignment = {**stable_nodes_assignment, **not_null_external_assignment, **null_external_assignment}
#         is_solution = True
#         for s_idx in states_nodes.keys():
#             res = network.nodes[network.index_to_name[s_idx]].assign_values_to_boolean_function(all_assignment)
#             if res != states_nodes[s_idx]:
#                 is_solution = False
#                 break
#         if not_null_external_assignment in all_possible_externals:
#             if not is_solution:
#                 print("Problem 1")
#         else:
#             if is_solution:
#                 tmp = [[x for x in not_null_external_assignment
#                         if not_null_external_assignment[x] != all_possible_externals[i][x]]
#                        for i in range(len(all_possible_externals))]
#                 print(tmp)
#                 print(f"Problem 2")
#
#
# def validate_external(network, stable_states, assignment):
#     is_solution = True
#     for s_idx in stable_states.keys():
#         res = network.nodes[network.index_to_name[s_idx]].assign_values_to_boolean_function(assignment)
#         if res != stable_states[s_idx]:
#             is_solution = False
#             break
#     return is_solution
