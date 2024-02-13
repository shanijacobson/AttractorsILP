import itertools
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import reachability


def get_included_external(solutions, external_id, bigger_external_id):
    external_list = solutions.external_assignments[external_id]
    bigger_external_list = solutions.external_assignments[bigger_external_id]
    external_size = external_list.shape[-1]
    not_null_externals = np.argwhere(bigger_external_list[0] != -1).reshape(-1)
    external_full_assignment = external_list
    for i in not_null_externals:
        if external_list[0][i] == -1:
            external_full_assignment = np.tile(external_full_assignment, 2).reshape(-1, external_size)
            external_full_assignment[np.arange(0, len(external_full_assignment), 2), i] = 0
            external_full_assignment[np.arange(1, len(external_full_assignment), 2), i] = 1
    unique_external_1, inverse_idx = np.unique(external_full_assignment[:, not_null_externals], axis=0, return_inverse=True)
    unique_external_1_encoded = [e.tobytes() for e in unique_external_1]
    external_2_encoded = [e[not_null_externals].tobytes() for e in bigger_external_list]
    _, idx, _ = np.intersect1d(unique_external_1_encoded, external_2_encoded, return_indices=True)
    return external_full_assignment[np.where([x in idx for x in inverse_idx])[0]]


def build_solutions_hierarchy_tree(network, solutions):
    edges_list = defaultdict(list)
    included_externals = {}
    ordered_solution = sorted(list(solutions.solutions.values()), key=lambda s: len(s.stable_nodes))
    for i, solution in tqdm(enumerate(ordered_solution)):
        if len(solution.stable_nodes) == network.state_size:
            break
        external_id = solutions.solution_to_externals[solution.solution_id]
        for bigger_solution in ordered_solution[i + 1: len(ordered_solution)]:
            if len(bigger_solution.stable_nodes) == len(solution.stable_nodes):
                continue
            if all(item in bigger_solution.stable_nodes.items() for item in solution.stable_nodes.items()):
                bigger_external_id = solutions.solution_to_externals[bigger_solution.solution_id]
                if external_id == bigger_external_id:
                    edges_list[solution.solution_id].append(bigger_solution.solution_id)
                    continue
                key = (external_id, bigger_external_id)
                if key not in included_externals:
                    included_externals[key] = get_included_external(solutions, external_id, bigger_external_id)
                if len(included_externals[key]) != 0:
                    edges_list[solution.solution_id].append(bigger_solution.solution_id)
    solutions.all_included_solutions = edges_list.copy()
    solutions.all_included_externals = included_externals
    return solutions


def check_if_included(network, solutions, solution_id, included_solution, externals, done, is_verify_sub_solutions):
    if not is_verify_sub_solutions:
        solutions.solutions[solution_id].mark_as_included_solution()
        return
    tmp = [x for x in included_solution if x in solutions.all_included_solutions
           and len(solutions.all_included_solutions) > 0]
    tmp = [xx for x in tmp for xx in solutions.all_included_solutions[x]]
    included_idx = [x for x in solutions.all_included_solutions[solution_id] if x not in tmp]
    included_solutions = [solutions.solutions[x] for x in included_idx]
    stable_nodes = solutions.solutions[solution_id].stable_nodes
    exploration_functions = reachability.get_reduced_threshold_functions(network, stable_nodes, externals, included_solutions)
    key = str(sorted(exploration_functions.items()))
    if key in done:
        res = done[key]
    else:
        res = reachability.check_if_reachable(solutions.solutions[solution_id], included_solutions, exploration_functions)
        done[key] = res

    if res <= 0:
        solutions.solutions[solution_id].mark_as_included_solution()


def remove_included_solutions(network, solutions, is_verify_sub_solutions):
    original_solution_size = len(solutions.solutions)
    solutions = build_solutions_hierarchy_tree(network, solutions)
    done = {}
    for solution_id in solutions.all_included_solutions:
        if len(solutions.all_included_solutions[solution_id]) == 0:
            continue
        external_id = solutions.solution_to_externals[solution_id]
        not_stable_state = [idx for idx in range(network.state_size)
                            if idx not in solutions.solutions[solution_id].stable_nodes]
        not_empty_externals = sorted(list(set([k for i in not_stable_state for f, _ in network.threshold_functions[i]
                                               for k in f.keys() if k >= network.state_size])))
        not_empty_externals = np.array(not_empty_externals) - network.state_size
        external_list = solutions.external_assignments[external_id]
        if len(not_empty_externals) == 0:
            included_solution = solutions.all_included_solutions[solution_id]
            externals = {}
            check_if_included(network, solutions, solution_id, included_solution, externals, done, is_verify_sub_solutions)
            continue
        relevant_external_list, original_idx = np.unique(external_list[:, not_empty_externals], axis=0, return_inverse=True)
        not_included_externals = np.array([], dtype=int).reshape(-1, network.external_size)
        for i, e_l in enumerate(relevant_external_list):
            included_solutions_per_external = []
            for s_id in solutions.all_included_solutions[solution_id]:
                key = (s_id, solution_id)
                if key not in solutions.all_included_externals:
                    included_solutions_per_external.append(s_id)
                else:
                    included_external = solutions.all_included_externals[key][:, not_empty_externals]
                    if len(included_external) > 0:
                        if any(np.array_equal(e_l[e >= 0], e[e >= 0]) for e in included_external):
                            included_solutions_per_external.append(s_id)
            if len(included_solutions_per_external) == 0:
                not_included_externals = np.append(not_included_externals, external_list[original_idx == i], axis=0)
                continue
            externals = {node_idx: e_l[i] for i, node_idx in enumerate(not_empty_externals)}
            if solutions.solutions[solution_id]:
                check_if_included(network, solutions, solution_id, included_solutions_per_external, externals, done, is_verify_sub_solutions)
            solutions.solutions[solution_id].included_solution = True

        if 0 < len(not_included_externals) < len(original_idx):
            stable_nodes = solutions.solutions[solution_id].stable_nodes.copy()
            new_solution_id = solutions.add_solution(stable_nodes)
            new_external_id = solutions.add_externals_assignments(np.array(not_included_externals))
            solutions.update_external_to_solution(new_solution_id, new_external_id)
            print(f"new solution!!!: {solution_id} ----> {new_solution_id}")

    for s in list(solutions.solutions.keys()):
        if solutions.solutions[s].included_solution:
            solutions.solutions.pop(s)
    print(f"num of solution: {len(solutions.solutions)} / {original_solution_size}")
    print(f"not reachable: {[d for d in done.values() if d > 0]}")
    print(f"too big: {len([d for d in done.values() if d == -1])}")
    return solutions


# def add_null_external_solutions(network, solutions):
#     externals = np.array([np.frombuffer(ee, dtype=int) for ee in
#                           set([e.tobytes() for e_list in solutions.external_assignments.values() for e in e_list])])
#     not_empty_externals = np.array([i for i in range(network.external_size) if max(externals[:, i]) > -1])
#     new_external = externals[:, not_empty_externals]
#     null_externals = []
#     print(f"null: {network.external_size - len(not_empty_externals)} / {network.external_size}")
#     for ibinary in tqdm(itertools.product((0, 1), repeat=len(not_empty_externals))):
#         ibinary = np.array(ibinary)
#         if not np.any([np.all(ibinary[e >= 0] == e[e >= 0]) for e in new_external]):
#             null_externals.append(ibinary)
#
#     if len(null_externals) > 0:
#         print("ibinary")
#         # solution_id = solutions.add_solution(stable_nodes={})
#         # external_id = solutions.add_externals_assignments(np.array(null_externals))
#         # solutions.update_external_to_solution(solution_id, external_id)
#     return solutions
