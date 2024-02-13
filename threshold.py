import gurobipy as gp
import numpy as np


def get_parents_unate_sign(assignment_len, unate_dict):
    sign = []
    for p_idx in range(assignment_len):
        if p_idx in unate_dict['unate']:
            sign.append(1)
        elif p_idx in unate_dict['anti_unate']:
            sign.append(-1)
        elif p_idx in unate_dict['static']:
            sign.append(0)
        else:
            raise "error"
    return sign


def threshold_func_res(threshold_func, assignment):
    res = [np.sum(weights[p] * assignment[p] for p in range(len(assignment))) >= t
           for weights, t in threshold_func]
    return all(res)


def validate_threshold_function(truth_table, threshold_func):
    counter_false = 0
    for assignment, truth_val in truth_table():
        if truth_val != threshold_func_res(threshold_func, assignment):
            counter_false += 1
    return counter_false


def convert_boolean_to_second_order_threshold_function(assignment_len, parents_sign, truth_table):
    # Build model
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    threshold_first = model.addVar(vtype=gp.GRB.INTEGER, name="threshold", lb=-gp.GRB.INFINITY)
    weights_first = model.addVars(assignment_len, vtype=gp.GRB.INTEGER, name="weight_first")
    threshold_second = model.addVar(vtype=gp.GRB.INTEGER, name="threshold", lb=-gp.GRB.INFINITY)
    weights_second = model.addVars(assignment_len, vtype=gp.GRB.INTEGER, name="weight_second")
    model.setObjective(gp.quicksum(weights_first) + gp.quicksum(weights_second), gp.GRB.MINIMIZE)

    model.addConstrs(weights_first[i] >= 0 for i in range(assignment_len))
    model.addConstrs(weights_second[i] >= 0 for i in range(assignment_len))

    for assignment, res in truth_table():
        threshold_hold_first = model.addVar(vtype=gp.GRB.BINARY, name="threshold_hold")
        threshold_hold_second = model.addVar(vtype=gp.GRB.BINARY, name="threshold_hold")
        if res:
            model.addConstr(
                gp.quicksum(assignment[p] * parents_sign[p] * weights_first[p] for p in range(assignment_len))
                >= threshold_first)
            model.addConstr(
                gp.quicksum(assignment[p] * parents_sign[p] * weights_second[p] for p in range(assignment_len))
                >= threshold_second)
        else:
            model.addConstr(
                gp.quicksum(assignment[p] * parents_sign[p] * weights_first[p] for p in range(assignment_len))
                <= threshold_first - 1 + 50_000 * threshold_hold_first)
            model.addConstr(
                gp.quicksum(assignment[p] * parents_sign[p] * weights_second[p] for p in range(assignment_len))
                <= threshold_second - 1 + 50_000 * threshold_hold_second)
            model.addConstr(threshold_hold_first + threshold_hold_second <= 1)
    model.update()
    model.optimize()
    if model.status != gp.GRB.OPTIMAL:
        return

    first_weights_to_parent = {p_idx: parents_sign[p_idx] * round(weights_first[p_idx].X) for p_idx in range(assignment_len)}
    threshold_first = round(threshold_first.X)
    second_weights_to_parent = {p_idx: parents_sign[p_idx] * round(weights_second[p_idx].X) for p_idx in range(assignment_len)}
    threshold_second = round(threshold_second.X)
    model.dispose()
    return [(first_weights_to_parent, threshold_first), (second_weights_to_parent, threshold_second)]


def convert_boolean_to_first_order_threshold_function(assignment_len, parents_sign, truth_table):
    # Build model
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    threshold = model.addVar(vtype=gp.GRB.INTEGER, name="threshold", lb=-gp.GRB.INFINITY)
    weights = model.addVars(assignment_len, vtype=gp.GRB.INTEGER, name="weight")
    model.setObjective(gp.quicksum(weights), gp.GRB.MINIMIZE)

    model.addConstrs(weights[i] >= 0 for i in range(assignment_len))
    for assignment, res in truth_table():
        if res:
            model.addConstr(
                gp.quicksum(assignment[p] * parents_sign[p] * weights[p] for p in range(assignment_len))
                >= threshold)
        else:
            model.addConstr(
                gp.quicksum(assignment[p] * parents_sign[p] * weights[p] for p in range(assignment_len))
                <= threshold - 1)
    model.update()
    model.optimize()
    if model.status != gp.GRB.OPTIMAL:
        return

    weights_to_parent = {p_idx: parents_sign[p_idx] * round(weights[p_idx].X) for p_idx in range(assignment_len)}
    res = [(weights_to_parent, round(threshold.X))]
    model.dispose()
    return res


def get_node_unate_dict(truth_table, assignment_len):
    truth_table_res = [1 if t[1] else 0 for t in truth_table()]
    unate_dict = {"unate": [], "anti_unate": [], "static": [], "both": []}
    for p_idx in range(assignment_len):
        unate, anti_unate = False, False
        for i in range(2 ** p_idx):
            for j in range(2 ** (assignment_len - p_idx - 1)):
                f0 = truth_table_res[2**(assignment_len - p_idx)*i + j]
                f1 = truth_table_res[2**(assignment_len - p_idx)*i + 2**(assignment_len - p_idx - 1) + j]
                if f0 < f1:
                    unate = True
                if f1 < f0:
                    anti_unate = True
                if unate and anti_unate:
                    break
            if unate and anti_unate:
                unate_dict["both"].append(p_idx)
                break
        if not unate and not anti_unate:
            unate_dict["static"].append(p_idx)
        elif unate and not anti_unate:
            unate_dict["unate"].append(p_idx)
        elif not unate and anti_unate:
            unate_dict["anti_unate"].append(p_idx)
    return unate_dict


def convert_boolean_truth_table_to_thresholds_functions(assignment_len, truth_table):
    unate_dict = get_node_unate_dict(truth_table, assignment_len)
    if len(unate_dict['both']) > 0:
        print("not unate")
        return None, None
    if len(unate_dict['unate']) + len(unate_dict['anti_unate']) == 0:
        weight = {i: 0 for i in range(assignment_len)}
        t = 0 if next(truth_table())[1] else 1
        return unate_dict, [(weight, t)]

    parents_sign = get_parents_unate_sign(assignment_len, unate_dict)
    first_order = convert_boolean_to_first_order_threshold_function(assignment_len, parents_sign, truth_table)
    if first_order is not None:
        # error = validate_threshold_function(truth_table, first_order)
        # if error != 0:
        #     raise error
        return unate_dict, first_order
    second_order = convert_boolean_to_second_order_threshold_function(assignment_len, parents_sign, truth_table)
    if second_order is not None:
        # error = validate_threshold_function(truth_table, second_order)
        # if error != 0:
        #     raise error
        return unate_dict, second_order
    print("not second order unate")
    return None, None


def convert_network_function_to_thresholds_functions(network):
    network_threshold_function = {}
    for state_idx in range(network.state_size):
        node = network.nodes[network.index_to_name[state_idx]]
        assignment_len = len(node.get_boolean_functions_free_symbols())
        if assignment_len > 15:
            continue
        truth_table = node.get_node_truth_table
        unate_dict = get_node_unate_dict(truth_table, assignment_len)
        if len(unate_dict['both']) > 0:
            continue
        parents_sign = get_parents_unate_sign(assignment_len, unate_dict)
        first_order = convert_boolean_to_first_order_threshold_function(assignment_len, parents_sign, truth_table)
        if first_order is not None:
            network_threshold_function[state_idx] = first_order
            continue
        second_order = convert_boolean_to_second_order_threshold_function(assignment_len, parents_sign, truth_table)
        if second_order is not None:
            network_threshold_function[state_idx] = second_order
            continue
        raise "not threshold!"

    np.save(f"Networks/{network.name}/thresholds_constraints.npy", network_threshold_function, allow_pickle=True)
    return network_threshold_function


def get_threshold_status_for_network(network):
    network_stats = {"no_unate": 0, "too_long": 0, "no_solution": 0, "first_order": 0, "second_order": 0}
    for node_name in network.state_nodes_names:
        node = network.nodes[node_name]
        unate_dict = node.get_node_unate_dict()
        if len(unate_dict['both']) > 0:
            network_stats["no_unate"] += 1
            continue
        parents = node.get_boolean_functions_free_symbols()
        if len(parents) > 15:
            network_stats["too_long"] += 1
            continue

        if convert_boolean_to_first_order_threshold_function(node, unate_dict) is not None:
            network_stats["first_order"] += 1
        elif convert_boolean_to_second_order_threshold_function(node, unate_dict) is not None:
            network_stats["second_order"] += 1
        else:
            network_stats["no_solution"] += 1

    np.save(f"Networks/{network.name}/thresholds_stats.npy", network_stats, allow_pickle=True)
