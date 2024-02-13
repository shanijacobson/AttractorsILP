import pickle
import sympy as sp
import threshold


class BooleanNetwork(object):
    def __init__(self, network_name, path=None):
        self.name = network_name
        self.nodes = {}
        self.edges = 0
        self.size = 0
        self.state_size = 0
        self.external_size = 0
        self.state_nodes_names = []
        self.external_nodes_names = []
        self.external_only_depended_nodes_names = []
        self.hole_nodes_names = []
        self.index_to_name = []
        self.threshold_functions = {}
        self.unate_dict = {}

        functions = {}
        tmp = f'Networks/{network_name}/Boolean_Functions/expressions_clean.txt' if path is None else path
        with open(tmp) as f:
            for line in f.readlines():
                name, func = line.split('\n')[0].split("*=")
                functions[name] = func

        for name in sorted(functions.keys()):
            node = BooleanNode(name, functions[name])
            self.edges += len(node.get_parents())
            self.nodes[name] = node
            if node.external:
                self.external_nodes_names.append(name)
            else:
                self.state_nodes_names.append(name)
            self.size += 1
        self.__updated_network()
        self.get_threshold_functions()
        self.save_network()

    def get_threshold_functions(self):
        for node_name in self.state_nodes_names + self.external_only_depended_nodes_names + self.hole_nodes_names:
            node = self.nodes[node_name]
            parents = node.get_parents()
            truth_table = self.nodes[node_name].get_node_truth_table
            unate_dict, threshold_func = threshold.convert_boolean_truth_table_to_thresholds_functions(len(parents), truth_table)
            if threshold_func is None:
                print(f"No threshold function to: {node_name}")
                print(node.name, len(node.get_parents()), node.boolean_function)
                continue
            update_threshold_func = [({self.nodes[parents[i]].index: val for i, val in w.items()}, t)
                                     for w, t in threshold_func]
            self.threshold_functions[node.index] = update_threshold_func
            self.unate_dict[node.index] = {k: [self.nodes[parents[i]].index for i in val] for k, val in unate_dict.items()}

    def save_network(self):
        with open(f'Networks/{self.name}/BooleanNetwork.pickle', 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __updated_network(self):
        self.index_to_name = []
        self.state_size = 0
        self.external_size = 0
        self.__delete_not_influence_nodes()
        # add index
        for node_name in self.state_nodes_names:
            node = self.nodes[node_name]
            node.boolean_function = node.boolean_function.simplify()
            node.index = self.state_size
            self.index_to_name.append(node.name)
            self.state_size += 1

        for name in self.external_nodes_names:
            self.nodes[name].index = self.external_size + self.state_size
            self.index_to_name.append(name)
            self.external_size += 1

        for name in sorted(self.hole_nodes_names):
            node = self.nodes[name]
            node.boolean_function = node.boolean_function.simplify()
            if all(p in self.state_nodes_names for p in node.get_parents()):
                self.external_nodes_names.append(name)
                self.hole_nodes_names.remove(name)

        idx = self.external_size + self.state_size
        for name in self.external_only_depended_nodes_names:
            node = self.nodes[name]
            node.index = idx
            node.boolean_function = node.boolean_function.simplify()
            self.index_to_name.append(name)
            idx += 1

        for name in self.hole_nodes_names:
            node = self.nodes[name]
            node.index = idx
            node.boolean_function = node.boolean_function.simplify()
            self.index_to_name.append(name)
            idx += 1

        possible_parents = self.state_nodes_names + self.external_nodes_names
        for node in self.nodes.values():
            if any(p not in possible_parents for p in node.get_parents()):
                print(node.index, node.name, "problem")

    def __delete_not_influence_nodes(self):
        while True:
            new_external_depended = self.__delete_external_only_depended()
            if len(new_external_depended) == 0:
                break
        if self.size > 300:
            while True:
                new_hole = self.__delete_hole_nodes()
                if len(new_hole) == 0:
                    break

    def __delete_hole_nodes(self):
        new_holes = []
        for name in self.state_nodes_names:
            state_successors = [s for s in self.get_updated_successors_for_node(name) if s in self.state_nodes_names]
            if len(state_successors) == 0:
                new_holes.append(name)

        for name in new_holes:
            successors_assignment = {name: self.nodes[name].boolean_function}
            for s in self.get_updated_successors_for_node(name):
                self.nodes[s].update_boolean_function(successors_assignment)
            self.state_nodes_names.remove(name)
            if all(p in self.external_nodes_names for p in self.nodes[name].get_parents()):
                self.external_nodes_names.append(name)
            elif len(self.nodes[name].get_parents()) == 0:
                self.external_nodes_names.append(name)
            else:
                self.hole_nodes_names.append(name)
        return new_holes

    def __delete_external_only_depended(self):
        external_only_depended_node = []
        for name in self.state_nodes_names:
            node = self.nodes[name]
            state_parents = [p for p in node.get_parents() if p in self.state_nodes_names]
            if len(state_parents) > 0:
                continue
            node_parents = [p for p in node.get_parents() if p in self.external_nodes_names] + state_parents
            state_successors = [s for s in self.get_updated_successors_for_node(name) if s in self.state_nodes_names]
            to_add = True
            for s in state_successors:
                successor_parents = self.nodes[s].get_parents()
                if len(set(successor_parents + node_parents)) > 16:
                    to_add = False
                    break
                if any(p in successor_parents for p in node_parents):
                    to_add = False
                    break
            if to_add:
                external_only_depended_node.append(name)

        for name in external_only_depended_node:
            if len(self.state_nodes_names) == 1:
                external_only_depended_node = []
                break
            successors_assignment = {name: self.nodes[name].boolean_function}
            for s in self.get_updated_successors_for_node(name):
                self.nodes[s].update_boolean_function(successors_assignment)
            self.state_nodes_names.remove(name)
            self.external_only_depended_nodes_names.append(name)
        return external_only_depended_node

    def get_updated_successors_for_node(self, node_name):
        successors_name = []
        for node in self.nodes.values():
            if node_name in node.get_parents():
                successors_name.append(node.name)
        return successors_name

    def display_network_threshold_function(self):
        print(f"{self.name}, total_size: {self.size}, state_size: {self.state_size}, "
              f"external_size: {self.external_size}")
        for s_idx, func in self.threshold_functions.items():
            for i in range(len(func)):
                display_func = "".join([f"{'+' if w > 0 else ''}{int(w)}"
                                        f"*{'s' if p_idx < self.state_size else 'e'}"
                                        f"[{p_idx}]"
                                        for p_idx, w in sorted(list(func[i][0].items())) if w != 0])
                print(f"s[{s_idx}]: {display_func} >= {func[i][1]}")


class BooleanNode(object):
    def __init__(self, name, boolean_function):
        self.index = None
        self.name = name.replace(" ", "")
        self.function_str = boolean_function
        self.boolean_function = sp.parsing.sympy_parser.parse_expr(boolean_function)
        self.original_boolean_function = self.boolean_function.simplify()
        self.boolean_function = self.original_boolean_function
        self.external = boolean_function.replace("(", "").replace(")", "").replace(" ", "") == self.name
        self.static = False\
            # self.assign_values_to_boolean_function({name: 0}) == 0 or \
            #           self.assign_values_to_boolean_function({name: 1}) == 1

    def get_boolean_functions_free_symbols(self, with_external=False):
        free_symbols = list(self.boolean_function.free_symbols)
        if not with_external and self.external:
            return []
        return [p.name for p in free_symbols]

    def get_parents(self):
        if isinstance(self.boolean_function, int):
            return []
        return sorted([n.name.replace(" ", "") for n in list(self.boolean_function.free_symbols)])

    def update_boolean_function(self, assignment):
        self.boolean_function = self.assign_values_to_boolean_function(assignment)

    def assign_values_to_boolean_function(self, assignment):
        val = self.boolean_function.subs(assignment)
        if len(val.free_symbols) == 0:
            return 1 if val else 0
        return val

    def get_node_truth_table(self):
        return sp.logic.boolalg.truth_table(self.boolean_function, self.get_parents())
