import numpy as np


class SolutionObjects(object):

    def __init__(self):
        self.solutions = {}
        self.external_assignments = {}
        self.int_externals = {}
        self.solution_to_externals = {}
        self.solution_id_counter = 0
        self.external_id_counter = 0
        self.solutions_hierarchy_tree = {}
        self.all_included_solutions = {}
        self.all_included_externals = {}
        self.included_solutions = {}

    def add_solution(self, stable_nodes):
        solution = TrapSpace(self.solution_id_counter, stable_nodes)
        self.solutions[solution.solution_id] = solution
        self.solution_id_counter = self.solution_id_counter + 1
        return solution.solution_id

    def remove_solution(self, solution_id):
        self.solutions.pop(solution_id)
        self.solution_to_externals.pop(solution_id)

    def remove_external(self, external_id):
        solution_for_external = [sid for sid in self.solutions if self.solution_to_externals[sid] == external_id]
        if len(solution_for_external) != 0:
            print(f"error: {external_id} has {len(solution_for_external)} related solutions.")
            print(solution_for_external)
            return
        self.external_assignments.pop(external_id)

    def update_external_to_solution(self, solution_id, externals_id):
        self.solution_to_externals[solution_id] = externals_id

    def add_externals_assignments(self, externals_assignments):
        external_id = self.external_id_counter
        self.external_assignments[external_id] = np.array(externals_assignments)
        self.external_id_counter += 1
        return external_id

    def get_not_null_externals_for_solution(self, solution_id):
        externals_assignment = self.external_assignments[self.solution_to_externals[solution_id]]
        if len(externals_assignment) == 0:
            return externals_assignment
        return [i for i, v in enumerate(externals_assignment[0]) if v != -1]

    def get_external_assignment_for_solution(self, solution_id):
        external_id = self.solution_to_externals[solution_id]
        return self.external_assignments[external_id]


class TrapSpace(object):
    def __init__(self, solution_id, stable_nodes):
        self.solution_id = solution_id
        self.stable_nodes = stable_nodes
        self.included_solution = False

    def mark_as_included_solution(self):
        self.included_solution = True
