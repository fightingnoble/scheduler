from ortools.sat.python import cp_model

class BinPackingCPSolver:
    def __init__(self):
        self.data = self.create_data_model()
        self.model = cp_model.CpModel()
        self.b = []
        self.max_height = None

    @staticmethod
    def create_data_model():
        """Create the data for the example."""
        data = {}
        weights = [48, 30, 19, 36, 36, 27, 42, 42, 36, 24, 30]
        data["weights"] = weights
        data["items"] = list(range(len(weights)))
        data["bin_capacity"] = sum(weights) 
        return data

    def create_variables(self):
        # Variables
        for i in self.data["items"]:
            self.b.append(self.model.NewIntVar(0, self.data["bin_capacity"], f'b[{i}]'))

        # Define the auxiliary variable for the maximum height
        self.max_height = self.model.NewIntVar(0, self.data["bin_capacity"], "max_height")

    def define_constraints(self):
        # Constraints
        # Ensure the rectangle containment:
        # If item i is placed in bin j, items cannot exceed the bin size in height
        for i in self.data["items"]:
            self.model.Add(self.b[i] + self.data["weights"][i] <= self.max_height)

        for i in self.data["items"]:
            for j in range(i+1, len(self.data["items"])):
                xor_expr = self.model.NewBoolVar(f'xor_{i}_{j}')
                self.model.Add(self.b[i] + self.data["weights"][i] <= self.b[j]).OnlyEnforceIf(xor_expr)
                self.model.Add(self.b[j] + self.data["weights"][j] <= self.b[i]).OnlyEnforceIf(xor_expr.Not())

    def init_a_solution(self, bundle_ij):
        for i, j in bundle_ij:
            self.model.Add(self.b[i] == j)

    def solve(self):
        # Objective: minimize the highest point of the packing
        self.model.Minimize(self.max_height)

        solver = cp_model.CpSolver()
        status = solver.Solve(self.model)

        if status == cp_model.OPTIMAL:
            highest_point = solver.Value(self.max_height)
            print("Highest point of the packing:", highest_point)
            for i in self.data["items"]:
                print(f'Item {i} is placed at {solver.Value(self.b[i])}')
        else:
            print("The problem solution is not feasible.")

    def run(self):
        self.create_variables()
        self.define_constraints()
        self.solve()

if __name__ == "__main__":
    bin_packing_solver = BinPackingCPSolver()
    import numpy as np
    pos = np.cumsum(bin_packing_solver.data['weights']).tolist()
    pos = [0] + pos[:-1]

    bin_packing_solver.create_variables()
    bin_packing_solver.define_constraints()
    # bin_packing_solver.init_a_solution(zip(range(len(pos)), pos))
    bin_packing_solver.solve()
    # bin_packing_solver.run()
