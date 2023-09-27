from ortools.linear_solver import pywraplp


class BinPackingMPSolver:
    def __init__(self):
        self.data = self.create_data_model()
        self.solver = pywraplp.Solver.CreateSolver("SCIP")

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
        self.b = []
        for i in self.data["items"]:
            self.b.append(self.solver.IntVar(0, self.data["bin_capacity"], f'b[{i}]'))

        # big-M relaxation and auxiliary variables
        self.z = {}
        for i in self.data["items"]:
            for j in range(i+1, len(self.data["items"])):
                for l in range(2):
                    self.z[i, j, l] = self.solver.IntVar(0, 1, f'z[{i},{j},{l}]'
                    )
        # Define the auxiliary variable for the maximum height
        self.max_height = self.solver.IntVar(0, self.data["bin_capacity"], "max_height")
    def define_constraints(self):
        # Constraints
        # Ensure the rectangle containment: 
        # if item i is placed in bin j, items cannot exceed the bin size in height
        for i in self.data["items"]:
            self.solver.Add(
                self.b[i] + self.data["weights"][i] <= self.max_height
                )

        for i in self.data["items"]:
            for j in range(i+1, len(self.data["items"])):
                self.solver.Add(
                    self.b[i] + self.data["weights"][i] - self.data["bin_capacity"] * self.z[i, j, 0] <= self.b[j]
                )
                self.solver.Add(
                    self.b[j] + self.data["weights"][j]  - self.data["bin_capacity"] * self.z[i, j, 1] <= self.b[i] 
                )
                self.solver.Add(
                    self.z[i, j, 0] + self.z[i, j, 1] == 1
                )
    
    def init_a_solution(self, bundle_ij):
        for i, j in bundle_ij:
            self.solver.SetHint([self.b[i]], [j])


    def solve(self):
        # Objective: minimize the highest point of the packing
        self.solver.Minimize(self.max_height)

        status = self.solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            highest_point = self.max_height.solution_value()
            print("Highest point of the packing:", highest_point)
            for i in self.data["items"]:
                print(f'Item {i} is placed at {self.b[i].solution_value()}')
            # print z
            for i in self.data["items"]:
                for j in range(i+1, len(self.data["items"])):
                        print(f'z[{i},{j},{0}] = {self.z[i,j,0].solution_value()}, z[{i},{j},{1}] = {self.z[i,j,1].solution_value()}')
        else:
            print("The problem solution is not feasible.")

    def run(self):
        self.create_variables()
        self.define_constraints()
        self.solve()


if __name__ == "__main__":
    bin_packing_solver = BinPackingMPSolver()
    import numpy as np
    pos = np.cumsum(bin_packing_solver.data['weights']).tolist()
    pos = [0] + pos[:-1]
    
    bin_packing_solver.create_variables()
    bin_packing_solver.define_constraints()
    # bin_packing_solver.init_a_solution(zip(range(len(pos)), pos))
    bin_packing_solver.solve()
    # bin_packing_solver.run()
