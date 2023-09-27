from docplex.mp.model import Model
from utils import time_cnt

class BinPackingDocplexSolver:
    def __init__(self):
        self.data = self.create_data_model()
        self.model = Model(name="1DBP")
        self.b = []

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
        # position of the bottom-left corner of the rectangle
        for i in self.data["items"]:
            self.b.append(self.model.integer_var(lb=0, ub=self.data["bin_capacity"], name=f'b[{i}]'))

        # indicator constraints for the overlap
        self.left_or_right = self.model.binary_var_matrix(self.data["items"], self.data["items"], name="left_or_right")

        # Define the auxiliary variable for the maximum height
        self.height = [self.model.integer_var(lb=0, ub=self.data["bin_capacity"], name=f'height[{i}]') for i in self.data["items"]]
        self.max_height = self.model.integer_var(lb=0, ub=self.data["bin_capacity"], name="max_height")

    def define_constraints(self):
        # Constraints
        # Ensure the rectangle containment:
        # If item i is placed in bin j, items cannot exceed the bin size in height
        for i in self.data["items"]:
            self.model.add_constraint(self.b[i] + self.data["weights"][i] <= self.data["bin_capacity"], ctname=f'height[{i}]')

        for i in self.data["items"]:
            for j in range(i + 1, len(self.data["items"])):

                # self.model.addGenConstrIndicator(self.left_or_right[i, j], True, self.b[i] + self.data["weights"][i] <= self.b[j])
                # self.model.addGenConstrIndicator(self.left_or_right[i, j], False, self.b[j] + self.data["weights"][j] <= self.b[i])
                # self.model.add_indicator(self.left_or_right[i, j], self.b[i] + self.data["weights"][i] <= self.b[j], active_value=1, name=f'left[{i},{j}]')
                # self.model.add_indicator(self.left_or_right[i, j], self.b[j] + self.data["weights"][j] <= self.b[i], active_value=0, name=f'right[{i},{j}]')
                # Big-M relaxation
                self.model.add_constraint(self.b[i] + self.data["weights"][i] - self.data["bin_capacity"] * self.left_or_right[i, j] <= self.b[j])
                self.model.add_constraint(self.b[j] + self.data["weights"][j] - self.data["bin_capacity"] * (1 - self.left_or_right[i, j]) <= self.b[i])


        self.model.add_constraints(self.height[i] == (self.b[i] + self.data["weights"][i]) for i in self.data["items"])
        self.model.add_constraint(self.max_height == self.model.max(self.height))

    @time_cnt("solve")
    def solve(self):
        # Objective: minimize the highest point of the packing
        self.model.minimize(self.max_height)

        try:
            # Optimize model
            solution = self.model.solve()

            for v in self.model.iter_integer_vars():
                print(f'{v.get_name()} {v.solution_value}')

            print(f'Obj: {solution.get_objective_value()}')

        except Exception as e:
            print(f'Error: {str(e)}')

if __name__ == "__main__":
    bin_packing_solver = BinPackingDocplexSolver()
    bin_packing_solver.create_variables()
    bin_packing_solver.define_constraints()
    bin_packing_solver.solve()
