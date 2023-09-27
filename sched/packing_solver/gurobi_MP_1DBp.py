import gurobipy as gp
from gurobipy import GRB
import sys
from utils import time_cnt

class BinPackingGurobiSolver:
    def __init__(self):
        self.data = self.create_data_model()
        self.model = gp.Model("1DBP")
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
            self.b.append(self.model.addVar(lb=0, ub=self.data["bin_capacity"], vtype=GRB.INTEGER, name=f'b[{i}]'))

        # indicator constraints for the overlap
        # isLine = model.addVars(lines, vtype=GRB.BINARY, name="isLine")
        self.left_or_right = self.model.addVars(((i, j) for i in self.data["items"] for j in range(i+1, len(self.data["items"]))), vtype=GRB.BINARY, name="left_or_right")

        # Define the auxiliary variable for the maximum height
        self.max_height = self.model.addVar(lb=0, ub=self.data["bin_capacity"], vtype=GRB.INTEGER, name="max_height")
        self.height = self.model.addVars(self.data["items"], lb=0, ub=self.data["bin_capacity"], vtype=GRB.INTEGER, name="height")


    def define_constraints(self):
        # Constraints
        # Ensure the rectangle containment:
        # If item i is placed in bin j, items cannot exceed the bin size in height
        for i in self.data["items"]:
            self.model.addConstr(self.b[i] + self.data["weights"][i] <= self.data["bin_capacity"], name=f'height[{i}]')

        for i in self.data["items"]:
            for j in range(i+1, len(self.data["items"])):
                # self.model.addGenConstrIndicator(self.left_or_right[i, j], True, self.b[i] + self.data["weights"][i] <= self.b[j])
                # self.model.addGenConstrIndicator(self.left_or_right[i, j], False, self.b[j] + self.data["weights"][j] <= self.b[i])
                # Big-M relaxation
                self.model.addConstr(self.b[i] + self.data["weights"][i] - self.data["bin_capacity"] * self.left_or_right[i, j] <= self.b[j])
                self.model.addConstr(self.b[j] + self.data["weights"][j] - self.data["bin_capacity"] * (1 - self.left_or_right[i, j]) <= self.b[i])

        self.model.addConstrs(((self.height[i] == (self.b[i] + self.data["weights"][i])) for i in self.data["items"]), name="height")
        self.model.addGenConstrMax(self.max_height, self.height, name="max_height")

    @time_cnt("solve")
    def solve(self):
        # Objective: minimize the highest point of the packing
        self.model.setObjective(self.max_height, GRB.MINIMIZE)

        try:
            # Optimize model
            self.model.optimize()
            status = self.model.Status

            for v in self.model.getVars():
                print('%s %g' % (v.VarName, v.X))

            print('Obj: %g' % self.model.ObjVal)

        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))

        except AttributeError:
            print('Encountered an attribute error')


        # if self.model.status == GRB.OPTIMAL:
        #     highest_point = self.max_height.X
        #     print("Highest point of the packing:", highest_point)
        #     for i in self.data["items"]:
        #         print(f'Item {i} is placed at {self.b[i].X}')
        # else:
        #     print("The problem solution is not feasible.")

        # if status == GRB.UNBOUNDED:
        #     print('The model cannot be solved because it is unbounded')
        #     sys.exit(0)
        # if status == GRB.OPTIMAL:
        #     print('The optimal objective is %g' % self.model.ObjVal)
        #     sys.exit(0)
        # if status != GRB.INF_OR_UNBD and status != GRB.INFEASIBLE:
        #     print('Optimization was stopped with status %d' % status)
        #     sys.exit(0)

        # # do IIS
        # print('The model is infeasible; computing IIS')
        # removed = []

        # # Loop until we reduce to a model that can be solved
        # while True:

        #     self.model.computeIIS()
        #     print('\nThe following constraint cannot be satisfied:')
        #     for c in self.model.getConstrs():
        #         if c.IISConstr:
        #             print('%s' % c.ConstrName)
        #             # Remove a single constraint from the model
        #             removed.append(str(c.ConstrName))
        #             self.model.remove(c)
        #             break
        #     print('')

        #     self.model.optimize()
        #     status = self.model.Status

        #     if status == GRB.UNBOUNDED:
        #         print('The model cannot be solved because it is unbounded')
        #         sys.exit(0)
        #     if status == GRB.OPTIMAL:
        #         break
        #     if status != GRB.INF_OR_UNBD and status != GRB.INFEASIBLE:
        #         print('Optimization was stopped with status %d' % status)
        #         sys.exit(0)

        # print('\nThe following constraints were removed to get a feasible LP:')
        # print(removed)

        if status == GRB.UNBOUNDED:
            print('The model cannot be solved because it is unbounded')
            sys.exit(0)
        if status == GRB.OPTIMAL:
            print('The optimal objective is %g' % self.model.ObjVal)
            sys.exit(0)
        if status != GRB.INF_OR_UNBD and status != GRB.INFEASIBLE:
            print('Optimization was stopped with status %d' % status)
            sys.exit(0)

        # Relax the constraints to make the model feasible
        print('!!!!!!!!!!The model is infeasible; relaxing the constraints')
        orignumvars = self.model.NumVars
        self.model.feasRelaxS(0, False, False, True)
        self.model.optimize()

        status = self.model.Status
        if status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
            print('The relaxed model cannot be solved \
                because it is infeasible or unbounded')
            sys.exit(1)

        if status != GRB.OPTIMAL:
            print('Optimization was stopped with status %d' % status)
            sys.exit(1)

        print('\nSlack values:')
        slacks = self.model.getVars()[orignumvars:]
        for sv in slacks:
            if sv.X > 1e-6:
                print('%s = %g' % (sv.VarName, sv.X))
if __name__ == "__main__":
    bin_packing_solver = BinPackingGurobiSolver()
    bin_packing_solver.create_variables()
    bin_packing_solver.define_constraints()
    bin_packing_solver.solve()
