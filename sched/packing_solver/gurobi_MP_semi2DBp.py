import gurobipy as gp
from gurobipy import GRB
import sys
from utils import time_cnt
from typing import List, Dict, Tuple

class BinPackingGurobiSolverSemi2D:
    """
    Problem define:
    constant:
        M: number of bins
        N: number of items to be placed
        K: number of items all ready placed
        J: Number of placement problem should be considered
        Bins_size: (M,) array, the size of each bin
        (N,) The size list of items to be placed
        (K, 2) The size and the bin selection of items already placed
        problems: J list, each element is a list of the index of the items to be placed
        duation: (J,) array, the duration of each placement problem

    variable:
        which bin the item is selected to be placed
            (2) $x[i][j]$ whose value is 1 if the item i is placed in bin j, and 0 otherwise.
    Constraints:
        (1) Each item must be in exactly one bin, $\sum_{i=1}^{N} x[i][j]=1$
        (2) In each placement problem, the amount packed in each bin cannot exceed its capacity.
    Objective:
        As even as possible: for each problem, 
        utilization of bin m = sum(sum (the items capacity in bin m) * duration_j)
        maximize the min utilization of bins
    """
    def __init__(self, M, N, K, J, Bins_size, Items_tbd_size:Dict[int, int],
                 Items_done_size_sel:Dict[int, Tuple[int, int]], problems:List[List[int]],
                 duation:List[int]):
        self.M = M
        self.N = N
        self.K = K
        self.J = J
        self.Bins_size:Dict[int, int] = Bins_size
        self.Items_tbd_size:Dict[int, (int, int)] = Items_tbd_size
        self.Items_done_size_sel = Items_done_size_sel
        self.problems:List[List[int]] = problems
        self.duation = duation
        # dimension check
        assert len(self.Bins_size) == self.M
        assert len(self.Items_tbd_size) == self.N
        assert len(self.Items_done_size_sel) == self.K
        assert len(self.problems) == self.J
        assert len(self.duation) == self.J
        # model
        self.model = gp.Model("2DBP")
        self.used_size = [] # size of (J, M), the used size of each bin in each placement problem

        for j in range(J):
            item_idx:List[int] = self.problems[j]
            # pop the items that are already placed in this bin, and add to the used size
            used_size_j = []
            for m in range(M):
                used_size_j_m = 0
                for i in item_idx:
                    if i in self.Items_done_size_sel:
                        if self.Items_done_size_sel[i][1] == m:
                            used_size_j_m += self.Items_done_size_sel[i][0]
                            item_idx.remove(i)
                used_size_j.append(used_size_j_m)
            self.used_size.append(used_size_j)
        self.x = {}
        self.util = []
        self.usage = []
        

    def create_variables(self):
        # Variables
        # x[i, j] = 1 if item i is packed in bin j.
        for i in self.Items_tbd_size.keys():
            for m in range(self.M):
                self.x[(i, m)] = self.model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f'x_{i}_{m}')
        
        # for m in range(self.M):
        #     self.util.append(self.model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f'bin_{m}_util'))

        # for j in range(self.J):
        #     self.usage.append(self.model.addVars(self.M, lb=0, ub=self.Bins_size[m], vtype=GRB.INTEGER, name=f'bin_{j}_usage'))
        
        # self.min_util = self.model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f'min_util')
        # self.max_util = self.model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f'max_util')
        
    def create_constraints(self):
        # Constraints
        # Each item must be in exactly one bin.
        for i in self.Items_tbd_size.keys():
            self.model.addConstr(
                # gp.quicksum(self.x[(i, j)] for j in range(self.M)) == 1, 
                gp.quicksum([self.x[(i, m)] for m in range(self.M)]) == 1,
                name=f'item_{i}_in_one_bin')
        
        # In each placement problem, the amount packed in each bin cannot exceed its capacity.
        for j in range(self.J):
            item_idx:List[int] = self.problems[j]
            for m in range(self.M):
                self.model.addConstr(
                    # gp.quicksum([self.x[(i, m)] * self.Items_tbd_size[i] for i in item_idx]) + self.used_size[j][m] <= self.Bins_size[m], 
                    gp.quicksum([self.x[(i, m)] * self.Items_tbd_size[i] for i in item_idx]) + self.used_size[j][m] <= self.Bins_size[m],
                    name=f'bin_{m}_capacity')
                # self.model.addConstr(
                #     self.usage[j][m] == gp.quicksum([self.x[(i, m)] * self.Items_tbd_size[i] for i in item_idx]) + self.used_size[j][m],
                #     name=f'bin_{m}_usage')
    
        # for m in range(self.M):
        #     sum([self.x[(i, m)] * self.Items_tbd_size[i] for i in item_idx]) + self.used_size[j][m]
        #     self.model.addConstr(
        #         self.util[m] == sum([self.usage[j][m] * self.duation[j] / self.Bins_size[m] for j in range(self.J)])/sum(self.duation),
        #                          name=f'bin_{m}_util')

        # add max constraint
        # self.model.addGenConstrMax(self.min_util, self.util, name="min_util")
        # self.model.addGenConstrMax(self.max_util, self.util, name="max_util")

    @time_cnt("solve")
    def solve(self):
        # Objective: minimize the max utilization of bins
        # self.model.setObjective(self.min_util, GRB.MAXIMIZE)
        # self.model.setObjective(self.max_util, GRB.MINIMIZE)
        try:
            # Optimize model
            self.model.optimize()
            status = self.model.Status
            print('Status: %g' % status)

            # build solution dic of (core, lat)
            # print(self.model.display())
            sol = {}
            if status == GRB.OPTIMAL:
                print('Obj: %g' % self.model.ObjVal)
                for pid in self.Items_tbd_size.keys():
                    for m in range(self.M):
                        if self.x[(pid, m)].X > 0:
                            sol[pid]=m
            elif status == GRB.INF_OR_UNBD or status == GRB.INFEASIBLE:
                print('Optimization was stopped with status %d' % status)

                # do IIS
                print('The model is infeasible; computing IIS')
                removed = []

                # Loop until we reduce to a model that can be solved
                while True:
                    self.model.computeIIS()
                    print('\nThe following constraint cannot be satisfied:')
                    for c in self.model.getConstrs():
                        if c.IISConstr:
                            print('%s' % c.ConstrName)
                            # Remove a single constraint from the model
                            removed.append(str(c.ConstrName))
                            self.model.remove(c)
                            break
                    print('')
            return sol

        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))

        except AttributeError:
            print('Encountered an attribute error')


if __name__ == "__main__":
    bin_packing_solver = BinPackingGurobiSolverSemi2D()
    bin_packing_solver.create_variables()
    bin_packing_solver.define_constraints()
    bin_packing_solver.solve()
