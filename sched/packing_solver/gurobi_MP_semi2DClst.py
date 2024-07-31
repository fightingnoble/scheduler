import gurobipy as gp
from gurobipy import GRB
import sys
from utils import time_cnt
from typing import List, Dict, Tuple

class ClusterGurobiSolverSemi2D:
    """
    Problem define:
    constant:
        M: number of bins
        N: number of items to be placed
        K: number of items all ready placed
        J: Number of placement problem should be considered
        (N,) The size list of items to be placed
        (K, 2) The size and the bin selection of items already placed
        problems: J list, each element is a list of the index of the items to be placed
        duation: (J,) array, the duration of each placement problem
        affinity1: (N, M) preference to bin
        affinity2: (N, N) array, the affinity of items to be clustered in the same bin, sumation of each row is less than 1

    variable:
        Bin_usage: (J, M) array, the size of each bin at each round
        which bin the item is selected to be placed
            (2) $x[i][j]$ whose value is 1 if the item i is placed in bin j, and 0 otherwise.
        Bin_size: (M,) array, the max usage of all round for each bin
        tot_size: sum(max([Bins_size[:, m]]) for m in range(M))
        affinity_score: (N, N) array, the affinity score of each pair of items

    Constraints:
        (1) Each item must be in exactly one bin, $\sum_{i=1}^{N} x[i][j]=1$
    Objective:
        As even as possible: for each problem, 
        utilization of bin m = sum(sum (the items capacity in bin m) * duration_j)
        maximize the min utilization of bins
        affinity score as large as possible
    """
    def __init__(self, M, N, K, J, Items_tbd_size:Dict[int, int],
                 Items_done_size_sel:Dict[int, Tuple[int, int]], problems:List[List[int]],
                 duation:List[int], max_size:int, 
                 affinity1:Dict[Tuple[int, int], float], affinity2:Dict[Tuple[int, int], float]):
        self.M = M
        self.N = N
        self.K = K
        self.J = J
        self.Items_tbd_size:Dict[int, (int, int)] = Items_tbd_size
        self.Items_done_size_sel = Items_done_size_sel
        self.problems:List[List[int]] = problems
        self.duation:List[int] = duation
        self.max_size = max_size
        self.affinity1:Dict[Tuple[int, int], float] = affinity1
        self.affinity2:Dict[Tuple[int, int], float] = affinity2
        # dimension check
        assert len(self.Items_tbd_size) == self.N
        assert len(self.Items_done_size_sel) == self.K
        assert len(self.problems) == self.J
        assert len(self.duation) == self.J
        # model
        self.model = gp.Model("2DBP")
        self.used_size = {} # size of (J, M), the used size of each bin in each placement problem

        for j in range(self.J):
            for m in range(self.M):
                self.used_size[j, m] = 0

        for j in range(J):
            item_idx:List[int] = self.problems[j]
            # pop the items that are already placed in this bin, and add to the used size
            # print(item_idx)
            for i in list(item_idx):
                if i in self.Items_done_size_sel:
                    tgt_m = self.Items_done_size_sel[i][1]
                    self.used_size[j, tgt_m] += self.Items_done_size_sel[i][0]
                    item_idx.remove(i)
                    # print(f"remove {i}")
        self.util = {}
        self.usage = {}
        self.bin_size = {}
        self.max_constr = {}

    def create_variables(self):
        # Variables
        # x[i, j] = 1 if item i is packed in bin j.
        # self.Items_tbd_size.keys() combine with range(self.M)
        self.x = self.model.addVars(((i, m) for i in self.Items_tbd_size.keys() for m in range(self.M)), vtype=GRB.BINARY, name="x")
        
        for m in range(self.M):
            self.bin_size[m] = self.model.addVar(lb=0, ub=self.max_size, vtype=GRB.INTEGER, name=f'bin_{m}_size')
            self.util[m] = self.model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f'bin_{m}_util')
            for j in range(self.J):
                self.usage[j, m] = self.model.addVar(lb=0, ub=self.max_size, vtype=GRB.INTEGER, name=f'bin_{m}_usage_Prob{j}')
        
        self.tot_size = self.model.addVar(lb=0, ub=self.max_size, vtype=GRB.INTEGER, name="tot_size")
        self.affinity_score = self.model.addVar(lb=-self.N, ub=self.N, vtype=GRB.CONTINUOUS, name="affinity_score")
        
        self.min_util = self.model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f'min_util')
        self.max_util = self.model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f'max_util')
        
    def create_constraints(self):
        try:
            # Constraints
            # Each item must be in exactly one bin.
            for i in self.Items_tbd_size.keys():
                self.model.addConstr(
                    gp.quicksum([self.x[(i, m)] for m in range(self.M)]) == 1,
                    name=f'item_{i}_in_one_bin_constr')
            
            # define the usage of each bin in each placement problem
            for j in range(self.J):
                item_idx:List[int] = self.problems[j]
                for m in range(self.M):
                    self.model.addConstr(
                        gp.quicksum([self.x[i, m] * self.Items_tbd_size[i] for i in item_idx]) + self.used_size[j, m] == self.usage[j, m],
                        name=f'bin_{m}_usage_Prob{j}_constr')
            
            # define the bin size constraint
            for m in range(self.M):
                self.max_constr[m] = self.model.addGenConstrMax(self.bin_size[m], [self.usage[j, m] for j in range(self.J)]
                                        , name=f'bin_{m}_size_constr')
            self.model.addConstr(self.tot_size == gp.quicksum(self.bin_size[m] for m in range(self.M)), name="tot_size_constr")

            # calculate the affinity score
            if self.N > 0:
                self.model.addConstr(
                (gp.quicksum((self.x[n, m] * self.affinity1[n, m]) for n, m in self.affinity1.keys()) + \
                    gp.quicksum([self.x[n1, m] * self.x[n2, m] * self.affinity2[n1, n2] for m in range(self.M) for (n1, n2) in self.affinity2])/self.N) == self.affinity_score,
                    name="affinity_score")

            for m in range(self.M):
                    # sum([self.x[(i, m)] * self.Items_tbd_size[i] for i in item_idx]) + self.used_size[j][m]
                    self.model.addConstr(
                        self.util[m] == sum([self.usage[j][m] * self.duation[j] for j in range(self.J)])/sum(self.duation)/ self.Bins_size[m] ,
                                         name=f'bin_{m}_util')

            # add max constraint
            self.model.addGenConstrMin(self.min_util, self.util, name="min_util")
            self.model.addGenConstrMax(self.max_util, self.util, name="max_util")
        except Exception as e:
            print(f"Error adding constraint for bin {m}: {e}")

    @time_cnt("solve")
    def solve(self):
        # self.model.setObjective(self.tot_size, GRB.MINIMIZE) 
        self.model.setObjectiveN(self.tot_size, index=0, priority=2, name="min_tot_size")
        self.model.setObjectiveN(-self.affinity_score, index=1, priority=1, name="max_affinity_score")
        self.model.setObjectiveN(self.max_util-self.min_util, index=2, priority=0, name="util")
        try:
            # Optimize model
            self.model.optimize()
            status = self.model.Status
            print('Status: %g' % status)

            # build solution dic of (core, lat)
            # print(self.model.display())
            sel = {}
            if status == GRB.OPTIMAL:
                print('Obj: %g' % self.model.ObjVal)
                for pid in self.Items_tbd_size.keys():
                    for m in range(self.M):
                        if self.x[pid, m].X > 0:
                            sel[pid]=m
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
            # sel and bin_size
            return sel, [round(self.bin_size[m].X) for m in range(self.M)]

        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))

        except AttributeError:
            print('Encountered an attribute error')


if __name__ == "__main__":
    bin_packing_solver = ClusterGurobiSolverSemi2D()
    bin_packing_solver.create_variables()
    bin_packing_solver.define_constraints()
    bin_packing_solver.solve()
