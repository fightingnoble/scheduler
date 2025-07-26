task_graph_sinks={
    "O1": [],
    "O2": [],
}

task_graph_srcs = {
    # "Entry": ["surr_view_camera_pub", "streo_camera_pub", "LiDAR_pub"],
    "S1": ["A"],
    "S2": ["C", "G", ],
}

# S1->A->B-> O1
# S1->A->E->F -> O2
# S2->C->D->E->F -> O2
# S2->G-> O2
task_graph_ops = {   
    "A": ["B", "E"],
    "B": ["O1"],
    "C": ["D",],
    "D": ["E"], 
    "E": ["F"],
    "F": ["O2"],
    "G": ["O2"],
}

# Chain 1	S1	A	B		
# Chian 2	S2	C	D	E	F
# Chain 3		G			
					
# Slack distribution (ddl-ert)					
# Chain 1	10.0 	70.0 	50.0 		
# Chian 2	10.0 	40.0 	30.0 	35.0 	15.0 
# Chain 3		    120.0 			
					
# DDL assignment (ddl)					
# Chain 1	10.0 	70.0 	120.0 		
# Chian 2	10.0 	40.0 	70.0 	105.0 	120.0 
# Chain 3		    120.0 			

# case 0:					
# Typical latency (exp_comp_t)					
# Chain 1	0.0 	40.0 	33.3 	0.0 	0.0 
# Chian 2	0.0 	20.0 	20.0 	23.3 	10.0 
# Chain 3		    73.3 	0.0 	0.0 	0.0 

# Case 1					
# Chain 1	30.0 	42.0 	33.3 	0.0 	0.0 
# Chian 2	20.0 	21.0 	20.0 	23.3 	10.0 
# Chain 3		80.0 	0.0 	0.0 	0.0 

# Case 2					
# Chain 1	30.0 	42.0 	50.0 	0.0 	0.0 
# Chian 2	20.0 	21.0 	25.0 	23.3 	10.0 
# Chain 3		100.0 	0.0 	0.0 	0.0 

# record the exp_comp_t of ops, and offset of srcs
case0 = {
    "S1": 0,
    "S2": 0,
    "A": 40,
    "B": 33.3,
    "C": 20,
    "D": 20,
    "E": 23.3,
    "F": 10,
    "G": 73.3,
}

case1 = {
    "S1": 30,
    "S2": 20,
    "A": 42, # ++
    "B": 33.3,
    "C": 21.0, # ++
    "D": 20.0,
    "E": 23.3,
    "F": 10,
    "G": 80, # ++
}
case2 = {
    "S1": 30,
    "S2": 20,
    "A": 42, # ++
    "B": 50, # ++
    "C": 21, # ++
    "D": 20, 
    "E": 23.3,
    "F": 10,
    "G": 100, # +++
}# longer B

case3 = {
    "S1": 30,
    "S2": 20,
    "A": 42, # ++
    "B": 33.3, 
    "C": 21, # ++
    "D": 25, # ++
    "E": 23.3,
    "F": 10,
    "G": 100, # +++
} # longer D

case4 = {
    "S1": 30,
    "S2": 20,
    "A": 42, # ++
    "B": 33.3, 
    "C": 21, # ++
    "D": 20, 
    "E": 23.3,
    "F": 10,
    "G": 110, # +++
}# longer g

case5 = {
    "S1": 30,
    "S2": 20,
    "A": 42, # ++
    "B": 40, # ++
    "C": 21, # ++
    "D": 20, 
    "E": 23.3,
    "F": 10,
    "G": 105, # +++
}# longer B

case = case5
src_attr={
    "S1": case["S1"], 
    "S2": case["S2"],
}

task_attr = {
    "A": {"ert": 0, "ddl": 70, "exp_comp_t": case["A"], "base_size": 20},
    "B": {"ert": 70, "ddl": 120, "exp_comp_t": case["B"], "base_size": 20},
    "C": {"ert": 0, "ddl": 40, "exp_comp_t": case["C"], "base_size": 10},
    "D": {"ert": 40, "ddl": 70, "exp_comp_t": case["D"], "base_size": 10},
    "E": {"ert": 70, "ddl": 105, "exp_comp_t": case["E"], "base_size": 10},
    "F": {"ert": 105, "ddl": 120, "exp_comp_t": case["F"], "base_size": 10},
    "G": {"ert": 0, "ddl": 120, "exp_comp_t": case["G"], "base_size": 10},
}

tot_core = 40 