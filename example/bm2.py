task_graph_sinks={
    "O1": [],
    "O2": [],
}

task_graph_srcs = {
    "S1": ["A"],
    "S2": ["B"],
    "S3": ["C"],
}

task_graph_ops = {   
    "A": ["O1"],
    "B": ["C"],
    "C": ["D",],
    "D": ["O2"], 
}

# record the exp_lat of ops, and offset of srcs
case0 = {
    "S1": 60,
    "S2": 60,
    "S3": 10,
    "A": 60,
    "B": 20,
    "C": 25,
    "D": 15,
}
case1 = {
    "S1": 20,
    "S2": 20,
    "S3": 10,
    "A": 60,
    "B": 20,
    "C": 25,
    "D": 15,
}

case = case1
src_attr={
    "S1": case["S1"], 
    "S2": case["S2"],
    "S3": case["S3"],
}

task_attr = {
    "A": {"ert": 0, "ddl": 140, "exp_lat": case["A"], "base_size": 30},
    "B": {"ert": 0, "ddl": 60, "exp_lat": case["B"], "base_size": 20},
    "C": {"ert": 60, "ddl": 110, "exp_lat": case["C"], "base_size": 20},
    "D": {"ert": 110, "ddl": 140, "exp_lat": case["D"], "base_size": 20},
}

tot_core = 50
swt_lat = 5