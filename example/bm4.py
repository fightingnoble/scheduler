task_graph_sinks={
    "O1": [],
    "O2": [],
}

task_graph_srcs = {
    "S1": ["D"],
    "S2": ["C"],
    "S3": ["A"],
}

task_graph_ops = {   
    "A": ["B"],
    "B": ["O2"],
    "C": ["D",],
    "D": ["O1"], 
}

# record the exp_comp_t of ops, and offset of srcs
case0 = {
    "S1": 0, # 30hz
    "S2": 0, # 10hz
    "S3": 0, # 20hz
    "A": 20,
    "B": 25,
    "C": 15,
    "D": 45,
} # ideal case

case1 = {
    "S1": 30,
    "S2": 0,
    "S3": 10,
    "A": 38,
    "B": 54,
    "C": 22,
    "D": 50,
} # A unexpacted case

case = case1
src_attr={
    "S1": {"offset":0, "exp_comp_t":case["S1"], "tgt_device": "sen_p0", "base_size": 1, "comp_ratio": 0, "freq": 1/100},
    "S2": {"offset":0, "exp_comp_t":case["S2"], "tgt_device": "sen_p0", "base_size": 1, "comp_ratio": 0, "freq": 1/100},
    "S3": {"offset":0, "exp_comp_t":case["S3"], "tgt_device": "sen_p0", "base_size": 1, "comp_ratio": 0, "freq": 1/100},
}

task_attr = {
    "A": {"offset":0, "exp_comp_t": case["A"], "base_size": 25, "tgt_device": "acc_p0", "ert": 0, "ddl": 40, "var_factor": 1},
    "B": {"offset":0, "exp_comp_t": case["B"], "base_size": 25, "tgt_device": "acc_p0", "ert": 40, "ddl": 100, "var_factor": 1},
    "C": {"offset":0, "exp_comp_t": case["C"], "base_size": 25, "tgt_device": "acc_p0", "ert": 0, "ddl": 30, "var_factor": 1},
    "D": {"offset":0, "exp_comp_t": case["D"], "base_size": 25, "tgt_device": "acc_p0", "ert": 30, "ddl": 100, "var_factor": 1},
}

sink_attr = {
    "O1": {"ddl": 100},
    "O2": {"ddl": 100},
}

tot_core = 50
swt_lat = 5