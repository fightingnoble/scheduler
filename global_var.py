FLOPS_PER_CORE = 0.5

overhead_pushpull_per_core = {0.6:83, 0.8:87,} # cycles
overhead_of_enqueuing_op = 1000 # cycles
overhead_of_dequeuing_op = 1000 # cycles

clock_period = 1e-9 # Seconds
SRAM_size_per_core = 1.25 # MB

trace_file = "trace.txt"
trace_list = []

numerical_error_tol_abs = 1e-12
numerical_tol_bit = 12
numerical_error_tol_rel = 0.01

flop_error_tol_abs = 5e-7
flop_error_tol_bit = 7

fork_pid_base = 1000
max_fork_candi_num = 20
max_fork_pid = 100
