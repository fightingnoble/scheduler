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

GLB_BUFFER_SIZE = 40E6 # Bytes
GLB_BUFFER_SIZE_PER_CORE = 156.25E3 # Bytes
BW_DRAM = 100E9 # Bytes/s
LAT_PER_HOP = 10e-9 # Seconds
AVG_HOP_NUM = 10
MIN_CORE_NUM = 256

W_perc = 1/3
A_perc = 1/3
O_perc = 1/3