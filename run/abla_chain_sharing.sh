#!/bin/bash
# set -x

PYASRGS="--profiling_filename profiling/profiling_light.csv --n_p 3 --root_dir chain_ablation --gen_benchmark --aux_scale_factor 1 --e2e_latency 0.1 --plot True"

RATE_CFG_2VAR="--exec_t_comp_ratioA 0.30 --exec_t_comp_ratioB 0.20 --wsc_slack_ratio 1 --bin_pack_cfg Bp_coalescing.json "
RATE_CFG_1VAR="--exec_t_comp_ratioA 0 --exec_t_comp_ratioB 0.20 --wsc_slack_ratio 1 --bin_pack_cfg Bp_coalescing.json "

CHAIN_SHARING_1VAR_CFG="--exec_t_comp_ratioA 0.05 --exec_t_comp_ratioB 0 --wsc_slack_ratio 0.9 --num_cores 300 --bin_pack_cfg Bp_reside.json"
CHAIN_SHARING_2VAR_CFG="--exec_t_comp_ratioA 0.3 --exec_t_comp_ratioB 0.20 --wsc_slack_ratio 0.65 --num_cores 300 --bin_pack_cfg Bp_reside.json"

WC_DYN_ARGS="--var_sim_cfg var_sim_cfg.json --seed -1 --file_suffix var_0.2 --jitter_sim_en"
UE_DYN_ARGS="--var_sim_cfg Pos_case_tmp_2var.json --seed -1 --file_suffix var_0.49_0.2 --jitter_sim_en --exec_var_en "

python ablation_main.py --test_case bin_pack_new $PYASRGS $RATE_CFG_2VAR &
python ablation_main.py --test_case bin_pack_new $PYASRGS $RATE_CFG_1VAR &
python ablation_main.py --test_case bin_pack_new $PYASRGS $CHAIN_SHARING_1VAR_CFG &
python ablation_main.py --test_case bin_pack_new $PYASRGS $CHAIN_SHARING_2VAR_CFG &
wait

# # isolated case
# python ablation_main.py --test_case cyclic $PYASRGS $RATE_CFG_2VAR &
# python ablation_main.py --test_case cyclic $PYASRGS $RATE_CFG_1VAR &
# python ablation_main.py --test_case cyclic $PYASRGS $RATE_CFG_2VAR $WC_DYN_ARGS --exec_var_en &
# python ablation_main.py --test_case cyclic $PYASRGS $RATE_CFG_1VAR $WC_DYN_ARGS &

# # shared case：glb
# python ablation_main.py --test_case glb_dynamic --num_cores 101 $PYASRGS &
# python ablation_main.py --test_case glb_dynamic --num_cores 101 $PYASRGS &
# python ablation_main.py --test_case glb_dynamic --num_cores 101 $PYASRGS $WC_DYN_ARGS --exec_var_en &
# python ablation_main.py --test_case glb_dynamic --num_cores 101 $PYASRGS $WC_DYN_ARGS --exec_var_en &

# # shared case：ours
# python ablation_main.py --test_case dynamic $PYASRGS $CHAIN_SHARING_1VAR_CFG $WC_DYN_ARGS &
# python ablation_main.py --test_case dynamic $PYASRGS $CHAIN_SHARING_2VAR_CFG $WC_DYN_ARGS --exec_var_en &

# wait
# # UE_DYN_ARGS
# python ablation_main.py --test_case dynamic $PYASRGS $CHAIN_SHARING_2VAR_CFG $UE_DYN_ARGS &
# python ablation_main.py --test_case cyclic $PYASRGS $RATE_CFG_2VAR $UE_DYN_ARGS &
# wait

