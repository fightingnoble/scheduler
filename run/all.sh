mkdir -p log/barycenter/core_scan
mkdir -p log/barycenter/aux_scan
mkdir -p log/barycenter/lat_scan

echo "========================start (barycenter) $cfg/$x" `date "+%Y-%m-%d %H:%M:%S.%3N"` "========================" 

# nohup ./run/exp_cmd_min_core.sh barycenter/core_scan True True True True 0 --bin_sort barycenter --plot > log/barycenter/core_scan/cmd.log.txt 2>&1
# nohup ./run/exp_cmd_max_tp.sh 0 25 200 0 1 9  True True True True barycenter/aux_scan 0 --bin_sort barycenter --plot > log/barycenter/aux_scan/cmd.log.txt 2>&1&
# nohup ./run/exp_cmd_max_tp.sh 0 25 200 0 1 9  True True True True barycenter/aux_scan 0 --lateness_mode all_soft --bin_sort barycenter --plot > log/barycenter/aux_scan/all_soft_cmd.log.txt 2>&1&
# nohup ./run/exp_cmd_ctx_switch.sh 0 25 0 0 1 16 True True True True barycenter/lat_scan 0 --bin_sort barycenter --plot > log/barycenter/lat_scan/cmd.log.txt 2>&1&
# nohup ./run/exp_cmd_ctx_switch.sh 0 25 0 0 1 16 True True True True barycenter/lat_scan 0 --lateness_mode all_soft --bin_sort barycenter --plot > log/barycenter/lat_scan/all_soft_cmd.log.txt 2>&1&

wait

# nohup ./run/sweep_seed.sh 1 9 > log/barycenter/seed_scan.log.txt 2>&1&
wait

# python -m analyze.analyze_tp --profiling_filename profiling/profiling_light.csv --root_dir barycenter/aux_scan --folder_search_seq num_cores,cfg_n --output_dir log/barycenter
# python -m analyze.analyze_timing --root_dir barycenter/core_scan --output_dir log/barycenter
# python -m analyze.analyze_ctx_switch --profiling_filename profiling/profiling_light.csv --root_dir barycenter/core_scan --folder_search_seq cfg_n,num_cores --filename min_core --output_dir log/barycenter
# python -m analyze.analyze_tp  --root_dir barycenter/lat_scan  --output_dir log/barycenter --filename tp_w_fx_n_core
wait

# python analyze/xlsl_min_core.py --root_dir_dyn log/barycenter --root_dir_glb log/barycenter
# python analyze/xlsl_e2e_latency.py --root_dir_dyn log/barycenter --root_dir_glb log/barycenter
# python analyze/xlsl_max_tp.py --root_dir_dyn log/barycenter --root_dir_glb log/barycenter
# python analyze/xlsl_safe_scalable.py --root_dir_dyn log/barycenter --root_dir_glb log/barycenter