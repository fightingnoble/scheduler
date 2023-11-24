mkdir -p log/barycenter/core_scan
mkdir -p log/barycenter/aux_scan
mkdir -p log/barycenter/lat_scan

echo "========================start (barycenter) $cfg/$x" `date "+%Y-%m-%d %H:%M:%S.%3N"` "========================" 

# nohup ./run/sweep_seed.sh > log/barycenter/seed_scan.log.txt 2>&1&
nohup ./run/sweep_seed.sh 0 9 True > log/barycenter/seed_scan.log.txt 2>&1&
wait

python -m analyze.stat_num_exec --root_dir barycenter/core_scan --folder_search_seq cfg_n,num_cores
python -m analyze.stat_num_exec --root_dir barycenter/aux_scan --folder_search_seq num_cores,cfg_n 

python -m analyze.analyze_tp --output_dir log/barycenter --root_dir barycenter/aux_scan --folder_search_seq num_cores,cfg_n 
python -m analyze.analyze_ctx_switch --output_dir log/barycenter --root_dir barycenter/core_scan --folder_search_seq cfg_n,num_cores --filename min_core 
# python -m analyze.analyze_timing --output_dir log/barycenter --root_dir barycenter/core_scan --index_seq cfg_n,num_cores
# python -m analyze.analyze_tp --output_dir log/barycenter --root_dir barycenter/lat_scan --filename tp_w_fx_n_core
# wait

python analyze/xlsl_max_tp.py --root_dir_dyn log/barycenter --root_dir_glb log/barycenter
python analyze/xlsl_min_core.py --root_dir_dyn log/barycenter --root_dir_glb log/barycenter
# python analyze/xlsl_e2e_latency.py --root_dir_dyn log/barycenter --root_dir_glb log/barycenter
# python analyze/xlsl_safe_scalable.py --root_dir_dyn log/barycenter --root_dir_glb log/barycenter

wait


