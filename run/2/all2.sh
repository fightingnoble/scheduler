bash run/aba_scalability_scan.sh tp     True > log/coalescing_scan/scan_aux_bin_26_full.log 2>&1&
# bash run/aba_scalability_scan.sh bin    True > log/coalescing_scan/scan_bin_x9_full.log 2>&1&
# bash run/aba_scalability_scan.sh cfg    True > log/coalescing_scan/scan_cfg_x9_bin_26.log 2>&1&
# bash run/aba_scalability_scan.sh ratioB True > log/coalescing_scan/scan_ratioB_x9_bin_26.log 2>&1&

wait
python -m analyze.stat_num_exec --root_dir coalescing_scan --folder_search_seq num_bins,cfg_n
python -m analyze.analyze_timing --output_dir log/coalescing_scan --root_dir coalescing_scan --index_seq cfg_n --folder_search_seq num_bins,cfg_n
# python -m analyze.analyze_timing --output_dir log/coalescing_scan/n_bins_4 --root_dir coalescing_scan/n_bins_4 --index_seq cfg_n
# python -m analyze.analyze_timing --output_dir log/coalescing_scan/n_bins_8 --root_dir coalescing_scan/n_bins_8 --index_seq cfg_n
python -m analyze.analyze_ctx_switch --output_dir log/coalescing_scan --root_dir coalescing_scan --folder_search_seq num_bins,cfg_n --filename min_core 
