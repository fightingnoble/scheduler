# nohup python allocator_agent.py  --test_case bin_pack --num_cores 256 --n_p 3 > log/bin_pack_256.log.txt 2>&1
# nohup python allocator_agent.py  --test_case bin_pack --num_cores 266 --n_p 3 > log/bin_pack_266.log.txt 2>&1

nohup python allocator_agent.py --test_case glb_dynamic --num_cores 266 --n_p 3 > log/glb_dyn_266_jitter_dis.log.txt 2>&1 &
nohup python allocator_agent.py --test_case glb_dynamic --jitter_sim_en --num_cores 266 --file_suffix var_0.2 --n_p 3 > log/glb_dyn_266_jitter_en.log.txt 2>&1 &
nohup python allocator_agent.py --test_case dynamic --num_cores 266 --n_p 3 > log/dyn_266_jitter_dis.log.txt 2>&1 &
nohup python allocator_agent.py --test_case dynamic --jitter_sim_en --num_cores 266 --file_suffix var_0.2 --n_p 3 > log/dyn_266_jitter_en.log.txt 2>&1 &

nohup python allocator_agent.py --test_case glb_dynamic --num_cores 256 --n_p 3 > log/glb_dyn_256_jitter_dis.log.txt 2>&1 &
nohup python allocator_agent.py --test_case glb_dynamic --jitter_sim_en --file_suffix var_0.2 --num_cores 256 --n_p 3 > log/glb_dyn_256_jitter_en.log.txt 2>&1 &
nohup python allocator_agent.py --test_case dynamic --num_cores 256 --n_p 3 > log/dyn_256_jitter_dis.log.txt 2>&1 &
nohup python allocator_agent.py --test_case dynamic --jitter_sim_en --file_suffix var_0.2 --num_cores 256 --n_p 3 > log/dyn_256_jitter_en.log.txt 2>&1 &
