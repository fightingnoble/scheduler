nohup python allocator_agent.py --test_case glb_dynamic --jitter_sim_en --num_cores 266 > glb_dyn_266_jitter_en.log.txt 2>&1 &
nohup python allocator_agent.py --test_case dynamic --jitter_sim_en --num_cores 266 > dyn_266_jitter_en.log.txt 2>&1 &
nohup python allocator_agent.py --test_case glb_dynamic --num_cores 266 > glb_dyn_266_jitter_dis.log.txt 2>&1 &
nohup python allocator_agent.py --test_case dynamic --num_cores 266 > dyn_266_jitter_dis.log.txt 2>&1 &