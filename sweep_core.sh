#!/bin/bash
for x in $(seq 196 10 256); do
    nohup python allocator_agent.py --test_case glb_dynamic --num_cores $x > glb_dynamic_sched_$x.log.txt 2>&1 &
done