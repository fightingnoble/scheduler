1. 纠正执行次数， 让所有执行次数都等于4
   1. 加入timestamp alignment
   2. Lidar_based_3dDet_0 is triggered twice by Semantic_segm_1 unintentionally.
2. dyn_sched 对于正在运行的任务，发生配置变化的时候，不能触发调度决策
3. input data prefetching, detailed mapping
4. get a convergent scheduling table in a hyperperiod
5. distinguish the tile flops, task flops, and flops on path
6. gen_event_modB, gen_event_modA 现在第一个输出就是有用的，不兼容预激活携程
7. Jitter_en 的flag 在trace和log中定义的不一样