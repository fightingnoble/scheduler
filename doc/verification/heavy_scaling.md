python -m sched.slack_estim --profiling_file profiling_light.csv --e2e_latency 0.09 --aux_scale_factor 6 > doc/verification/heavy_scaling.md
```
{'ImageBB': (73, 0.06556172289170492, None), 'MultiCameraFusion': (73, 0.009034737328295088, None), 'Pure_camera_path_head': (6, 0.007008, 'upb'), 'Prediction': (20, 0.00170769268, 'upb'), 'Planning': (2, 0.0017076926800000001, 'upb'), 'Steering_speed': (1, 0.00048015442, 'upb'), 'Stereo_feature_enc': (7, 0.055064126790920764, None), 'Semantic_segm': (7, 0.020066044077390342, None), 'Lidar_based_3dDet': (7, 0.006474289351688905, None), 'Traffic_light_detection': (6, 0.0855, None), 'Lane_drivable_area_det': (7, 0.03993587320907924, None), 'Optical_Flow': (4, 0.03993587320907926, None), 'Depth_estimation': (4, 0.03993587320907924, None)}
{'LiDAR_pub': 0, 'surr_view_camera_pub': 0, 'IMU_pub': 0, 'streo_camera_pub': 0, 'Traffic_light_detection': 0, 'ImageBB': 0, 'Stereo_feature_enc': 0, 'MultiCameraFusion': 0.06901233988600519, 'Semantic_segm': 0.057962238727285016, 'Lane_drivable_area_det': 0.057962238727285016, 'Optical_Flow': 0.057962238727285016, 'Depth_estimation': 0.057962238727285016, 'Pure_camera_path_head': 0.07852258970526318, 'Lidar_based_3dDet': 0.07908439038769591, 'Sink_screen': 0.10000000000000003, 'Prediction': 0.08589943181052634, 'Planning': 0.0876970030526316, 'Steering_speed': 0.08949457429473687, 'Sink_control': 0.09000000000000002} {'LiDAR_pub': 0, 'surr_view_camera_pub': 0, 'IMU_pub': 0, 'streo_camera_pub': 0, 'Traffic_light_detection': 0.09000000000000001, 'ImageBB': 0.06901233988600519, 'Stereo_feature_enc': 0.057962238727285016, 'MultiCameraFusion': 0.07852258970526318, 'Semantic_segm': 0.07908439038769591, 'Lane_drivable_area_det': 0.1, 'Optical_Flow': 0.10000000000000003, 'Depth_estimation': 0.1, 'Pure_camera_path_head': 0.08589943181052634, 'Lidar_based_3dDet': 0.08589943181052634, 'Sink_screen': 0.10000000000000003, 'Prediction': 0.0876970030526316, 'Planning': 0.08949457429473687, 'Steering_speed': 0.09000000000000002, 'Sink_control': 0.09000000000000002}
Traffic_light_detection TaskIntAttr(name='Traffic_light_detection', freq=30, timing_flag='deadline', criticality='hard', trigger_mode='event', core_max=1000.0, core_min=0, core_list=None, parallel_mode=None, core_max_compile=1000.0, core_min_compile=0, core_list_compile=None, thread_scaling_factor=1, freq_division_factor=3, var_factor=1, jitter_max=0, ERT=0, ddl=0.09000000000000001, exp_comp_t=0.0855, flops=0.23324, task_flag='stationary', pre_assigned_resource_flag=True, num_exec=3, no_stall_latency=0.07774666666666667, min_tot_rsc=18, max_tot_rsc=24, flops_ModelSum=0.69972, flops_max=0.69972, equiv_core=13.994399999999999, util=0.7774666666666666, main_size=6, rda_size=2)

ImageBB TaskIntAttr(name='ImageBB', freq=30, timing_flag='deadline', criticality='soft', trigger_mode='event', core_max=1000.0, core_min=0, core_list=None, parallel_mode=None, core_max_compile=1000.0, core_min_compile=0, core_list_compile=None, thread_scaling_factor=1, freq_division_factor=3, var_factor=1, jitter_max=0, ERT=0, ddl=0.06901233988600519, exp_comp_t=0.06556172289170492, flops=2.391886, task_flag='stationary', pre_assigned_resource_flag=True, num_exec=3, no_stall_latency=0.06553112328767123, min_tot_rsc=219, max_tot_rsc=261, flops_ModelSum=7.175658, flops_max=7.175658, equiv_core=143.51316, util=0.6553112328767123, main_size=73, rda_size=14)

MultiCameraFusion TaskIntAttr(name='MultiCameraFusion', freq=30, timing_flag='deadline', criticality='soft', trigger_mode='N', core_max=1000.0, core_min=0, core_list=None, parallel_mode=None, core_max_compile=1000.0, core_min_compile=0, core_list_compile=None, thread_scaling_factor=1, freq_division_factor=3, var_factor=1, jitter_max=0, ERT=0.06901233988600519, ddl=0.009510249819257993, exp_comp_t=0.009034737328295088, flops=0.32961399999999996, task_flag='stationary', pre_assigned_resource_flag=False, num_exec=3, no_stall_latency=0.009030520547945205, min_tot_rsc=219, max_tot_rsc=261, flops_ModelSum=0.9888419999999998, flops_max=0.9888419999999998, equiv_core=19.776839999999993, util=0.09030520547945202, main_size=73, rda_size=14)

Pure_camera_path_head TaskIntAttr(name='Pure_camera_path_head', freq=30, timing_flag='deadline', criticality='soft', trigger_mode='N', core_max=6, core_min=0, core_list=None, parallel_mode='upb', core_max_compile=6, core_min_compile=0, core_list_compile=None, thread_scaling_factor=1, freq_division_factor=3, var_factor=1, jitter_max=0, ERT=0.07852258970526318, ddl=0.007376842105263154, exp_comp_t=0.007008, flops=0.021024, task_flag='moveable', pre_assigned_resource_flag=False, num_exec=3, no_stall_latency=0.007008, min_tot_rsc=18, max_tot_rsc=18, flops_ModelSum=0.063072, flops_max=0.063072, equiv_core=1.26144, util=0.07007999999999999, main_size=6, rda_size=0)

Prediction TaskIntAttr(name='Prediction', freq=30, timing_flag='deadline', criticality='soft', trigger_mode='N', core_max=20, core_min=0, core_list=None, parallel_mode='upb', core_max_compile=20, core_min_compile=0, core_list_compile=None, thread_scaling_factor=1, freq_division_factor=3, var_factor=3, jitter_max=0, ERT=0.08589943181052634, ddl=0.0017975712421052642, exp_comp_t=0.00170769268, flops=0.0170769268, task_flag='moveable', pre_assigned_resource_flag=False, num_exec=3, no_stall_latency=0.00170769268, min_tot_rsc=60, max_tot_rsc=180, flops_ModelSum=0.051230780399999995, flops_max=0.15369234119999997, equiv_core=1.0246156079999997, util=0.017076926799999996, main_size=20, rda_size=0)

Planning TaskIntAttr(name='Planning', freq=30, timing_flag='deadline', criticality='hard', trigger_mode='N', core_max=2, core_min=0, core_list=None, parallel_mode='upb', core_max_compile=2, core_min_compile=0, core_list_compile=None, thread_scaling_factor=1, freq_division_factor=3, var_factor=1, jitter_max=0, ERT=0.0876970030526316, ddl=0.0017975712421052642, exp_comp_t=0.0017076926800000001, flops=0.0017076926800000001, task_flag='moveable', pre_assigned_resource_flag=False, num_exec=3, no_stall_latency=0.0017076926800000001, min_tot_rsc=6, max_tot_rsc=6, flops_ModelSum=0.00512307804, flops_max=0.00512307804, equiv_core=0.1024615608, util=0.0170769268, main_size=2, rda_size=0)

Steering_speed TaskIntAttr(name='Steering_speed', freq=240, timing_flag='deadline', criticality='hard', trigger_mode='event', core_max=2, core_min=0, core_list=None, parallel_mode='upb', core_max_compile=1, core_min_compile=0, core_list_compile=None, thread_scaling_factor=1, freq_division_factor=3, var_factor=1, jitter_max=0, ERT=0.08949457429473687, ddl=0.0005054257052631572, exp_comp_t=0.00048015442, flops=0.00024007721, task_flag='moveable', pre_assigned_resource_flag=False, num_exec=24, no_stall_latency=0.00048015442, min_tot_rsc=3, max_tot_rsc=6, flops_ModelSum=0.00576185304, flops_max=0.00576185304, equiv_core=0.11523706079999999, util=0.0384123536, main_size=1, rda_size=1)

Stereo_feature_enc TaskIntAttr(name='Stereo_feature_enc', freq=20, timing_flag='deadline', criticality='soft', trigger_mode='event', core_max=1000.0, core_min=0, core_list=None, parallel_mode=None, core_max_compile=1000.0, core_min_compile=0, core_list_compile=None, thread_scaling_factor=1, freq_division_factor=2, var_factor=1, jitter_max=0, ERT=0, ddl=0.057962238727285016, exp_comp_t=0.055064126790920764, flops=0.178674, task_flag='stationary', pre_assigned_resource_flag=False, num_exec=2, no_stall_latency=0.05104971428571429, min_tot_rsc=14, max_tot_rsc=18, flops_ModelSum=0.357348, flops_max=0.357348, equiv_core=7.14696, util=0.5104971428571429, main_size=7, rda_size=2)

Semantic_segm TaskIntAttr(name='Semantic_segm', freq=20, timing_flag='deadline', criticality='soft', trigger_mode='N', core_max=1000.0, core_min=0, core_list=None, parallel_mode=None, core_max_compile=1000.0, core_min_compile=0, core_list_compile=None, thread_scaling_factor=1, freq_division_factor=2, var_factor=1, jitter_max=0, ERT=0.057962238727285016, ddl=0.02112215166041089, exp_comp_t=0.020066044077390342, flops=0.065111, task_flag='stationary', pre_assigned_resource_flag=True, num_exec=2, no_stall_latency=0.018603142857142856, min_tot_rsc=14, max_tot_rsc=18, flops_ModelSum=0.130222, flops_max=0.130222, equiv_core=2.60444, util=0.18603142857142857, main_size=7, rda_size=2)

Lidar_based_3dDet TaskIntAttr(name='Lidar_based_3dDet', freq=10, timing_flag='deadline', criticality='soft', trigger_mode='event', core_max=1000.0, core_min=0, core_list=None, parallel_mode=None, core_max_compile=1000.0, core_min_compile=0, core_list_compile=None, thread_scaling_factor=1, freq_division_factor=1, var_factor=1, jitter_max=0, ERT=0.07908439038769591, ddl=0.006815041422830431, exp_comp_t=0.006474289351688905, flops=0.021008, task_flag='moveable', pre_assigned_resource_flag=False, num_exec=1, no_stall_latency=0.006002285714285714, min_tot_rsc=7, max_tot_rsc=9, flops_ModelSum=0.021008, flops_max=0.021008, equiv_core=0.42016, util=0.06002285714285714, main_size=7, rda_size=2)

Lane_drivable_area_det TaskIntAttr(name='Lane_drivable_area_det', freq=20, timing_flag='realtime', criticality='soft', trigger_mode='N', core_max=1000.0, core_min=0, core_list=None, parallel_mode=None, core_max_compile=1000.0, core_min_compile=0, core_list_compile=None, thread_scaling_factor=6, freq_division_factor=2, var_factor=1, jitter_max=0, ERT=0.057962238727285016, ddl=0.04203776127271499, exp_comp_t=0.03993587320907924, flops=0.1285636, task_flag='moveable', pre_assigned_resource_flag=True, num_exec=12, no_stall_latency=0.036732457142857146, min_tot_rsc=84, max_tot_rsc=84, flops_ModelSum=1.5427632, flops_max=1.5427632, equiv_core=30.855264, util=0.3673245714285714, main_size=7, rda_size=0)

Optical_Flow TaskIntAttr(name='Optical_Flow', freq=20, timing_flag='realtime', criticality='soft', trigger_mode='N', core_max=1000.0, core_min=0, core_list=None, parallel_mode=None, core_max_compile=1000.0, core_min_compile=0, core_list_compile=None, thread_scaling_factor=6, freq_division_factor=2, var_factor=1, jitter_max=0, ERT=0.057962238727285016, ddl=0.04203776127271502, exp_comp_t=0.03993587320907926, flops=0.079514, task_flag='moveable', pre_assigned_resource_flag=False, num_exec=12, no_stall_latency=0.039757, min_tot_rsc=48, max_tot_rsc=48, flops_ModelSum=0.9541679999999999, flops_max=0.9541679999999999, equiv_core=19.083359999999995, util=0.3975699999999999, main_size=4, rda_size=0)

Depth_estimation TaskIntAttr(name='Depth_estimation', freq=20, timing_flag='realtime', criticality='soft', trigger_mode='N', core_max=1000.0, core_min=0, core_list=None, parallel_mode=None, core_max_compile=1000.0, core_min_compile=0, core_list_compile=None, thread_scaling_factor=6, freq_division_factor=2, var_factor=1, jitter_max=0, ERT=0.057962238727285016, ddl=0.04203776127271499, exp_comp_t=0.03993587320907924, flops=0.0642818, task_flag='moveable', pre_assigned_resource_flag=False, num_exec=12, no_stall_latency=0.0321409, min_tot_rsc=48, max_tot_rsc=48, flops_ModelSum=0.7713816, flops_max=0.7713816, equiv_core=15.427632, util=0.321409, main_size=4, rda_size=0)

Traffic_light_detection_0_0 Task 0: Traffic_light_detection_0_0
	timing_flag: deadline, criticality: hard, trigger_mode: event
	num of cores: no constraint
	spatial factor: thread 1, freq 3, var 1
	slack info: 0.00e+00 ~ 9.00e-02 (8.55e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 2.33e-01, io_time: 1.00e-06, totcpu: 2.33e-01
	flops: 2.33e-01, req.: 6
	main_size: 6, RDA_size: 2, pre_assigned: True


Traffic_light_detection_0_1 Task 1: Traffic_light_detection_0_1
	timing_flag: deadline, criticality: hard, trigger_mode: event
	num of cores: no constraint
	spatial factor: thread 1, freq 3, var 1
	slack info: 0.00e+00 ~ 9.00e-02 (8.55e-02)
	period: 1.00e-01, i_offset: 3.33e-02
	jitter_max: 0
	cpu_time: 2.33e-01, io_time: 1.00e-06, totcpu: 2.33e-01
	flops: 2.33e-01, req.: 6
	main_size: 6, RDA_size: 2, pre_assigned: True


Traffic_light_detection_0_2 Task 2: Traffic_light_detection_0_2
	timing_flag: deadline, criticality: hard, trigger_mode: event
	num of cores: no constraint
	spatial factor: thread 1, freq 3, var 1
	slack info: 0.00e+00 ~ 9.00e-02 (8.55e-02)
	period: 1.00e-01, i_offset: 6.67e-02
	jitter_max: 0
	cpu_time: 2.33e-01, io_time: 1.00e-06, totcpu: 2.33e-01
	flops: 2.33e-01, req.: 6
	main_size: 6, RDA_size: 2, pre_assigned: True


ImageBB_0_0 Task 3: ImageBB_0_0
	timing_flag: deadline, criticality: soft, trigger_mode: event
	num of cores: no constraint
	spatial factor: thread 1, freq 3, var 1
	slack info: 0.00e+00 ~ 6.90e-02 (6.56e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 2.39e+00, io_time: 1.00e-06, totcpu: 2.39e+00
	flops: 2.39e+00, req.: 73
	main_size: 73, RDA_size: 14, pre_assigned: True


ImageBB_0_1 Task 4: ImageBB_0_1
	timing_flag: deadline, criticality: soft, trigger_mode: event
	num of cores: no constraint
	spatial factor: thread 1, freq 3, var 1
	slack info: 0.00e+00 ~ 6.90e-02 (6.56e-02)
	period: 1.00e-01, i_offset: 3.33e-02
	jitter_max: 0
	cpu_time: 2.39e+00, io_time: 1.00e-06, totcpu: 2.39e+00
	flops: 2.39e+00, req.: 73
	main_size: 73, RDA_size: 14, pre_assigned: True


ImageBB_0_2 Task 5: ImageBB_0_2
	timing_flag: deadline, criticality: soft, trigger_mode: event
	num of cores: no constraint
	spatial factor: thread 1, freq 3, var 1
	slack info: 0.00e+00 ~ 6.90e-02 (6.56e-02)
	period: 1.00e-01, i_offset: 6.67e-02
	jitter_max: 0
	cpu_time: 2.39e+00, io_time: 1.00e-06, totcpu: 2.39e+00
	flops: 2.39e+00, req.: 73
	main_size: 73, RDA_size: 14, pre_assigned: True


MultiCameraFusion_0_0 Task 6: MultiCameraFusion_0_0
	timing_flag: deadline, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 1, freq 3, var 1
	slack info: 6.90e-02 ~ 7.85e-02 (9.03e-03)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 3.30e-01, io_time: 1.00e-06, totcpu: 3.30e-01
	flops: 3.30e-01, req.: 73
	main_size: 0, RDA_size: 0, pre_assigned: False


MultiCameraFusion_0_1 Task 7: MultiCameraFusion_0_1
	timing_flag: deadline, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 1, freq 3, var 1
	slack info: 6.90e-02 ~ 7.85e-02 (9.03e-03)
	period: 1.00e-01, i_offset: 3.33e-02
	jitter_max: 0
	cpu_time: 3.30e-01, io_time: 1.00e-06, totcpu: 3.30e-01
	flops: 3.30e-01, req.: 73
	main_size: 0, RDA_size: 0, pre_assigned: False


MultiCameraFusion_0_2 Task 8: MultiCameraFusion_0_2
	timing_flag: deadline, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 1, freq 3, var 1
	slack info: 6.90e-02 ~ 7.85e-02 (9.03e-03)
	period: 1.00e-01, i_offset: 6.67e-02
	jitter_max: 0
	cpu_time: 3.30e-01, io_time: 1.00e-06, totcpu: 3.30e-01
	flops: 3.30e-01, req.: 73
	main_size: 0, RDA_size: 0, pre_assigned: False


Pure_camera_path_head_0_0 Task 9: Pure_camera_path_head_0_0
	timing_flag: deadline, criticality: soft, trigger_mode: N
	num of cores (compile): 1 ~ 6
	num of cores (run): 1 ~ 6
	spatial factor: thread 1, freq 3, var 1
	slack info: 7.85e-02 ~ 8.59e-02 (7.01e-03)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 2.10e-02, io_time: 1.00e-06, totcpu: 2.10e-02
	flops: 2.10e-02, req.: 6
	main_size: 0, RDA_size: 0, pre_assigned: False


Pure_camera_path_head_0_1 Task 10: Pure_camera_path_head_0_1
	timing_flag: deadline, criticality: soft, trigger_mode: N
	num of cores (compile): 1 ~ 6
	num of cores (run): 1 ~ 6
	spatial factor: thread 1, freq 3, var 1
	slack info: 7.85e-02 ~ 8.59e-02 (7.01e-03)
	period: 1.00e-01, i_offset: 3.33e-02
	jitter_max: 0
	cpu_time: 2.10e-02, io_time: 1.00e-06, totcpu: 2.10e-02
	flops: 2.10e-02, req.: 6
	main_size: 0, RDA_size: 0, pre_assigned: False


Pure_camera_path_head_0_2 Task 11: Pure_camera_path_head_0_2
	timing_flag: deadline, criticality: soft, trigger_mode: N
	num of cores (compile): 1 ~ 6
	num of cores (run): 1 ~ 6
	spatial factor: thread 1, freq 3, var 1
	slack info: 7.85e-02 ~ 8.59e-02 (7.01e-03)
	period: 1.00e-01, i_offset: 6.67e-02
	jitter_max: 0
	cpu_time: 2.10e-02, io_time: 1.00e-06, totcpu: 2.10e-02
	flops: 2.10e-02, req.: 6
	main_size: 0, RDA_size: 0, pre_assigned: False


Prediction_0_0 Task 12: Prediction_0_0
	timing_flag: deadline, criticality: soft, trigger_mode: N
	num of cores (compile): 1 ~ 20
	num of cores (run): 1 ~ 20
	spatial factor: thread 1, freq 3, var 3
	slack info: 8.59e-02 ~ 8.77e-02 (1.71e-03)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 1.71e-02, io_time: 1.00e-06, totcpu: 1.71e-02
	flops: 1.71e-02, req.: 20
	main_size: 0, RDA_size: 0, pre_assigned: False


Prediction_0_1 Task 13: Prediction_0_1
	timing_flag: deadline, criticality: soft, trigger_mode: N
	num of cores (compile): 1 ~ 20
	num of cores (run): 1 ~ 20
	spatial factor: thread 1, freq 3, var 3
	slack info: 8.59e-02 ~ 8.77e-02 (1.71e-03)
	period: 1.00e-01, i_offset: 3.33e-02
	jitter_max: 0
	cpu_time: 1.71e-02, io_time: 1.00e-06, totcpu: 1.71e-02
	flops: 1.71e-02, req.: 20
	main_size: 0, RDA_size: 0, pre_assigned: False


Prediction_0_2 Task 14: Prediction_0_2
	timing_flag: deadline, criticality: soft, trigger_mode: N
	num of cores (compile): 1 ~ 20
	num of cores (run): 1 ~ 20
	spatial factor: thread 1, freq 3, var 3
	slack info: 8.59e-02 ~ 8.77e-02 (1.71e-03)
	period: 1.00e-01, i_offset: 6.67e-02
	jitter_max: 0
	cpu_time: 1.71e-02, io_time: 1.00e-06, totcpu: 1.71e-02
	flops: 1.71e-02, req.: 20
	main_size: 0, RDA_size: 0, pre_assigned: False


Planning_0_0 Task 15: Planning_0_0
	timing_flag: deadline, criticality: hard, trigger_mode: N
	num of cores (compile): 1 ~ 2
	num of cores (run): 1 ~ 2
	spatial factor: thread 1, freq 3, var 1
	slack info: 8.77e-02 ~ 8.95e-02 (1.71e-03)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 1.71e-03, io_time: 1.00e-06, totcpu: 1.71e-03
	flops: 1.71e-03, req.: 2
	main_size: 0, RDA_size: 0, pre_assigned: False


Planning_0_1 Task 16: Planning_0_1
	timing_flag: deadline, criticality: hard, trigger_mode: N
	num of cores (compile): 1 ~ 2
	num of cores (run): 1 ~ 2
	spatial factor: thread 1, freq 3, var 1
	slack info: 8.77e-02 ~ 8.95e-02 (1.71e-03)
	period: 1.00e-01, i_offset: 3.33e-02
	jitter_max: 0
	cpu_time: 1.71e-03, io_time: 1.00e-06, totcpu: 1.71e-03
	flops: 1.71e-03, req.: 2
	main_size: 0, RDA_size: 0, pre_assigned: False


Planning_0_2 Task 17: Planning_0_2
	timing_flag: deadline, criticality: hard, trigger_mode: N
	num of cores (compile): 1 ~ 2
	num of cores (run): 1 ~ 2
	spatial factor: thread 1, freq 3, var 1
	slack info: 8.77e-02 ~ 8.95e-02 (1.71e-03)
	period: 1.00e-01, i_offset: 6.67e-02
	jitter_max: 0
	cpu_time: 1.71e-03, io_time: 1.00e-06, totcpu: 1.71e-03
	flops: 1.71e-03, req.: 2
	main_size: 0, RDA_size: 0, pre_assigned: False


Steering_speed_0_0 Task 18: Steering_speed_0_0
	timing_flag: deadline, criticality: hard, trigger_mode: event
	num of cores (compile): 1 ~ 1
	num of cores (run): 1 ~ 2
	spatial factor: thread 1, freq 3, var 1
	slack info: 8.95e-02 ~ 9.00e-02 (4.80e-04)
	period: 4.17e-03, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 2.40e-04, io_time: 1.00e-06, totcpu: 2.40e-04
	flops: 2.40e-04, req.: 1
	main_size: 0, RDA_size: 0, pre_assigned: False


Steering_speed_0_1 Task 19: Steering_speed_0_1
	timing_flag: deadline, criticality: hard, trigger_mode: event
	num of cores (compile): 1 ~ 1
	num of cores (run): 1 ~ 2
	spatial factor: thread 1, freq 3, var 1
	slack info: 8.95e-02 ~ 9.00e-02 (4.80e-04)
	period: 4.17e-03, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 2.40e-04, io_time: 1.00e-06, totcpu: 2.40e-04
	flops: 2.40e-04, req.: 1
	main_size: 0, RDA_size: 0, pre_assigned: False


Steering_speed_0_2 Task 20: Steering_speed_0_2
	timing_flag: deadline, criticality: hard, trigger_mode: event
	num of cores (compile): 1 ~ 1
	num of cores (run): 1 ~ 2
	spatial factor: thread 1, freq 3, var 1
	slack info: 8.95e-02 ~ 9.00e-02 (4.80e-04)
	period: 4.17e-03, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 2.40e-04, io_time: 1.00e-06, totcpu: 2.40e-04
	flops: 2.40e-04, req.: 1
	main_size: 0, RDA_size: 0, pre_assigned: False


Stereo_feature_enc_0_0 Task 21: Stereo_feature_enc_0_0
	timing_flag: deadline, criticality: soft, trigger_mode: event
	num of cores: no constraint
	spatial factor: thread 1, freq 2, var 1
	slack info: 0.00e+00 ~ 5.80e-02 (5.51e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 1.79e-01, io_time: 1.00e-06, totcpu: 1.79e-01
	flops: 1.79e-01, req.: 7
	main_size: 0, RDA_size: 0, pre_assigned: False


Stereo_feature_enc_0_1 Task 22: Stereo_feature_enc_0_1
	timing_flag: deadline, criticality: soft, trigger_mode: event
	num of cores: no constraint
	spatial factor: thread 1, freq 2, var 1
	slack info: 0.00e+00 ~ 5.80e-02 (5.51e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 1.79e-01, io_time: 1.00e-06, totcpu: 1.79e-01
	flops: 1.79e-01, req.: 7
	main_size: 0, RDA_size: 0, pre_assigned: False


Semantic_segm_0_0 Task 23: Semantic_segm_0_0
	timing_flag: deadline, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 1, freq 2, var 1
	slack info: 5.80e-02 ~ 7.91e-02 (2.01e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 6.51e-02, io_time: 1.00e-06, totcpu: 6.51e-02
	flops: 6.51e-02, req.: 7
	main_size: 7, RDA_size: 2, pre_assigned: True


Semantic_segm_0_1 Task 24: Semantic_segm_0_1
	timing_flag: deadline, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 1, freq 2, var 1
	slack info: 5.80e-02 ~ 7.91e-02 (2.01e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 6.51e-02, io_time: 1.00e-06, totcpu: 6.51e-02
	flops: 6.51e-02, req.: 7
	main_size: 7, RDA_size: 2, pre_assigned: True


Lidar_based_3dDet_0_0 Task 25: Lidar_based_3dDet_0_0
	timing_flag: deadline, criticality: soft, trigger_mode: event
	num of cores: no constraint
	spatial factor: thread 1, freq 1, var 1
	slack info: 7.91e-02 ~ 8.59e-02 (6.47e-03)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 2.10e-02, io_time: 1.00e-06, totcpu: 2.10e-02
	flops: 2.10e-02, req.: 7
	main_size: 0, RDA_size: 0, pre_assigned: False


Lane_drivable_area_det_0_0 Task 26: Lane_drivable_area_det_0_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 1.29e-01, io_time: 1.00e-06, totcpu: 1.29e-01
	flops: 1.29e-01, req.: 7
	main_size: 7, RDA_size: 0, pre_assigned: True


Lane_drivable_area_det_0_1 Task 27: Lane_drivable_area_det_0_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 1.29e-01, io_time: 1.00e-06, totcpu: 1.29e-01
	flops: 1.29e-01, req.: 7
	main_size: 7, RDA_size: 0, pre_assigned: True


Lane_drivable_area_det_1_0 Task 28: Lane_drivable_area_det_1_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 1.29e-01, io_time: 1.00e-06, totcpu: 1.29e-01
	flops: 1.29e-01, req.: 7
	main_size: 7, RDA_size: 0, pre_assigned: True


Lane_drivable_area_det_1_1 Task 29: Lane_drivable_area_det_1_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 1.29e-01, io_time: 1.00e-06, totcpu: 1.29e-01
	flops: 1.29e-01, req.: 7
	main_size: 7, RDA_size: 0, pre_assigned: True


Lane_drivable_area_det_2_0 Task 30: Lane_drivable_area_det_2_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 1.29e-01, io_time: 1.00e-06, totcpu: 1.29e-01
	flops: 1.29e-01, req.: 7
	main_size: 7, RDA_size: 0, pre_assigned: True


Lane_drivable_area_det_2_1 Task 31: Lane_drivable_area_det_2_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 1.29e-01, io_time: 1.00e-06, totcpu: 1.29e-01
	flops: 1.29e-01, req.: 7
	main_size: 7, RDA_size: 0, pre_assigned: True


Lane_drivable_area_det_3_0 Task 32: Lane_drivable_area_det_3_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 1.29e-01, io_time: 1.00e-06, totcpu: 1.29e-01
	flops: 1.29e-01, req.: 7
	main_size: 7, RDA_size: 0, pre_assigned: True


Lane_drivable_area_det_3_1 Task 33: Lane_drivable_area_det_3_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 1.29e-01, io_time: 1.00e-06, totcpu: 1.29e-01
	flops: 1.29e-01, req.: 7
	main_size: 7, RDA_size: 0, pre_assigned: True


Lane_drivable_area_det_4_0 Task 34: Lane_drivable_area_det_4_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 1.29e-01, io_time: 1.00e-06, totcpu: 1.29e-01
	flops: 1.29e-01, req.: 7
	main_size: 7, RDA_size: 0, pre_assigned: True


Lane_drivable_area_det_4_1 Task 35: Lane_drivable_area_det_4_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 1.29e-01, io_time: 1.00e-06, totcpu: 1.29e-01
	flops: 1.29e-01, req.: 7
	main_size: 7, RDA_size: 0, pre_assigned: True


Lane_drivable_area_det_5_0 Task 36: Lane_drivable_area_det_5_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 1.29e-01, io_time: 1.00e-06, totcpu: 1.29e-01
	flops: 1.29e-01, req.: 7
	main_size: 7, RDA_size: 0, pre_assigned: True


Lane_drivable_area_det_5_1 Task 37: Lane_drivable_area_det_5_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 1.29e-01, io_time: 1.00e-06, totcpu: 1.29e-01
	flops: 1.29e-01, req.: 7
	main_size: 7, RDA_size: 0, pre_assigned: True


Optical_Flow_0_0 Task 38: Optical_Flow_0_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 7.95e-02, io_time: 1.00e-06, totcpu: 7.95e-02
	flops: 7.95e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Optical_Flow_0_1 Task 39: Optical_Flow_0_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 7.95e-02, io_time: 1.00e-06, totcpu: 7.95e-02
	flops: 7.95e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Optical_Flow_1_0 Task 40: Optical_Flow_1_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 7.95e-02, io_time: 1.00e-06, totcpu: 7.95e-02
	flops: 7.95e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Optical_Flow_1_1 Task 41: Optical_Flow_1_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 7.95e-02, io_time: 1.00e-06, totcpu: 7.95e-02
	flops: 7.95e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Optical_Flow_2_0 Task 42: Optical_Flow_2_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 7.95e-02, io_time: 1.00e-06, totcpu: 7.95e-02
	flops: 7.95e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Optical_Flow_2_1 Task 43: Optical_Flow_2_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 7.95e-02, io_time: 1.00e-06, totcpu: 7.95e-02
	flops: 7.95e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Optical_Flow_3_0 Task 44: Optical_Flow_3_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 7.95e-02, io_time: 1.00e-06, totcpu: 7.95e-02
	flops: 7.95e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Optical_Flow_3_1 Task 45: Optical_Flow_3_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 7.95e-02, io_time: 1.00e-06, totcpu: 7.95e-02
	flops: 7.95e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Optical_Flow_4_0 Task 46: Optical_Flow_4_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 7.95e-02, io_time: 1.00e-06, totcpu: 7.95e-02
	flops: 7.95e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Optical_Flow_4_1 Task 47: Optical_Flow_4_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 7.95e-02, io_time: 1.00e-06, totcpu: 7.95e-02
	flops: 7.95e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Optical_Flow_5_0 Task 48: Optical_Flow_5_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 7.95e-02, io_time: 1.00e-06, totcpu: 7.95e-02
	flops: 7.95e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Optical_Flow_5_1 Task 49: Optical_Flow_5_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 7.95e-02, io_time: 1.00e-06, totcpu: 7.95e-02
	flops: 7.95e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Depth_estimation_0_0 Task 50: Depth_estimation_0_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 6.43e-02, io_time: 1.00e-06, totcpu: 6.43e-02
	flops: 6.43e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Depth_estimation_0_1 Task 51: Depth_estimation_0_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 6.43e-02, io_time: 1.00e-06, totcpu: 6.43e-02
	flops: 6.43e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Depth_estimation_1_0 Task 52: Depth_estimation_1_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 6.43e-02, io_time: 1.00e-06, totcpu: 6.43e-02
	flops: 6.43e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Depth_estimation_1_1 Task 53: Depth_estimation_1_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 6.43e-02, io_time: 1.00e-06, totcpu: 6.43e-02
	flops: 6.43e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Depth_estimation_2_0 Task 54: Depth_estimation_2_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 6.43e-02, io_time: 1.00e-06, totcpu: 6.43e-02
	flops: 6.43e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Depth_estimation_2_1 Task 55: Depth_estimation_2_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 6.43e-02, io_time: 1.00e-06, totcpu: 6.43e-02
	flops: 6.43e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Depth_estimation_3_0 Task 56: Depth_estimation_3_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 6.43e-02, io_time: 1.00e-06, totcpu: 6.43e-02
	flops: 6.43e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Depth_estimation_3_1 Task 57: Depth_estimation_3_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 6.43e-02, io_time: 1.00e-06, totcpu: 6.43e-02
	flops: 6.43e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Depth_estimation_4_0 Task 58: Depth_estimation_4_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 6.43e-02, io_time: 1.00e-06, totcpu: 6.43e-02
	flops: 6.43e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Depth_estimation_4_1 Task 59: Depth_estimation_4_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 6.43e-02, io_time: 1.00e-06, totcpu: 6.43e-02
	flops: 6.43e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Depth_estimation_5_0 Task 60: Depth_estimation_5_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 6.43e-02, io_time: 1.00e-06, totcpu: 6.43e-02
	flops: 6.43e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Depth_estimation_5_1 Task 61: Depth_estimation_5_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 6.43e-02, io_time: 1.00e-06, totcpu: 6.43e-02
	flops: 6.43e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


```

python -m sched.slack_estim --profiling_file profiling.csv --e2e_latency 0.09

```
{'ImageBB': (73, 0.06556172289170492, None), 'MultiCameraFusion': (73, 0.009034737328295088, None), 'Pure_camera_path_head': (6, 0.007008, 'upb'), 'Prediction': (20, 0.00170769268, 'upb'), 'Planning': (2, 0.0017076926800000001, 'upb'), 'Steering_speed': (1, 0.00048015442, 'upb'), 'Stereo_feature_enc': (7, 0.055064126790920764, None), 'Semantic_segm': (7, 0.020066044077390342, None), 'Lidar_based_3dDet': (7, 0.006474289351688905, None), 'Traffic_light_detection': (6, 0.0855, None), 'Lane_drivable_area_det': (7, 0.03993587320907924, None), 'Optical_Flow': (4, 0.03993587320907926, None), 'Depth_estimation': (4, 0.03993587320907924, None)}
{'LiDAR_pub': 0, 'surr_view_camera_pub': 0, 'IMU_pub': 0, 'streo_camera_pub': 0, 'Traffic_light_detection': 0, 'ImageBB': 0, 'Stereo_feature_enc': 0, 'MultiCameraFusion': 0.06901233988600519, 'Semantic_segm': 0.057962238727285016, 'Lane_drivable_area_det': 0.057962238727285016, 'Optical_Flow': 0.057962238727285016, 'Depth_estimation': 0.057962238727285016, 'Pure_camera_path_head': 0.07852258970526318, 'Lidar_based_3dDet': 0.07908439038769591, 'Sink_screen': 0.10000000000000003, 'Prediction': 0.08589943181052634, 'Planning': 0.0876970030526316, 'Steering_speed': 0.08949457429473687, 'Sink_control': 0.09000000000000002} {'LiDAR_pub': 0, 'surr_view_camera_pub': 0, 'IMU_pub': 0, 'streo_camera_pub': 0, 'Traffic_light_detection': 0.09000000000000001, 'ImageBB': 0.06901233988600519, 'Stereo_feature_enc': 0.057962238727285016, 'MultiCameraFusion': 0.07852258970526318, 'Semantic_segm': 0.07908439038769591, 'Lane_drivable_area_det': 0.1, 'Optical_Flow': 0.10000000000000003, 'Depth_estimation': 0.1, 'Pure_camera_path_head': 0.08589943181052634, 'Lidar_based_3dDet': 0.08589943181052634, 'Sink_screen': 0.10000000000000003, 'Prediction': 0.0876970030526316, 'Planning': 0.08949457429473687, 'Steering_speed': 0.09000000000000002, 'Sink_control': 0.09000000000000002}
Traffic_light_detection TaskIntAttr(name='Traffic_light_detection', freq=30, timing_flag='deadline', criticality='hard', trigger_mode='event', core_max=1000.0, core_min=0, core_list=None, parallel_mode=None, core_max_compile=1000.0, core_min_compile=0, core_list_compile=None, thread_scaling_factor=1, freq_division_factor=3, var_factor=1, jitter_max=0, ERT=0, ddl=0.09000000000000001, exp_comp_t=0.0855, flops=0.23324, task_flag='stationary', pre_assigned_resource_flag=True, num_exec=3, no_stall_latency=0.07774666666666667, min_tot_rsc=18, max_tot_rsc=24, flops_ModelSum=0.69972, flops_max=0.69972, equiv_core=13.994399999999999, util=0.7774666666666666, main_size=6, rda_size=2)

ImageBB TaskIntAttr(name='ImageBB', freq=30, timing_flag='deadline', criticality='soft', trigger_mode='event', core_max=1000.0, core_min=0, core_list=None, parallel_mode=None, core_max_compile=1000.0, core_min_compile=0, core_list_compile=None, thread_scaling_factor=1, freq_division_factor=3, var_factor=1, jitter_max=0, ERT=0, ddl=0.06901233988600519, exp_comp_t=0.06556172289170492, flops=2.391886, task_flag='stationary', pre_assigned_resource_flag=True, num_exec=3, no_stall_latency=0.06553112328767123, min_tot_rsc=219, max_tot_rsc=261, flops_ModelSum=7.175658, flops_max=7.175658, equiv_core=143.51316, util=0.6553112328767123, main_size=73, rda_size=14)

MultiCameraFusion TaskIntAttr(name='MultiCameraFusion', freq=30, timing_flag='deadline', criticality='soft', trigger_mode='N', core_max=1000.0, core_min=0, core_list=None, parallel_mode=None, core_max_compile=1000.0, core_min_compile=0, core_list_compile=None, thread_scaling_factor=1, freq_division_factor=3, var_factor=1, jitter_max=0, ERT=0.06901233988600519, ddl=0.009510249819257993, exp_comp_t=0.009034737328295088, flops=0.32961399999999996, task_flag='stationary', pre_assigned_resource_flag=False, num_exec=3, no_stall_latency=0.009030520547945205, min_tot_rsc=219, max_tot_rsc=261, flops_ModelSum=0.9888419999999998, flops_max=0.9888419999999998, equiv_core=19.776839999999993, util=0.09030520547945202, main_size=73, rda_size=14)

Pure_camera_path_head TaskIntAttr(name='Pure_camera_path_head', freq=30, timing_flag='deadline', criticality='soft', trigger_mode='N', core_max=6, core_min=0, core_list=None, parallel_mode='upb', core_max_compile=6, core_min_compile=0, core_list_compile=None, thread_scaling_factor=1, freq_division_factor=3, var_factor=1, jitter_max=0, ERT=0.07852258970526318, ddl=0.007376842105263154, exp_comp_t=0.007008, flops=0.021024, task_flag='moveable', pre_assigned_resource_flag=False, num_exec=3, no_stall_latency=0.007008, min_tot_rsc=18, max_tot_rsc=18, flops_ModelSum=0.063072, flops_max=0.063072, equiv_core=1.26144, util=0.07007999999999999, main_size=6, rda_size=0)

Prediction TaskIntAttr(name='Prediction', freq=30, timing_flag='deadline', criticality='soft', trigger_mode='N', core_max=20, core_min=0, core_list=None, parallel_mode='upb', core_max_compile=20, core_min_compile=0, core_list_compile=None, thread_scaling_factor=1, freq_division_factor=3, var_factor=3, jitter_max=0, ERT=0.08589943181052634, ddl=0.0017975712421052642, exp_comp_t=0.00170769268, flops=0.0170769268, task_flag='moveable', pre_assigned_resource_flag=False, num_exec=3, no_stall_latency=0.00170769268, min_tot_rsc=60, max_tot_rsc=180, flops_ModelSum=0.051230780399999995, flops_max=0.15369234119999997, equiv_core=1.0246156079999997, util=0.017076926799999996, main_size=20, rda_size=0)

Planning TaskIntAttr(name='Planning', freq=30, timing_flag='deadline', criticality='hard', trigger_mode='N', core_max=2, core_min=0, core_list=None, parallel_mode='upb', core_max_compile=2, core_min_compile=0, core_list_compile=None, thread_scaling_factor=1, freq_division_factor=3, var_factor=1, jitter_max=0, ERT=0.0876970030526316, ddl=0.0017975712421052642, exp_comp_t=0.0017076926800000001, flops=0.0017076926800000001, task_flag='moveable', pre_assigned_resource_flag=False, num_exec=3, no_stall_latency=0.0017076926800000001, min_tot_rsc=6, max_tot_rsc=6, flops_ModelSum=0.00512307804, flops_max=0.00512307804, equiv_core=0.1024615608, util=0.0170769268, main_size=2, rda_size=0)

Steering_speed TaskIntAttr(name='Steering_speed', freq=240, timing_flag='deadline', criticality='hard', trigger_mode='event', core_max=2, core_min=0, core_list=None, parallel_mode='upb', core_max_compile=1, core_min_compile=0, core_list_compile=None, thread_scaling_factor=1, freq_division_factor=3, var_factor=1, jitter_max=0, ERT=0.08949457429473687, ddl=0.0005054257052631572, exp_comp_t=0.00048015442, flops=0.00024007721, task_flag='moveable', pre_assigned_resource_flag=False, num_exec=24, no_stall_latency=0.00048015442, min_tot_rsc=3, max_tot_rsc=6, flops_ModelSum=0.00576185304, flops_max=0.00576185304, equiv_core=0.11523706079999999, util=0.0384123536, main_size=1, rda_size=1)

Stereo_feature_enc TaskIntAttr(name='Stereo_feature_enc', freq=20, timing_flag='deadline', criticality='soft', trigger_mode='event', core_max=1000.0, core_min=0, core_list=None, parallel_mode=None, core_max_compile=1000.0, core_min_compile=0, core_list_compile=None, thread_scaling_factor=1, freq_division_factor=2, var_factor=1, jitter_max=0, ERT=0, ddl=0.057962238727285016, exp_comp_t=0.055064126790920764, flops=0.178674, task_flag='stationary', pre_assigned_resource_flag=False, num_exec=2, no_stall_latency=0.05104971428571429, min_tot_rsc=14, max_tot_rsc=18, flops_ModelSum=0.357348, flops_max=0.357348, equiv_core=7.14696, util=0.5104971428571429, main_size=7, rda_size=2)

Semantic_segm TaskIntAttr(name='Semantic_segm', freq=20, timing_flag='deadline', criticality='soft', trigger_mode='N', core_max=1000.0, core_min=0, core_list=None, parallel_mode=None, core_max_compile=1000.0, core_min_compile=0, core_list_compile=None, thread_scaling_factor=1, freq_division_factor=2, var_factor=1, jitter_max=0, ERT=0.057962238727285016, ddl=0.02112215166041089, exp_comp_t=0.020066044077390342, flops=0.065111, task_flag='stationary', pre_assigned_resource_flag=True, num_exec=2, no_stall_latency=0.018603142857142856, min_tot_rsc=14, max_tot_rsc=18, flops_ModelSum=0.130222, flops_max=0.130222, equiv_core=2.60444, util=0.18603142857142857, main_size=7, rda_size=2)

Lidar_based_3dDet TaskIntAttr(name='Lidar_based_3dDet', freq=10, timing_flag='deadline', criticality='soft', trigger_mode='event', core_max=1000.0, core_min=0, core_list=None, parallel_mode=None, core_max_compile=1000.0, core_min_compile=0, core_list_compile=None, thread_scaling_factor=1, freq_division_factor=1, var_factor=1, jitter_max=0, ERT=0.07908439038769591, ddl=0.006815041422830431, exp_comp_t=0.006474289351688905, flops=0.021008, task_flag='moveable', pre_assigned_resource_flag=False, num_exec=1, no_stall_latency=0.006002285714285714, min_tot_rsc=7, max_tot_rsc=9, flops_ModelSum=0.021008, flops_max=0.021008, equiv_core=0.42016, util=0.06002285714285714, main_size=7, rda_size=2)

Lane_drivable_area_det TaskIntAttr(name='Lane_drivable_area_det', freq=20, timing_flag='realtime', criticality='soft', trigger_mode='N', core_max=1000.0, core_min=0, core_list=None, parallel_mode=None, core_max_compile=1000.0, core_min_compile=0, core_list_compile=None, thread_scaling_factor=6, freq_division_factor=2, var_factor=1, jitter_max=0, ERT=0.057962238727285016, ddl=0.04203776127271499, exp_comp_t=0.03993587320907924, flops=0.1285636, task_flag='moveable', pre_assigned_resource_flag=True, num_exec=12, no_stall_latency=0.036732457142857146, min_tot_rsc=84, max_tot_rsc=84, flops_ModelSum=1.5427632, flops_max=1.5427632, equiv_core=30.855264, util=0.3673245714285714, main_size=7, rda_size=0)

Optical_Flow TaskIntAttr(name='Optical_Flow', freq=20, timing_flag='realtime', criticality='soft', trigger_mode='N', core_max=1000.0, core_min=0, core_list=None, parallel_mode=None, core_max_compile=1000.0, core_min_compile=0, core_list_compile=None, thread_scaling_factor=6, freq_division_factor=2, var_factor=1, jitter_max=0, ERT=0.057962238727285016, ddl=0.04203776127271502, exp_comp_t=0.03993587320907926, flops=0.079514, task_flag='moveable', pre_assigned_resource_flag=False, num_exec=12, no_stall_latency=0.039757, min_tot_rsc=48, max_tot_rsc=48, flops_ModelSum=0.9541679999999999, flops_max=0.9541679999999999, equiv_core=19.083359999999995, util=0.3975699999999999, main_size=4, rda_size=0)

Depth_estimation TaskIntAttr(name='Depth_estimation', freq=20, timing_flag='realtime', criticality='soft', trigger_mode='N', core_max=1000.0, core_min=0, core_list=None, parallel_mode=None, core_max_compile=1000.0, core_min_compile=0, core_list_compile=None, thread_scaling_factor=6, freq_division_factor=2, var_factor=1, jitter_max=0, ERT=0.057962238727285016, ddl=0.04203776127271499, exp_comp_t=0.03993587320907924, flops=0.0642818, task_flag='moveable', pre_assigned_resource_flag=False, num_exec=12, no_stall_latency=0.0321409, min_tot_rsc=48, max_tot_rsc=48, flops_ModelSum=0.7713816, flops_max=0.7713816, equiv_core=15.427632, util=0.321409, main_size=4, rda_size=0)

Traffic_light_detection_0_0 Task 0: Traffic_light_detection_0_0
	timing_flag: deadline, criticality: hard, trigger_mode: event
	num of cores: no constraint
	spatial factor: thread 1, freq 3, var 1
	slack info: 0.00e+00 ~ 9.00e-02 (8.55e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 2.33e-01, io_time: 1.00e-06, totcpu: 2.33e-01
	flops: 2.33e-01, req.: 6
	main_size: 6, RDA_size: 2, pre_assigned: True


Traffic_light_detection_0_1 Task 1: Traffic_light_detection_0_1
	timing_flag: deadline, criticality: hard, trigger_mode: event
	num of cores: no constraint
	spatial factor: thread 1, freq 3, var 1
	slack info: 0.00e+00 ~ 9.00e-02 (8.55e-02)
	period: 1.00e-01, i_offset: 3.33e-02
	jitter_max: 0
	cpu_time: 2.33e-01, io_time: 1.00e-06, totcpu: 2.33e-01
	flops: 2.33e-01, req.: 6
	main_size: 6, RDA_size: 2, pre_assigned: True


Traffic_light_detection_0_2 Task 2: Traffic_light_detection_0_2
	timing_flag: deadline, criticality: hard, trigger_mode: event
	num of cores: no constraint
	spatial factor: thread 1, freq 3, var 1
	slack info: 0.00e+00 ~ 9.00e-02 (8.55e-02)
	period: 1.00e-01, i_offset: 6.67e-02
	jitter_max: 0
	cpu_time: 2.33e-01, io_time: 1.00e-06, totcpu: 2.33e-01
	flops: 2.33e-01, req.: 6
	main_size: 6, RDA_size: 2, pre_assigned: True


ImageBB_0_0 Task 3: ImageBB_0_0
	timing_flag: deadline, criticality: soft, trigger_mode: event
	num of cores: no constraint
	spatial factor: thread 1, freq 3, var 1
	slack info: 0.00e+00 ~ 6.90e-02 (6.56e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 2.39e+00, io_time: 1.00e-06, totcpu: 2.39e+00
	flops: 2.39e+00, req.: 73
	main_size: 73, RDA_size: 14, pre_assigned: True


ImageBB_0_1 Task 4: ImageBB_0_1
	timing_flag: deadline, criticality: soft, trigger_mode: event
	num of cores: no constraint
	spatial factor: thread 1, freq 3, var 1
	slack info: 0.00e+00 ~ 6.90e-02 (6.56e-02)
	period: 1.00e-01, i_offset: 3.33e-02
	jitter_max: 0
	cpu_time: 2.39e+00, io_time: 1.00e-06, totcpu: 2.39e+00
	flops: 2.39e+00, req.: 73
	main_size: 73, RDA_size: 14, pre_assigned: True


ImageBB_0_2 Task 5: ImageBB_0_2
	timing_flag: deadline, criticality: soft, trigger_mode: event
	num of cores: no constraint
	spatial factor: thread 1, freq 3, var 1
	slack info: 0.00e+00 ~ 6.90e-02 (6.56e-02)
	period: 1.00e-01, i_offset: 6.67e-02
	jitter_max: 0
	cpu_time: 2.39e+00, io_time: 1.00e-06, totcpu: 2.39e+00
	flops: 2.39e+00, req.: 73
	main_size: 73, RDA_size: 14, pre_assigned: True


MultiCameraFusion_0_0 Task 6: MultiCameraFusion_0_0
	timing_flag: deadline, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 1, freq 3, var 1
	slack info: 6.90e-02 ~ 7.85e-02 (9.03e-03)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 3.30e-01, io_time: 1.00e-06, totcpu: 3.30e-01
	flops: 3.30e-01, req.: 73
	main_size: 0, RDA_size: 0, pre_assigned: False


MultiCameraFusion_0_1 Task 7: MultiCameraFusion_0_1
	timing_flag: deadline, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 1, freq 3, var 1
	slack info: 6.90e-02 ~ 7.85e-02 (9.03e-03)
	period: 1.00e-01, i_offset: 3.33e-02
	jitter_max: 0
	cpu_time: 3.30e-01, io_time: 1.00e-06, totcpu: 3.30e-01
	flops: 3.30e-01, req.: 73
	main_size: 0, RDA_size: 0, pre_assigned: False


MultiCameraFusion_0_2 Task 8: MultiCameraFusion_0_2
	timing_flag: deadline, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 1, freq 3, var 1
	slack info: 6.90e-02 ~ 7.85e-02 (9.03e-03)
	period: 1.00e-01, i_offset: 6.67e-02
	jitter_max: 0
	cpu_time: 3.30e-01, io_time: 1.00e-06, totcpu: 3.30e-01
	flops: 3.30e-01, req.: 73
	main_size: 0, RDA_size: 0, pre_assigned: False


Pure_camera_path_head_0_0 Task 9: Pure_camera_path_head_0_0
	timing_flag: deadline, criticality: soft, trigger_mode: N
	num of cores (compile): 1 ~ 6
	num of cores (run): 1 ~ 6
	spatial factor: thread 1, freq 3, var 1
	slack info: 7.85e-02 ~ 8.59e-02 (7.01e-03)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 2.10e-02, io_time: 1.00e-06, totcpu: 2.10e-02
	flops: 2.10e-02, req.: 6
	main_size: 0, RDA_size: 0, pre_assigned: False


Pure_camera_path_head_0_1 Task 10: Pure_camera_path_head_0_1
	timing_flag: deadline, criticality: soft, trigger_mode: N
	num of cores (compile): 1 ~ 6
	num of cores (run): 1 ~ 6
	spatial factor: thread 1, freq 3, var 1
	slack info: 7.85e-02 ~ 8.59e-02 (7.01e-03)
	period: 1.00e-01, i_offset: 3.33e-02
	jitter_max: 0
	cpu_time: 2.10e-02, io_time: 1.00e-06, totcpu: 2.10e-02
	flops: 2.10e-02, req.: 6
	main_size: 0, RDA_size: 0, pre_assigned: False


Pure_camera_path_head_0_2 Task 11: Pure_camera_path_head_0_2
	timing_flag: deadline, criticality: soft, trigger_mode: N
	num of cores (compile): 1 ~ 6
	num of cores (run): 1 ~ 6
	spatial factor: thread 1, freq 3, var 1
	slack info: 7.85e-02 ~ 8.59e-02 (7.01e-03)
	period: 1.00e-01, i_offset: 6.67e-02
	jitter_max: 0
	cpu_time: 2.10e-02, io_time: 1.00e-06, totcpu: 2.10e-02
	flops: 2.10e-02, req.: 6
	main_size: 0, RDA_size: 0, pre_assigned: False


Prediction_0_0 Task 12: Prediction_0_0
	timing_flag: deadline, criticality: soft, trigger_mode: N
	num of cores (compile): 1 ~ 20
	num of cores (run): 1 ~ 20
	spatial factor: thread 1, freq 3, var 3
	slack info: 8.59e-02 ~ 8.77e-02 (1.71e-03)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 1.71e-02, io_time: 1.00e-06, totcpu: 1.71e-02
	flops: 1.71e-02, req.: 20
	main_size: 0, RDA_size: 0, pre_assigned: False


Prediction_0_1 Task 13: Prediction_0_1
	timing_flag: deadline, criticality: soft, trigger_mode: N
	num of cores (compile): 1 ~ 20
	num of cores (run): 1 ~ 20
	spatial factor: thread 1, freq 3, var 3
	slack info: 8.59e-02 ~ 8.77e-02 (1.71e-03)
	period: 1.00e-01, i_offset: 3.33e-02
	jitter_max: 0
	cpu_time: 1.71e-02, io_time: 1.00e-06, totcpu: 1.71e-02
	flops: 1.71e-02, req.: 20
	main_size: 0, RDA_size: 0, pre_assigned: False


Prediction_0_2 Task 14: Prediction_0_2
	timing_flag: deadline, criticality: soft, trigger_mode: N
	num of cores (compile): 1 ~ 20
	num of cores (run): 1 ~ 20
	spatial factor: thread 1, freq 3, var 3
	slack info: 8.59e-02 ~ 8.77e-02 (1.71e-03)
	period: 1.00e-01, i_offset: 6.67e-02
	jitter_max: 0
	cpu_time: 1.71e-02, io_time: 1.00e-06, totcpu: 1.71e-02
	flops: 1.71e-02, req.: 20
	main_size: 0, RDA_size: 0, pre_assigned: False


Planning_0_0 Task 15: Planning_0_0
	timing_flag: deadline, criticality: hard, trigger_mode: N
	num of cores (compile): 1 ~ 2
	num of cores (run): 1 ~ 2
	spatial factor: thread 1, freq 3, var 1
	slack info: 8.77e-02 ~ 8.95e-02 (1.71e-03)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 1.71e-03, io_time: 1.00e-06, totcpu: 1.71e-03
	flops: 1.71e-03, req.: 2
	main_size: 0, RDA_size: 0, pre_assigned: False


Planning_0_1 Task 16: Planning_0_1
	timing_flag: deadline, criticality: hard, trigger_mode: N
	num of cores (compile): 1 ~ 2
	num of cores (run): 1 ~ 2
	spatial factor: thread 1, freq 3, var 1
	slack info: 8.77e-02 ~ 8.95e-02 (1.71e-03)
	period: 1.00e-01, i_offset: 3.33e-02
	jitter_max: 0
	cpu_time: 1.71e-03, io_time: 1.00e-06, totcpu: 1.71e-03
	flops: 1.71e-03, req.: 2
	main_size: 0, RDA_size: 0, pre_assigned: False


Planning_0_2 Task 17: Planning_0_2
	timing_flag: deadline, criticality: hard, trigger_mode: N
	num of cores (compile): 1 ~ 2
	num of cores (run): 1 ~ 2
	spatial factor: thread 1, freq 3, var 1
	slack info: 8.77e-02 ~ 8.95e-02 (1.71e-03)
	period: 1.00e-01, i_offset: 6.67e-02
	jitter_max: 0
	cpu_time: 1.71e-03, io_time: 1.00e-06, totcpu: 1.71e-03
	flops: 1.71e-03, req.: 2
	main_size: 0, RDA_size: 0, pre_assigned: False


Steering_speed_0_0 Task 18: Steering_speed_0_0
	timing_flag: deadline, criticality: hard, trigger_mode: event
	num of cores (compile): 1 ~ 1
	num of cores (run): 1 ~ 2
	spatial factor: thread 1, freq 3, var 1
	slack info: 8.95e-02 ~ 9.00e-02 (4.80e-04)
	period: 4.17e-03, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 2.40e-04, io_time: 1.00e-06, totcpu: 2.40e-04
	flops: 2.40e-04, req.: 1
	main_size: 0, RDA_size: 0, pre_assigned: False


Steering_speed_0_1 Task 19: Steering_speed_0_1
	timing_flag: deadline, criticality: hard, trigger_mode: event
	num of cores (compile): 1 ~ 1
	num of cores (run): 1 ~ 2
	spatial factor: thread 1, freq 3, var 1
	slack info: 8.95e-02 ~ 9.00e-02 (4.80e-04)
	period: 4.17e-03, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 2.40e-04, io_time: 1.00e-06, totcpu: 2.40e-04
	flops: 2.40e-04, req.: 1
	main_size: 0, RDA_size: 0, pre_assigned: False


Steering_speed_0_2 Task 20: Steering_speed_0_2
	timing_flag: deadline, criticality: hard, trigger_mode: event
	num of cores (compile): 1 ~ 1
	num of cores (run): 1 ~ 2
	spatial factor: thread 1, freq 3, var 1
	slack info: 8.95e-02 ~ 9.00e-02 (4.80e-04)
	period: 4.17e-03, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 2.40e-04, io_time: 1.00e-06, totcpu: 2.40e-04
	flops: 2.40e-04, req.: 1
	main_size: 0, RDA_size: 0, pre_assigned: False


Stereo_feature_enc_0_0 Task 21: Stereo_feature_enc_0_0
	timing_flag: deadline, criticality: soft, trigger_mode: event
	num of cores: no constraint
	spatial factor: thread 1, freq 2, var 1
	slack info: 0.00e+00 ~ 5.80e-02 (5.51e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 1.79e-01, io_time: 1.00e-06, totcpu: 1.79e-01
	flops: 1.79e-01, req.: 7
	main_size: 0, RDA_size: 0, pre_assigned: False


Stereo_feature_enc_0_1 Task 22: Stereo_feature_enc_0_1
	timing_flag: deadline, criticality: soft, trigger_mode: event
	num of cores: no constraint
	spatial factor: thread 1, freq 2, var 1
	slack info: 0.00e+00 ~ 5.80e-02 (5.51e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 1.79e-01, io_time: 1.00e-06, totcpu: 1.79e-01
	flops: 1.79e-01, req.: 7
	main_size: 0, RDA_size: 0, pre_assigned: False


Semantic_segm_0_0 Task 23: Semantic_segm_0_0
	timing_flag: deadline, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 1, freq 2, var 1
	slack info: 5.80e-02 ~ 7.91e-02 (2.01e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 6.51e-02, io_time: 1.00e-06, totcpu: 6.51e-02
	flops: 6.51e-02, req.: 7
	main_size: 7, RDA_size: 2, pre_assigned: True


Semantic_segm_0_1 Task 24: Semantic_segm_0_1
	timing_flag: deadline, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 1, freq 2, var 1
	slack info: 5.80e-02 ~ 7.91e-02 (2.01e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 6.51e-02, io_time: 1.00e-06, totcpu: 6.51e-02
	flops: 6.51e-02, req.: 7
	main_size: 7, RDA_size: 2, pre_assigned: True


Lidar_based_3dDet_0_0 Task 25: Lidar_based_3dDet_0_0
	timing_flag: deadline, criticality: soft, trigger_mode: event
	num of cores: no constraint
	spatial factor: thread 1, freq 1, var 1
	slack info: 7.91e-02 ~ 8.59e-02 (6.47e-03)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 2.10e-02, io_time: 1.00e-06, totcpu: 2.10e-02
	flops: 2.10e-02, req.: 7
	main_size: 0, RDA_size: 0, pre_assigned: False


Lane_drivable_area_det_0_0 Task 26: Lane_drivable_area_det_0_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 1.29e-01, io_time: 1.00e-06, totcpu: 1.29e-01
	flops: 1.29e-01, req.: 7
	main_size: 7, RDA_size: 0, pre_assigned: True


Lane_drivable_area_det_0_1 Task 27: Lane_drivable_area_det_0_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 1.29e-01, io_time: 1.00e-06, totcpu: 1.29e-01
	flops: 1.29e-01, req.: 7
	main_size: 7, RDA_size: 0, pre_assigned: True


Lane_drivable_area_det_1_0 Task 28: Lane_drivable_area_det_1_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 1.29e-01, io_time: 1.00e-06, totcpu: 1.29e-01
	flops: 1.29e-01, req.: 7
	main_size: 7, RDA_size: 0, pre_assigned: True


Lane_drivable_area_det_1_1 Task 29: Lane_drivable_area_det_1_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 1.29e-01, io_time: 1.00e-06, totcpu: 1.29e-01
	flops: 1.29e-01, req.: 7
	main_size: 7, RDA_size: 0, pre_assigned: True


Lane_drivable_area_det_2_0 Task 30: Lane_drivable_area_det_2_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 1.29e-01, io_time: 1.00e-06, totcpu: 1.29e-01
	flops: 1.29e-01, req.: 7
	main_size: 7, RDA_size: 0, pre_assigned: True


Lane_drivable_area_det_2_1 Task 31: Lane_drivable_area_det_2_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 1.29e-01, io_time: 1.00e-06, totcpu: 1.29e-01
	flops: 1.29e-01, req.: 7
	main_size: 7, RDA_size: 0, pre_assigned: True


Lane_drivable_area_det_3_0 Task 32: Lane_drivable_area_det_3_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 1.29e-01, io_time: 1.00e-06, totcpu: 1.29e-01
	flops: 1.29e-01, req.: 7
	main_size: 7, RDA_size: 0, pre_assigned: True


Lane_drivable_area_det_3_1 Task 33: Lane_drivable_area_det_3_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 1.29e-01, io_time: 1.00e-06, totcpu: 1.29e-01
	flops: 1.29e-01, req.: 7
	main_size: 7, RDA_size: 0, pre_assigned: True


Lane_drivable_area_det_4_0 Task 34: Lane_drivable_area_det_4_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 1.29e-01, io_time: 1.00e-06, totcpu: 1.29e-01
	flops: 1.29e-01, req.: 7
	main_size: 7, RDA_size: 0, pre_assigned: True


Lane_drivable_area_det_4_1 Task 35: Lane_drivable_area_det_4_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 1.29e-01, io_time: 1.00e-06, totcpu: 1.29e-01
	flops: 1.29e-01, req.: 7
	main_size: 7, RDA_size: 0, pre_assigned: True


Lane_drivable_area_det_5_0 Task 36: Lane_drivable_area_det_5_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 1.29e-01, io_time: 1.00e-06, totcpu: 1.29e-01
	flops: 1.29e-01, req.: 7
	main_size: 7, RDA_size: 0, pre_assigned: True


Lane_drivable_area_det_5_1 Task 37: Lane_drivable_area_det_5_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 1.29e-01, io_time: 1.00e-06, totcpu: 1.29e-01
	flops: 1.29e-01, req.: 7
	main_size: 7, RDA_size: 0, pre_assigned: True


Optical_Flow_0_0 Task 38: Optical_Flow_0_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 7.95e-02, io_time: 1.00e-06, totcpu: 7.95e-02
	flops: 7.95e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Optical_Flow_0_1 Task 39: Optical_Flow_0_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 7.95e-02, io_time: 1.00e-06, totcpu: 7.95e-02
	flops: 7.95e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Optical_Flow_1_0 Task 40: Optical_Flow_1_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 7.95e-02, io_time: 1.00e-06, totcpu: 7.95e-02
	flops: 7.95e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Optical_Flow_1_1 Task 41: Optical_Flow_1_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 7.95e-02, io_time: 1.00e-06, totcpu: 7.95e-02
	flops: 7.95e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Optical_Flow_2_0 Task 42: Optical_Flow_2_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 7.95e-02, io_time: 1.00e-06, totcpu: 7.95e-02
	flops: 7.95e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Optical_Flow_2_1 Task 43: Optical_Flow_2_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 7.95e-02, io_time: 1.00e-06, totcpu: 7.95e-02
	flops: 7.95e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Optical_Flow_3_0 Task 44: Optical_Flow_3_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 7.95e-02, io_time: 1.00e-06, totcpu: 7.95e-02
	flops: 7.95e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Optical_Flow_3_1 Task 45: Optical_Flow_3_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 7.95e-02, io_time: 1.00e-06, totcpu: 7.95e-02
	flops: 7.95e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Optical_Flow_4_0 Task 46: Optical_Flow_4_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 7.95e-02, io_time: 1.00e-06, totcpu: 7.95e-02
	flops: 7.95e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Optical_Flow_4_1 Task 47: Optical_Flow_4_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 7.95e-02, io_time: 1.00e-06, totcpu: 7.95e-02
	flops: 7.95e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Optical_Flow_5_0 Task 48: Optical_Flow_5_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 7.95e-02, io_time: 1.00e-06, totcpu: 7.95e-02
	flops: 7.95e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Optical_Flow_5_1 Task 49: Optical_Flow_5_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 7.95e-02, io_time: 1.00e-06, totcpu: 7.95e-02
	flops: 7.95e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Depth_estimation_0_0 Task 50: Depth_estimation_0_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 6.43e-02, io_time: 1.00e-06, totcpu: 6.43e-02
	flops: 6.43e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Depth_estimation_0_1 Task 51: Depth_estimation_0_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 6.43e-02, io_time: 1.00e-06, totcpu: 6.43e-02
	flops: 6.43e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Depth_estimation_1_0 Task 52: Depth_estimation_1_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 6.43e-02, io_time: 1.00e-06, totcpu: 6.43e-02
	flops: 6.43e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Depth_estimation_1_1 Task 53: Depth_estimation_1_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 6.43e-02, io_time: 1.00e-06, totcpu: 6.43e-02
	flops: 6.43e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Depth_estimation_2_0 Task 54: Depth_estimation_2_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 6.43e-02, io_time: 1.00e-06, totcpu: 6.43e-02
	flops: 6.43e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Depth_estimation_2_1 Task 55: Depth_estimation_2_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 6.43e-02, io_time: 1.00e-06, totcpu: 6.43e-02
	flops: 6.43e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Depth_estimation_3_0 Task 56: Depth_estimation_3_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 6.43e-02, io_time: 1.00e-06, totcpu: 6.43e-02
	flops: 6.43e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Depth_estimation_3_1 Task 57: Depth_estimation_3_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 6.43e-02, io_time: 1.00e-06, totcpu: 6.43e-02
	flops: 6.43e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Depth_estimation_4_0 Task 58: Depth_estimation_4_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 6.43e-02, io_time: 1.00e-06, totcpu: 6.43e-02
	flops: 6.43e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Depth_estimation_4_1 Task 59: Depth_estimation_4_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 6.43e-02, io_time: 1.00e-06, totcpu: 6.43e-02
	flops: 6.43e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Depth_estimation_5_0 Task 60: Depth_estimation_5_0
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 0.00e+00
	jitter_max: 0
	cpu_time: 6.43e-02, io_time: 1.00e-06, totcpu: 6.43e-02
	flops: 6.43e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


Depth_estimation_5_1 Task 61: Depth_estimation_5_1
	timing_flag: realtime, criticality: soft, trigger_mode: N
	num of cores: no constraint
	spatial factor: thread 6, freq 2, var 1
	slack info: 5.80e-02 ~ 1.00e-01 (3.99e-02)
	period: 1.00e-01, i_offset: 5.00e-02
	jitter_max: 0
	cpu_time: 6.43e-02, io_time: 1.00e-06, totcpu: 6.43e-02
	flops: 6.43e-02, req.: 4
	main_size: 0, RDA_size: 0, pre_assigned: False


```