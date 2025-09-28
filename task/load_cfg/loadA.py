__all__ = ['task_graph_srcs', 'task_graph_ops', 'task_graph_sinks', 
           'affinity_cfg', 'task_sink_attr', 'task_src_attr', 'pre_assign_priority']

task_graph_srcs = {
    # "Entry": ["surr_view_camera_pub", "streo_camera_pub", "LiDAR_pub"],
    "LiDAR_pub": ["Lidar_based_3dDet"],
    "surr_view_camera_pub": ["Traffic_light_detection", "ImageBB", ],
    "IMU_pub": ["Steering_speed"],
    "streo_camera_pub": ["Stereo_feature_enc"],
}
task_graph_sinks = {
    "Sink_control": [],
    "Sink_screen": [],
}
task_sink_attr={
    "Sink_control": ["deadline", float("nan")],
    "Sink_screen": ["realtime", float("nan")]
}
task_src_attr={
    "LiDAR_pub": 10,
    "surr_view_camera_pub": 30,
    "IMU_pub": 240,
    "streo_camera_pub": 20
}

task_graph_ops = {   
    "Traffic_light_detection": ["Sink_control"],
    "ImageBB": ["MultiCameraFusion"],
    "MultiCameraFusion": ["Pure_camera_path_head"],
    "Pure_camera_path_head": ["Prediction"],
    "Prediction": ["Planning"],
    "Planning": ["Steering_speed"],
    "Steering_speed": ["Sink_control"],
    "Stereo_feature_enc": ["Semantic_segm", "Lane_drivable_area_det", "Optical_Flow", "Depth_estimation"],
    "Semantic_segm": ["Lidar_based_3dDet","Sink_screen"],
    "Lidar_based_3dDet": ["Prediction"],
    "Lane_drivable_area_det": ["Sink_screen"],
    "Optical_Flow": ["Sink_screen"],
    "Depth_estimation": ["Sink_screen"],
}

# affnity of a task is set to be a list that contains user-specified tasks, itself, it predecessors and its successors.
affinity_cfg = {
    "Traffic_light_detection": [],
    "ImageBB": ["MultiCameraFusion"],
    "MultiCameraFusion": ["ImageBB", "Pure_camera_path_head"],
    "Pure_camera_path_head": ["MultiCameraFusion", "Prediction"],
    "Prediction": ["Lidar_based_3dDet", "Pure_camera_path_head", "Planning"],
    "Planning": ["Prediction", "Steering_speed"],
    "Steering_speed": ["Planning"],
    "Stereo_feature_enc": ["Semantic_segm", "Lane_drivable_area_det", "Optical_Flow", "Depth_estimation"],
    "Semantic_segm": ["Stereo_feature_enc", "LiDAR_based_3dDet"],
    "Lidar_based_3dDet": ["Stereo_feature_enc", "Semantic_segm", "Prediction"],
    "Lane_drivable_area_det": ["Stereo_feature_enc"],
    "Optical_Flow": ["Stereo_feature_enc", "Lane_drivable_area_det",],
    "Depth_estimation": ["Stereo_feature_enc", "Lane_drivable_area_det",],
}
# post-processing
# For the task that is pre-assigned with the resource, the affinity is set to be itself

pre_assign_priority = [
    ["Traffic_light_detection",
    "ImageBB",
    "Stereo_feature_enc"],
[    "Lane_drivable_area_det",
    "Optical_Flow",
    "Depth_estimation"
]]
