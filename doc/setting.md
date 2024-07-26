## folder setting 

global_var.py
- cfg_dir: 
    - setting of the var distribution (runtime)
    - setting of the bin packing algorithm (compile time)
- log_dir: 


## Simulation setting (`var_sim_cfg.json`)

This JSON file contains configuration settings for four different categories: `e2e_var`, `load_var`, `jitter`, and `exec`. Below is a detailed explanation for each category.

### e2e_var

The `e2e_var` object contains settings related to end-to-end variations.

| Key | Description | Type | Example |
| --- | ----------- | ---- | ------- |
| event_list | A list of events, each event is represented as a list with two elements. The first element is the time of the event, and the second element is the event's value. | List of Lists | `[[0.08, 6], [0.09, 11], [0.1, 11]]` |
| period | The period of time between events. | Float | `0.1` |

### load_var

The `load_var` object contains settings related to load variations.

#### Objects
| Key | Description | Type | Example |
| --- | ----------- | ---- | ------- |
| dyn_obj_num | An object that contains settings related to the number of dynamic objects. | Object | See below |
| dyn_model_sel | The dynamic model selection. | String | `random` |
| dyn_traj_plan | The dynamic trajectory planning. | String | `random` |
| dyn_region_proposed | The proposed region for dynamic objects. | List of Lists | `[[0, 0, 10, 10], [0, 0, 10, 10]]` |

#### Format of load variations

| Key | Description | Type | Example |
| --- | ----------- | ---- | ------- |
| src_name | A list of source nodes that events is injected from. | List of Strings | `["Stereo_feature_enc", "Lidar_based_3dDet", "ImageBB"]` |
| tgt_name | A list of target nodes that events influence. | List of Strings | `["Prediction"]` |
| typical | The typical value. | Integer | `10` |
| maxsize | The maximum size. | Integer | `30` |
| period | The period of time between events. | Float | `0.03333333333333333` |

### jitter/exec

The `jitter/exec` object contains settings related to jitter.

| Key | Description | Type | Example |
| --- | ----------- | ---- | ------- |
| loc | The location parameter for the distribution. | Float | `0` |
| scale | The scale parameter for the distribution. | Float | `0.2/0.3` |
| enforce_wc | A boolean value that indicates whether to enforce worst-case scenario. | Boolean | `false` |

## Scheduler setting
latency model:
$$
    \lat_{\xi^{j}_{k}, \ETEs} = \sum_{\tau_i \in \xi^{j}_{k}} {\var_{ld}\cdot \lat_{i, typ} }{(1\!+\!\var_{sl})}\!+\! \frac{\var_{\jitter}}{\freq_{src}}. \label{eq::lat_model}
$$

compensation model:

$$
    \slack({\tau_i}) \geq \lat_{i, typ}  (1 + S_{sl}) + S_{\jitter}
$$

Compensation stages:
- Compile-time overprovisioning:  

    As each task is constrainted to not start until getting budget, 
    that's to say a isolated time-space slice is assign to tasks in each slot. 
    
- Runtime overprovisioning: 

    Getting resources from redundant or idle resources or by preempting other tasks. 

Compensation ratio:
- jitter_t_comp_ratio: 

    percentage of jitter compensated at compile time. (1/src_freq*jitter_t_comp_ratio) 
- exec_t_comp_ratioA:

    The percent of slowdown assumed at runtime compared to the planned execution time, which is also seen as the overprovisioning ratio. 
    This option results in using by latency model for shrinking the time budget in cfg slection, 
    to gurrantee the task completes within the planned time, with at most RatioA times slowdown.
    This factor affects both the minimal resource requirements as well as the ERT and ddl of each task. 
- exec_t_comp_ratioB:

    Similar to RatioA, but a runtime factor used for estimating the resource requirements. 
- wsc_slack_ratio: 

    The temporal shrinking ratio with respect to the worst-case end-to-end slack. 
    ``` python
    compact_slcak = slack_from_WC_analysis/wsc_slack_ratio 
    RDA_size = math.ceil(size * wsc_comp / wsc_estm1 * wsc_slack_ratio) - size 
    ```
    This option influences the amount of maximum avalable resources used for overcoming the corner cases. 


