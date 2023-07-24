from __future__ import annotations
from typing import List
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from task.task_agent import ProcessBase
    from model.buffer import EventCache, TriggerCache
import warnings
import math, copy
import numpy as np
from typing import Dict, Tuple
from global_var import numerical_tol_bit

class WatermarkStrategy(object):
    """
    check the watermark
    Watermarks are generated at, or directly after, source functions, 
    Each parallel subtask of a source function usually generates its watermarks independently, 
    These watermarks define the event time at that particular parallel source.

    As the watermarks flow through the streaming program, 
    they advance the event time at the operators where they arrive, 
    Whenever an operator advances its event time, 
    it generates a new watermark downstream for its successor operators.
    Some operators consume multiple input streams; a union, for example,
    Such an operator’s current event time is the minimum of its input streams’ event times. 
    As its input streams update their event times, so does the operator.

    """

    @staticmethod
    def extract_stream_from_buffer(_p, buffer, glb_n_task_dict, pred_data:Dict[int, Dict]=None):
        """
        extract the stream of the predecessor operator's output from the buffer
        """
        stream_dict = {}
        period = None
        if pred_data is None:
            pred_data = _p.pred_data

        for key in pred_data:
            pid = glb_n_task_dict[key].pid
            stream_t = buffer.buffer_mux("output").get(pid, [])
            if len(stream_t)>0:
                if period is None:
                    period = stream_t[0].period
                else:
                    assert period == stream_t[0].period
                stream_dict[key] = list(filter(lambda x: x.ctx.get_timestamp() > _p.event_time-period, stream_t))
            else:
                stream_dict[key] = []
        return stream_dict, period

    @staticmethod
    def join_downscaling_streams(_p, stream_dict, pred_data:Dict[int, Dict]=None):
        joint_stream_dict = {}
        if pred_data is None:
            pred_data = _p.pred_data

        for key in joint_stream_dict:
            attr_dict = pred_data[key]
            if attr_dict["reDistPattn"] == "downscaling":
                # name parse
                # remove the thread number at the end of the name
                thread_n = key.split('_')[-2]
                troughput_n = key.split('_')[-1]
                task_n = key.replace("_"+thread_n, "").replace("_"+troughput_n, "")
                if task_n not in joint_stream_dict:
                    joint_stream_dict[task_n] = []
                joint_stream_dict[task_n].extend(stream_dict[key])
            else:
                joint_stream_dict[key] = stream_dict[key]
        return joint_stream_dict

    @staticmethod
    def join_valid(_p:ProcessBase, matched_pair:dict, pred_data:Dict[int, Dict]=None):
        """
        check the joint valid
        """
        # check the joint valid
        joint_valid = {}
        if pred_data is None:
            pred_data = _p.pred_data
        for key in pred_data:
            attr_dict = pred_data[key]
            valid = key in matched_pair
            if attr_dict["reDistPattn"] == "downscaling":
                # name parse
                # remove the thread number at the end of the name
                thread_n = key.split('_')[-2]
                troughput_n = key.split('_')[-1]
                task_n = key.replace("_"+thread_n, "").replace("_"+troughput_n, "")
                joint_valid.update({task_n:joint_valid.get(task_n, False) + valid})
            elif not valid:
                return False

        assert (np.array(list(joint_valid.values()))<=1).all()
        return np.array(list(joint_valid.values())).all()

    @classmethod
    def check_data_depends(cls, _p, buffer=None, glb_n_task_dict=None, min_event_time=None, event_cache:EventCache=None):
        """
        The computation of an operator with multiple input streams is triggered 
        whenever all of its input streams have emitted at least one element with a timestamp 
        equal to or greater than the current watermark. 
        """

        stream_dict, period = cls.extract_stream_from_buffer(_p, buffer, glb_n_task_dict)
        # join the downscaling streams
        # stream_dict = cls.join_downscaling_streams(_p, stream_dict)
        # return cls.chk_data_trigger(_p, stream_dict, min_event_time)

        if period is None:
            return {}, False

        matched_pair = {}
        if event_cache is None:
            pred_data = _p.pred_data
        else:
            pred_data = event_cache[_p.pid]
        if min_event_time < float("inf"):
            # iterate the stream dict reversely, event time is in the descending order
            for key in pred_data:
                attr_dict = pred_data[key]
                if attr_dict["reDistPattn"] == "downscaling":
                    interval = period / attr_dict["factor"]
                else:
                    interval = period
                stream = stream_dict[key]
                if len(stream)>0:
                    # find the closest event \textbf{before} the min_event_time
                    # i.e., the event time is in the range of [min_event_time-period, min_event_time)
                    # Assersion: the event time is in the ascending order
                    # Assersion: every stream has same period
                    for element in stream:
                        assert element.period == period
                        if round(element.ctx.get_timestamp(), numerical_tol_bit) <= round(min_event_time, numerical_tol_bit) < round(element.ctx.get_timestamp()+interval, numerical_tol_bit):
                            matched_pair.update({key:element})
                            break
        else:
            return {}, False
        # check the data trigger
        if cls.join_valid(_p, matched_pair, pred_data):
            # pop matched event
            for key in pred_data:
                if key in matched_pair:
                    stream_dict[key].pop()
            return matched_pair, True
        else:
            # TODO: watermark strategy
            pass
            return {}, False

    @classmethod
    def chk_data_trigger(cls, _p:ProcessBase, stream_dict:dict=None, min_event_time=None, pred_data:Dict[int, Dict]=None):
        if pred_data is None:
            pred_data = _p.pred_data

        if stream_dict is None:
            # stream_dict = {key:pred_data[key]['event_queue'].queue for key in pred_data}
            stream_dict = {key:pred_data[key]['event_queue'] for key in pred_data}

        for key in pred_data:
            stream = stream_dict[key]
            while len(stream)>0:
                if stream[0].ctx.get_timestamp() <= _p.event_time:
                    # the timestamp of the element is in the past
                    # stream.pop(0)
                    stream.get()
                else:
                    break

        # get the minimum event time
        min_event_time_t = float("inf")

        # detect the data trigger
        for key in pred_data:
            stream = stream_dict[key]
            if len(stream)>0:
                # get the minimum event time
                if stream[0].ctx.get_timestamp() < min_event_time_t:
                    min_event_time_t = stream[0].ctx.get_timestamp()
                    period = stream[0].period
        if not min_event_time_t < float("inf") or min_event_time is None:
            min_event_time = min_event_time_t

        matched_pair = {}
        if min_event_time < float("inf"):
            # iterate the stream dict reversely, event time is in the descending order
            for key in pred_data:
                stream = stream_dict[key]
                if len(stream)>0:
                    # find the closest event \textbf{after} the min_event_time
                    # i.e., the event time is in the range of [min_event_time, min_event_time+period)
                    # Assersion: the event time is in the ascending order
                    # Assersion: every stream has same period
                    for element in stream:
                        assert element.period == period
                        if round(element.ctx.get_timestamp() - period, numerical_tol_bit) < round(min_event_time, numerical_tol_bit) <= round(element.ctx.get_timestamp(), numerical_tol_bit):
                            matched_pair.update({key:element})
                            break
        else:
            return {}, False
        
        # check the data trigger
        if cls.join_valid(_p, matched_pair, pred_data):
            # event time is advanced
            for key in pred_data:
                if key in matched_pair:
                    # stream_dict[key].pop()
                    stream_dict[key].remove(matched_pair[key])
            return matched_pair, True
        else:
            # TODO: watermark strategy
            pass
            return {}, False
    
    @classmethod
    def check_trigger(cls, _p:ProcessBase, event_cache:EventCache=None, 
                      trigger_cache:TriggerCache=None):
        """
        if all the predecessor tasks are completed, return True
        """
        if len(_p.pred_ctrl):
            if trigger_cache is None:
                pred_ctrl = _p.pred_ctrl
                event_triggers = _p.event_triggers
            else:
                pred_ctrl = trigger_cache[_p.pid]
                event_triggers = trigger_cache.sensor_cache[_p.pid]

            for key in pred_ctrl.keys():
                if not pred_ctrl[key]["valid"]:
                    return False
            # to avoid the duplicated context in the condition of job migration
            if len(_p.msg_cache) == 0:
                _p.build_ctx()
            _p.update_ctx("trigger", pred_ctrl=pred_ctrl)
            # clear the pred_ctrl valid flag
            _p.reset_depends(type="ctrl", pred_ctrl=pred_ctrl)
            event_triggers.pop(0)
            return True
        else:
            if event_cache is None:
                pred_data = _p.pred_data
            else:
                pred_data = event_cache[_p.pid]

            matched_pair, status = cls.chk_data_trigger(_p, pred_data=pred_data)
            if status:
                # to avoid the duplicated context in the condition of job migration
                if len(_p.msg_cache) == 0:
                    _p.build_ctx()
                # cache the context of the upstream src node
                _p.update_ctx('upstream', matched_pair=matched_pair)
                return True

    @classmethod
    def chk_release(cls, curr_t, inactive_list:List[ProcessBase], active_list, 
                    event_cache:EventCache=None, trigger_cache:TriggerCache=None,
                    bin_event_flg:bool=False, 
                    bin_name:str="", DEBUG_FG:bool=False,
                    ):
        """
        check release
            1. check the dependencies of the tasks in inactive list
            2. if the dependencies are satisfied, move the task to the wait queue
        """

        l_active:List[ProcessBase] = []

        for _p in inactive_list:
            if cls.check_trigger(_p, event_cache=event_cache, trigger_cache=trigger_cache):
                l_active.append(_p)


        if bin_name and len(l_active) and not bin_event_flg:
            bin_event_flg = True
            print(f"({bin_name})")
            
        for _p in l_active:
            inactive_list.remove(_p)
            _p.update_deadline_from_timestamp()
            if _p.deadline < curr_t and _p.task.criticality == "hard":
                print(f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) MISSED DEADLINE @ {curr_t:.6f}/{_p.msg_cache[0].get_timestamp():.6f}!!")
                print(f"		{_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) deadline: {_p.deadline:.6f}")
                _p.task.missed_deadline_count += 1
            else:
                if _p.deadline < curr_t: 
                    warnings.warn(f"Task {_p.task.id}:{_p.task.name}({_p.pid}) violate timing constraint @ {_p.deadline:.6f}/{_p.msg_cache[0].get_timestamp():.6f}!!")
                    print(f"		{_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) deadline: {_p.deadline:.6f}")
                _p.handle_process_load_var()
                _p.release_util(curr_t, active_list)
        return bin_event_flg 


if __name__ == "__main__":
    from model.buffer import Buffer
    class ProcessSample(ProcessBase):
        def __init__(self, pid=0):
            self.pred_data = {}
            self.event_time = -1
            self.counter = 0
            self.new_event = False
            self.pid = pid

    # Create some sample data streams
    class Data:
        def __init__(self, timestamp, period, pid=0, data_type='output'):
            self.ctx = Context(timestamp)
            self.period = period
            self.pid = pid
            self.data_type = data_type
            self.size = 1

    class Context:
        def __init__(self, timestamp):
            self.timestamp = timestamp
        
        def get_timestamp(self):
            return self.timestamp

    sim_hyper_p = 6
    hyper_p_size = 24

    def test_chk_data_depends():

        def inject_data2buffer(buffer:Buffer, pred_name, glb_n_task_dict, period=24, offset=0):
            pid = glb_n_task_dict[pred_name].pid
            queue = buffer.buffer_mux('output')[pid] if pid in buffer.buffer_mux('output') else []
            buffer.put(Data(len(queue)*period+offset, period, pid, 'output'))

        buffer = Buffer(sort_fn=lambda x:(x.ctx.get_timestamp()))
        # Test case
        p_Lidar_based_3dDet_0, p_Prediction_0, p_Prediction_1, p_Prediction_2, p_Steering_speed_0 = build_sample_process()

        glb_n_task_dict = {"stream_lidar":ProcessSample(0), "stream_imu":ProcessSample(1), "stream_camera_stereo_0":ProcessSample(2), 
                           "stream_camera_stereo_1":ProcessSample(3), "stream_camera_surround_0":ProcessSample(4), 
                           "stream_camera_surround_1":ProcessSample(5), "stream_camera_surround_2":ProcessSample(6)}

        # Create some sample data
        # 0.096132, 0.142027, 0.242028 0.342029 *240
        # stream_lidar add a data [(24*i, 24)] @[23, 34, 58, 82]
        
        # 0.085619, 0.185621, 0.285628, 0.385624 *240
        # stream_camera_stereo_0 add [(24*i, 24)] @[22, 46, 70, 94]

        # 0.097755, 0.197751, 0.297763, 0.397754 *240
        # stream_camera_surround_0 add [(24*i, 24)] @[23, 47, 71, 95]

        # 0.131092, 0.231083, 0.331084, 0.431086 *240
        # stream_camera_surround_1 add [(24*i, 24)] @[31, 55, 79, 103]

        # 0.164419 0.264426 0.364416 0.464418 *240
        # stream_camera_surround_2 add [(24*i+16, 24)] @[39, 63, 87, 111]

        for i in range(sim_hyper_p):
            print('hyper_p', i)
            for j in range(hyper_p_size):
                t = i * hyper_p_size + j
                # if t in [23, 34, 58, 82]:
                #     inject_data2buffer(p_Lidar_based_3dDet_0, 'stream_lidar')
                #     inject_data2buffer(p_Prediction_0, 'stream_lidar')
                #     inject_data2buffer(p_Prediction_1, 'stream_lidar')
                #     inject_data2buffer(p_Prediction_2, 'stream_lidar')
                # if t in [22, 46, 70, 94]:
                #     inject_data2buffer(p_Lidar_based_3dDet_0, 'stream_camera_stereo_0')
                # if t in [23, 47, 71, 95]:
                #     inject_data2buffer(p_Prediction_0, 'stream_camera_surround_0')
                # if t in [31, 55, 79, 103]:
                #     inject_data2buffer(p_Prediction_1, 'stream_camera_surround_1', offset=8)
                # if t in [39, 63, 87, 111]:
                #     inject_data2buffer(p_Prediction_2, 'stream_camera_surround_2', offset=16)

                if t in [23, 34, 58, 82]:
                    inject_data2buffer(buffer, 'stream_lidar', glb_n_task_dict)
                if t in [22, 46, 70, 94]:
                    inject_data2buffer(buffer, 'stream_camera_stereo_0', glb_n_task_dict)
                if t in [23, 47, 71, 95]:
                    inject_data2buffer(buffer, 'stream_camera_surround_0', glb_n_task_dict)
                if t in [31, 55, 79, 103]:
                    inject_data2buffer(buffer, 'stream_camera_surround_1', glb_n_task_dict, offset=8)
                if t in [39, 63, 87, 111]:
                    inject_data2buffer(buffer, 'stream_camera_surround_2', glb_n_task_dict, offset=16)

                for _p, _p_name in zip([p_Lidar_based_3dDet_0, p_Prediction_0, p_Prediction_1, p_Prediction_2], 
                                       ['p_Lidar_based_3dDet_0', 'p_Prediction_0', 'p_Prediction_1', 'p_Prediction_2']):
                    if True:
                        matched_pair, status = WatermarkStrategy.check_data_depends(_p, buffer, glb_n_task_dict)
                        if status:
                            _p.event_time = max([matched_pair[key].ctx.get_timestamp() for key in matched_pair])
                            print(f"Event time {_p_name}: {_p.event_time}")
                            for key, data_event in matched_pair.items():
                                print(f"Stream: {key}, Timestamp: {data_event.ctx.get_timestamp()}")
                            print()
                        else:
                            # print("No matched events")
                            pass
                        _p.new_event = False

    # Test case
    def test_chk_data_trigger():
        def inject_a_data_event(_p, pred_name, period=24, offset=0):
            queue = _p.pred_data[pred_name]['event_queue']
            counter =  _p.pred_data[pred_name]['counter']
            queue.append(Data(counter*period+offset, period))
            counter =  _p.pred_data[pred_name].update({'counter': counter+1})
            _p.new_event = True

        p_Lidar_based_3dDet_0, p_Prediction_0, p_Prediction_1, p_Prediction_2, p_Steering_speed_0 = build_sample_process()

        # Create some sample data

        # 0.096132, 0.142027, 0.242028 0.342029 *240
        # stream_lidar add a data [(24*i, 24)] @[23, 34, 58, 82]
        
        # 0.085619, 0.185621, 0.285628, 0.385624 *240
        # stream_camera_stereo_0 add [(24*i, 24)] @[22, 46, 70, 94]

        # 0.097755, 0.197751, 0.297763, 0.397754 *240
        # stream_camera_surround_0 add [(24*i, 24)] @[23, 47, 71, 95]

        # 0.131092, 0.231083, 0.331084, 0.431086 *240
        # stream_camera_surround_1 add [(24*i, 24)] @[31, 55, 79, 103]

        # 0.164419 0.264426 0.364416 0.464418 *240
        # stream_camera_surround_2 add [(24*i+16, 24)] @[39, 63, 87, 111]

        for i in range(sim_hyper_p):
            print('hyper_p', i)
            for j in range(hyper_p_size):
                t = i * hyper_p_size + j
                if t in [23, 34, 58, 82]:
                    inject_a_data_event(p_Lidar_based_3dDet_0, 'stream_lidar')
                    inject_a_data_event(p_Prediction_0, 'stream_lidar')
                    inject_a_data_event(p_Prediction_1, 'stream_lidar')
                    inject_a_data_event(p_Prediction_2, 'stream_lidar')
                if t in [22, 46, 70, 94]:
                    inject_a_data_event(p_Lidar_based_3dDet_0, 'stream_camera_stereo_0')
                if t in [23, 47, 71, 95]:
                    inject_a_data_event(p_Prediction_0, 'stream_camera_surround_0')
                if t in [31, 55, 79, 103]:
                    inject_a_data_event(p_Prediction_1, 'stream_camera_surround_1', offset=8)
                if t in [39, 63, 87, 111]:
                    inject_a_data_event(p_Prediction_2, 'stream_camera_surround_2', offset=16)
                # Call chk_data_trigger
                # matched_pair, status = WatermarkStrategy.chk_data_trigger(p_Lidar_based_3dDet_0)
                for _p, _p_name in zip([p_Lidar_based_3dDet_0, p_Prediction_0, p_Prediction_1, p_Prediction_2], 
                                       ['p_Lidar_based_3dDet_0', 'p_Prediction_0', 'p_Prediction_1', 'p_Prediction_2']):
                    if _p.new_event:
                        matched_pair, status = WatermarkStrategy.chk_data_trigger(_p)
                        if status:
                            _p.event_time = max([matched_pair[key].ctx.get_timestamp() for key in matched_pair])
                            print(f"Event time {_p_name}: {_p.event_time}")
                            for key, data_event in matched_pair.items():
                                print(f"Stream: {key}, Timestamp: {data_event.ctx.get_timestamp()}")
                            print()
                        else:
                            # print("No matched events")
                            pass
                        _p.new_event = False

    def build_sample_process():
        p_Lidar_based_3dDet_0 = ProcessSample()
        p_Prediction_0 = ProcessSample()
        p_Prediction_1 = ProcessSample()
        p_Prediction_2 = ProcessSample()
        p_Steering_speed_0 = ProcessSample()

        # Create some sample data streams: lidar, imu, camera_stereo (phase 0-1), camera_surround (phase 0-2)
        stream_lidar = [] # cycle = 24, offset=0
        stream_imu = [] # cycle = 1, offset=0
        stream_camera_stereo_0 = [] # cycle = 24, offset=0
        stream_camera_stereo_1 = [] # cycle = 24, offset=12
        stream_camera_surround_0 = [] # cycle = 24, offset=0
        stream_camera_surround_1 = [] # cycle = 24, offset=8
        stream_camera_surround_2 = [] # cycle = 24, offset=16
        
        # Initialize the process data
        p_Lidar_based_3dDet_0.pred_data = {
            'stream_lidar': {'event_queue': [], 'counter':0, 'reDistPattn':'one2one'},
            'stream_camera_stereo_0': {'event_queue': [], 'counter':0, 'reDistPattn':'one2one'},
        }
        p_Lidar_based_3dDet_0.event_time = -1

        p_Prediction_0.pred_data = {
            'stream_lidar': {'event_queue': [], 'counter':0, 'reDistPattn':'one2one'},
            'stream_camera_surround_0': {'event_queue': [], 'counter':0, 'reDistPattn':'one2one'},
        }
        p_Prediction_0.event_time = -1

        p_Prediction_1.pred_data = {
            'stream_lidar': {'event_queue': [], 'counter':0, 'reDistPattn':'one2one'},
            'stream_camera_surround_1': {'event_queue': [], 'counter':0, 'reDistPattn':'one2one'},
        }
        p_Prediction_1.event_time = -1

        p_Prediction_2.pred_data = {
            'stream_lidar': {'event_queue': [], 'counter':0, 'reDistPattn':'one2one'},
            'stream_camera_surround_2': {'event_queue': [], 'counter':0, 'reDistPattn':'one2one'},
        }
        p_Prediction_2.event_time = -1

        p_Steering_speed_0.pred_data = {
            'stream_imu': {'event_queue': [], 'counter':0, 'reDistPattn':'one2one'},
            'stream_camera_surround_0': {'event_queue': [], 'counter':0, 'reDistPattn':'one2one'},
            'stream_camera_surround_1': {'event_queue': [], 'counter':0, 'reDistPattn':'one2one'},
            'stream_camera_surround_2': {'event_queue': [], 'counter':0, 'reDistPattn':'one2one'},
        }
        p_Steering_speed_0.event_time = -1
        return p_Lidar_based_3dDet_0,p_Prediction_0,p_Prediction_1,p_Prediction_2,p_Steering_speed_0
        
    # Run the test
    # test_chk_data_trigger()
    test_chk_data_depends()

    
