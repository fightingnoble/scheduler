"""Unused message-reading helpers, preserved without behavior changes."""

from model.buffer import Data


def msg_read(msg_queue, curr_t, glb_name_p_dict, process_dict, buffer, bin_name, bin_event_flg):
    msg_list = []
    # read out all message and clear the message pipe
    while not msg_queue.empty():
        msg_list.append(msg_queue.get())
    if msg_list:
        for _p in process_dict.values():
            for key, attr in _p.pred_data.items():
                # if msg_pipe.filter(key):
                if msg_filter(msg_list, key):
                    attr["valid"] = True
                    attr["time"] = curr_t
                    # TODO: fix the event time as the actual time
                    if bin_name and not bin_event_flg:
                        bin_event_flg = True
                        print(f"({bin_name})")
                    print(f"		{_p.task.name} received event {key:s} @ {curr_t:.6f}")
                    buffer.put(Data(glb_name_p_dict[key].pid, glb_name_p_dict[key].io_time, (0,), "output", curr_t, 1/glb_name_p_dict[key].task.freq, glb_name_p_dict[key].task.period))
    return bin_event_flg


def msg_filter(msg_list:list, keyword:str):
    return [msg for msg in msg_list if keyword in msg]
