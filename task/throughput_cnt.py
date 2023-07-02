def throughput_counter(t_s, window_size):
    t_e = t_s + window_size  # End time of the window
    cnt = 0  # Number of tasks completed within the window

    while True:
        curr_t = yield cnt  # Receive current time from the user

        if curr_t > t_e:
            break  # Exit the generator if current time exceeds end time

        cnt += 1  # Increment the task count
    return cnt

def Tp_cnt_caller(t_s, window_size, result_list=None):
    while True:
        cnt = yield from throughput_counter(t_s, window_size)
        print("Number of tasks completed within the window: ", cnt)
        if result_list is not None:
            result_list.append(cnt)