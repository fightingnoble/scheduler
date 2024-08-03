from global_var import AVG_HOP_NUM, LAT_PER_HOP, BW_DRAM, BROADCAST_SCALER, AVG_HEAD_LAT


# cal_lat = lambda lat, lat_jitter=0, var_sl=1, var_ld=1: lat*(1+var_sl) * var_ld + lat_jitter
# slack_comp = lambda slack, lat_jitter=0, var_sl=1, var_ld=1: (slack - lat_jitter) / (1+var_sl) / var_ld 
# def avg_trasfer_time(size, slow_down=1):
#     return cal_lat(AVG_HEAD_LAT + size / BW_DRAM, 0, slow_down, 1) 

cal_lat = lambda lat, lat_jitter=0, var_sl=0, var_ld=1: lat/(1-var_sl) * var_ld + lat_jitter
slack_comp = lambda slack, lat_jitter=0, var_sl=1, var_ld=1: (slack - lat_jitter) * (1-var_sl) / var_ld 
def avg_trasfer_time(size, slow_down=0):
    return cal_lat(AVG_HEAD_LAT + size / BW_DRAM, 0, slow_down, 1) 


def multicast_lat_scaler(data):
    data.io_time *= BROADCAST_SCALER
