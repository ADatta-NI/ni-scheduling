import numpy as np

J = 3
O = 4
M = 3
R = 8

def read_init_reconfig_times():
    rt_m_0k = np.empty((0, R))
    f = open('rt_m_0k.data', 'r')
    for line in f.readlines():
        l = np.array([float(x) for x in line.split('\t')])
        rt_m_0k = np.append(rt_m_0k, np.array([l]), axis=0)
    
    rt_m_0k = np.reshape(rt_m_0k, (M, R))
    return rt_m_0k


print(read_init_reconfig_times())
