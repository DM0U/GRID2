#!/usr/bin/env python3

import h5py
import numpy as np
import denoise as dn



###############################################################################
# 固定参数的修改模块

# 光子计数的时间步长，单位s
time_interval = 100 / 1000

# 当弱信号接受卫星的最大峰时刻与强卫星的最大峰时刻差距超过其时，认为弱信号无法读取
time_delta_max = 3

# 判断是否为有效峰 (波峰 / sigma判据)
# TODO: Not used yet
rate_level = 4


###############################################################################
# 以下参数不需要手动修改

# 最大允许时差对应的最大允许索引差
id_delta_max = time_delta_max / time_interval



###############################################################################
# 能量边界值读取

with h5py.File('data/deposition.h5', 'r') as source:
    Ebounds = np.array(source['Matrix'].attrs['Edges'])


###############################################################################
class gamma_event:
    '''
    gamma暴事件的记录
    允许调用函数
        __init__(self, file_path, event)
            事件初始化函数
    包含变量
        self.Time
            事件的时间轴，一维numpy数组，
            间隔由全局变量 time_interval 决定
        self.Eng_out
            每个卫星每个探测器每个输出能道的每个时间间隔的总能量
            索引顺序：cube, det, eng, time ，4维numpy数组
        self.Eng_out_rate
            每个卫星每个探测器上的信号质量参数，越大越好
            (均值为0的)信号峰值对信号标准差的比例
            2维numpy数组，对应 (卫星, 探测器)
        self.Cube_witness
            每个卫星上信号最好的探测器编号
            1维numpy数组，对应各个卫星
        self.Id_happen
            每个卫星上测得的gamma暴的峰值时刻的索引值
            为-1说明无法测到信号
            1维numpy数组，对应各个卫星
        self.cube_best
            信号质量最高的卫星编号，int
        self.Id_cube_time_delta
            每个卫星测得的信号始末位置，采用波形跨越0时为标准
            为 (-1,-1) 说明无可测信号
            二维numpy数组，大小为 (num_cube, 2)，
            第二维分别对应开始，结束时刻的索引
        self.Eng_out_gamma_all
            gamma暴中各个卫星各个输出能道的有效能量结果
            3维numpy数组，分别对应(卫星, 探测器, 输出能道)
    '''


    def __init__(self, file_path, event):
        '''
        gamma暴事件的初始化
        包括数据的读入，gamma暴时刻的分析与各个输出能道的能量分析
            input:
                file_path
                    .h5文件的路径
                event
                    需要阅读的事件在.h5文件中的名称
            output:
                self.Time
                    事件的时间轴，一维numpy数组，
                    间隔由全局变量 time_interval 决定
                self.Eng_out
                    每个卫星每个探测器每个输出能道的每个时间间隔的总能量
                    索引顺序：cube, det, eng, time ，4维numpy数组
                self.Eng_out_rate
                    每个卫星每个探测器上的信号质量参数，越大越好
                    (均值为0的)信号峰值对信号标准差的比例
                    2维numpy数组，对应 (卫星, 探测器)
                self.Cube_witness
                    每个卫星上信号最好的探测器编号
                    1维numpy数组，对应各个卫星
                self.Id_happen
                    每个卫星上测得的gamma暴的峰值时刻的索引值
                    为-1说明无法测到信号
                    1维numpy数组，对应各个卫星
                self.cube_best
                    信号质量最高的卫星编号，int
                self.Id_cube_time_delta
                    每个卫星测得的信号始末位置，采用波形跨越0时为标准
                    为 (-1,-1) 说明无可测信号
                    二维numpy数组，大小为 (num_cube, 2)，
                    第二维分别对应开始，结束时刻的索引
                self.Eng_out_gamma_all
                    gamma暴中各个卫星各个输出能道的有效能量结果
                    3维numpy数组，分别对应(卫星, 探测器, 输出能道)
        '''

        # 数据读入
        self.__read_event__(file_path, event)
        

        # 时间分析
        self.__analyse_event_time__()
        

        # 输出能道分析
        self.__analyse_event_eng__()
        




    def __read_event__(self, file_path, event):
        '''
        根据输入的路径名在__init__函数中初始化对应事件
        ！！！禁止外部调用！！！
            input:
                file_path
                    .h5文件的路径
                event
                    需要阅读的事件在.h5文件中的名称
            output:
                self.Time
                    事件的时间轴，一维numpy数组，
                    间隔由全局变量 time_interval 决定
                self.Eng_out
                    每个卫星每个探测器每个输出能道的每个时间间隔的总能量
                    索引顺序：cube, det, eng, time ，4维numpy数组 
        '''
        global time_interval, Ebounds

        # 事件的读入
        # 需要一次性读完，否则大量的io操作会消耗过多时间
        with h5py.File(file_path, 'r') as source:

            light_doc = {}
            for cube in range(7):
                light_doc[cube] = {}
                for det in range(4):
                    light_doc[cube][det] = source[event]['response_E_T'] \
                                                ['D' + str(cube)][str(det)][...]


        # 对应同一个gamma事件，采用同一个时间段进行衡量
        # 计算共同的时间间隔
        Time_sup = np.empty((7, 4))
        Time_inf = np.empty((7, 4))
        # 此处的 for 没有替换方法，但是几乎不对运行时长造成影响
        for cube in range(7):
            for det in range(4):
                Time_sup[cube, det] = light_doc[cube][det][:,0].max()
                Time_inf[cube, det] = light_doc[cube][det][:,0].min()
        # 每个时刻t的计数的范围对应t~t+dt
        time_sup = np.floor(Time_sup.min() - time_interval)
        time_inf = np.ceil(Time_inf.max())


        # 得出事件时间与时间边界
        self.Time = np.arange(time_inf, time_sup, time_interval)
        time_num = self.Time.size    # 不直接计算，防止float的精度错误
        Tbounds = np.array([*(self.Time), self.Time[-1] + time_interval])
        

        # 广播规则的预备，对应时间上下限的准备
        Inf = np.empty((time_num, 100, 2))
        Inf[:,:,0] = np.expand_dims(self.Time ,1)
        Inf[:,:,1] = np.expand_dims(Ebounds[:100], 0)

        Sup = np.empty((time_num, 100, 2))
        Sup[:,:,0] = np.expand_dims(self.Time + time_interval ,1)
        Sup[:,:,1] = np.expand_dims(Ebounds[1:], 0)


        # 光子计数模块
        # 初始索引顺序：cube, det, time, Eng
        # time, Eng多一个长度用于存储超界的记录
        self.Eng_out = np.zeros((7, 4, time_num + 1, 100 + 1))
        for cube in range(7):
            for det in range(4):
                # id = -1 , 对应尾部废弃项
                Id_time = (np.expand_dims(light_doc[cube][det][:, 0], 1)\
                            >= Tbounds).sum(axis=1) - 1
                # Id_time[Id_time == -1] = Tbounds.size - 1
                Id_eng = (np.expand_dims(light_doc[cube][det][:, 1], 1)\
                           >= Ebounds).sum(axis=1) - 1
                # Id_eng[Id_eng == -1] = Ebounds.size - 1

                # 对能量计数并储存到Eng_out中
                np.add.at(self.Eng_out[cube, det], (Id_time, Id_eng), \
                          light_doc[cube][det][:,1])

        # 使得时间值从 0 开始，方便阅读   
        self.Time -= self.Time[0] 

        # 最终索引顺序：cube, det, eng, time
        # 去除范围外的部分（对应eng, time的最后一项）
        self.Eng_out = self.Eng_out[:, :, :-1, :-1].swapaxes(2, 3)
    
    def __analyse_event_time__(self):
        '''
        对读入的事件 self.Time, self.Eng_out 信息做分析
            得出信号发生时间（中心时刻与始末时刻）
        ！！！禁止外部调用！！！
            input:
                self.Time
                    时刻表
                    一维numpy数组，对应各个时间
                self.Eng_out
                    每个卫星每个探测器每个输出能道的每个时间间隔的总能量
                    索引顺序：cube, det, eng, time ，4维numpy数组 
            output:
                self.Eng_out_rate
                    每个卫星每个探测器上的信号质量参数，越大越好
                    (均值为0的)信号峰值对信号标准差的比例
                    2维numpy数组，对应 (卫星, 探测器)
                self.Cube_witness
                    每个卫星上信号最好的探测器编号
                    1维numpy数组，对应各个卫星
                self.Id_happen
                    每个卫星上测得的gamma暴的峰值时刻的索引值
                    为-1说明无法测到信号
                    1维numpy数组，对应各个卫星
                self.cube_best
                    信号质量最高的卫星编号，int
                self.Id_cube_time_delta
                    每个卫星测得的信号始末位置，采用波形跨越0时为标准
                    为 (-1,-1) 说明无可测信号
                    二维numpy数组，大小为 (num_cube, 2)，
                    第二维分别对应开始，结束时刻的索引
        '''
        global time_interval, id_delta_max

        # 处理依据： 平均值降为0后以最大峰跨越0为始末时刻
        Eng_dn = dn.denoise_curve_by_fft(self.Eng_out[:,:,0], 40)
        Eng_dn -= np.expand_dims(Eng_dn.mean(axis=-1), 2)

        # Eng_dn 的处理，需要指出，当信号过弱时可能会出现最大峰在边缘的情况
        # 由于采用的判断依据是最大峰两边为 0 的位置，
        #   为了保证算法的可行性需要保证最大峰两侧存在0位置
        # 为了运行效率，此处采用暴力将始末时刻的数值设为 0 的做法
        '''
        实际上应该截取的始末时刻，但该方法运算速度慢
            time_size = Eng_dn.shape[-1]
            Id_inf = np.argmax((Eng_dn < 0), axis=-1)
            Id_sup = time_size - 1 - np.argmax((Eng_dn < 0)[:,:,::-1], axis=-1)
        '''
        Eng_dn[:,:,0] = 0
        Eng_dn[:,:,-1] = 0

        # gamma信号质量的判据，采用 (max / sigma) 为依据
        id_max = Eng_dn.argmax(axis=-1)
        Eng_max = Eng_dn.max(axis=-1)
        Eng_min = Eng_dn.min(axis=-1)  # TODO：maybe it should be deleted.
        Eng_sigma = np.sqrt(Eng_dn.var(axis=-1))

        # 为每个卫星寻找信赖的探测器
        # TODO: maybe the best choice is to sum all detectors?
        self.Eng_rate = Eng_max / Eng_sigma
        self.Cube_witness = self.Eng_rate.argmax(axis=-1)

        # 每个卫星认为的gamma暴最强的时刻
        self.Id_happen = id_max[np.arange(7), self.Cube_witness]

        # 信号质量最好的卫星，作为gamma暴发生时刻的基准
        self.cube_best = self.Eng_rate.max(axis=-1).argmax()

        # 信号的始末时刻以最高质量的探测器上的最低能道为依据
        Eng_cube_evidence = Eng_dn[np.arange(7), self.Cube_witness]
        

        # 每个卫星对应的gamma暴始末时刻，为-1,-1说明无法检测到信号
        #   判据： 与最佳信号的中心时刻差值过大
        # 对于for循环，当time_interval = 0.1，重复运行以下代码10w次约22s
        self.Id_cube_time_delta = np.empty((7,2), dtype=int)
        for cube in range(7):
            if np.abs(self.Id_happen[cube] - self.Id_happen[self.cube_best]) > \
                    id_delta_max:
                self.Id_cube_time_delta[cube] = -1, -1
                self.Id_happen[cube] = -1
            else:
                Id_filt = np.argwhere(Eng_cube_evidence[cube] <= 0).squeeze()
                self.Id_cube_time_delta[cube] = \
                    Id_filt[Id_filt < self.Id_happen[cube]].max(),\
                    Id_filt[Id_filt > self.Id_happen[cube]].min()

    def __analyse_event_eng__(self):
        '''
        对读入的事件计算各输出能道的能量值
        ！！！禁止外部调用！！！
            input:
                self.Eng_out
                    每个卫星每个探测器每个输出能道的每个时间间隔的总能量
                    索引顺序：cube, det, eng, time ，4维numpy数组 
                self.Id_cube_time_delta
                    每个卫星测得的信号始末位置，采用波形跨越0时为标准
                    为 (-1,-1) 说明无可测信号
                    二维numpy数组，大小为 (num_cube, 2)，
                    第二维分别对应开始，结束时刻的索引
            output:
                self.Eng_out_gamma_all
                    gamma暴中各个卫星各个输出能道的有效能量结果
                    3维numpy数组，分别对应(卫星, 探测器, 输出能道)
        '''

        # 必须采用for循环（各个卫星的时间段不等长，不能广播）
        #   运行10w次，time_interval=0.1s，总时长约19s
        # 若 Id_cube_time_delta 相等(即-1),说明无信号
        self.Eng_out_gamma_all = np.empty((7, 4, 100))
        for cube in range(7):
            if self.Id_cube_time_delta[cube,0] < self.Id_cube_time_delta[cube,1]:
                eng_gamma = self.Eng_out[cube,:,:,\
                    self.Id_cube_time_delta[cube,0]:self.Id_cube_time_delta[cube,1]]

                self.Eng_out_gamma_all[cube] = eng_gamma.sum(axis=-1) - \
                    self.Eng_out[cube].mean(axis=-1) * \
                        (self.Id_cube_time_delta[cube,1] - self.Id_cube_time_delta[cube,0])
            else:
                self.Eng_out_gamma_all[cube] = 0
