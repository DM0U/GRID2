#!/usr/bin/env python3

import h5py
import numpy as np
from numpy import sin, cos, arcsin, arccos, arctan

###############################################################################
# 初始化全局变量
file_path = ''
event_name = ''

Cube_info = None
Cube_coor_frame_x = None
Cube_coor_frame_y = None
Cube_coor_frame_z = None

###############################################################################
# 用户接口

# 事件初始化
def event_init(event, path=None):
    '''
    单次测量事件的初始化，并记录初始化的事件名称
        500次读取约1.5s, 认为不需要直接读入所有数据
        input:
            event:
                h5文件中的事件名，str
            path:
                h5文件路径名，str 如果没有默认采用上一次的路径
        output:
            /
        error:
            'No path to get documents'
                文件路径未初始化
            'No such event in this document'
                文件中不存在目标事件
    '''
    global file_path, event_name
    global Cube_info


    # 读入路径的确认
    if path != None:
        file_path = path
    if file_path == '':
        raise Exception('No path to get documents')
    
    # event_name用于在之后的其它函数中检测是否有更新事件
    if event_name == event:
        return
    else:
        event_name = event
    
    Cube_info = []
    with h5py.File(file_path, 'r') as ipt:

        if event not in ipt.keys():
            raise Exception('No such event in this document\nevent: ' + event)

        for i in range(7):
            cube_name = 'detector' + str(i)
            Cube_info.append(ipt[event]['detector_info'][cube_name][...])

    Cube_info = np.array(Cube_info)
    coor_init()

# 功能实现模块
# 包括相对角度的计算，被遮挡的计算，延时的计算等
def get_angular_coor_in_cubes(dir_from_earth_cart):
    '''
    输入星源相对地球的方向单位笛卡尔矢量，
    计算其相对各个卫星的入射方向（顶角与旋转角）
        input:
            dir_from_earth_cart
                从地球出发指向入射方向的笛卡尔单位矢量
        output:
            Theta, Phi
                相对各个卫星的入射顶角与旋转角
    '''
    global Cube_coor_frame_x, Cube_coor_frame_y, Cube_coor_frame_z

    dir_from_earth = dir_from_earth_cart

    # 入射的顶角
    Theta = arccos((dir_from_earth * Cube_coor_frame_z).sum(axis=1))

    # 若sin(theta) = 0, 即theta = 0或pi，phi不重要，只需要注意除以0报错
    Sin_theta = sin(Theta)
    Sin_theta[Sin_theta == 0] = 1 
    # 判断y分量正负性以决定phi的范围
    Comp_y = ((dir_from_earth * Cube_coor_frame_y).sum(axis=1) < 0)
    
    Phi = arccos((dir_from_earth * Cube_coor_frame_x).sum(axis=1) / Sin_theta)
    Phi[Comp_y] = 2 * np.pi - Phi[Comp_y]

    return Theta, Phi


###############################################################################
def coor_init():
    '''
    警告: 禁止外部文件调用！！！
    根据Cube_info初始化7个卫星的坐标架，内部调用保证正确初始化
        input:
            Cube_info
                各个Cube的信息，全局变量
        output:
            Cube_coor_frame_[xi]
                立方星3个坐标正方向的方向单位矢量，全局变量
                xi 可取 x, y, z
    '''
    global Cube_info
    global Cube_coor_frame_x, Cube_coor_frame_y, Cube_coor_frame_z


    _Phi = Cube_info[:, 0]
    _Theta = np.pi / 2 - Cube_info[:, 1]

    Cube_coor_frame_x = np.empty((7, 3))
    Cube_coor_frame_x[:, 0] = sin(_Theta) * cos(_Phi)
    Cube_coor_frame_x[:, 1] = sin(_Theta) * sin(_Phi)
    Cube_coor_frame_x[:, 2] = cos(_Theta)


    _Phi = Cube_info[:, 2]
    _Theta = np.pi / 2 - Cube_info[:, 3]

    Cube_coor_frame_z = np.empty((7, 3))
    Cube_coor_frame_z[:, 0] = sin(_Theta) * cos(_Phi)
    Cube_coor_frame_z[:, 1] = sin(_Theta) * sin(_Phi)
    Cube_coor_frame_z[:, 2] = cos(_Theta)

    Cube_coor_frame_y = np.cross(Cube_coor_frame_z, Cube_coor_frame_x)