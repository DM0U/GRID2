#!/usr/bin/env python3

import h5py
import numpy as np

###############################################################################
# 响应矩阵基础数据的读入与预备
# 最终索引：(theta, phi, Det, Eng_in, Engout)
# 大小：    (7, 12, 4, 40, 100)


#  数据读入
with h5py.File('data/deposition.h5') as ipt_deposition:
    Response_Matrix = ipt_deposition['Matrix'][...]


# 调整 Response_Matrix 形态
Response_Matrix = Response_Matrix.reshape(40, 7, 12, 4, 100)
Response_Matrix = Response_Matrix.swapaxes(0, 1).swapaxes(1, 2).swapaxes(2, 3)



###############################################################################
# 计算需要的响应矩阵并输出
# 使用 get_Response_Matrix 方法统一theta, phi的输入数据类型

def get_Response_Matrix(theta, phi, method='float'):
    '''
    双线性插值，计算 theta, phi 方向入射的响应矩阵
        input:
            theta       
                入射的顶角, 一维 numpy 数组 / float
            phi         
                入射的旋转角,一维 numpy 数组 / float
            method      
                表征输入的类型，默认为'float'
                允许类型'float', 'array'，对应输入数字/numpy数组
        output:
            大小为(size?, det, Engin, Engout)的numpy数组
                第一维的size仅在method为'array'时存在
                表示Theta, Phi的size中各个方向对应的响应矩阵
        Error:
            方法(method)错误                
                'No such method'
            theta/phi非数（int/float)
                'Theta/phi should be a number'
            数组theta与phi规模不匹配        
                'Theta and Phi do not match in size'
            theta/phi类型非method要求的数组 
                'Type of Theta/Phi should be numpy.ndarray'
            theta/phi非一维数组
                'Theta/Phi should be One-dimensional'
    '''
    if method == 'float':
        return get_Response_Matrix_float(theta, phi)
    elif method == 'array':
        return get_Response_Matrix_array(theta, phi)
    else:
        raise Exception('No such method')





def get_Response_Matrix_float(theta, phi):
    '''
    利用双线性插值，计算单一确定值的theta, phi方向入射的响应矩阵
        intput:
            theta       
                入射的顶角, [0, pi], float
            phi         
                入射的旋转角, [0, 2pi), float
        output:
            大小为(det, Engin, Engout)的numpy数组，表示theta, phi方向的响应矩阵
    '''
    global Response_Matrix

    # 类型错误的异常抛出
    if ((type(theta) != float) and (type(theta) != int)) or \
       ((type(phi) != float) and (type(phi) != int)):
       raise Exception('Theta/phi should be a number')

    # phi 方向的处理
    id_phi = phi / 2 / np.pi * 12
    id_phi_inf = int(np.floor(id_phi))
    id_phi_sup = (id_phi_inf + 1) % 12

    delta_id_phi = id_phi - id_phi_inf

    # theta 方向的处理
    id_theta = theta / np.pi * 6
    id_theta_inf = int(np.floor(id_theta))
    id_theta_sup = (id_theta_inf + 1) % 6

    delta_id_theta = id_theta - id_theta_inf


    return \
    (1 - delta_id_phi) * \
        (Response_Matrix[id_theta_inf, id_phi_inf] * (1 - delta_id_theta) + \
        Response_Matrix[id_theta_sup, id_phi_inf] * delta_id_theta) + \
    delta_id_phi * \
        (Response_Matrix[id_theta_inf, id_phi_sup] * (1 - delta_id_theta) + \
        Response_Matrix[id_theta_sup, id_phi_sup] * delta_id_theta)

def get_Response_Matrix_array(Theta, Phi):
    '''
    双线性插值，计算等长一维numpy数组中各项 Theta[i], Phi[i] 方向入射的响应矩阵
        intput:
            Theta       
                入射的顶角, 一维 numpy 数组，与Phi等长
            Phi         
                入射的旋转角,一维 numpy 数组，与Theta等长
        output:
            大小为(size, det, Engin, Engout)的numpy数组
            表示Theta, Phi的size中各个方向对应的响应矩阵
    '''

    size_Theta = Theta.size
    size_Phi = Phi.size

    # 异常的处理与抛出（大小不匹配/类型错误/数组维度错误）
    if size_Theta != size_Phi:
        raise Exception('Theta and Phi do not match in size')
    elif (type(Phi) != np.ndarray) or (type(Theta) != np.ndarray):
        raise Exception('Type of Theta/Phi should be numpy.ndarray')
    elif (size_Phi != Phi.shape[0]) or (size_Theta != Theta.shape[0]):
        raise Exception('Theta/Phi should be One-dimensional')
    
    # phi 方向处理
    id_phi = (Phi / 2 / np.pi * 12)
    id_phi_inf = np.int0(np.floor(id_phi))
    id_phi_sup = (id_phi_inf + 1) % 12

    delta_id_phi = (id_phi - id_phi_inf).reshape(size_Phi, 1, 1, 1)

    # theta方向处理
    id_theta = (Theta / np.pi * 6)
    id_theta_inf = np.int0(np.floor(id_theta))
    id_theta_sup = (id_theta_inf + 1) % 6

    delta_id_theta = (id_theta - id_theta_inf).reshape(size_Theta, 1, 1, 1)

    return \
    (1 - delta_id_phi) * \
        (Response_Matrix[id_theta_inf, id_phi_inf] * (1 - delta_id_theta) + \
        Response_Matrix[id_theta_sup, id_phi_inf] * delta_id_theta) + \
    delta_id_phi * \
        (Response_Matrix[id_theta_inf, id_phi_sup] * (1 - delta_id_theta) + \
        Response_Matrix[id_theta_sup, id_phi_sup] * delta_id_theta)