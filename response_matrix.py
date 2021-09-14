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

def get_Response_Matrix(theta, phi):
    '''
    利用双线性插值，计算theta, phi方向入射的响应矩阵
        TODO: 或许应该改为接受一维numpy列表的theta,phi？
        intput:
            theta       入射的顶角, [0, pi], float
            phi         入射的旋转角, [0, 2pi), float
        output:
            大小为(det, Engin, Engout)的numpy数组，表示theta, phi方向的响应矩阵
    '''
    global Response_Matrix

    id_phi = phi / 2 / np.pi * 12
    id_phi_inf = int(np.floor(id_phi))
    id_phi_sup = (id_phi_inf + 1) % 12

    delta_id_phi = id_phi - id_phi_inf


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