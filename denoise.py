#!/usr/bin/env python3

import numpy as np
from numpy import fft as nf


###############################################################################
# 降噪处理部分，提供两种方法(高斯波包/FFT)

def denoise_curve_by_gauss(virgin_curve, sigma=1):
    '''
    对读入的 virgin_curve 中各个点做高斯波包的弥散化
        可以理解为原序列 virgin_curve 对应多个 delta函数相加，
        本函数将 delta 函数变换为高斯波包
        input:
            virgin_curve
                1维numpy数组，对应各个时刻的曲线值
                ！需要保证波包不位于两侧，否则弥散化后会超出边界
            sigma：
                高斯波包的标准差。默认为1
                由于离散取值的特性，sigma应取1e0至1e1量级，否则序列失去意义
        output:
            curve_denoised
                1维numpy数组，对应处理后的平滑曲线
                与virgin_curve等长
    '''
    Id_arr = np.arange(virgin_curve.size)

    return np.sum(np.exp(- ((np.expand_dims(Id_arr, 1) - Id_arr) ** 2) / \
                           (2 * (sigma ** 2))) / \
                  (np.sqrt(2 * np.pi) * sigma) * \
                 np.expand_dims(virgin_curve, 0), axis=1)

# TODO：截断依据：小于最大频率的某一比例？直接取前若干项？
def denoise_curve_by_fft(virgin_curve, freq_num=40):
    '''
    对读入的 virgin_curve 做傅里叶变换，并截断高频项
        input:
            virgin_curve
                多维numpy数组，最后一维对应各个时刻的曲线值
            freq_num
                根据频谱数量的截断依据，对频谱保留前 freq_num 项
        output:
            curve_denoise
                1维numpy数组，对应处理后的平滑曲线
                与virgin_curve等长
    '''
    # 傅里叶变换至频谱空间
    four_curve = nf.fft(virgin_curve)

    # 截断高频项
    '''
    if method == 'rate':
        four_curve_abs = np.abs(four_curve)
        four_abs_level = max(four_curve_abs[1:]) * freq_level
        four_curve[four_curve_abs < four_abs_level] = 0
    '''

    four_curve = four_curve * (np.arange(four_curve.shape[-1]) < freq_num)



    return nf.ifft(four_curve).real