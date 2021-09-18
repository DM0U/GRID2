import h5py
import matplotlib.pyplot as plt
import numpy as np
import denoise as dn
from tqdm import tqdm

with h5py.File('data/deposition.h5', 'r') as source:
    ebounds = np.array(source['Matrix'].attrs['Edges'])
    index = np.array(source['Matrix'].attrs['Index'])

Cube_list = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6']

for cube_i in range(7):
    with h5py.File('data/train_1.h5', 'r') as source:
        response_E_T = source['313']['response_E_T'][Cube_list[cube_i]]['0'][:]

    time_inf = int(np.ceil(response_E_T[0][0]))
    time_sup = int(np.floor(response_E_T[-1][0]))

    total_time = time_sup - time_inf
    time_interval = 0.1
    time_num = int(total_time/time_interval)

    Spectrum = np.zeros(time_num)

    # 对于每一个粒子，按时间和能量归类
    for k in tqdm(range(response_E_T.shape[0])):
        # 对时间归类
        for i in range(time_num):
            if response_E_T[k][0] > time_inf + time_interval*i and response_E_T[k][0] < time_inf + time_interval*(i + 1):
                if response_E_T[k][1] > ebounds[0] and response_E_T[k][1] < ebounds[1]:
                    Spectrum[i] += response_E_T[k][1]
                break

    time_axis = np.zeros(time_num)
    for i in range(time_num):
        time_axis[i] = time_inf + time_interval*i

    plt.plot(time_axis, Spectrum, 'red')
    plt.plot(time_axis, dn.denoise_curve_by_gauss(Spectrum, 2), 'black')
    plt.plot(time_axis, dn.denoise_curve_by_fft(Spectrum, method='number', freq_num=50), 'lime')

    #plt.cla()
    #plt.plot(response_E_T[:, 0], np.log10(response_E_T[:, 1]), 'o', ms=1)
    plt.savefig('spectrum{}.png'.format(cube_i))