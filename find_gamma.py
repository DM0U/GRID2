import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks_cwt
import denoise as dn

with h5py.File('train_1.h5', 'r') as source:
    response_E_T = source['1']['response_E_T']['D6']['0'][:]

with h5py.File('deposition.h5', 'r') as source:
    ebounds = np.array(source['Matrix'].attrs['Edges'])

time_inf = int(np.ceil(response_E_T[0][0]))
time_sup = int(np.floor(response_E_T[-1][0]))

total_time = time_sup - time_inf
time_interval = 0.1
time_num = int(total_time/time_interval)

Spectrum = np.zeros([time_num, 100])

# 对于每一个粒子，按时间和能量归类
for k in range(response_E_T.shape[0]):
    # 对时间归类
    for i in range(time_num):
        if response_E_T[k][0] > time_inf + time_interval*i and response_E_T[k][0] < time_inf + time_interval*(i + 1):
            # 对能量归类
            for j in range(100):
                if response_E_T[k][1] > ebounds[j] and response_E_T[k][1] < ebounds[j+1]:
                    Spectrum[i][j] += response_E_T[k][1]
                    break
            break

# 降噪，去平均
Spectrum_denoise = dn.denoise_curve_by_fft(Spectrum[:, 0], method='number', freq_num=50)
energy_aver = np.sum(Spectrum_denoise)/time_num
Spectrum_denoise[:] = Spectrum_denoise[:] - energy_aver

###########################
# 寻找峰和峰的起始，结束位置
###########################

# 寻找极值点
extremum = []
#spec_max = Spectrum.max()
#id_max = np.argmax(Spectrum_denoise)
for i in range(1, time_num-1):
    if Spectrum_denoise[i] < Spectrum_denoise[i-1] and Spectrum_denoise[i] < Spectrum_denoise[i+1]:
        extremum.append([i, -1])
    if Spectrum_denoise[i] > Spectrum_denoise[i-1] and Spectrum_denoise[i] > Spectrum_denoise[i+1]:
        extremum.append([i, 1])

# 判断是否有Gamma射线暴输入
energy_max = [0, 0]
energy_min = [0, 0]
for i in range(len(extremum)):
    if Spectrum_denoise[extremum[i][0]] > energy_max[0]:
        energy_max = [Spectrum_denoise[extremum[i][0]], i]
    elif Spectrum_denoise[extremum[i][0]] < energy_min[0]:
        energy_min = [Spectrum_denoise[extremum[i][0]], i]

if energy_max[0] + 2 * energy_min[0] > 0:
    gamma_exist = True
else:
    gamma_exist = False

gamma_location = [time_inf + time_interval*extremum[energy_max[1]][0], energy_max[0]]

time_axis = np.zeros(time_num)
for i in range(time_num):
    time_axis[i] = time_inf + time_interval*i

plt.scatter(gamma_location[0], gamma_location[1])
plt.plot(time_axis, Spectrum_denoise, 'black')
#plt.plot(time_axis, Spectrum[:, 0], 'red')
#plt.plot(time_axis, dn.denoise_curve_by_gauss(Spectrum[:, 0], 2), 'red')
#plt.plot(time_axis, dn.denoise_curve_by_fft(Spectrum[:, 0], method='number', freq_num=50), 'lime')
plt.show()


with h5py.File('Spectrum.h5', 'w') as output:
    output['Spectrum'] = Spectrum

plt.plot(response_E_T[5700:6700,0], np.log10(response_E_T[5700:6700,1]), 'o', ms=1)
plt.savefig('test1_x.png')