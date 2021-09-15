import h5py
import matplotlib.pyplot as plt
import numpy as np

with h5py.File('train_1.h5', 'r') as source:
    response_E_T = source['1']['response_E_T']['D1']['0'][:]

# 能道
with h5py.File('deposition.h5', 'r') as source:
    ebounds = np.array(source['Matrix'].attrs['Edges'])

# 1000为时间分的份数（选取时间间隔为1950~2050），100为能道数
Spectrum = np.zeros([1000,100])

# 对于每一个粒子，按时间和能量归类
for k in range(response_E_T.shape[0]):
    # 对时间归类
    for i in range(1000):
        if response_E_T[k][0]>1950+0.1*i+1559450000 and response_E_T[k][0]<1950.1+0.1*i+1559450000:
            # 对能量归类
            for j in range(100):
                if response_E_T[k][1]>ebounds[j] and response_E_T[k][1]<ebounds[j+1]:
                    Spectrum[i][j] += response_E_T[k][1]
                    break
            break


noise = np.zeros(100)
for i in range(50, 55):
    noise += Spectrum[i][:]
noise = noise/5

gamma_ray = np.zeros([10, 100])
for i in range(10):
    gamma_ray[i] = Spectrum[i+55] - noise

with h5py.File('Spectrum.h5', 'w') as output:
    output['Spectrum'] = Spectrum

time_axis = np.zeros(1000)
for i in range(1000):
    time_axis[i] = 1950 + 0.1*i
plt.plot(time_axis, Spectrum[:, 0])
plt.show()

'''
plt.plot(response_E_T[5700:6700,0], np.log10(response_E_T[5700:6700,1]), 'o', ms=1)
plt.savefig('test1_x.png')
'''