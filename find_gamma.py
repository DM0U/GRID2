import h5py
import matplotlib.pyplot as plt
import numpy as np
import denoise as dn
import coordinate_system as coor
import response_matrix as res
from tqdm import tqdm

with h5py.File('data/deposition.h5', 'r') as source:
    ebounds = np.array(source['Matrix'].attrs['Edges'])
    index = np.array(source['Matrix'].attrs['Index'])

# 将球坐标转化为直角坐标
def transformer(spherical):
    cartesian = np.empty(3)
    cartesian[0] = np.sin(spherical[0])*np.cos(spherical[1])
    cartesian[1] = np.sin(spherical[0])*np.sin(spherical[1])
    cartesian[2] = np.cos(spherical[0])
    return cartesian

# 将直角坐标转化为球坐标
def inverse_transformer(cartesian):
    spherical = np.empty(2)
    spherical[0] = np.arccos(cartesian[2])
    if cartesian[1] > 0:
        spherical[1] = np.arccos(cartesian[0]/np.sqrt(1-(cartesian[2] ** 2)))
    else:
        spherical[1] = 2*np.pi - np.arccos(cartesian[0]/np.sqrt(1-(cartesian[2] ** 2)))
    return spherical

# 矢量归一化
def renormalize(array):
    norm = np.sqrt(np.sum(array ** 2))
    return array/norm


################################################
# 以下部分处理真实输入
################################################

Cube_list = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6']
energy_total_real = []

for cube_i in range(7):
    with h5py.File('data/train_1.h5', 'r') as source:
        response_E_T = source['1']['response_E_T'][Cube_list[cube_i]]['0'][:]

    # 读取起始时间和结束时间
    time_inf = int(np.ceil(response_E_T[0][0]))
    time_sup = int(np.floor(response_E_T[-1][0]))

    total_time = time_sup - time_inf
    time_interval = 0.1    # 设定的时间间隔
    time_num = int(total_time/time_interval)

    Spectrum = np.zeros(time_num)

    # 对于每一个粒子，按时间和能量（只考虑第0能道）归类
    for k in tqdm(range(response_E_T.shape[0])):
        # 对时间归类
        for i in range(time_num):
            if response_E_T[k][0] > time_inf + time_interval*i and response_E_T[k][0] < time_inf + time_interval*(i + 1):
                if response_E_T[k][1] > ebounds[0] and response_E_T[k][1] < ebounds[1]:
                    Spectrum[i] += response_E_T[k][1]
                break 
                # 对能量归类
                '''
                for j in range(100):
                    if response_E_T[k][1] > ebounds[j] and response_E_T[k][1] < ebounds[j+1]:
                        Spectrum[i][j] += response_E_T[k][1]
                        break
                break
                '''

    # 降噪，去平均
    Spectrum_denoise = dn.denoise_curve_by_fft(Spectrum, method='number', freq_num=50)
    energy_aver = np.sum(Spectrum_denoise)/time_num
    Spectrum_denoise -= energy_aver

    ###########################
    # 寻找峰和峰的起始，结束位置
    ###########################

    # 寻找最值点
    energy_max = Spectrum_denoise.max()
    max_location = np.argmax(Spectrum_denoise)
    energy_min = Spectrum_denoise.min()

    # 判断是否存在峰
    if energy_max + 2 * energy_min > 0:

        # 寻找峰起始和结束的位置
        gamma_range = []

        # 从最大位置向前寻找零点
        i = max_location
        while True:
            if Spectrum_denoise[i-1] < 0 and Spectrum_denoise[i+1] > 0:
                gamma_range.append(i)
                break
            i -= 1

        # 从最大位置向后寻找零点
        i = max_location
        while True:
            if Spectrum_denoise[i+1] < 0 and Spectrum_denoise[i-1] > 0:
                gamma_range.append(i)
                break
            i += 1
        
        # 计算Gamma爆总能量（使用了原始能谱并减去了噪声）
        noise_aver = (gamma_range[1] - gamma_range[0])*np.sum(Spectrum[:gamma_range[0]])/gamma_range[0]
        energy_total_real.append([np.sum(Spectrum[gamma_range[0]:gamma_range[1]]) - noise_aver, cube_i])

energy_total_real = np.array(energy_total_real)


   
################################################
# 以下部分在不同角度模拟输入
################################################

# 读取输入能道
in_ebounds = []
for i in index[::84]:
    in_ebounds.append(i[0])
in_ebounds = np.array(in_ebounds)

# 生成模拟的输入（伽马源）能谱
input_energy = []
for i in range(40):
    if i == 0:
        input_energy.append((in_ebounds[1] - in_ebounds[0])/(in_ebounds[i] ** 2))
    elif i == 39:
        input_energy.append((in_ebounds[39] - in_ebounds[38])/(in_ebounds[i] ** 2))
    else:
        input_energy.append((in_ebounds[i + 1] - in_ebounds[i-1])/(2 * in_ebounds[i] ** 2))
input_energy = np.array(input_energy)


# 初始化并读取各卫星朝向
coor.event_init('1', 'data/train_1.h5')
cube_normal = coor.Cube_coor_frame_z

# 将接收到信号的卫星的法向矢量相加（并归一）作为初始（猜测）矢量
guess_cartesian = np.zeros(3)
for i in energy_total_real[:, 1]:
    guess_cartesian += cube_normal[int(i)]
guess_spherical = inverse_transformer(renormalize(guess_cartesian))

################################
# 生成900个位置的笛卡尔坐标
################################

# 生成以初始矢量为z轴的900个位置的球坐标
theta_list = np.linspace(0, np.pi/2, 15)
phi_list = np.linspace(0, 2*np.pi, 60)
theta, phi = np.meshgrid(theta_list, phi_list)
guess_points = np.stack((theta, phi), axis=2)

# 生成以初始矢量为z轴的900个位置的笛卡尔坐标
guess_points_cartesian = np.empty([60, 15, 3])
for i in range(60):
    for j in range(15):
        guess_points_cartesian[i][j] = transformer(guess_points[i][j])

# 将以初始矢量为z轴的笛卡尔坐标转化为地球系中的笛卡尔坐标
transform_matrix = np.matmul([[np.cos(guess_spherical[1]), -np.sin(guess_spherical[1]), 0],
                              [np.sin(guess_spherical[1]), np.cos(guess_spherical[1]), 0],
                              [0, 0, 1]], 
                              [[np.cos(guess_spherical[0]), 0, np.sin(guess_spherical[0])],
                               [0, 1, 0],
                               [-np.sin(guess_spherical[0]), 0, np.cos(guess_spherical[0])]])

transform_matrix = transform_matrix.T

for i in range(60):
    for j in range(15):
        guess_points_cartesian[i][j] = np.matmul(guess_points_cartesian[i][j], transform_matrix)

# 对每个位置模拟输入并计算评分（当前评分是理论与实际能量比的方差）
var_result = np.empty([60, 15])

for i in tqdm(range(60)):
    for j in range(15):
        cube_theta, cube_phi = coor.get_angular_coor_in_cubes(guess_points_cartesian[i][j])
        res_matrix = res.get_Response_Matrix(cube_theta, cube_phi, method='array')

        energy_total_theoretical = []
        for k in energy_total_real[:, 1]:
            theoretical_spectrum = np.matmul(input_energy, res_matrix[int(k)][0])
            energy_total_theoretical.append(theoretical_spectrum[0])

        energy_total_theoretical = np.array(energy_total_theoretical)
        energy_total_theoretical = renormalize(energy_total_theoretical)
        beta = energy_total_theoretical/energy_total_real[:, 0]

        var_result[i][j] = np.var(beta)

# 将评分最高的位置作为输出（此处输出的是theta, phi）
min_location_temp = np.argmin(var_result)
print(inverse_transformer(guess_points_cartesian[int(min_location_temp // 15)][int(min_location_temp % 15)]))

# 保存各位置的评分以方便查看
with h5py.File('var_result.h5', 'w') as output:
    output['var_result'] = var_result


# 以下为画图功能
'''
time_axis = np.zeros(time_num)
for i in range(time_num):
    time_axis[i] = time_inf + time_interval*i

#plt.scatter(gamma_location[0], gamma_location[1])
#plt.plot(time_axis, Spectrum_denoise, 'black')
plt.plot(time_axis, Spectrum[:, 0], 'red')
plt.plot(time_axis, dn.denoise_curve_by_gauss(Spectrum[:, 0], 2), 'black')
plt.plot(time_axis, dn.denoise_curve_by_fft(Spectrum[:, 0], method='number', freq_num=50), 'lime')
plt.show()


with h5py.File('Spectrum.h5', 'w') as output:
    output['Spectrum'] = Spectrum

plt.plot(response_E_T[5700:6700,0], np.log10(response_E_T[5700:6700,1]), 'o', ms=1)
plt.savefig('test1_x.png')
'''