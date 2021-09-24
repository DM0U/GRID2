import h5py
import matplotlib.pyplot as plt
import numpy as np
import denoise as dn
import response_matrix as res
from tqdm import tqdm
import gamma_detector as gd
import cube_system as cs
import pandas as pd

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
    if norm != 0.0:
        return array/norm
    else:
        return array

# 以输入矢量为中心生成模拟入射的位置
def generate(guess_spherical):

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

    return guess_points_cartesian

# 将球坐标转为GCRS
def spherical_to_gcrs(spherical):
    ra = spherical[1]
    dec = np.pi / 2 - spherical[0]
    return np.array([ra, dec])

# 生成模拟输入的能谱
E_edges = pd.read_csv('data/Eedges.csv', header=None).to_numpy().squeeze()
Eng_in = (1 / E_edges - np.roll(1 / E_edges, -1))[:-1]

# 初始化总误差,误差列表和输出
# error = 0
# error_list = np.empty(500)
source = []

test_list = ['test_1.h5', 'test_2.h5']
for test_i in range(2):

    # 对500个事件循环
    for event_num in tqdm(range(500)):

        file_path = 'data/{}'.format(test_list[test_i])
        event = '{}'.format(event_num + 1 + 500*test_i)

        # 读入真实信号源方位
        '''
        with h5py.File('data/{}'.format(list[test_i]), 'r') as source:
            real_source = np.array(source['{}'.format(event_num+1+500*test_i)]['source'])

        real_source_spherical = np.empty(2)
        real_source_spherical[0] = np.pi/2 - real_source[0][1]
        real_source_spherical[1] = real_source[0][0]
        real_source_cartesian = transformer(real_source_spherical)
        '''

        # 初始化
        Gamma = gd.gamma_event(file_path, event)

        Cubes = cs.cube_system(file_path, event)

        ################################################
        # 以下部分处理真实输入
        ################################################
        
        # 对每个卫星的四块晶体取平均
        gamma_energy_real = np.sum(Gamma.Eng_out_gamma_all, axis=1)

        # 判断每次事件有几颗卫星接收到高质量峰
        rate_limit = 3.5
        gamma_rate = np.sum(Gamma.Eng_rate, axis=1)/4
        cube_count = np.count_nonzero(gamma_rate > rate_limit)

        ################################################
        # 以下部分根据接收到高质量峰的卫星数的不同进行不同的模拟
        ################################################

        #########################
        # 没有卫星接收到可靠的信号
        #########################
        if cube_count == 0:

            # 生成模拟入射的位置
            theta_list = np.linspace(0, np.pi, 30)
            phi_list = np.linspace(0, 2*np.pi, 60)
            theta, phi = np.meshgrid(theta_list, phi_list)
            guess_points = np.stack((theta, phi), axis=2)

            guess_points_cartesian = np.empty([60, 30, 3])
            for i in range(60):
                for j in range(30):
                    guess_points_cartesian[i][j] = transformer(guess_points[i][j])

            # 对每个位置模拟输入并计算评分
            score = np.empty([60, 30])

            # 评分方案一：响应矩阵元之和
            '''
            for i in range(60):
                for j in range(30):                
                    res_matrix = Cubes.get_response(guess_points_cartesian[i][j])

                    score[i][j] = res_matrix.sum()
            '''

            # 评分方案二： 模拟的各卫星的第零能道能量平方和
            for i in range(60):
                for j in range(30):
                    
                    res_matrix = Cubes.get_response(guess_points_cartesian[i][j])

                    gamma_energy_theoretical = []
                    
                    for k in range(7):
                        theoretical_spectrum = np.zeros(100)
                        for l in range(4):
                            theoretical_spectrum += np.matmul(Eng_in, res_matrix[k][l])
                        gamma_energy_theoretical.append(theoretical_spectrum[0:3])

                    gamma_energy_theoretical = np.array(gamma_energy_theoretical)

                    # alpha和beta为第1，2能道的权重                    
                    alpha = 0
                    beta = 0
                    
                    score[i][j] = np.sum(gamma_energy_theoretical[:, 0] ** 2) \
                                + alpha * np.sum(gamma_energy_theoretical[:, 1] ** 2) \
                                + beta * np.sum(gamma_energy_theoretical[:, 2] ** 2)

            # 将评分最高的位置作为输出
            min_location_temp = np.argmin(score)
            result_cartesian = guess_points_cartesian[int(min_location_temp // 30)][int(min_location_temp % 30)]


        ###########################
        # 有一颗卫星接收到可靠的信号
        ###########################
           
        ###############################
        # 有一颗以上卫星接收到可靠的信号
        ###############################
        else:

            # 处理真实能谱
            gamma_energy_real = renormalize(gamma_energy_real[:, 0:3])
            # 读取各卫星朝向
            cube_normal = Cubes.Cube_coor_frame_z

            # 将接收到高质量峰的卫星的法向矢量相加（并归一）作为初始（猜测）矢量
            guess_cartesian = np.zeros(3)
            for i in range(7):
                if (gamma_rate > rate_limit)[i]:
                    guess_cartesian += cube_normal[i]
            guess_spherical = inverse_transformer(renormalize(guess_cartesian))

            # 在以初始矢量为中心的半球上均匀生成模拟的入射点
            guess_points_cartesian = generate(guess_spherical)
            
            # 对每个位置模拟输入并计算评分
            score = np.empty([60, 15])

            for i in range(60):
                for j in range(15):
                    
                    res_matrix = Cubes.get_response(guess_points_cartesian[i][j])

                    gamma_energy_theoretical = []
                    
                    for k in range(7):
                        theoretical_spectrum = np.zeros(100)
                        for l in range(4):
                            theoretical_spectrum += np.matmul(Eng_in, res_matrix[k][l])
                        gamma_energy_theoretical.append(theoretical_spectrum[0:3])

                    gamma_energy_theoretical = renormalize(np.array(gamma_energy_theoretical))
                    
                    alpha = 0
                    beta = 0
                    
                    score[i][j] = np.sum((gamma_energy_real[:, 0] - gamma_energy_theoretical[:, 0]) ** 2) \
                                + alpha * np.sum((gamma_energy_real[:, 1] - gamma_energy_theoretical[:, 1]) ** 2) \
                                + beta * np.sum((gamma_energy_real[:, 2] - gamma_energy_theoretical[:, 2]) ** 2)
                    
            # 将评分最高的位置作为输出
            min_location_temp = np.argmin(score)
            result_cartesian = guess_points_cartesian[int(min_location_temp // 15)][int(min_location_temp % 15)]
            #print(int(min_location_temp // 15), int(min_location_temp % 15))
        

        #error_list[event_num] = np.arccos(np.dot(result_cartesian, real_source_cartesian))
        #error += np.arccos(np.dot(result_cartesian, real_source_cartesian))
        
        result_spherical = inverse_transformer(result_cartesian)
        source.append(spherical_to_gcrs(result_spherical))
    '''
    with h5py.File('error_list{}.h5'.format(test_i), 'w') as output:
        output['error'] = error_list

    print(error)
    '''

with h5py.File('source.h5', 'w') as output:
    output['source'] = np.array(source)


