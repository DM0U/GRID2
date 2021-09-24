import h5py
import numpy as np
import gamma_detector as gd
import cube_system as cs
import pandas as pd

###############################################################################
# 固定参数
rate_limit = 3.5


###############################################################################
# 预备函数：矢量归一化
def renormalize(array):
    norm = np.sqrt(np.sum(array ** 2))
    if norm != 0.0:
        return array/norm
    else:
        return array

###############################################################################
# 生成模拟输入的能谱

E_edges = pd.read_csv('data/Eedges.csv', header=None).to_numpy().squeeze()
Eng_in = (1 / E_edges - np.roll(1 / E_edges, -1))[:-1]

###############################################################################
# 模拟打表的准备函数


# 指定打表密度与范围
param_points_0 = {'num_theta':30, 'num_phi':60, \
                  'theta_inf':0, 'theta_sup':np.pi, \
                  'phi_inf':0, 'phi_sup':2 * np.pi}

param_points_1 = {'num_theta':15, 'num_phi':60, \
                  'theta_inf':0, 'theta_sup':np.pi / 2, \
                  'phi_inf':0, 'phi_sup':2 * np.pi}


# 相对位置的打表

def get_point_list(param_points):
    '''
    根据输入的参数生成对应范围与密度的若干相对范围表打表
        每种参数对应的表只需初始化时调用一次
        input:
            param_points
                打表所需的参数字典，包含以下选项
                    num_theta/num_phi:
                        int, theta/phi方向的打点数量
                    [theta/phi]_[inf/sup]
                        float, theta/phi方向打点的上下界，sup对应上界，inf对应下界
        output:
            3维numpy数组，依次对应(phi, theta, cartesian)
            以 theta = 0 为中心生成的打点表
    '''
    # 参数字典读入
    num_theta, num_phi = param_points['num_theta'], param_points['num_phi']
    theta_inf, theta_sup = param_points['theta_inf'], param_points['theta_sup']
    phi_inf, phi_sup = param_points['phi_inf'], param_points['phi_sup']

    theta_list = np.linspace(theta_inf, theta_sup, num_theta)
    phi_list = np.linspace(phi_inf, phi_sup, num_phi)
    theta_, phi_ = np.meshgrid(theta_list, phi_list)
    points_list_cart = cs.spec_to_cart(theta_.reshape(num_theta * num_phi), \
        phi_.reshape(num_theta * num_phi)).reshape(num_phi, num_theta, 3)

    return points_list_cart

points_list_cart_0 = get_point_list(param_points_0)
points_list_cart_1 = get_point_list(param_points_1)


def generate(guess_spherical, points_list_cart):
    '''
    以输入矢量为中心生成模拟入射的若干位置
        input:
            guess_spherical
                模拟入射的中心方向（在该方向周围打点）
            points_list_cart
                在相对方向中采用的打表结果
        output:
            guess_points_cartesian
                猜测的入射方向列表
    '''
    # 将以初始矢量为z轴的笛卡尔坐标转化为地球系中的笛卡尔坐标
    transform_matrix = np.matmul(\
        [[np.cos(guess_spherical[1]), -np.sin(guess_spherical[1]), 0], \
         [np.sin(guess_spherical[1]),  np.cos(guess_spherical[1]), 0], \
         [0, 0, 1]], \
        [[ np.cos(guess_spherical[0]), 0, np.sin(guess_spherical[0])], \
         [0, 1, 0], \
         [-np.sin(guess_spherical[0]), 0, np.cos(guess_spherical[0])]]  )

    transform_matrix = transform_matrix.T

    guess_points_cartesian = np.matmul(points_list_cart, transform_matrix)

    return guess_points_cartesian



###############################################################################
# 实际的查找函数

def find_gamma(file_path, event, method='test'):
    '''
    对指定的事件反溯其源
        input:
            file_path
                事件记录文件路径
            event
                事件名
            method
                'test'
                    只进行反溯计算
                'validation'
                    反溯完成后根据正确答案计算角度差
                    只能在训练集上使用
        output:
            result_gcrs
                长为2的一维numpy，对应参数 ra,dec
            error_rad
                当 method 为 'validation'时，
                额外输出计算角度与真值的差
    '''

    global points_list_cart_0, points_list_cart_1, rate_limit
    global param_points_0, param_points_1

 
    ################################################
    # 基类初始化
    Gamma = gd.gamma_event(file_path, event)
    Cubes = cs.cube_system(file_path, event)

    ################################################
    # 原始分析信号的进一步处理
    
    # 对每个卫星的四块晶体取平均
    gamma_energy_real = np.sum(Gamma.Eng_out_gamma_all, axis=1)

    # 判断每次事件有几颗卫星接收到高质量峰
    gamma_rate = np.sum(Gamma.Eng_rate, axis=1) / 4
    cube_count = np.count_nonzero(gamma_rate > rate_limit)

    ################################################
    # 以下部分根据接收到高质量峰的卫星数的不同进行不同的模拟
    ################################################

    #########################
    # 没有卫星接收到可靠的信号
    #########################
    if cube_count == 0:

        # 生成模拟入射的位置
        guess_points_cartesian = points_list_cart_0

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
        for i in range(param_points_0['num_phi']):
            for j in range(param_points_0['num_theta']):
                
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
            result_cartesian = guess_points_cartesian[int(min_location_temp // 30)] \
                                                     [int(min_location_temp % 30)]

           
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
        _cart = renormalize(guess_cartesian)
        guess_spherical = cs.trans_coor('cart', 'spec', x=_cart[0], y=_cart[1], z=_cart[2])

        # 在以初始矢量为中心的半球上均匀生成模拟的入射点
        guess_points_cartesian = generate(guess_spherical, points_list_cart_1)
        
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
                
                score[i][j] = np.sum((gamma_energy_real[:, 0] - \
                                        gamma_energy_theoretical[:, 0]) ** 2) + \
                                        alpha * np.sum((gamma_energy_real[:, 1] - \
                                        gamma_energy_theoretical[:, 1]) ** 2) + \
                                        beta * np.sum((gamma_energy_real[:, 2] - \
                                        gamma_energy_theoretical[:, 2]) ** 2)
                
        # 将评分最高的位置作为输出
        min_location_temp = np.argmin(score)
        result_cartesian = guess_points_cartesian[int(min_location_temp // 15)]\
                                                    [int(min_location_temp % 15)]
        
    # end of  if cube_count == 0


    ###################################################
    # 对计算得的角度的最后处理与输出
        
    ra, dec = cs.trans_coor('cart', 'gcrs', x=result_cartesian[0], \
                                    y=result_cartesian[1], z=result_cartesian[2])
    result_gcrs = np.array([ra, dec])
        
    if method == 'validation':
        with h5py.File(file_path, 'r') as source:
            _ra, _dec = source[event]['source'][0]

        real_source_cart = cs.trans_coor('gcrs', 'cart', ra=_ra, dec=_dec)
        error_rad = np.arccos(np.dot(result_cartesian, real_source_cart))
        return result_gcrs, error_rad
    else:
        return result_gcrs




