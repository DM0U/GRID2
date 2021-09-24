#!/usr/bin/env python3

import h5py
import numpy as np
from numpy import sin, cos, arccos
import response_matrix as rm

###############################################################################
# 类中需要调用的常量

Radius_Earth = 6400    # 地球半径，单位km
c0 = 299792458 / 1000  # 真空光速，单位km/s



###############################################################################
# cube_system类，负责所有卫星相关参数

class cube_system:
    '''
    卫星位形信息的存储类
        对外函数接口
            __init__(self, file_path, event)
                卫星系统初始化
            cal_time_delay(self, dir_from_earth_cart)
                计算入射方向gamma相对地心的延时
            get_response(self, dir_from_earth_cart)
                输入入射方向，给出七个卫星对应相应矩阵
            check_not_blocked(self, dir_from_earth_cart)
                输入入射方向，计算各个卫星的是否被遮挡(也体现在响应矩阵中)

        内部变量(原则上所有内部变量都不应该也不需要被读取)        
            self.Cube_info
                卫星的位置信息
                二维numpy数组，第一维对应卫星编号，
                第二维依次对应xra, xdec, zra, zdec, 
                Xgrcs, Ygcrs, Zgcrs, longitude, latitude
            self.Cube_coor_frame_[xi]
                立方星3个坐标正方向的方向单位矢量，全局变量
                一维numpy数组，三个分量分别对应x,y,z方向
                xi 可取 x, y, z
            self.Cube_radius
                卫星距离地心的距离，一维numpy数组，全局变量，单位km
            self.Cube_coor_unit
                卫星坐标方向单位矢量，7*3的numpy二维数组，全局变量
                用于减少重复计算

    '''

    def __init__(self, file_path, event):
        '''
        对单次测量时间中的卫星信息初始化
            input:
                file_path:
                    h5文件路径名，str
                event:
                    h5文件中的事件名，str
            output:
                self.Cube_info
                    卫星的位置信息
                    二维numpy数组，第一维对应卫星编号，
                    第二维依次对应xra, xdec, zra, zdec, 
                    Xgrcs, Ygcrs, Zgcrs, longitude, latitude  
                elf.Cube_coor_frame_[xi]
                    立方星3个坐标正方向的方向单位矢量，全局变量
                    一维numpy数组，三个分量分别对应x,y,z方向
                    xi 可取 x, y, z
                self.Cube_coor
                    卫星坐标，7*3的numpy二维数组，全局变量，单位km
                self.Cube_coor_unit
                    卫星坐标方向单位矢量，7*3的numpy二维数组，全局变量
                    用于减少重复计算
        '''

        # 数据读入
        self.__read_cube_coor__(file_path, event)

        # 数据预处理
        self.__analyse_cube_info__()

    def __read_cube_coor__(self, file_path, event):
        '''
        单次测量事件的卫星信息，
        !!! 禁止外部调用
            500次读取约1.5s, 认为不需要直接读入所有数据
            input:
                file_path:
                    h5文件路径名，str
                event:
                    h5文件中的事件名，str        
            output:
                self.Cube_info
                    卫星的位置信息
                    二维numpy数组，第一维对应卫星编号，
                    第二维依次对应xra, xdec, zra, zdec, 
                    Xgrcs, Ygcrs, Zgcrs, longitude, latitude
        '''

        self.Cube_info = []
        with h5py.File(file_path, 'r') as ipt:

            for i in range(7):
                cube_name = 'detector' + str(i)
                self.Cube_info.append(ipt[event]['detector_info']\
                                         [cube_name][...])

        self.Cube_info = np.array(self.Cube_info)

    def __analyse_cube_info__(self):
        '''
        对读入的 self.Cube_info信息进行分析与保存
        !!! 禁止外部调用
            input:
                self.Cube_info
                    卫星的位置信息
                    二维numpy数组，第一维对应卫星编号，
                    第二维依次对应xra, xdec, zra, zdec, 
                    Xgrcs, Ygcrs, Zgcrs, longitude, latitude
            output:
                self.Cube_coor_frame_[xi]
                    立方星3个坐标正方向的方向单位矢量，全局变量
                    一维numpy数组，三个分量分别对应x,y,z方向
                    xi 可取 x, y, z
                self.Cube_radius
                    卫星距离地心的距离，一维numpy数组，全局变量，单位km
                self.Cube_coor_unit
                    卫星坐标方向单位矢量，7*3的numpy二维数组，全局变量
                    用于减少重复计算
        '''

        # x坐标架的计算
        _Phi = self.Cube_info[:, 0]
        _Theta = np.pi / 2 - self.Cube_info[:, 1]

        self.Cube_coor_frame_x = np.empty((7, 3))
        self.Cube_coor_frame_x[:, 0] = sin(_Theta) * cos(_Phi)
        self.Cube_coor_frame_x[:, 1] = sin(_Theta) * sin(_Phi)
        self.Cube_coor_frame_x[:, 2] = cos(_Theta)

        # z坐标架的计算
        _Phi = self.Cube_info[:, 2]
        _Theta = np.pi / 2 - self.Cube_info[:, 3]

        self.Cube_coor_frame_z = np.empty((7, 3))
        self.Cube_coor_frame_z[:, 0] = sin(_Theta) * cos(_Phi)
        self.Cube_coor_frame_z[:, 1] = sin(_Theta) * sin(_Phi)
        self.Cube_coor_frame_z[:, 2] = cos(_Theta)

        # y坐标架的计算（矢量叉乘）
        self.Cube_coor_frame_y = \
            np.cross(self.Cube_coor_frame_z, self.Cube_coor_frame_x)

        # 卫星坐标的存储
        Cube_coor = self.Cube_info[:, 4:7]
        self.Cube_radius = np.sqrt((Cube_coor ** 2).sum(axis=-1))

        # 卫星方向单位矢量
        self.Cube_coor_unit = Cube_coor / np.expand_dims(self.Cube_radius, 1)


    def get_angular_coor_in_cubes(self, dir_from_earth_cart):
        '''
        输入星源相对地球的方向单位笛卡尔矢量，
        计算其相对各个卫星的入射方向（顶角与旋转角）
            input:
                dir_from_earth_cart
                    从地球出发指向入射方向的笛卡尔单位矢量,一维numpy数组，size = 3
                self.Cube_coor_frame_[xi]
                    立方星3个坐标正方向的方向单位矢量，全局变量
                    一维numpy数组，三个分量分别对应x,y,z方向
                    xi 可取 x, y, z
            output:
                Theta, Phi
                    相对各个卫星的入射顶角与旋转角，两个一维numpy数组
        '''

        # 入射的顶角
        # 需要对上下界做出限定防止浮点数错误
        _Cos_Theta = (dir_from_earth_cart * self.Cube_coor_frame_z).sum(axis=1)
        _Cos_Theta = np.maximum(np.minimum(_Cos_Theta, 1), -1)
        Theta = arccos(_Cos_Theta)

        # 若sin(theta) = 0, 即theta = 0或pi，phi不重要，只需要注意除以0报错
        Sin_theta = sin(Theta)
        Sin_theta[Sin_theta == 0] = 1 
        # 判断y分量正负性以决定phi的范围
        Comp_y = \
            ((dir_from_earth_cart * self.Cube_coor_frame_y).sum(axis=1) < 0)
        
        # 需要对上下界做出限定防止浮点数错误
        _Cos_Phi = (dir_from_earth_cart * self.Cube_coor_frame_x).sum(axis=1) /\
            Sin_theta
        _Cos_Phi = np.maximum(np.minimum(_Cos_Phi, 1), -1)
        Phi = arccos(_Cos_Phi)
        Phi[Comp_y] = 2 * np.pi - Phi[Comp_y]

        # 防止浮点数精度问题
        Theta[Theta >= np.pi] = np.pi
        Phi[Phi >= 2 * np.pi] = 0

        return Theta, Phi

    def check_not_blocked(self, dir_from_earth_cart):
        '''
        输入星源相对地球的方向单位笛卡尔矢量，
        计算各个卫星的是否被遮挡
            input:
                dir_from_earth_cart
                    从地球出发指向入射方向的笛卡尔单位矢量,
                    一维numpy数组，size = 3，分别对应x,y,z
                self.Cube_radius
                    卫星距离地心的距离，一维numpy数组，全局变量，单位km
                self.Cube_coor_unit
                    卫星坐标方向单位矢量，7*3的numpy二维数组，全局变量
                    用于减少重复计算
            output:
                Cube_not_blocked
                    是否未被遮挡，一维numpy数组，对应7卫星
                    遮挡为0，否则1
        '''
        global Radius_Earth

        _Cos = (dir_from_earth_cart * self.Cube_coor_unit).sum(axis=-1)

        # 若其在地球前方，必然不会被挡住
        Front = (_Cos >= 0)

        # 截距大于地球半径，也不会被挡住
        _Sin = np.sqrt(1 - (_Cos ** 2))
        Intercept = self.Cube_radius * _Sin

        return Front | (Intercept >= Radius_Earth)

    def cal_time_delay(self, dir_from_earth_cart):
        '''
        输入星源相对地球的方向单位笛卡尔矢量，
        计算各个卫星相对地心的时延（不判断遮挡）
            理论上可以与check_not_blocked共同进行从而节约时间，
                但实际上由于该步骤几乎不耗时，所以没有必要将功能混在一起
            input:
                dir_from_earth_cart
                    从地球出发指向入射方向的笛卡尔单位矢量,
                    一维numpy数组，size = 3，分别对应x,y,z
                self.Cube_radius
                    卫星距离地心的距离，一维numpy数组，全局变量，单位km
                self.Cube_coor_unit
                    卫星坐标方向单位矢量，7*3的numpy二维数组，全局变量
                    用于减少重复计算
            output:
                Time_delay
                    理论时延，一维numpy数组，对应7卫星,单位s
        '''
        global c0

        _Cos = (dir_from_earth_cart * self.Cube_coor_unit).sum(axis=-1)


        return - self.Cube_radius * _Cos / c0       

    def get_response(self, dir_from_earth_cart):
        '''
        输入入射方向，给出七个卫星对应相应矩阵
            若被遮挡则输出0矩阵
            input:
                dir_from_earth_cart
                    gamma源方向单位矢量，一维numpy数组，size=3
                self
                    本地信息与相关处理函数
            output:
                    大小为(cube, det, Engin, Engout)的numpy数组
                    表示各个卫星的对应探测器的响应矩阵
        '''

        Theta, Phi = self.get_angular_coor_in_cubes(dir_from_earth_cart)

        Resp = rm.get_Response_Matrix(Theta, Phi, method='array')

        Not_blocked = self.check_not_blocked(dir_from_earth_cart)

        Resp *= Not_blocked.reshape(Not_blocked.size, 1, 1, 1)

        return Resp



###############################################################################
# 辅助的坐标转换函数

def spec_to_cart(theta, phi):
    '''
    用于快速实现 球坐标系 至 笛卡尔坐标系 的转换
        input:
            theta
                顶角，float 或 np.array
            phi
                旋转角，float 或 np.array
        output:
            dir_cart
                笛卡尔坐标系中的归一化矢量
                一维或二维numpy数组，最后一维对应x,y,z
    '''
    cart = np.array([sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)])

    if len(cart.shape) == 2:
        cart = cart.swapaxes(0, 1)

    return cart

def gcrs_to_cart(ra, dec):
    '''
    用于快速实现 天球坐标系 至 笛卡尔坐标系 的转换
        input:
            ra
                赤经，float 或 np.array
            dec
                赤纬，float 或 np.array
        output:
            dir_cart
                笛卡尔坐标系中的归一化矢量
                一维或二维numpy数组，最后一维对应x,y,z
    '''

    theta, phi = gcrs_to_spec(ra, dec)

    return spec_to_cart(theta, phi)

def gcrs_to_spec(ra, dec):
    '''
    用于快速实现 天球坐标系 至 球坐标系 的转换
        input:
            ra
                赤经，float 或 np.array
            dec
                赤纬，float 或 np.array
        output:
            theta
                顶角，float 或 np.array
            phi
                旋转角，float 或 np.array
    '''

    theta = np.pi / 2 - dec
    phi = ra

    return theta, phi







