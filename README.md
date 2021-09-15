# GRID2
---

## 工作同步报告：
DM0U：卫星坐标与假定伽马源方位角计算 

---
## 数据环境约定

**请将所有数据的路径设为`./data/*`**

数据集下载地址：

1.  [基本数据集](https://cloud.tsinghua.edu.cn/d/44b1bd37ee444ecb84b6/)
2. ？？？



***

## 部分文件使用说明

### `response_matrix.py`:  响应矩阵的读取与计算

该程序基于双线性插值算法，针对 $7\times 12$ 的响应矩阵，利用双线性差值计算 $(\theta, \phi)$ 方向入射的能量对应的响应矩阵[^1]。

使用时请使用语句 `from response_matrix import get_Response_Matrix ` ，之后所有的响应矩阵计算请通过函数 `get_Response_Matrix(theta, phi, method)` 实现[^2]。该函数将返回对应对应4个探测晶体的响应矩阵，规模 `(4, Engin, Engout)`。


### `coordinate_system.py`: 卫星坐标与坐标架的处理
此为坐标处理模块，基于直接的笛卡尔坐标系的矢量进行运算。
提供接口 `event_init`，目前阶段请使用 `Cube_coor_frame_[xi]` 调用卫星坐标系，之后需要将其进行封装（一般来说我觉得应该不需要调用`Cube_coor_frame_[xi]`，一切都用封装好的函数处理）。
<todel>反正我懒得用`astropy`判断角度，单位整来整去太麻烦。如果有谁想用可以自己写模块。唯一需要确认的是是否满足theta = pi / 2 - dec, phi = ra。(如果这个炸了真就全炸了，但这个写法就astropy来看似乎没毛病)</todel>



### `response_matrix_test.py`:  检验响应矩阵读取模块`response_matrix.py`的正确性
该文件并非研究中需要直接使用的文件，默认置于`tool`文件夹中作为备份。需要使用时请将其移至根工作目录。
模块功能:
1. 响应矩阵重新分割的检验
2. 双线性算法正确性的简单验证
3. 批量输入 $\theta$ 与 $\phi$ 时矢量化模块的正确性验证(对拍程序) 






[^1]: 由于响应矩阵的得出基于蒙特卡洛模拟，故即使 $\theta=0$ ，针对不同的 $\phi$ 也可能有不同的响应矩阵。此处的处理是认为这些差异很小，不进行进一步的修正。
[^2]: `method` 字符串类型，可选值 `'float'`, `'array'` ，分别对应不同的输入类型