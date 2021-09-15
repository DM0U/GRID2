# GRID2
---

## 工作同步报告：
DM0U：卫星坐标与假定伽马源方位角计算 



## 思路讨论：

参数处理可行的思路：

1. 时间发生值取最高值作为参数，每次事件的每个卫星提取1个（TODO：遮挡者处理）

2.  利用最低能道截断得整个伽马暴的有效时长（思路：参考衍射的半角宽/斜率最大）。 对有效时长内的伽马暴求和算平均能谱

---
## 数据环境约定

**请将所有数据的路径设为`./data/*`**

数据集下载地址：

1.  [基本数据集](https://cloud.tsinghua.edu.cn/d/44b1bd37ee444ecb84b6/)
2.  [Spectrum.h5](https://cloud.tsinghua.edu.cn/d/4e735ca0bb4244b697ea/)



***

## 部分文件使用说明

### `response_matrix.py`:  响应矩阵的读取与计算

该程序基于双线性插值算法，针对 $7\times 12$ 的响应矩阵，利用双线性差值计算 $(\theta, \phi)$ 方向入射的能量对应的响应矩阵[^1]。

使用时请使用语句 `from response_matrix import get_Response_Matrix ` ，之后所有的响应矩阵计算请通过函数 `get_Response_Matrix(theta, phi, method)` 实现[^2]。该函数将返回对应对应4个探测晶体的响应矩阵，规模 `(4, Engin, Engout)`。


### `coordinate_system.py`: 卫星坐标与坐标架的处理
此为坐标处理模块，基于直接的笛卡尔坐标系的矢量进行运算。
提供接口 `event_init`，目前阶段请使用 `Cube_coor_frame_[xi]` 调用卫星坐标系，之后需要将其进行封装（一般来说我觉得应该不需要调用`Cube_coor_frame_[xi]`，一切都用封装好的函数处理）。
<todel>反正我懒得用`astropy`判断角度，单位整来整去太麻烦。如果有谁想用可以自己写模块。唯一需要确认的是是否满足theta = pi / 2 - dec, phi = ra。(如果这个炸了真就全炸了，但这个写法就astropy来看似乎没毛病)</todel>


### `denoise.py` :  对最短能道的光变曲线做降噪处理

依据每个时刻数据点的高斯波包扩散或傅里叶变换截断高频项的方法，对波包特征明显的尖锐曲线[^3]进行平滑化处理。



### `response_matrix_test.py`:  检验响应矩阵读取模块`response_matrix.py`的正确性
该文件并非研究中需要直接使用的文件，默认置于`tool`文件夹中作为备份。需要使用时请将其移至根工作目录。
模块功能:
1. 响应矩阵重新分割的检验
2. 双线性算法正确性的简单验证
3. 批量输入 $\theta$ 与 $\phi$ 时矢量化模块的正确性验证(对拍程序) 

---

## 最小二乘法反推输入能道的方法

考虑矩阵
$$
Ax=b
$$
其中 $x, b$ 为列向量，$A$ 为变换系数矩阵。$A, b$ 已知，尝试求解 $x$ 。
若 $A$ 的行数大于列数，此时 $x$ 一般不存在解（且非特例情况[^4]下不会有多重解）。那么，此时 $x$ 的最佳取值可依据最小二乘法决定。在几何意义上而言，这等效于先将 $b$ 投影至矩阵 $A$ 的列空间得 $b'$ ，之后求解线性方程组 $A x = b' = Pb$ 。

投影矩阵 $P$ 满足
$$
P = A (A^{T} A)^{-1} A^{T}
$$
考虑 $A$ 的QR分解，
$$
\begin{align}
\because& A = Q R,\,Q^{T} Q = I\\
\therefore& P = A (A^{T} A)^{-1} A^{T} = A (R^{T} Q^{T} Q R)^{-1} R^{T} Q^{T} = A (R^{T} R)^{-1} A^{T}\\
\because& Ax = Pb = A (R^{T} R)^{-1} A^{T} b\\
\therefore& x = (R^{T} R)^{-1} A^{T} b
\end{align}
$$
对应代码
```python
Q, R = np.linalg.qr(A)

L = np.linalg.inv(np.matmul(R.T, R))
L = np.matmul(L, A.T)
x = np.matmul(L, b)
```



对于预设的入射方向，可以得到对应响应矩阵 $RM$ ，要求 
$$
Eng_{in} RM = Eng_{out}
$$
即
$$
RM^{T} Eng_{in}^{T} = Eng_{out}^{T}
$$

---
[^1]: 由于响应矩阵的得出基于蒙特卡洛模拟，故即使 $\theta=0$ ，针对不同的 $\phi$ 也可能有不同的响应矩阵。此处的处理是认为这些差异很小，不进行进一步的修正。
[^2]: `method` 字符串类型，可选值 `'float'`, `'array'` ，分别对应不同的输入类型
[^3]: 由于高能道的本身有效光子数过小，故只可对最低能道进行处理。
[^4]: $A, b$ 均为 $\bold{0}$ 矩阵