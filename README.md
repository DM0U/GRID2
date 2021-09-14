# GRID2
---

## 数据环境约定

**请将所有数据的路径设为`./data/*`**

数据集下载地址：

1.  [基本数据集](https://cloud.tsinghua.edu.cn/d/44b1bd37ee444ecb84b6/)
2. ？？？



***

## 部分文件使用说明

### `response_matrix.py`  响应矩阵的读取与计算

该函数基于双线性插值算法，针对 $7\times 12$ 的响应矩阵，利用双线性差值计算 $(\theta, \phi)$ 方向入射的能量对应的响应矩阵[^1]。

使用时请使用语句 `from response_matrix import get_Response_Matrix ` ，之后所有的响应矩阵计算请通过函数 `get_Response_Matrix(theta, phi)` 实现。该函数将返回对应对应4个探测晶体的响应矩阵，规模 `(4, Engin, Engout)`。





[^1]: 由于响应矩阵的得出基于蒙特卡洛模拟，故即使 $\theta=0$ ，针对不同的 $\phi$ 也可能有不同的响应矩阵。此处的处理是认为这些差异很小，不进行进一步的修正。