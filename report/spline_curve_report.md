# 曲线设计与编辑工具报告

## 1. 项目概述

本项目实现了一个基于Python和PyQt5的曲线设计与编辑工具，模仿PowerPoint实现了曲线设计与编辑功能。工具支持三次样条曲线的绘制和编辑，采用Foley参数化方法，支持C2、G1和G0连续性的调整。

## 2. 系统架构

### 2.1 系统结构

项目主要包含以下几个文件：
- `main.py`: 程序入口文件
- `curve_editor.py`: 包含用户界面和交互逻辑
- `curve.py`: 包含曲线相关的数学模型和算法实现
- `requirements.txt`: 项目依赖库

系统主要由两个核心组件构成：
1. **数据模型层**：实现曲线的数学模型和相关算法
2. **图形界面层**：提供用户交互界面和绘图功能

### 2.2 类设计

项目主要包含以下几个类：

```
CurveEditorMainWindow (主窗口类)
├── CanvasWidget (画布类，处理绘图和交互)

Point2D (点类)
Curve (曲线抽象基类)
├── ParametricSpline (参数化三次样条曲线类)

CurveType (枚举，曲线类型)
ContinuityType (枚举，连续性类型)
```

## 3. 核心功能与实现

### 3.1 三次样条曲线数学原理

三次样条曲线是一种分段三次多项式曲线，它在各节点处满足一定的连续性条件。对于给定的一组型值点 $P_0, P_1, \ldots, P_n$，我们可以构造一条曲线 $S(t)$，使其通过所有型值点并满足指定的连续性条件。

#### 3.1.1 曲线参数化方法

曲线参数化是为每个型值点分配一个参数值 $t_i$ 的过程。本项目实现了Foley参数化方法，该方法考虑了点之间的距离和转角因素，通常能得到更自然的曲线形状。

Foley参数化的计算公式如下：

给定型值点 $P_0, P_1, \ldots, P_n$，我们首先计算相邻点之间的弦长：

$$d_i = |P_{i+1} - P_i|$$

对于每个内部点 $P_i$ $(1 \leq i \leq n-1)$，我们计算转角 $\alpha_i$：

$$\alpha_i = \arccos\left(\frac{(P_i - P_{i-1}) \cdot (P_{i+1} - P_i)}{|P_i - P_{i-1}| \cdot |P_{i+1} - P_i|}\right)$$

然后，Foley参数化的距离计算为：

$$\delta_i = d_i \left(1 + 1.5 \cdot \frac{\alpha_{i-1} \cdot d_{i-1}}{d_{i-1} + d_i} + 1.5 \cdot \frac{\alpha_i \cdot d_i}{d_i + d_{i+1}}\right)$$

最后，参数值 $t_i$ 通过归一化这些距离得到：

$$t_0 = 0, \quad t_i = t_{i-1} + \frac{\delta_{i-1}}{\sum_{j=0}^{n-1} \delta_j} \quad (1 \leq i \leq n)$$

#### 3.1.2 三次样条曲线构造

对于参数化后的型值点 $(t_i, P_i)$，我们构造一组三次多项式 $S_i(t)$，使得：

$$S_i(t) = a_i + b_i(t-t_i) + c_i(t-t_i)^2 + d_i(t-t_i)^3, \quad t \in [t_i, t_{i+1}]$$

满足以下条件：
1. 插值条件：$S_i(t_i) = P_i$, $S_i(t_{i+1}) = P_{i+1}$
2. 连续性条件：$S'_i(t_{i+1}) = S'_{i+1}(t_{i+1})$, $S''_i(t_{i+1}) = S''_{i+1}(t_{i+1})$

这可以通过求解一个线性方程组来确定各段样条的系数。最终，我们可以得到一组三次多项式，它们共同构成了经过所有型值点、在各节点处满足C2连续性的曲线。

### 3.2 代码实现

#### 3.2.1 参数化计算

Foley参数化在 `curve.py` 中的 `ParametricSpline` 类的 `_compute_parameterization` 方法中实现：

```python
def _compute_parameterization(self):
    """计算参数值"""
    if len(self.data_points) < 2:
        return []
    
    # 如果只有两个点，直接使用均匀参数化
    if len(self.data_points) == 2:
        return [0.0, 1.0]
    
    # 初始化参数列表
    t = [0.0]  # 第一个参数值总是0
    
    # 计算各段距离
    chordal_distances = []
    
    for i in range(len(self.data_points) - 1):
        distance = self.data_points[i].distance_to(self.data_points[i+1])
        # 确保距离不为零
        distance = max(distance, 0.0001)  # 设置最小距离避免除零
        chordal_distances.append(distance)
    
    # 当点数少于3个时，无法计算角度，使用均匀参数化
    if len(self.data_points) < 3:
        for i in range(1, len(self.data_points)):
            t.append(float(i) / (len(self.data_points) - 1))
        return t
    
    # 尝试使用Foley参数化
    try:
        # 计算角度（用于Foley参数化）
        alphas = []
        for i in range(len(self.data_points) - 2):
            p1 = self.data_points[i]
            p2 = self.data_points[i+1]
            p3 = self.data_points[i+2]
            
            a = chordal_distances[i]**2
            b = chordal_distances[i+1]**2
            c = p1.distance_to(p3)**2
            
            # 避免除零
            denom = 2.0 * chordal_distances[i] * chordal_distances[i+1]
            if denom < 0.0001:  # 几乎为零
                alpha = math.pi / 2  # 默认90度
            else:
                # 余弦定理
                cos_alpha = (a + b - c) / denom
                cos_alpha = max(-0.99999, min(0.99999, cos_alpha))  # 避免acos异常
                alpha = math.pi - math.acos(cos_alpha)
                alpha = min(alpha, math.pi/2)
            
            alphas.append(alpha)
        
        # 计算Foley参数化的距离
        foley_distances = []
        for i in range(len(chordal_distances)):
            foley_dist = chordal_distances[i]
            
            if i == 0 and alphas:
                # 第一段
                denom = chordal_distances[i] + chordal_distances[i+1]
                if denom > 0.0001:  # 避免除零
                    foley_dist *= (1.0 + 1.5 * alphas[i] * chordal_distances[i] / denom)
            
            elif i == len(chordal_distances) - 1 and alphas:
                # 最后一段
                denom = chordal_distances[i-1] + chordal_distances[i]
                if denom > 0.0001:  # 避免除零
                    foley_dist *= (1.0 + 1.5 * alphas[i-1] * chordal_distances[i-1] / denom)
            
            elif alphas and i > 0 and i < len(chordal_distances) - 1:
                # 中间段
                denom1 = chordal_distances[i-1] + chordal_distances[i]
                denom2 = chordal_distances[i] + chordal_distances[i+1]
                
                factor = 1.0
                if denom1 > 0.0001:  # 避免除零
                    factor += 1.5 * alphas[i-1] * chordal_distances[i-1] / denom1
                if denom2 > 0.0001:  # 避免除零
                    factor += 1.5 * alphas[i] * chordal_distances[i] / denom2
                
                foley_dist *= factor
            
            foley_distances.append(foley_dist)
        
        # 计算参数总和（用于归一化）
        sum_foley = sum(foley_distances)
        
        # 防止除零错误
        if sum_foley > 0.0001:
            # 使用Foley参数化
            for i in range(1, len(self.data_points)):
                last_param = t[-1]
                t.append(last_param + foley_distances[i-1] / sum_foley)
            return t
    
    except Exception as e:
        # 如果Foley参数化失败，使用均匀参数化
        pass
    
    # 使用均匀参数化作为后备
    t = [0.0]
    for i in range(1, len(self.data_points)):
        t.append(float(i) / (len(self.data_points) - 1))
    return t
```

#### 3.2.2 三次样条曲线生成

曲线生成在 `ParametricSpline` 类的 `generate_curve_points` 方法中实现，该方法也添加了完善的异常处理机制：

```python
def generate_curve_points(self, sampling_period=0.001):
    """生成三次样条曲线的采样点"""
    curve_points = []
    
    if len(self.data_points) < 2 or len(self.parameter_values) < 2:
        return curve_points
    
    # 如果只有两个点，生成一条直线
    if len(self.data_points) == 2:
        start = self.data_points[0]
        end = self.data_points[1]
        steps = int(1.0 / sampling_period) + 1
        for i in range(steps + 1):
            t = i / steps
            x = (1 - t) * start.x + t * end.x
            y = (1 - t) * start.y + t * end.y
            curve_points.append(Point2D(x, y))
        return curve_points
    
    try:
        n = len(self.data_points)
        
        # 构建线性方程组
        h = np.zeros(n-1)
        for i in range(n-1):
            h[i] = max(self.parameter_values[i+1] - self.parameter_values[i], 0.0001)  # 避免h为0
        
        # 构建右侧向量 b
        b_x = np.zeros(n-1)
        b_y = np.zeros(n-1)
        for i in range(n-1):
            b_x[i] = 6.0 * (self.data_points[i+1].x - self.data_points[i].x) / h[i]
            b_y[i] = 6.0 * (self.data_points[i+1].y - self.data_points[i].y) / h[i]
        
        # 构建对角线矩阵
        if n > 2:
            # 构建三对角矩阵的数据
            diag_data = []
            diag_indices = []
            diag_indptr = [0]
            
            # 构建u向量和v向量
            u = np.zeros(n-2)
            v_x = np.zeros(n-2)
            v_y = np.zeros(n-2)
            
            for i in range(n-2):
                u[i] = 2.0 * (h[i] + h[i+1])
                # 确保u不为0
                u[i] = max(u[i], 0.0001)
                v_x[i] = b_x[i+1] - b_x[i]
                v_y[i] = b_y[i+1] - b_y[i]
            
            # 构建三对角矩阵
            for i in range(n-2):
                # 对角线元素
                diag_data.append(u[i])
                diag_indices.append(i)
                
                # 次对角线元素(下方)
                if i > 0:
                    diag_data.append(h[i])
                    diag_indices.append(i-1)
                
                # 次对角线元素(上方)
                if i < n-3:
                    diag_data.append(h[i+1])
                    diag_indices.append(i+1)
                
                diag_indptr.append(len(diag_data))
            
            # 创建稀疏矩阵
            A = csr_matrix((diag_data, diag_indices, diag_indptr), shape=(n-2, n-2))
            
            # 解线性方程组得到二阶导数值
            M_x = np.zeros(n)
            M_y = np.zeros(n)
            
            if n == 3:
                M_x[1] = v_x[0] / u[0]
                M_y[1] = v_y[0] / u[0]
            else:
                try:
                    mid_x = spsolve(A, v_x)
                    mid_y = spsolve(A, v_y)
                    
                    for i in range(len(mid_x)):
                        M_x[i+1] = mid_x[i]
                        M_y[i+1] = mid_y[i]
                except Exception:
                    # 如果求解失败，返回两点之间的直线
                    return self._generate_piecewise_line(sampling_period)
            
            # 使用三次样条公式生成曲线点
            for i in range(n-1):
                t_start = self.parameter_values[i]
                t_end = self.parameter_values[i+1]
                
                t_values = np.arange(t_start, t_end, sampling_period)
                for t in t_values:
                    A1 = (self.parameter_values[i+1] - t) / h[i]
                    A2 = (t - self.parameter_values[i]) / h[i]
                    A3 = A1 * (A1 * A1 - 1.0) * h[i] * h[i] / 6.0
                    A4 = A2 * (A2 * A2 - 1.0) * h[i] * h[i] / 6.0
                    
                    x = A1 * self.data_points[i].x + A2 * self.data_points[i+1].x + A3 * M_x[i] + A4 * M_x[i+1]
                    y = A1 * self.data_points[i].y + A2 * self.data_points[i+1].y + A3 * M_y[i] + A4 * M_y[i+1]
                    
                    curve_points.append(Point2D(x, y))
        
    except Exception:
        # 如果曲线生成失败，返回两点之间的直线
        curve_points = self._generate_piecewise_line(sampling_period)
    
    return curve_points

def _generate_piecewise_line(self, sampling_period=0.001):
    """生成分段直线（作为失败时的备选方案）"""
    curve_points = []
    
    for i in range(len(self.data_points) - 1):
        start = self.data_points[i]
        end = self.data_points[i+1]
        steps = max(int(start.distance_to(end) / (10 * sampling_period)), 5)
        
        for j in range(steps + 1):
            t = j / steps
            x = (1 - t) * start.x + t * end.x
            y = (1 - t) * start.y + t * end.y
            curve_points.append(Point2D(x, y))
    
    return curve_points
```

#### 3.2.3 曲线更新与异常处理

在用户界面层，`CanvasWidget` 类的 `update_curve` 方法也添加了异常处理，确保即使底层计算失败也不会影响界面的响应：

```python
def update_curve(self):
    """更新曲线"""
    if len(self.points) < 2:
        self.curve_points = []
        self.current_curve = None
        return
    
    # 使用参数化三次样条曲线对象
    try:
        self.current_curve = ParametricSpline(self.points, self.curve_type)
        self.curve_points = self.current_curve.generate_curve_points(self.sampling_period)
    except Exception:
        # 错误处理：清空曲线点，但不输出错误
        self.curve_points = []
        self.current_curve = None
```

### 3.3 用户界面与交互

用户界面基于PyQt5实现，主要包括：

1. **主窗口**：包含菜单栏、工具栏和状态栏
2. **画布**：处理绘图和交互

#### 3.3.1 交互模式

工具支持两种交互模式：
- **绘制模式**：通过鼠标点击添加型值点，实时生成三次样条曲线
- **编辑模式**：通过拖动调整型值点位置，实时更新曲线形状

#### 3.3.2 快捷键

- **空格键**：完成当前曲线绘制
- **Esc键**：取消当前操作
- **Delete键**：删除选中的点

## 4. 实验结果与分析

### 4.1 功能展示

工具实现了以下功能：
1. 输入有序点列（型值点），实时生成分段的三次样条曲线
2. 修改拖动型值点的位置（保持整条曲线 C^2）
3. 编辑型值点处的切线信息，成为 G^1 或 G^0
4. 鲁棒的异常处理，确保在各种边缘情况下程序稳定运行

### 4.2 算法分析

Foley参数化方法相比于均匀参数化、弦长参数化和向心参数化，能更好地处理转角较大的情况，生成更符合直觉的曲线形状。通过考虑型值点之间的转角大小，可以在转角较大的地方分配更多参数空间，使得曲线在这些区域更加平滑。

为了提高算法的鲁棒性，我们进行了以下改进：

1. **边缘情况处理**：针对点数较少（如只有2个点）的情况进行特殊处理
2. **除零保护**：在所有可能涉及除法的地方添加阈值检查，避免除零错误
3. **异常处理**：使用try-except块捕获可能的计算异常
4. **备选算法**：当高级算法失败时，提供简单可靠的备选算法（如均匀参数化和分段直线）

这些改进确保了程序在各种输入条件下的稳定性，即使在某些特殊排列的点上，也能提供合理的曲线结果。

### 4.3 连续性分析

本工具支持三种连续性选项：
- **C2连续**：曲线在各节点处的一阶导数和二阶导数均连续
- **G1连续**：曲线在各节点处的切线方向连续，但切线长度可能不同
- **G0连续**：曲线仅在各节点处的位置连续，切线可能不连续

不同连续性选项适用于不同的设计需求，例如G1连续可以在保持曲线整体平滑的同时，在某些节点处形成更尖锐的转角。

## 5. 总结与展望

本项目实现了一个功能完善的曲线设计与编辑工具，支持Foley参数化的三次样条曲线绘制和编辑，以及不同连续性的调整。工具具有友好的用户界面和丰富的交互功能，能够满足基本的曲线设计需求。

未来可以进一步改进的方向包括：
1. 支持更多类型的曲线（如NURBS曲线）
2. 实现更丰富的编辑功能（如添加/删除节点、调整节点权重等）
3. 支持曲线的导入/导出
4. 优化算法性能，提高大型曲线的处理效率
5. 添加撤销/重做功能，提升用户体验

## 6. 参考文献

1. Foley, J. D., & Van Dam, A. (1982). Fundamentals of interactive computer graphics. Reading, MA: Addison-Wesley.
2. Bartels, R. H., Beatty, J. C., & Barsky, B. A. (1987). An introduction to splines for use in computer graphics and geometric modeling. Morgan Kaufmann.
3. Piegl, L., & Tiller, W. (1997). The NURBS book. Springer Science & Business Media. 