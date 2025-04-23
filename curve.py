import numpy as np
import math
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from enum import Enum

class CurveType(Enum):
    FOLEY = 4         # Foley参数化

class ContinuityType(Enum):
    C2 = 1            # C2连续
    G1 = 2            # G1连续
    G0 = 3            # G0连续

class Point2D:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        if isinstance(other, Point2D):
            return self.x == other.x and self.y == other.y
        return False
    
    def distance_to(self, other):
        """计算两点间的欧几里得距离"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class Curve:
    """曲线抽象基类"""
    def __init__(self):
        pass
    
    def generate_curve_points(self, sampling_period=0.001):
        """生成曲线上的采样点"""
        raise NotImplementedError("Subclass must implement abstract method")
    
    def get_curve_type(self):
        """获取曲线类型"""
        raise NotImplementedError("Subclass must implement abstract method")

class ParametricSpline(Curve):
    """参数化三次样条曲线"""
    def __init__(self, data_points, curve_type=CurveType.FOLEY):
        super().__init__()
        self.data_points = data_points.copy()
        self.curve_type = curve_type
        self.parameter_values = self._compute_parameterization() if len(data_points) > 1 else []
    
    def set_data_points(self, data_points):
        """设置数据点"""
        self.data_points = data_points.copy()
        if len(data_points) > 1:
            self.parameter_values = self._compute_parameterization()
    
    def get_curve_type(self):
        return self.curve_type
    
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
            print(f"Foley参数化失败: {e}, 使用均匀参数化")
        
        # 使用均匀参数化作为后备
        t = [0.0]
        for i in range(1, len(self.data_points)):
            t.append(float(i) / (len(self.data_points) - 1))
        return t
    
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
                    except Exception as e:
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
            
        except Exception as e:
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