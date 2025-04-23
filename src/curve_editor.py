import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QComboBox, QRadioButton, 
                            QButtonGroup, QCheckBox, QPushButton, QStatusBar, 
                            QMessageBox, QAction, QToolBar, QColorDialog, QActionGroup)
from PyQt5.QtGui import (QPainter, QPen, QBrush, QColor, QPainterPath, 
                         QKeyEvent, QMouseEvent)
from PyQt5.QtCore import Qt, QPoint, QRect, QSize, pyqtSignal

from curve import Point2D, CurveType, ContinuityType, ParametricSpline

class CanvasWidget(QWidget):
    """绘图画布类"""
    statusMessage = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 设置窗口属性
        self.setMinimumSize(600, 400)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        
        # 初始化数据
        self.points = []              # 型值点
        self.curve_points = []        # 曲线采样点
        
        # 参数设置
        self.curve_type = CurveType.FOLEY
        self.continuity_type = ContinuityType.C2
        self.current_mode = 0         # 0: 绘制三次样条曲线, 1: 编辑三次样条曲线
        
        # 交互状态
        self.adding_line = False
        self.editing_points = False
        self.editing_type_point = False
        self.selected_type_point_index = -1
        
        # 显示设置
        self.show_grid = True
        
        # 样式设置
        self.grid_size = 20
        self.point_radius = 4
        self.sampling_period = 0.001
        
        # 当前曲线对象
        self.current_curve = None
        
        # 颜色设置
        self.colors = {
            "background": QColor(255, 255, 255),
            "grid": QColor(200, 200, 200, 80),
            "type_point": QColor(255, 0, 0),
            "curve_foley": QColor(0, 255, 255)
        }
    
    def clear_all(self):
        """清除所有点和曲线"""
        self.points = []
        self.curve_points = []
        self.current_curve = None
        self.adding_line = False
        self.editing_points = False
        self.editing_type_point = False
        self.selected_type_point_index = -1
        self.update()
    
    def set_curve_type(self, curve_type):
        """设置曲线类型"""
        self.curve_type = curve_type
        self.update_curve()
        self.update()
    
    def set_continuity_type(self, continuity_type):
        """设置连续性类型"""
        self.continuity_type = continuity_type
        
        # 更新曲线
        self.update()
    
    def set_mode(self, mode):
        """设置编辑模式"""
        # 切换模式时，如果正在添加线段，取消添加
        if self.adding_line and len(self.points) > 0:
            self.points.pop()
            self.adding_line = False
        
        # 重置编辑状态
        self.editing_points = False
        self.editing_type_point = False
        self.selected_type_point_index = -1
        
        # 设置新模式
        self.current_mode = mode
        
        # 根据模式更新曲线类型
        self.update_curve()
        self.update()
    
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
        except Exception as e:
            # 错误处理：清空曲线点，但不输出错误
            self.curve_points = []
            self.current_curve = None
    
    def find_nearest_point(self, pos, point_list, max_distance=5.0):
        """查找最近的点"""
        nearest_index = -1
        min_distance = max_distance
        
        test_point = Point2D(pos.x(), pos.y())
        
        for i, point in enumerate(point_list):
            distance = test_point.distance_to(point)
            if distance < min_distance:
                min_distance = distance
                nearest_index = i
        
        return nearest_index
    
    def paintEvent(self, event):
        """绘制画布"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制背景
        painter.fillRect(self.rect(), self.colors["background"])
        
        # 绘制网格
        if self.show_grid:
            self.draw_grid(painter)
        
        # 绘制曲线
        self.draw_curve(painter)
        
        # 绘制点
        self.draw_points(painter)
    
    def draw_grid(self, painter):
        """绘制网格"""
        pen = QPen(self.colors["grid"])
        pen.setStyle(Qt.DotLine)
        painter.setPen(pen)
        
        # 绘制水平线
        for y in range(0, self.height(), self.grid_size):
            painter.drawLine(0, y, self.width(), y)
        
        # 绘制垂直线
        for x in range(0, self.width(), self.grid_size):
            painter.drawLine(x, 0, x, self.height())
    
    def draw_curve(self, painter):
        """绘制曲线"""
        if not self.curve_points:
            return
        
        # 设置画笔
        pen = QPen(self.colors["curve_foley"])
        pen.setWidth(2)
        painter.setPen(pen)
        
        # 绘制曲线
        path = QPainterPath()
        path.moveTo(self.curve_points[0].x, self.curve_points[0].y)
        
        for point in self.curve_points[1:]:
            path.lineTo(point.x, point.y)
        
        painter.drawPath(path)
    
    def draw_points(self, painter):
        """绘制型值点"""
        if not self.points:
            return
        
        # 设置型值点的画笔和画刷
        painter.setPen(Qt.black)
        painter.setBrush(QBrush(self.colors["type_point"]))
        
        # 绘制型值点
        for point in self.points:
            painter.drawEllipse(QPoint(int(point.x), int(point.y)), 
                               self.point_radius, self.point_radius)
        
        # 如果正在添加线段，绘制临时线段
        if self.adding_line and len(self.points) >= 2:
            painter.setPen(QPen(Qt.black, 1, Qt.DashLine))
            painter.drawLine(
                QPoint(int(self.points[-2].x), int(self.points[-2].y)),
                QPoint(int(self.points[-1].x), int(self.points[-1].y))
            )
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.LeftButton:
            # 获取鼠标位置
            pos = event.pos()
            mouse_point = Point2D(pos.x(), pos.y())
            
            if self.current_mode == 0:  # 绘制模式
                # 添加点
                if not self.adding_line:
                    self.points.append(mouse_point)
                    self.adding_line = True
                else:
                    self.points.append(mouse_point)
                
                self.update_curve()
            
            elif self.current_mode == 1:  # 编辑模式
                if not self.editing_type_point:
                    # 尝试选择型值点
                    self.selected_type_point_index = self.find_nearest_point(
                        pos, self.points
                    )
                    
                    if self.selected_type_point_index >= 0:
                        self.editing_points = True
                        self.editing_type_point = True
            
            self.update()
        
        elif event.button() == Qt.RightButton:
            # 如果正在添加线段，取消添加
            if self.adding_line and len(self.points) > 0:
                self.points.pop()
                self.adding_line = False
            
            # 停止编辑点
            self.editing_points = False
            self.editing_type_point = False
            
            self.update()
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.LeftButton:
            if self.editing_type_point:
                # 停止编辑点
                self.editing_type_point = False
                
                # 更新曲线
                self.update_curve()
                self.update()
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        pos = event.pos()
        mouse_point = Point2D(pos.x(), pos.y())
        
        if self.adding_line and len(self.points) > 0:
            # 更新最后一个点的位置
            self.points[-1] = mouse_point
            self.update_curve()
            self.update()
        
        elif self.editing_type_point and self.selected_type_point_index >= 0:
            # 更新选中型值点的位置
            self.points[self.selected_type_point_index] = mouse_point
            
            self.update_curve()
            
            self.update()
    
    def keyPressEvent(self, event):
        """键盘按键事件"""
        key = event.key()
        
        if key == Qt.Key_Escape:
            # 取消添加线段或编辑点
            if self.adding_line and len(self.points) > 0:
                self.points.pop()
                self.adding_line = False
            
            self.editing_points = False
            self.editing_type_point = False
            self.update()
        
        elif key == Qt.Key_Space:
            # 完成当前线段的添加
            if self.adding_line:
                self.adding_line = False
                self.update_curve()
                self.update()
        
        elif key == Qt.Key_Delete:
            # 删除选中的点
            if (self.selected_type_point_index >= 0 and 
                not self.editing_type_point and 
                len(self.points) > 0):
                
                del self.points[self.selected_type_point_index]
                self.selected_type_point_index = -1
                self.editing_points = False
                self.update_curve()
                self.update()

class CurveEditorMainWindow(QMainWindow):
    """曲线编辑器主窗口"""
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("曲线设计与编辑工具")
        self.setMinimumSize(800, 600)
        
        # 创建状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("就绪")
        
        # 创建画布
        self.canvas = CanvasWidget(self)
        self.canvas.statusMessage.connect(self.statusBar.showMessage)
        
        # 创建工具栏
        self.create_toolbars()
        
        # 创建菜单
        self.create_menus()
        
        # 设置中央窗口
        self.setCentralWidget(self.canvas)
    
    def create_toolbars(self):
        """创建工具栏"""
        # 模式工具栏
        mode_toolbar = QToolBar("编辑模式")
        mode_toolbar.setIconSize(QSize(32, 32))
        
        # 创建模式选择按钮组，使用QActionGroup而不是QButtonGroup
        mode_action_group = QActionGroup(self)
        mode_action_group.setExclusive(True)
        
        draw_spline_action = QAction("绘制三次样条曲线", self)
        draw_spline_action.setCheckable(True)
        draw_spline_action.setChecked(True)
        draw_spline_action.triggered.connect(lambda: self.canvas.set_mode(0))
        mode_toolbar.addAction(draw_spline_action)
        mode_action_group.addAction(draw_spline_action)
        
        edit_spline_action = QAction("编辑三次样条曲线", self)
        edit_spline_action.setCheckable(True)
        edit_spline_action.triggered.connect(lambda: self.canvas.set_mode(1))
        mode_toolbar.addAction(edit_spline_action)
        mode_action_group.addAction(edit_spline_action)
        
        self.addToolBar(mode_toolbar)
        
        # 参数化方法工具栏
        param_toolbar = QToolBar("参数化方法")
        
        # 创建参数化方法选择按钮组
        param_action_group = QActionGroup(self)
        param_action_group.setExclusive(True)
        
        foley_action = QAction("Foley参数化", self)
        foley_action.setCheckable(True)
        foley_action.setChecked(True)
        foley_action.triggered.connect(lambda: self.canvas.set_curve_type(CurveType.FOLEY))
        param_toolbar.addAction(foley_action)
        param_action_group.addAction(foley_action)
        
        self.addToolBar(param_toolbar)
        
        # 连续性工具栏
        continuity_toolbar = QToolBar("连续性设置")
        
        # 创建连续性选择按钮组
        continuity_action_group = QActionGroup(self)
        continuity_action_group.setExclusive(True)
        
        c2_action = QAction("C2连续", self)
        c2_action.setCheckable(True)
        c2_action.setChecked(True)
        c2_action.triggered.connect(lambda: self.canvas.set_continuity_type(ContinuityType.C2))
        continuity_toolbar.addAction(c2_action)
        continuity_action_group.addAction(c2_action)
        
        g1_action = QAction("G1连续", self)
        g1_action.setCheckable(True)
        g1_action.triggered.connect(lambda: self.canvas.set_continuity_type(ContinuityType.G1))
        continuity_toolbar.addAction(g1_action)
        continuity_action_group.addAction(g1_action)
        
        g0_action = QAction("G0连续", self)
        g0_action.setCheckable(True)
        g0_action.triggered.connect(lambda: self.canvas.set_continuity_type(ContinuityType.G0))
        continuity_toolbar.addAction(g0_action)
        continuity_action_group.addAction(g0_action)
        
        self.addToolBar(continuity_toolbar)
    
    def create_menus(self):
        """创建菜单"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件")
        
        clear_action = QAction("清除所有点", self)
        clear_action.triggered.connect(self.canvas.clear_all)
        file_menu.addAction(clear_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 视图菜单
        view_menu = menubar.addMenu("视图")
        
        toggle_grid_action = QAction("显示网格", self)
        toggle_grid_action.setCheckable(True)
        toggle_grid_action.setChecked(True)
        toggle_grid_action.triggered.connect(self.toggle_grid)
        view_menu.addAction(toggle_grid_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助")
        
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
    
    def toggle_grid(self, state):
        """切换网格显示"""
        self.canvas.show_grid = state
        self.canvas.update()
    
    def show_about_dialog(self):
        """显示关于对话框"""
        QMessageBox.about(
            self, 
            "关于曲线设计与编辑工具",
            "曲线设计与编辑工具 v1.0\n\n"
            "本工具模仿PowerPoint实现了曲线设计与编辑功能，\n"
            "支持参数化三次样条曲线的设计与编辑，采用Foley参数化方法。\n\n"
            "快捷键:\n"
            "空格 - 完成当前曲线绘制\n"
            "Esc - 取消当前操作\n"
            "Delete - 删除选中的点"
        )

def main():
    app = QApplication(sys.argv)
    window = CurveEditorMainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 