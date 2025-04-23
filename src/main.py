#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
曲线设计与编辑工具 - 主程序入口
"""

import sys
from PyQt5.QtWidgets import QApplication
from curve_editor import CurveEditorMainWindow

def main():
    """主函数"""
    app = QApplication(sys.argv)
    window = CurveEditorMainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()