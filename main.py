import csv
import numpy as np

from PyQt5.QtWidgets import (QApplication, QMainWindow, QScrollArea, QVBoxLayout, QWidget, QGridLayout,
                             QPushButton, QHBoxLayout, QFileDialog, QMessageBox, QSlider, QLabel,QLineEdit)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


from Automation.BDaq import *
from Automation.BDaq.InstantAiCtrl import InstantAiCtrl

deviceDescription = "USB-4704,BID#0"
profilePath = u"../../profile/DemoDevice.xml"
ret = ErrorCode.Success
instanceAiObj = InstantAiCtrl(deviceDescription)
instanceAiObj.loadProfile = profilePath

import time

def readAI():
    global count, ts

    _, scaledData = instanceAiObj.readDataF64(0, 8)
    count += 1
    if count == 100:
        count = 0
        print(f'{time.time() - ts}')
        ts = time.time()

    return scaledData



class FilterThread(QThread):
    filter_completed = pyqtSignal(list)

    def __init__(self, raw_data, lower_bound, upper_bound, sampling_rate):
        super().__init__()
        self.raw_data = raw_data
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.sampling_rate = sampling_rate

    def run(self):
        # 将原始数据转换为 numpy 数组
        raw_data_np = np.array(self.raw_data)
        # 进行 FFT
        fft_data = np.fft.fft(raw_data_np)

        freq = np.fft.fftfreq(len(fft_data), d=1 / self.sampling_rate)

        # 创建滤波器
        ideal_filter = np.zeros_like(freq)

        if self.lower_bound == 0:
            # 低通滤波器
            ideal_filter[freq <= self.upper_bound] = 1
        else:
            # 带通滤波器
            ideal_filter[(freq >= self.lower_bound) & (freq <= self.upper_bound)] = 1

        # 应用滤波器
        filtered_fft_data = fft_data * ideal_filter

        # 进行逆 FFT 得到滤波后的数据
        filtered_data = np.fft.ifft(filtered_fft_data).real

        # 发送滤波完成信号
        self.filter_completed.emit(filtered_data.tolist())


class SensorPlot(QWidget):
    def __init__(self, parent, index):
        super().__init__(parent)
        self.index = index
        self.is_running = False
        self.data = []
        self.raw_data = []
        self.time_data = []
        self.time_limit = 10
        self.voltage_limit = 10
        self.sampling_rate = 100
        self.current_scale = 1
        self.display_mode = 'time'


        self.canvas = PlotCanvas(self, width=5, height=4)
        self.canvas.setFixedWidth(600)
        self.canvas.setFixedHeight(400)
        self.start_button = QPushButton('Start', self)
        self.start_button.setFixedWidth(160)
        self.start_button.setFixedHeight(30)
        self.start_button.clicked.connect(self.toggle)
        self.save_button = QPushButton('Save Data', self)
        self.save_button.setFixedWidth(160)
        self.save_button.setFixedHeight(30)
        self.save_button.clicked.connect(self.save_data)
        self.read_button = QPushButton('Read Data', self)
        self.read_button.setFixedWidth(160)
        self.read_button.setFixedHeight(30)
        self.read_button.setEnabled(True)  # 默认禁用
        self.read_button.clicked.connect(self.read_data)

        self.scale_button = QPushButton('Apply Scale', self)
        self.scale_button.setFixedWidth(160)
        self.scale_button.setFixedHeight(30)
        self.scale_button.setEnabled(False)
        self.scale_button.clicked.connect(self.apply_scale)

        self.scale_slider = QSlider(Qt.Horizontal, self)
        self.scale_slider.setMinimum(1)
        self.scale_slider.setMaximum(5)
        self.scale_slider.setValue(1)
        self.scale_slider.setFixedWidth(160)
        self.scale_slider.valueChanged.connect(self.update_scale_label)
        self.scale_label = QLabel('Scale: x1', self)

        self.filter_button = QPushButton('Apply Filter', self)
        self.filter_button.setFixedWidth(140)
        self.filter_button.setFixedHeight(30)
        self.filter_button.setEnabled(False)
        self.filter_button.clicked.connect(self.apply_filter)

        self.restore_button = QPushButton('Restore Data', self)
        self.restore_button.setFixedWidth(140)
        self.restore_button.setFixedHeight(30)
        self.restore_button.setEnabled(False)
        self.restore_button.clicked.connect(self.restore_data)

        self.fft_button = QPushButton('Toggle FFT', self)
        self.fft_button.setFixedWidth(140)
        self.fft_button.setFixedHeight(30)
        self.fft_button.setEnabled(False)
        self.fft_button.clicked.connect(self.toggle_fft)

        self.lower_bound_slider = QSlider(Qt.Horizontal, self)
        self.lower_bound_slider.setMinimum(0)
        self.lower_bound_slider.setMaximum(499)
        self.lower_bound_slider.setValue(100)
        self.lower_bound_slider.setFixedHeight(30)
        self.lower_bound_slider.setFixedWidth(160)
        self.lower_bound_slider.valueChanged.connect(self.update_lower_bound_label)
        self.lower_bound_label = QLabel('Lower Bound: 100 Hz')

        self.upper_bound_slider = QSlider(Qt.Horizontal, self)
        self.upper_bound_slider.setMinimum(0)
        self.upper_bound_slider.setMaximum(499)
        self.upper_bound_slider.setValue(499)
        self.upper_bound_slider.setFixedHeight(30)
        self.upper_bound_slider.setFixedWidth(160)
        self.upper_bound_slider.valueChanged.connect(self.update_upper_bound_label)
        self.upper_bound_label = QLabel('Upper Bound: 499 Hz')

        self.time_slider = QSlider(Qt.Horizontal, self)
        self.time_slider.setMinimum(10)
        self.time_slider.setMaximum(50)
        self.time_slider.setValue(self.time_limit)
        self.time_slider.valueChanged.connect(self.update_time_limit)

        self.voltage_slider = QSlider(Qt.Horizontal, self)
        self.voltage_slider.setMinimum(2)
        self.voltage_slider.setMaximum(10)
        self.voltage_slider.setValue(10)
        self.voltage_slider.valueChanged.connect(self.update_voltage_limit)

        self.sampling_slider = QSlider(Qt.Horizontal, self)
        self.sampling_slider.setMinimum(1)
        self.sampling_slider.setMaximum(100)
        self.sampling_slider.setValue(self.sampling_rate)
        self.sampling_slider.valueChanged.connect(self.update_sampling_rate)

        self.time_input = QLineEdit(self)
        self.time_input.setFixedWidth(80)
        self.time_input.setText(str(self.time_limit))

        self.voltage_input = QLineEdit(self)
        self.voltage_input.setFixedWidth(80)
        self.voltage_input.setText(str(self.voltage_limit))

        self.sampling_input = QLineEdit(self)
        self.sampling_input.setFixedWidth(80)
        self.sampling_input.setText(str(self.sampling_rate))


        self.lower_bound_input = QLineEdit(self)
        self.lower_bound_input.setText(str(self.lower_bound_slider.value()))
        self.lower_bound_input.setFixedWidth(60)

        self.upper_bound_input = QLineEdit(self)
        self.upper_bound_input.setText(str(self.upper_bound_slider.value()))
        self.upper_bound_input.setFixedWidth(60)


        # 添加滑条的标签
        self.time_slider_label = QLabel('Time Limit (s):', self)
        self.voltage_slider_label = QLabel('Voltage Limit (V):', self)
        self.sampling_slider_label = QLabel('Sampling Rate (Hz):', self)

        self.time_slider.valueChanged.connect(self.update_time_limit_from_slider)
        self.voltage_slider.valueChanged.connect(self.update_voltage_limit_from_slider)
        self.sampling_slider.valueChanged.connect(self.update_sampling_rate_from_slider)
        self.lower_bound_slider.valueChanged.connect(self.update_lower_bound_label)
        self.upper_bound_slider.valueChanged.connect(self.update_upper_bound_label)

        self.voltage_input.returnPressed.connect(self.update_voltage_limit_from_input)
        self.time_input.returnPressed.connect(self.update_time_limit_from_input)
        self.sampling_input.returnPressed.connect(self.update_sampling_rate_from_input)
        self.lower_bound_input.textChanged.connect(self.update_lower_bound_from_input)
        self.upper_bound_input.textChanged.connect(self.update_upper_bound_from_input)

        # 新增 QLabel 用于显示悬停数据点的信息
        self.info_label = QLabel("Hover over a data point to see its value", self)

        canvas_layout = QVBoxLayout()  # 修改为垂直布局
        canvas_layout.addWidget(self.canvas)  # 添加画布
        canvas_layout.addWidget(self.info_label)  # 将信息标签添加到画布下方

        button_layout = QGridLayout()
        button_layout.setSpacing(30)
        button_layout.setContentsMargins(60, 60, 60, 10)
        button_layout.addWidget(self.start_button, 0, 0)  # 第一行，第一列
        button_layout.addWidget(self.save_button, 0, 1)  # 第一行，第二列
        button_layout.addWidget(self.read_button, 0, 2)  # 第一行，第二列

        button_layout.addWidget(self.scale_button, 1, 0)  # 第二行，第二列
        button_layout.addWidget(self.scale_label, 1 ,1 )
        button_layout.addWidget(self.scale_slider, 1, 2)

        filter_layout = QGridLayout()
        filter_layout.setSpacing(60)
        filter_layout.setContentsMargins(60, 60, 60, 10)
        filter_layout.addWidget(self.filter_button, 0, 0)
        filter_layout.addWidget(self.restore_button, 0, 1)
        filter_layout.addWidget(self.fft_button, 0, 2)
        filter_layout.addWidget(self.lower_bound_label, 1, 0)
        filter_layout.addWidget(self.lower_bound_slider, 1, 1)
        filter_layout.addWidget(self.upper_bound_label, 2, 0)
        filter_layout.addWidget(self.upper_bound_slider, 2, 1)

        slider_layout = QGridLayout()
        slider_layout.setSpacing(30)
        slider_layout.setContentsMargins(60, 60, 60, 10)
        slider_layout.addWidget(self.time_slider_label, 0, 0)
        slider_layout.addWidget(self.time_slider, 0, 1)
        slider_layout.addWidget(self.voltage_slider_label, 1, 0)
        slider_layout.addWidget(self.voltage_slider, 1, 1)
        slider_layout.addWidget(self.sampling_slider_label, 2, 0)
        slider_layout.addWidget(self.sampling_slider, 2, 1)

        # 更新滑条和输入框的布局
        slider_layout.addWidget(self.time_slider_label, 0, 0)
        slider_layout.addWidget(self.time_slider, 0, 1)
        slider_layout.addWidget(self.time_input, 0, 2)  # 新增输入框
        slider_layout.addWidget(self.voltage_slider_label, 1, 0)
        slider_layout.addWidget(self.voltage_slider, 1, 1)
        slider_layout.addWidget(self.voltage_input, 1, 2)  # 新增输入框
        slider_layout.addWidget(self.sampling_slider_label, 2, 0)
        slider_layout.addWidget(self.sampling_slider, 2, 1)
        slider_layout.addWidget(self.sampling_input, 2, 2)  # 新增输入框

        filter_layout.addWidget(self.lower_bound_input, 1, 2)  # 将输入框放在滑条旁边
        filter_layout.addWidget(self.upper_bound_input, 2, 2)  # 将输入框放在滑条旁边

        layout = QHBoxLayout()
        layout.addLayout(canvas_layout)
        layout.addLayout(button_layout)
        layout.addLayout(filter_layout)
        layout.addLayout(slider_layout)
        # layout.addWidget(self.info_label)  # 添加信息标签到布局中

        self.setLayout(layout)

        # 连接 canvas 的 hover 事件
        self.canvas.mpl_connect('motion_notify_event', self.on_hover)

    def on_hover(self, event):
        if event.inaxes == self.canvas.ax:  # 检查鼠标是否在绘图区域
            xdata = event.xdata
            if xdata is not None:
                try:
                # 找到最近的数据点
                    closest_index = (np.abs(np.array(self.time_data) - xdata)).argmin()
                    value = self.data[closest_index]
                    self.info_label.setText(f"Time: {self.time_data[closest_index]:.2f}s, Value: {value:.2f}")
                except:
                    pass
    def toggle(self):
        if self.is_running:
            self.stop()
        else:
            self.start()

    def start(self):
        self.is_running = True
        self.start_button.setText('Stop')
        self.data.clear()
        self.raw_data.clear()
        self.time_data.clear()
        self.canvas.clear_data()
        self.start_time = time.time()  # 记录开始时间
        self.update_data()
        self.scale_button.setEnabled(False)
        self.filter_button.setEnabled(False)
        self.restore_button.setEnabled(False)
        self.fft_button.setEnabled(False)  # 开始时禁用 FFT 按钮
        self.read_button.setEnabled(False)


    def update_data(self):
        if self.is_running:

            new_data = readAI()[self.index]
            self.data.append(new_data)
            self.raw_data.append(new_data)
            current_time = time.time() - self.start_time  # 计算相对时间
            self.time_data.append(current_time)

            if len(self.time_data) > 0 and (self.time_data[-1] > self.time_limit):
                self.data.pop(0)
                self.raw_data.pop(0)
                self.time_data.pop(0)

            self.update_plot()
            QTimer.singleShot(int(1000 / self.sampling_rate), self.update_data)

    def stop(self):
        self.is_running = False
        self.start_button.setText('Start')
        self.scale_button.setEnabled(True)
        self.filter_button.setEnabled(True)
        self.restore_button.setEnabled(True)
        self.fft_button.setEnabled(True)  # 停止时启用 FFT 按钮
        self.read_button.setEnabled(True)



    def apply_filter(self):
        if not self.is_running:
            if len(self.raw_data) < 2:
                QMessageBox.warning(self, "Insufficient Data",
                                    "Not enough data to apply filter. Please collect more data.")
                return

            lower_bound = self.lower_bound_slider.value()
            upper_bound = self.upper_bound_slider.value()

            # 添加参数检查
            if lower_bound >= upper_bound:
                QMessageBox.warning(self, "Invalid Filter Parameters",
                                    "Upper bound must be greater than lower bound.")
                return
            if lower_bound >= 0.5 * self.sampling_rate or upper_bound >= 0.5 * self.sampling_rate:
                QMessageBox.warning(self, "Invalid Filter Parameters",
                                    "Filter frequency must be less than half the sampling rate.")
                return

            self.filter_button.setEnabled(False)  # Disable the button during filtering
            self.filter_thread = FilterThread(self.raw_data, lower_bound, upper_bound, self.sampling_rate)
            self.filter_thread.filter_completed.connect(self.filter_completed)
            self.filter_thread.start()

    def filter_completed(self, filtered_data):
        self.data = filtered_data
        self.update_plot()
        self.filter_button.setEnabled(True)  # Re-enable the button after filtering

    def restore_data(self):
        if not self.is_running:
            self.data = self.raw_data.copy()
            self.update_plot()

    def toggle_fft(self):
        if not self.is_running:
            if len(self.data) < 2:
                QMessageBox.warning(self, "Insufficient Data",
                                    "Not enough data to compute FFT. Please collect more data.")
                return

            self.display_mode = 'fft' if self.display_mode == 'time' else 'time'
            self.update_plot()

    def update_plot(self):
        if self.display_mode == 'time':
            self.canvas.update_plot(self.data, self.time_data, self.time_limit, self.voltage_limit)
        elif self.display_mode == 'fft':
            self.plot_fft()

    def plot_fft(self):
        if len(self.data) > 1:
            # 进行FFT变换
            fft_data = np.fft.fft(self.data)

            # 计算对应的频率
            freq = np.fft.fftfreq(len(fft_data), d=1 / self.sampling_rate)
            # 只取正频率部分（从0到采样率的一半）
            positive_freq = freq[:len(freq) // 2]
            magnitude = np.abs(fft_data)[:len(fft_data) // 2]

            # 更新图像
            self.canvas.update_plot_fft(positive_freq, magnitude, self.voltage_limit)

    def apply_scale(self):
        if not self.is_running:
            self.current_scale = self.scale_slider.value()
            scaled_data = [d * self.current_scale for d in self.data]
            self.canvas.update_plot(scaled_data, self.time_data, self.time_limit, self.voltage_limit)

    def update_scale_label(self):
        self.scale_label.setText(f'Scale: x{self.scale_slider.value()}')

    def update_time_limit(self, value):
        self.time_limit = value
        self.update_plot()

    def update_voltage_limit(self, value):
        self.voltage_limit = value
        self.update_plot()

    def update_sampling_rate(self, value):
        self.sampling_rate = value

    def update_lower_bound_label(self):
        self.lower_bound_label.setText(f'Lower Bound: {self.lower_bound_slider.value()} Hz')
        self.lower_bound_input.setText(str(self.lower_bound_slider.value()))  # 同步到输入框

    def update_upper_bound_label(self):
        self.upper_bound_label.setText(f'Upper Bound: {self.upper_bound_slider.value()} Hz')
        self.upper_bound_input.setText(str(self.upper_bound_slider.value()))  # 同步到输入框

    def save_data(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv);;All Files (*)",
                                                   options=options)
        if file_name:
            try:
                with open(file_name, 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([f"Sensor {self.index + 1} Data"])
                    csvwriter.writerows([[data] for data in self.data])
                QMessageBox.information(self, "Success", f"Sensor {self.index + 1} data saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save data: {str(e)}")

    def read_data(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv);;All Files (*)",
                                                   options=options)
        if file_name:
            try:
                with open(file_name, 'r') as csvfile:
                    csvreader = csv.reader(csvfile)
                    header = next(csvreader)  # 跳过标题行
                    self.data = [float(row[0]) for row in csvreader]  # 假设每行只有一个数据点
                    self.raw_data = self.data.copy()  # 如果有原始数据字段，可修改
                    self.time_data = list(np.arange(0, len(self.data) / self.sampling_rate, 1 / self.sampling_rate))
                    self.update_plot()
                QMessageBox.information(self, "Success", "Data loaded successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to read data: {str(e)}")

    def update_time_limit_from_input(self):
        try:
            value = int(self.time_input.text())
            if 10 <= value <= 50:
                self.time_limit = value
                self.time_slider.setValue(value)
                self.update_plot()
        except ValueError:
            pass  # Ignore invalid input

    def update_voltage_limit_from_input(self):
        try:
            value = int(self.voltage_input.text())
            if 2 <= value <= 10:
                self.voltage_limit = value
                self.voltage_slider.setValue(value)
                self.update_plot()
        except ValueError:
            pass  # Ignore invalid input

    def update_sampling_rate_from_input(self):
        try:
            value = int(self.sampling_input.text())
            if 1 <= value <= 100:
                self.sampling_rate = value
                self.sampling_slider.setValue(value)
                self.update_plot()
        except ValueError:
            pass  # Ignore invalid input
    def update_time_limit_from_slider(self, value):
        self.time_limit = value
        self.time_input.setText(str(value))
        self.update_plot()

    def update_voltage_limit_from_slider(self, value):
        self.voltage_limit = value
        self.voltage_input.setText(str(value))
        self.update_plot()

    def update_sampling_rate_from_slider(self, value):
        self.sampling_rate = value
        self.sampling_input.setText(str(value))

    def update_lower_bound_from_input(self, text):
        try:
            value = int(text)
            if 0 <= value < self.upper_bound_slider.value():  # 检查范围
                self.lower_bound_slider.setValue(value)
            else:
                # 弹出提示窗口
                QMessageBox.warning(self, "输入错误", f"请输入 0 到 {self.upper_bound_slider.value() - 1} 范围内的数字")
                self.lower_bound_input.setText(str(self.lower_bound_slider.value()))  # 恢复之前的值
        except ValueError:
            # 输入无效时恢复滑条值
            self.lower_bound_input.setText(str(self.lower_bound_slider.value()))

    def update_upper_bound_from_input(self, text):
        try:
            value = int(text)
            if value > self.lower_bound_slider.value() and value <= 499:  # 检查范围
                self.upper_bound_slider.setValue(value)
            else:
                # 弹出提示窗口
                QMessageBox.warning(self, "输入错误",
                                    f"请输入 {self.lower_bound_slider.value() + 1} 到 499 范围内的数字")
                self.upper_bound_input.setText(str(self.upper_bound_slider.value()))  # 恢复之前的值
        except ValueError:
            # 输入无效时恢复滑条值
            self.upper_bound_input.setText(str(self.upper_bound_slider.value()))

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super(PlotCanvas, self).__init__(self.fig)
        self.setParent(parent)

    def update_plot(self, data, time_data, time_limit, voltage_limit):
        self.ax.cla()
        self.ax.set_title(f"Sensor Data")
        self.ax.set_ylim(-voltage_limit, voltage_limit)
        self.ax.grid(True)

        if len(time_data) > 0:
            self.ax.set_xlim(time_data[-1] - time_limit, time_data[-1])

        self.ax.plot(time_data, data, 'b-')
        self.ax.xaxis.set_ticklabels([])

        self.draw()

    def update_plot_fft(self, freq, magnitude, voltage_limit):
        self.ax.cla()
        self.ax.set_title("FFT of Sensor Data")
        self.ax.set_ylim(0, max(magnitude) * 1.1)
        self.ax.grid(True)

        self.ax.plot(freq[:len(magnitude)], magnitude[:len(magnitude)], 'r-')
        self.ax.set_xlabel('Frequency (Hz)')
        self.ax.set_ylabel('Magnitude')

        self.draw()

    def clear_data(self):
        self.ax.cla()
        self.draw()


class App(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Sensor Data Real-Time Plot")
        self.setGeometry(100, 100, 800, 600)

        # 创建QScrollArea实例
        scroll_area = QScrollArea(self)
        scroll_area.setGeometry(0, 0, 800, 600)  # 设置QScrollArea的尺寸

        # 创建一个QWidget作为QScrollArea的子控件
        central_widget = QWidget()
        scroll_area.setWidget(central_widget)  # 设置子控件

        # 创建布局并添加到子控件
        layout = QVBoxLayout(central_widget)

        # 添加SensorPlot实例到布局
        for i in range(8):
            sensor_plot = SensorPlot(central_widget, i)
            layout.addWidget(sensor_plot)

        # 设置QScrollArea的WidgetResizable属性为True，这样QScrollArea会根据子控件的大小自动调整滚动条
        scroll_area.setWidgetResizable(True)

        # 将QScrollArea设置为主窗口的中心控件
        self.setCentralWidget(scroll_area)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = App()
    main_window.show()
    sys.exit(app.exec_())
