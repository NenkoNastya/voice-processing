import sys
from PyQt5 import QtWidgets, uic
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
from scipy.io import wavfile  # scipy library to read wav files
import numpy as np
from PyQt5 import QtWidgets
from PyQt5 import uic
from PyQt5.QtCore import QIODevice
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import sys
import numpy
import matplotlib
import pyaudio
import wave
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
import numpy as np
import time
import struct
import pitch
from scipy.fftpack import fft
from numpy import argmax, diff, nonzero
from scipy.signal import correlate
from scipy.io import wavfile  # scipy library to read wav files
import tkinter as tk
from scipy.io import wavfile  # scipy library to read wav files

# установить размер блока в 1024 сэмпла
chunk = 1024 * 4
# образец формата
FORMAT = pyaudio.paInt16
# моно, если хотите стере измените на 2
channels = 1
# 44100 сэмплов в секунду
sample_rate = 44100
record_seconds = 6
TIME_VECTOR = np.arange (chunk) / sample_rate * record_seconds * 10



filename = "recorded.wav"
app = QtWidgets.QApplication([])
ui = uic.loadUi("C:\PYTHON\startwindow.ui")
ui.setWindowTitle("Исследование спектральных характеристик голосоречевого сигнала при дисфонии")


def btn_record():
    global i
    i = 1
    app.quit()


def btn_spectr():
    global i
    i = 2
    app.quit()

def update_waveform():
    p = pyaudio.PyAudio()
    # открыть объект потока как ввод и вывод
    stream = p.open(format=FORMAT,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)
    for a in range(int(44100 / chunk * record_seconds)):
        data = stream.read(chunk)
        # если вы хотите слышать свой голос во время записи
        # stream.write(data)
        # print(len(data))
        data_int = struct.unpack(str(chunk * 2) + 'B', data)
        data_np = np.array(data_int, dtype='b')[::2] + 128
        line.setData(x=TIME_VECTOR, y=data_np, pen=pen)
        print(data_np)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    # завершить работу объекта pyaudio
    p.terminate()
    # сохранить аудиофайл
    # открываем файл в режиме 'запись байтов'
    wf = wave.open(filename, "wb")
    # установить каналы
    wf.setnchannels(channels)
    # установить формат образца
    wf.setsampwidth(p.get_sample_size(FORMAT))
    # установить частоту дискретизации
    wf.setframerate(sample_rate)
    # записываем кадры как байты
    wf.writeframes(b"".join(frames))
    # закрыть файл
    wf.close()

ui.record.clicked.connect(btn_record)
ui.spectr.clicked.connect(btn_spectr)
ui.show()
app.exec()

if (i == 1):
    frames = []
    apps = QtWidgets.QApplication([])
    ui = uic.loadUi("C:\PYTHON\graphexample.ui")

    timer = QtCore.QTimer()
    timer.timeout.connect(update_waveform)
    timer.setSingleShot(True)
    timer.start(6000)
    # initialize PyAudio object
    start_time = time.time()
    x = np.arange(0, 2 * chunk, 2)
    line = ui.graph.plot(x, np.random.rand(chunk), '-', lw=2)
    grid = QtWidgets.QGridLayout(ui.centralwidget)
    grid.addWidget(ui.graph, 0, 0)
    ui.graph.setBackground('w')
    pen = pg.mkPen(color=(0, 0, 0))
    ui.show()
    apps.exec()


if (i == 2):
    apps1 = QtWidgets.QApplication([])
    ui = uic.loadUi("C:\PYTHON\spectrgraph.ui")
    # initialize PyAudio object
    p = pyaudio.PyAudio()
    # открыть объект потока как ввод и вывод
    stream = p.open(format=FORMAT,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)
    start_time = time.time()
    frames = []
    data = stream.read(chunk)

    # line, = ax.plot(x, np.random.rand(chunk), '-', lw=2)
    for a in range(int(44100 / chunk * record_seconds)):
        data = stream.read(chunk)
        # если вы хотите слышать свой голос во время записи
        # stream.write(data)
        # print(len(data))
        data_int = struct.unpack(str(chunk * 2) + 'B', data)
        data_np = np.array(data_int, dtype='b')[::2] + 128
        y_fft = fft(data_int)
        spec = np.abs(y_fft[0:chunk]) * 2 / (256 * chunk)
        frames.append(data)
        # line.set_ydata(data_np)
        # fig.canvas.draw()
        # fig.canvas.flush_events()
    x_fft = np.linspace(20, sample_rate, chunk)

    grid = QtWidgets.QGridLayout(ui.centralwidget)
    grid.addWidget(ui.graph, 0, 0)
    ui.graph.setBackground('w')
    pen = pg.mkPen(color=(0,0,0))
    ui.graph.setXRange(20,sample_rate/22, padding=0)
    ui.graph.plot(x_fft, spec, pen = pen)
    #p1.setXRange(20, sample_rate / 22)
    # остановить и закрыть поток
    stream.stop_stream()
    stream.close()
    # завершить работу объекта pyaudio
    p.terminate()
    # сохранить аудиофайл
    # открываем файл в режиме 'запись байтов'
    wf = wave.open(filename, "wb")
    # установить каналы
    wf.setnchannels(channels)
    # установить формат образца
    wf.setsampwidth(p.get_sample_size(FORMAT))
    # установить частоту дискретизации
    wf.setframerate(sample_rate)
    # записываем кадры как байты
    wf.writeframes(b"".join(frames))
    # закрыть файл
    wf.close()
    # fs, Audiodata = wavfile.read("recorded.wav")




    ui.show()
    apps1.exec()