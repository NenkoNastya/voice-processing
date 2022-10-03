import parselmouth
import wave
import pyaudio
import librosa
import matplotlib as plt
from matplotlib import pyplot as plt
import librosa.display
from tkinter import ttk
import tkinter as tk
from numpy import argmax, diff, nonzero
from scipy.signal import correlate
import tkinter.filedialog as fd
from scipy.io import wavfile # scipy library to read wav files
import numpy as np
from array import *
from scipy.fftpack import fft
from scipy.signal import find_peaks
import shutil

import parselmouth
from parselmouth.praat import call

sound = parselmouth.Sound("recorded.wav")

def recording():
    filename = "recorded.wav"
    chunk = 1024
    FORMAT = pyaudio.paInt16
    channels = 1
    sample_rate = 44100
    record_seconds = 5
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)
    frames = []
    print("Recording...")
    for i in range(int(44100 / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)
    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(sample_rate)
    wf.writeframes(b"".join(frames))
    wf.close()
    AudioName = "recorded.wav"
    fs, Audiodata = wavfile.read(AudioName)
    import matplotlib.pyplot as plt
    n = len(Audiodata)
    AudioFreq = fft(Audiodata)
    AudioFreq = AudioFreq[0:int(np.ceil((n + 1) / 2.0))]
    MagFreq = np.abs(AudioFreq)
    MagFreq = MagFreq / float(n)
    freqAxis = np.arange(0, int(np.ceil((n + 1) / 2.0)), 1.0) * (fs / n); # ось частот для спектра
    #peaks = find_peaks(MagFreq)
    #height = peaks[1]['peak_heights']  # list of the heights of the peaks
    #peak_pos = freqAxis[peaks[0]]  # list of the peaks positions
    plt.figure(figsize=(30, 10))
    plt.subplot(1, 2, 1)
    plt.plot(Audiodata)
    plt.ylabel('Audio signal in samples')
    plt.subplot(1, 2, 2)
    plt.plot(freqAxis, MagFreq)
    plt.xlim(0, 1700)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude spectrum')
    plt.show()

def parabolic(f, x):
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return xv, yv


def autocorr(Audiodata, fs):
    corr = correlate(Audiodata, Audiodata, mode='full')
    corr = corr[len(corr)//2:]
    d = diff(corr)
    start = nonzero(d > 0)[0][0]
    peak = argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)
    return fs / px


def freq():
        Audiodata, fs= librosa.load("recorded.wav")
        str(autocorr(Audiodata, fs))
        answer.delete(0, tk.END)
        answer.insert(0, '%.1f Hz' % autocorr(Audiodata, fs))
        return autocorr(Audiodata, fs)

def draw_spectrogram(spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")

def draw_intensity(intensity):
    plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
    plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
    plt.grid(False)
    plt.ylim(0)
    plt.ylabel("intensity [dB]")

def intensity():
    intensity = sound.to_intensity()
    spectrogram = sound.to_spectrogram()
    plt.figure()
    draw_spectrogram(spectrogram)
    plt.twinx()
    draw_intensity(intensity)
    plt.xlim([sound.xmin, sound.xmax])
    plt.show()

def draw_pitch(pitch):
    # Extract selected pitch contour, and
    # replace unvoiced samples by NaN to not plot
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values==0] = np.nan
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
    plt.grid(False)
    plt.ylim(0, pitch.ceiling)
    plt.ylabel("fundamental frequency [Hz]")

def Pitch():
    pitch = sound.to_pitch()
    # If desired, pre-emphasize the sound fragment before calculating the spectrogram
    pre_emphasized_snd = sound.copy()
    pre_emphasized_snd.pre_emphasize()
    spectrogram = pre_emphasized_snd.to_spectrogram(window_length=0.03, maximum_frequency=8000)
    plt.figure()
    draw_spectrogram(spectrogram)
    plt.twinx()
    draw_pitch(pitch)
    plt.xlim([sound.xmin, sound.xmax])
    plt.show()


def measurePitch(sound, f0min, f0max, unit):
    sound = parselmouth.Sound(sound)  # read the sound
    duration = call(sound, "Get total duration")  # duration
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max)  # create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit)  # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0, 0, unit)  # get standard deviation
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer = call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    print(duration, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer,
          localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer)
    list = [('Длительность = ', duration),
            ('Pitch = ', pitch)]
    my_file = open("Parameters.txt", "w")
    my_file.write('Длительность = %s\n' % duration)
    my_file.write('Jitter = %s\n'% localJitter)
    my_file.write('Высота = %s\n' % pitch)
    my_file.close()
   # return duration, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer

def PitchContour(sound, f0min, f0max, unit, interpolation):
    sound = parselmouth.Sound("recorded.wav")
    #sound = parselmouth.Sound(sound)
    my_file2 = open("Pitch listing.txt", "w")
    tmin = 0 #seconds
    tmax = 5 #seconds
    Pitch = call(sound, "To Pitch", 0.001, f0min, f0max)
    my_file2.write('Контур высоты =\n')
    arr_time = np.empty(501)
    arr_pitch = np.empty(501)
    for i in range(int((tmax-tmin)/0.01)):
        time = tmin + i*0.01
        PitchLine = call(Pitch, "Get value at time", time, unit, interpolation)
        print(time, PitchLine)
        arr_time[i] = time
        arr_pitch[i] = PitchLine
        #print(arr_time[i])
        #print(PitchLine)
        my_file2.write('%s\n' % PitchLine)
    plt.plot(arr_time, arr_pitch)
    plt.title("Dynamic Changes in FF")
    plt.ylabel('Fundamental Frequency, Hertz')
    plt.xlabel('Time, sec')
    plt.ylim(100, 500)
    plt.show()
    my_file2.close()

window = tk.Tk()
window.title('Детектирование эмоций')
window.geometry("400x400")
answer = tk.Entry(window, width=20)
btn1 = tk.Button(window, width=20, height=2, text='Запись голоса', command=recording)
btn2 = tk.Button(window, width=20, height=2, text='Вычислить ЧОТ', command=freq)
btn3 = tk.Button(window, width=20, height=2, text='Спектрограмма', command=intensity)
btn4 = tk.Button(window, width=20, height=2, text='Высота', command=Pitch)
btn5 = tk.Button(window, width=20, height=2, text='Посчитать параметры', command= lambda: measurePitch(sound, 75, 500, "Hertz"))
btn6 = tk.Button(window, width=20, height=2, text='Контур частоты', command= lambda: PitchContour(sound, 75, 500, "Hertz", "Linear"))
btn1.grid(row=3, column=2)
btn2.grid(row=4, column=2)
answer.grid(row=5, column=2)
btn3.grid(row=6, column=2)
btn4.grid(row=7, column=2)
btn5.grid(row=8, column=2)
btn6.grid(row=9, column=2)
window.grid_columnconfigure(0, minsize=140)
window.grid_rowconfigure(0, minsize=10)
window.mainloop()


shutil.copyfile
shutil.copyfile('recorded.wav', 'recorded2.wav')
