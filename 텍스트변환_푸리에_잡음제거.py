import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.signal import butter, filtfilt

# 시리얼 포트 설정 (포트 이름과 전송 속도를 환경에 맞게 설정)
py_serial = serial.Serial(port="COM8", baudrate=9600)
time.sleep(3)  # 시리얼 포트가 열릴 때까지 대기

# 텍스트 파일 경로 설정
file_path = 'exhibition/wow.txt'

# 텍스트 파일 열기 및 시리얼 포트에서 데이터 읽기
with open(file_path, 'w') as f:  # 'a' 모드 대신 'w' 모드를 사용하여 파일을 새로 씀
    start_time = time.time()
    while time.time() - start_time < 10:  # 10초 동안 데이터 읽기
        if py_serial.readable():
            response = py_serial.readline()
            f.write(response.decode('utf-8').strip() + '\n')

# 시리얼 포트 닫기
py_serial.close()

# 텍스트 파일에서 데이터 읽기
data = []
with open(file_path, 'r') as f:
    for line in f:
        try:
            voltage = float(line.strip())
            data.append(voltage)
        except ValueError:
            print(f"Invalid line: {line}")

data = np.array(data)

# 샘플 레이트 설정 (예: 100Hz, 아두이노 코드의 데이터 전송 간격에 따라 설정)
sample_rate = 100

# 대역 통과 필터 적용 함수
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    if low <= 0 or high >= 1:
        raise ValueError("Digital filter critical frequencies must be 0 < Wn < 1")
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# 뇌파 신호의 잡음 제거를 위한 필터링 (0-50Hz 대역 통과 필터 적용)
lowcut = 0.1  # 0Hz 대신 0.1Hz로 설정하여 필터링의 안정성 확보
highcut = 45.0  # 나이퀴스트 주파수를 넘지 않도록 설정 (50Hz 대신 45Hz로 설정)
filtered_data = bandpass_filter(data, lowcut, highcut, sample_rate, order=5)

# 푸리에 변환 수행
fft = np.fft.fft(filtered_data)
magnitude = np.abs(fft)
frequency = np.fft.fftfreq(len(magnitude), 1 / sample_rate)

# 주파수 및 스펙트럼 생성
f = np.linspace(0, sample_rate, len(magnitude))
left_spectrum = magnitude[:int(len(magnitude) / 2)]
left_f = f[:int(len(magnitude) / 2)]

# 스펙트럼 플롯 생성
plt.figure(figsize=(20, 10))
plt.plot(left_f, left_spectrum)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Power Spectrum')
plt.grid()
plt.show()

# 단시간 푸리에 변환 (STFT) 수행
n_fft = 2048
hop_length = 512
stft = librosa.stft(filtered_data, n_fft=n_fft, hop_length=hop_length)
spectrogram = np.abs(stft)

# STFT 스펙트로그램 시각화
plt.figure(figsize=(14, 5))
librosa.display.specshow(spectrogram, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='log')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.colorbar(format='%2.0f dB')
plt.title('Spectrogram')
plt.show()

# 로그 스펙트로그램 생성 및 시각화
log_spectrogram = librosa.amplitude_to_db(spectrogram)
plt.figure(figsize=(14, 5))
librosa.display.specshow(log_spectrogram, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='log')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.colorbar(format='%2.0f dB')
plt.title('Log-Scaled Spectrogram')
plt.show()
