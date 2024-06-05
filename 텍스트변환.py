import serial
import time

py_serial = serial.Serial(port="COM11", baudrate=9600)
time.sleep(3)

f = open('exhibition\\test_3.txt', 'a')
a = True
print("석섹스")
while a:
    if py_serial.readable():
        response = py_serial.readline()
        f.write(str(response.decode('utf-8')).strip() + '\n') 
    else:
        a = False

f.close()
