import socket
import time
import sys, string, os


CM_PATH = "/opt/ipg/carmaker/linux64/bin/CM"
TCP_IP = 'localhost'
TCP_PORT = 16660 
BUFFER_SIZE = 1024

os.system("%s -cmdport %s &" % (CM_PATH,TCP_PORT)) 

time.sleep(2)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))


MESSAGE = "LoadTestRun testrun2\rQuantSubscribe {Time}\rStartSim\r"
MESSAGE = MESSAGE.encode('utf-8')
s.send(MESSAGE)
time.sleep(500)


MESSAGE = "expr {$Qu(Time)*1000}\rStopSim\r"
MESSAGE = MESSAGE.encode('utf-8')
s.send(MESSAGE)
time.sleep(1)
s_string_val = s.recv(200)
# Ausgabe des empfangenen Wertes in Python
print (s_string_val)

# Schliessen des Ports
s.close()
