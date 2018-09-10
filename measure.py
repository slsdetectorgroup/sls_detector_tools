import time
from sls_detector_tools.keithley import SerialInstrument, PicoamMeter
from sls_detector_tools import BigXrayBox

import numpy as np
import matplotlib.pyplot as plt


b = BigXrayBox()
b.unlock()
s = PicoamMeter()
s.open_port()
s.configure()


s.voltage = -1.0
s.output = True

#mA = np.arange(30,41,5)
mA = np.array([40,40,40])
I = np.zeros(mA.size)

b.open_shutter('Direct beam')
print(b.shutter_status)
plt.figure()
for v in [60]:
    b.voltage = v
    time.sleep(0)
    for i,c in enumerate(mA):
        b.current = c
        while b.current != c:
            time.sleep(0.1)
        print('Starting:', b.current)

        I[i] = s.current

        print(c, I[i])

    plt.plot(np.arange(mA.size), I, 'o-')

b.close_shutter('Direct beam')

print(I.mean())

#plt.show()
#x = [0,25,50,75,100,125,200]
#y = [3.1, 2.37, 1.96, 1.64, 1.37,1.16, 0.56]
#plt.figure()
#plt.plot(x,y, 'o-')
