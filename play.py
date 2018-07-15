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
plt.figure()

N = 20
s.output = True
V = np.zeros(N)
I = np.zeros(N)

b.open_shutter('Direct beam')
print(b.shutter_status)
for c in [20, 30, 40]:
    b.current = c
    time.sleep(10)
    for i, v in enumerate(np.linspace(-1, 0.1, N)):
        s.voltage = v
        time.sleep(1)
        I[i] = s.current
        V[i] = s.voltage

        print(i, I[i], V[i])
    plt.plot(V, I, label = 'beam: {}mA'.format(c))

b.close_shutter('Direct beam')
for i, v in enumerate(np.linspace(-1, 0.1, N)):
    s.voltage = v
    time.sleep(0.1)
    I[i] = s.current
    V[i] = s.voltage

    print(i, I[i], V[i])
plt.plot(V, I, label = 'no beam')
plt.legend()
plt.show()