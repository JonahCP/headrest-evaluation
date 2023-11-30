import os
import sys
from tkinter import Tk, Canvas
import time


### Import LOOP packages and functions
dirP = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(dirP + '/headrest-evaluation/z1_ref_other/0_lib')
print(sys.path)

import cnbiloop
from cnbiloop import BCI, BCI_tid

sys.path.append(dirP + '/1_packages')
from serialCommunication import SerialWriter

def sendTiD(Event_):
    bci.id_msg_bus.SetEvent(Event_)
    bci.iDsock_bus.sendall(str.encode(bci.id_serializer_bus.Serialize()))

bci = BCI_tid.BciInterface()


### Define frequencies, duration, and other experimental constants
FREQ = [7.5, 8.57, 10, 12] # in Hz
STIMULUS_DURATION = 8   # in seconds
REST_DURATION = 20      # in seconds


### Define circle constants
COLOR = 'yellow'
RADIUS = 150


### Define functions
"""
Flickers a COLOR circle of radius RADIUS at a defined frequency

@param freq: frequency (in Hz) to flicker circle at
""" 
def flicker(freq):
    print('Beginning %.1f Hz: ' % (freq) + time.strftime('%Y-%m-%d %H:%M:%S'))
    duration = STIMULUS_DURATION
    period = 1 / freq
    total_time = 0
    cycles = 0
    while duration > 0:
        start = time.time()
        present = canvas.find_withtag('circle')
        if not present:
            canvas.create_oval(circle_x - RADIUS, circle_y - RADIUS,
                               circle_x + RADIUS, circle_y + RADIUS,
                               fill = COLOR, tags = 'circle')
        else:
            canvas.delete('circle')
        root.update()

        # Hold the frame for T seconds
        time.sleep((period / 2 - (time.time() - start)))
        duration -= (period / 2)
        cycles += 0.5
        total_time += (time.time() - start)
    if canvas.find_withtag('circle'):
        canvas.delete('circle')
        root.update()
    print('Average freq: %.3f' % (cycles / total_time))
    print('Ending %.1f Hz: ' % (freq) + time.strftime('%Y-%m-%d %H:%M:%S'))

"""
Rests at black screen for REST_DURATION seconds
""" 
def rest():
    print()
    time.sleep(REST_DURATION)


### Setup Tkinter window
root = Tk()
root.title('SSVEP')
root.config(cursor='none')
root.attributes('-fullscreen', True)

# Setup screen for drawing
canvas_width = root.winfo_screenwidth()
canvas_height = root.winfo_screenheight()
canvas = Canvas(root, width = canvas_width, height = canvas_height, bg='black')
canvas.pack()
root.update()

# Find center of the screen
circle_x = canvas.winfo_width() // 2
circle_y = canvas.winfo_height() // 2


### Begin flickering
print('Starting program:', time.strftime('%Y-%m-%d %H:%M:%S'))  # Output current timestamp

# Randomize order of frequencies presented
import random
freq_rand = FREQ.copy()
random.shuffle(freq_rand)

sendTiD(1)
time.sleep(2)
for freq in freq_rand:
    sendTiD(FREQ.index(freq) + 10)
    flicker(freq)       # Stimulation period
    sendTiD(FREQ.index(freq) + 10)
    rest()              # Rest period
sendTiD(1)
