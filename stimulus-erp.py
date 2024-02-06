from tkinter import Tk, Canvas
from datetime import datetime
import pandas as pd
import time
import os
import random
import sys

### Import LOOP packages and functions
dirP = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(dirP + '/headrest-evaluation/z1_ref_other/0_lib')

import cnbiloop
from cnbiloop import BCI, BCI_tid

sys.path.append(dirP + '/1_packages')
from serialCommunication import SerialWriter

def sendTiD(Event_):
    bci.id_msg_bus.SetEvent(Event_)
    bci.iDsock_bus.sendall(str.encode(bci.id_serializer_bus.Serialize()))


bci = BCI_tid.BciInterface()

### Define duration and other experimental constants
TRIAL_DURATION = 0.5  # trial in seconds
REST_DURATION = 1  # rest in seconds

### Define shape constants
COLOR = 'yellow'
RADIUS = 150


### Define functions
def draw_circle():
    canvas.create_oval(pos_x - RADIUS, pos_y - RADIUS,
                       pos_x + RADIUS, pos_y + RADIUS,
                       fill=COLOR, tags='shape')


def draw_square():
    canvas.create_rectangle(pos_x - RADIUS, pos_y - RADIUS,
                            pos_x + RADIUS, pos_y + RADIUS,
                            fill=COLOR, tags='shape')


### Setup Tkinter window
root = Tk()
root.config(cursor='none')
root.title('SSVEP')
root.attributes('-fullscreen', True)

### Setup screen for drawing
canvas_width = root.winfo_screenwidth()
canvas_height = root.winfo_screenheight()
canvas = Canvas(root, width=canvas_width, height=canvas_height, bg='black')
canvas.pack()
root.update()

### Find center of the screen
pos_x = canvas.winfo_width() // 2
pos_y = canvas.winfo_height() // 2

### Generate stimulus sequence
stims = [True, True, False, False, False, False, False, False, False, False]
sequence = []
for i in range(10):
    random.shuffle(stims)
    sequence.extend(stims)

datetimes = []
events = []

### Begin stimuli
print('Starting program: ', datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])  # Output current timestamp
datetimes.append(datetime.utcnow().strftime('%Y-%m-%d %H.%M.%S.%f')[:-3])
events.append('ERP start')
sendTiD(1)

for trial in sequence:
    # Present stimulus
    if trial:
        draw_circle()
        datetimes.append(datetime.utcnow().strftime('%Y-%m-%d %H.%M.%S.%f')[:-3])
        events.append('target')
        sendTiD(10)
    else:
        draw_square()
        datetimes.append(datetime.utcnow().strftime('%Y-%m-%d %H.%M.%S.%f')[:-3])
        events.append('base')
        sendTiD(20)
    root.update()
    time.sleep(TRIAL_DURATION)

    # Inter-trial-interval (black screen)
    canvas.delete('shape')
    root.update()
    time.sleep(REST_DURATION)

# sendTiD(1)
event_timestamps = pd.DataFrame(
    {'datetimes': datetimes,
     'events': events
     })

### uncomment for initial headrest testing trials
# event_timestamps.to_csv('erp_timestamps_' + datetime.utcnow().strftime('%Y-%m-%d_%H.%M.%S.%f')[:-3])
