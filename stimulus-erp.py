from tkinter import Tk, Canvas
import time
import os
import random
import sys


### Import LOOP packages and functions
dirP = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(dirP + '/z1_ref_other/0_lib')

import cnbiloop
from cnbiloop import BCI, BCI_tid

sys.path.append(dirP + '/1_packages')
from serialCommunication import SerialWriter

def sendTiD(Event_):
    bci.id_msg_bus.SetEvent(Event_)
    bci.iDsock_bus.sendall(str.encode(bci.id_serializer_bus.Serialize()))

bci = BCI_tid.BciInterface()


### Define duration and other experimental constants
DURATION = 0.5      # in seconds (for both stimulus and rest periods)

### Define shape constants
COLOR = 'yellow'
RADIUS = 150


### Define functions
def draw_circle():
    canvas.create_oval(pos_x - RADIUS, pos_y - RADIUS,
                       pos_x + RADIUS, pos_y + RADIUS, 
                       fill = COLOR, tags = 'shape')
    
def draw_square():
    canvas.create_rectangle(pos_x - RADIUS, pos_y - RADIUS, 
                            pos_x + RADIUS, pos_y + RADIUS, 
                            fill = COLOR, tags = 'shape')

def flicker():
    rand = random.randint(1, 9)
    i = 0
    while True:
        present = canvas.find_withtag('shape')
        if not present:
            if i == rand:
                draw_circle()
                sendTiD(10)
                i = 0
                rand = random.randint(1, 9)
            else:
                draw_square()
                sendTiD(20)
                i += 1
        else:
            canvas.delete('shape')
        root.update()

        # Hold the frame
        time.sleep(DURATION)
        

### Setup Tkinter window
root = Tk()
root.config(cursor='none')
root.title('SSVEP')
root.attributes('-fullscreen', True)

### Setup screen for drawing
canvas_width = root.winfo_screenwidth()
canvas_height = root.winfo_screenheight()
canvas = Canvas(root, width = canvas_width, height = canvas_height, bg='black')
canvas.pack()
root.update()

### Find center of the screen
pos_x = canvas.winfo_width() // 2
pos_y = canvas.winfo_height() // 2

### Begin stimuli
print('Starting program timestamp:', time.strftime('%Y-%m-%d %H:%M:%S'))  # Output current timestamp
sendTiD(1)
flicker()
sendTiD(1)
