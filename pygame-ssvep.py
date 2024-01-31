import os
from random import shuffle
import sys

from PIL import Image, ImageTk
import pygame


### Import LOOP packages and functions
dirP = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(dirP + '/headrest-evaluation/z1_ref_other/0_lib')

import cnbiloop
from cnbiloop import BCI, BCI_tid

sys.path.append(dirP + '/ssvep/1_packages')
from serialCommunication import SerialWriter

def sendTiD(Event_):
    bci.id_msg_bus.SetEvent(Event_)
    bci.iDsock_bus.sendall(str.encode(bci.id_serializer_bus.Serialize()))

bci = BCI_tid.BciInterface()


### Define frequencies, duration, and other experimental constants
FREQ = [7.5, 8.57, 10, 12] # in Hz
STIMULUS_DURATION = 8000   # in ms
REST_DURATION = 2000       # in ms

GREY = (140, 140, 140)


### Set up PyGame window
pygame.init()
win = pygame.display.set_mode((0, 0), pygame.FULLSCREEN, vsync=1)
win.fill(GREY)   # set background color to grey
pygame.display.update()
pygame.display.set_caption('SSVEP')
pygame.mouse.set_visible(False)
clock = pygame.time.Clock()

### Set up images
ch1 = 'images/checker1.png'
ch2 = 'images/checker2.png'
imgs = [ch1, ch2]

boards = []
for i in imgs:
    img = pygame.image.load(i).convert()
    boards.append(img)

# Find coordinates to render image
x, y = win.get_size()
im_x, im_y = boards[0].get_size()
x = (x / 2) - (im_x / 2)
y = (y / 2) - (im_y / 2)

### Begin flickering stimuli
sendTiD(1)

pygame.time.delay(2000)
freq_rand = FREQ.copy()
shuffle(freq_rand)
for freq in freq_rand:
    print("Beginning %.2f Hz" % freq)
    duration = STIMULUS_DURATION
    clock.tick(freq * 2)

    sendTiD(FREQ.index(freq) + 10)
    i = 0
    while duration > 0:
        win.blit(boards[i], (x, y))
        pygame.display.update()
        i = (i + 1) % 2

        duration -= clock.tick(freq * 2)

    sendTiD(FREQ.index(freq) + 10)
    pygame.draw.rect(win, GREY, pygame.Rect(x, y, im_x, im_y))
    pygame.display.update()
    pygame.time.delay(REST_DURATION)

sendTiD(1)