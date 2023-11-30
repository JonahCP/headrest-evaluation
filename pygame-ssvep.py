import os
from random import shuffle
import sys
# import time

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
STIMULUS_DURATION = 4000   # in ms
REST_DURATION = 2000       # in ms


### Define TiD values
START = 99  # Beginning of SSVEP trials
END = 199   # End of SSVEP trials


### Define functions
def flicker():
    # print('--- Trial Start ---')
    sendTiD(START)
    pygame.time.delay(REST_DURATION)
    img_at = 0
    freq_rand = FREQ.copy()
    shuffle(freq_rand)
    for freq in freq_rand:
        # print('Beginning %.2f Hz: ' % (freq) + time.strftime('%Y-%m-%d %H:%M:%S'))
        sendTiD(FREQ.index(freq) + 100)
        duration = STIMULUS_DURATION
        clock.tick(freq * 2)

        while duration > 0:
            win.blit(loaded_imgs[img_at], (0, 0))
            pygame.display.update()
            img_at = (img_at + 1) % len(imgs)

            duration -= clock.tick(freq * 2)

        win.blit(loaded_imgs[0], (0, 0))
        pygame.display.update()
        # print('Ending %.2f Hz: ' % (freq) + time.strftime('%Y-%m-%d %H:%M:%S'))
        sendTiD(FREQ.index(freq) + 100)

        pygame.time.delay(REST_DURATION)
    # print('--- Trial Ended ---')
    sendTiD(END)

def wait_for_input():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                return


### Main method
blank = 'images/blank.png'
circle = 'images/circle.png'
imgs = [blank, circle]

pygame.init()
win = pygame.display.set_mode((0, 0), pygame.FULLSCREEN, vsync=1)
pygame.display.set_caption('SSVEP')
clock = pygame.time.Clock()
pygame.mouse.set_visible(False)

loaded_imgs = []
for i in imgs:
    img = pygame.image.load(i).convert()
    loaded_imgs.append(img)

win.blit(loaded_imgs[0], (0, 0))
pygame.display.update()

while True:
    wait_for_input()
    flicker()