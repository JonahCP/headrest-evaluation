# import datetime
import os
import sys

import pygame


### Import LOOP packages and functions
dirP = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
# #print(dirP + '/4_ref_other')
sys.path.append(dirP + '/headrest-evaluation/z1_ref_other/0_lib')

import cnbiloop
from cnbiloop import BCI, BCI_tid

sys.path.append(dirP + '/1_packages')
from serialCommunication import SerialWriter

def sendTiD(Event_):
    bci.id_msg_bus.SetEvent(Event_)
    bci.iDsock_bus.sendall(str.encode(bci.id_serializer_bus.Serialize()))

bci = BCI_tid.BciInterface()


### Define TiD values
START       = 1     # Beginning of ERP trials
END         = 1     # End of ERP trials
STANDARD    = 10    # Standard stimulus
TARGET      = 20    # Target stimulus
KEY_PRESS   = 30    # Key pressed


### Main method 
blank = 'images/blank.png'
square = 'images/square.png'
circle = 'images/circle.png'
imgs = [square, blank, square, blank, circle, blank, square, blank]

pygame.init()
win = pygame.display.set_mode((0, 0), pygame.FULLSCREEN, vsync=1)
pygame.display.set_caption('ERP')
clock = pygame.time.Clock()
pygame.mouse.set_visible(False)

loaded_imgs = []
for i in imgs:
    img = pygame.image.load(i).convert()
    loaded_imgs.append(img)

sendTiD(START)
img_at = 0
running = True
# start = datetime.datetime.now()
counter = 0
while running:
    elapsed = clock.tick(60)
    counter += 1
    if counter == 30:
        win.blit(loaded_imgs[img_at], (0, 0))
        pygame.display.update()
        img_at = (img_at + 1) % len(imgs)
        counter = 0

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            keys = pygame.key.get_pressed()
            if event.type == pygame.QUIT or keys[pygame.K_ESCAPE]:
                running = False
            if event.key == pygame.K_SPACE:
                # print("Space Key Press at %d", datetime.datetime.now())
                sendTiD(KEY_PRESS)
sendTiD(END)
