from datetime import datetime
import pandas as pd
import os
import sys
import random
import pygame


### Import LOOP packages and functions
dirP = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
#print(dirP + '/4_ref_other')
sys.path.append(dirP + '/headrest-evaluation/z1_ref_other/0_lib')

# import cnbiloop
# from cnbiloop import BCI, BCI_tid

sys.path.append(dirP + '/1_packages')
from serialCommunication import SerialWriter

# def sendTiD(Event_):
#     bci.id_msg_bus.SetEvent(Event_)
#     bci.iDsock_bus.sendall(str.encode(bci.id_serializer_bus.Serialize()))

# bci = BCI_tid.BciInterface()


### Define TiD values
START       = 1     # Beginning of ERP trials
END         = 1     # End of ERP trials
STANDARD    = 10    # Standard stimulus
TARGET      = 20    # Target stimulus
KEY_PRESS   = 30    # Key pressed

### Define duration and other experimental constants
TRIAL_DURATION = 500    # trial in seconds
REST_DURATION = 1000    # rest in seconds

### Define shape constants
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
RADIUS = 150

### Define functions
def awaitInput(duration):
    while duration > 0:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                # print("Space key pressed")
                sendTiD(KEY_PRESS)
        duration -= clock.tick(60)


datetimes = []
events = []
def logEvent(event):
    datetimes.append(datetime.utcnow().strftime('%Y-%m-%d %H.%M.%S.%f')[:-3])
    events.append(event)

### Set up PyGame window 
pygame.init()
win = pygame.display.set_mode((0, 0), pygame.FULLSCREEN, vsync=1)
win.fill(BLACK)
pygame.display.update()
pygame.display.set_caption('ERP')
pygame.mouse.set_visible(False)
clock = pygame.time.Clock()

# Find coordinates to render image
pos_x, pos_y = win.get_size()
pos_x //= 2
pos_y //= 2

### Generate stimulus sequence
stims = [True, True, False, False, False, False, False, False, False, False]
sequence = []
for i in range(10):
    random.shuffle(stims)
    sequence.extend(stims)

logEvent('START PROTOCOL')
# sendTiD(START)

pygame.time.delay(5000)
for trial in sequence:
    if trial:
        pygame.draw.circle(win, YELLOW, (pos_x, pos_y), RADIUS)
        logEvent('TARGET')    
        # sendTiD(TARGET)
    else:
        pygame.draw.rect(win, YELLOW, pygame.Rect(pos_x - RADIUS, pos_y - RADIUS, RADIUS * 2, RADIUS * 2))
        logEvent('STANDARD')    
        # sendTiD(STANDARD)
    pygame.display.update()
    awaitInput(TRIAL_DURATION)

    pygame.draw.rect(win, BLACK, pygame.Rect(pos_x - RADIUS, pos_y - RADIUS, RADIUS * 2, RADIUS * 2))
    pygame.display.update()
    awaitInput(REST_DURATION)

# sendTiD(END)
logEvent('END PROTOCOL')    
    
event_timestamps = pd.DataFrame(
    {'datetimes': datetimes,
     'events': events
     })

### uncomment for initial headrest testing trials
event_timestamps.to_csv('erp_timestamps_' + datetime.utcnow().strftime('%Y-%m-%d_%H.%M.%S.%f')[:-3])
