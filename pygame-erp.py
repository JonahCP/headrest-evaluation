from datetime import datetime
from pandas import DataFrame
import pygame
from random import shuffle


### Define timestamping function
datetimes = []
events = []
def logEvent(event):
    datetimes.append(datetime.now().strftime('%Y-%m-%d %H.%M.%S.%f')[:-3])
    events.append(event)

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
                logEvent('KEY PRESS')
        duration -= clock.tick(60)


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
    shuffle(stims)
    sequence.extend(stims)

logEvent('START')

pygame.time.delay(5000)
for trial in sequence:
    if trial:
        pygame.draw.circle(win, YELLOW, (pos_x, pos_y), RADIUS)
        logEvent('TARGET')    
    else:
        pygame.draw.rect(win, YELLOW, pygame.Rect(pos_x - RADIUS, pos_y - RADIUS, RADIUS * 2, RADIUS * 2))
        logEvent('STANDARD')    
    pygame.display.update()
    awaitInput(TRIAL_DURATION)

    pygame.draw.rect(win, BLACK, pygame.Rect(pos_x - RADIUS, pos_y - RADIUS, RADIUS * 2, RADIUS * 2))
    pygame.display.update()
    awaitInput(REST_DURATION)

logEvent('END')

### Convert timestamp lists to CSV file
event_timestamps = DataFrame({
        'datetimes': datetimes,
        'events': events
    })

event_timestamps.to_csv('erp_timestamps_' + datetime.now().strftime('%Y-%m-%d_%H.%M.%S.%f')[:-7] + '.csv')
