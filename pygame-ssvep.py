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

### Define frequencies, duration, and other experimental constants
FREQ = [7.5, 8.57, 10, 12] # in Hz
STIMULUS_DURATION = 8000   # in ms
REST_DURATION = 15000      # in ms

### Define color and shape constants
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
RADIUS = 150


### Set up PyGame window
pygame.init()
win = pygame.display.set_mode((0, 0), pygame.FULLSCREEN, vsync=1)
win.fill(BLACK)         # set background color to black
pygame.display.update()
pygame.display.set_caption('SSVEP')
pygame.mouse.set_visible(False)
clock = pygame.time.Clock()

# Find coordinates to render image
pos_x, pos_y = win.get_size()
pos_x //= 2
pos_y //= 2

### Begin flickering stimuli
logEvent('START')

pygame.time.delay(5000)
freq_rand = FREQ.copy()
shuffle(freq_rand)
for freq in freq_rand:
    duration = STIMULUS_DURATION
    clock.tick(freq * 2)

    on_screen = False
    logEvent('%.2f Hz' % freq)
    while duration > 0:
        if not on_screen:
            pygame.draw.circle(win, YELLOW, (pos_x, pos_y), RADIUS)
        else:
            pygame.draw.circle(win, BLACK, (pos_x, pos_y), RADIUS)
        on_screen = not on_screen
        pygame.display.update()

        duration -= clock.tick(freq * 2)
    
    if on_screen:
        pygame.draw.circle(win, BLACK, (pos_x, pos_y), RADIUS)
        pygame.display.update()

    pygame.display.update()
    pygame.time.delay(REST_DURATION)

logEvent('END')

### Convert timestamp lists to CSV file
timestamps = DataFrame({
        'datetimes': datetimes,
        'events': events
    })

timestamps.to_csv('ssvep_timestamps_' + datetime.now().strftime('%Y-%m-%d_%H.%M.%S.%f')[:-7] + '.csv')