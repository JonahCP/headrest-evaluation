import sys

from PIL import Image, ImageTk
import pygame
import time

# Define frequencies, duration, and other experimental constants
FREQ = [7.5, 8.57, 10, 12, 15] # in Hz
STIMULUS_DURATION = 4000   # in ms
REST_DURATION = 2000       # in ms

blank = '../images/blank.png'
circle = '../images/circle.png'
imgs = [blank, circle]

pygame.init()
win = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption('SSVEP')
clock = pygame.time.Clock()
pygame.mouse.set_visible(False)

loaded_imgs = []
for i in imgs:
    img = pygame.image.load(i).convert()
    loaded_imgs.append(img)

img_at = 0
for freq in FREQ:
    print('Beginning %.2f Hz: ' % (freq) + time.strftime('%Y-%m-%d %H:%M:%S'))
    duration = STIMULUS_DURATION
    clock.tick(freq * 2)

    while duration > 0:
        win.blit(loaded_imgs[img_at], (0, 0))
        pygame.display.update()
        img_at = (img_at + 1) % len(imgs)

        duration -= clock.tick(freq * 2)

    win.blit(loaded_imgs[0], (0, 0))
    pygame.display.update()
    print('Ending %.2f Hz: ' % (freq) + time.strftime('%Y-%m-%d %H:%M:%S'))

    pygame.time.delay(REST_DURATION)
