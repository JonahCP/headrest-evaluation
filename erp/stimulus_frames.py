import sys

from PIL import Image, ImageTk
import pygame

blank = '../images/blank.png'
square = '../images/square.png'
circle = '../images/circle.png'
imgs = [square, blank, square, blank, circle, blank, square, blank]

pygame.init()
win = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption('ERP')
clock = pygame.time.Clock()

loaded_imgs = []
for i in imgs:
    img = pygame.image.load(i).convert()
    loaded_imgs.append(img)

elapsed_time = 0
img_at = 0
win.blit(loaded_imgs[img_at], (0, 0))
pygame.display.flip()

img_at = 1
while True:
    dt = clock.tick()
    elapsed_time += dt
    if elapsed_time > 500:
        win.blit(loaded_imgs[img_at], (0, 0))
        pygame.display.flip()
        if img_at == 7:
            img_at = 0
        else:
            img_at += 1
        elapsed_time = 0
