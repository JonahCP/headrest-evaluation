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
pygame.mouse.set_visible(False)

loaded_imgs = []
for i in imgs:
    img = pygame.image.load(i).convert()
    loaded_imgs.append(img)

img_at = 0
while True:
    elapsed = clock.tick(2)
    # print(elapsed)
    
    win.blit(loaded_imgs[img_at], (0, 0))
    pygame.display.update()

    img_at = (img_at + 1) % len(imgs)
