import datetime
import sys

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
running = True
start = datetime.datetime.now()
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
                print("Space Key Press at %d", datetime.datetime.now())
