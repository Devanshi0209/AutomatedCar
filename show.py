import pygame
import cv2
import numpy as np

pygame.init()
gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('Capstone Project')

black = (0,0,0)
white = (255,255,255)

clock = pygame.time.Clock()
crashed = False
#carImg = pygame.image.load('D:\WindowsNoEditor\PythonAPI\examples\dataset\out00745296.jpg')



x =  0
y = 0
count=745296
while not crashed:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            crashed = True


    gameDisplay.fill(white)
    carstring='D:\WindowsNoEditor\PythonAPI\examples\dataset\out%08d.jpg' % count
    imagearray=cv2.imread(carstring)
    imagearray=np.array(imagearray)
    carImg=pygame.surfarray.make_surface(imagearray)
    rotated_image = pygame.transform.rotate(carImg, -90)
    gameDisplay.blit(rotated_image, (x,y))
    pygame.display.update()
    clock.tick(60)
    count+=1

pygame.quit()
quit()